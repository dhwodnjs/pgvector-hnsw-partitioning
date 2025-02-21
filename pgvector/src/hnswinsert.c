#include "postgres.h"

#include <math.h>

#include "access/generic_xlog.h"
#include "hnsw.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/datum.h"
#include "utils/memutils.h"

/*
 * Get the insert page
 */
static BlockNumber
GetInsertPage(Relation index)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;
	BlockNumber insertPage;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	insertPage = metap->insertPage;

	UnlockReleaseBuffer(buf);

	return insertPage;
}

/*
 * Check for a free offset
 */
static bool
HnswFreeOffset(Relation index, Buffer buf, Page page, HnswElement element, Size etupSize, Size ntupSize, Buffer *nbuf, Page *npage, OffsetNumber *freeOffno, OffsetNumber *freeNeighborOffno, BlockNumber *newInsertPage, uint8 *tupleVersion)
{
	OffsetNumber offno;
	OffsetNumber maxoffno = PageGetMaxOffsetNumber(page);

	for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
	{
		ItemId		eitemid = PageGetItemId(page, offno);
		HnswElementTuple etup = (HnswElementTuple) PageGetItem(page, eitemid);

		/* Skip neighbor tuples */
		if (!HnswIsElementTuple(etup))
			continue;

		if (etup->deleted)
		{
			BlockNumber elementPage = BufferGetBlockNumber(buf);
			BlockNumber neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
			OffsetNumber neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
			ItemId		nitemid;
			Size		pageFree;
			Size		npageFree;

			if (!BlockNumberIsValid(*newInsertPage))
				*newInsertPage = elementPage;

			if (neighborPage == elementPage)
			{
				*nbuf = buf;
				*npage = page;
			}
			else
			{
				*nbuf = ReadBuffer(index, neighborPage);
				LockBuffer(*nbuf, BUFFER_LOCK_EXCLUSIVE);

				/* Skip WAL for now */
				*npage = BufferGetPage(*nbuf);
			}

			nitemid = PageGetItemId(*npage, neighborOffno);

			/* Ensure aligned for space check */
			Assert(etupSize == MAXALIGN(etupSize));
			Assert(ntupSize == MAXALIGN(ntupSize));

			/*
			 * Calculate free space individually since tuples are overwritten
			 * individually (in separate calls to PageIndexTupleOverwrite)
			 */
			pageFree = ItemIdGetLength(eitemid) + PageGetExactFreeSpace(page);
			npageFree = ItemIdGetLength(nitemid);
			if (neighborPage != elementPage)
				npageFree += PageGetExactFreeSpace(*npage);
			else if (pageFree >= etupSize)
				npageFree += pageFree - etupSize;

			/* Check for space */
			if (pageFree >= etupSize && npageFree >= ntupSize)
			{
				*freeOffno = offno;
				*freeNeighborOffno = neighborOffno;
				*tupleVersion = etup->version;
				return true;
			}
			else if (*nbuf != buf)
				UnlockReleaseBuffer(*nbuf);
		}
	}

	return false;
}

/*
 * Add a new page
 */
static void
HnswInsertAppendPage(Relation index, Buffer *nbuf, Page *npage, GenericXLogState *state, Page page, bool building)
{
	/* Add a new page */
	LockRelationForExtension(index, ExclusiveLock);
	*nbuf = HnswNewBuffer(index, MAIN_FORKNUM);
	UnlockRelationForExtension(index, ExclusiveLock);

	/* Init new page */
	if (building)
		*npage = BufferGetPage(*nbuf);
	else
		*npage = GenericXLogRegisterBuffer(state, *nbuf, GENERIC_XLOG_FULL_IMAGE);

	HnswInitPage(*nbuf, *npage);

	/* Update previous buffer */
	HnswPageGetOpaque(page)->nextblkno = BufferGetBlockNumber(*nbuf);
//    elog(WARNING, "new page blkno: %d", (int)HnswPageGetOpaque(page)->nextblkno);
}

/*
 * Add to element and neighbor pages
 */
static void
AddElementOnDisk(Relation index, HnswElement e, int m, BlockNumber insertPage, BlockNumber *updatedInsertPage, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	Size		etupSize;
	Size		ntupSize;
	Size		combinedSize;
	Size		maxSize;
	Size		minCombinedSize;
	HnswElementTuple etup;
	BlockNumber currentPage = insertPage;
	HnswNeighborTuple ntup;
	Buffer		nbuf;
	Page		npage;
	OffsetNumber freeOffno = InvalidOffsetNumber;
	OffsetNumber freeNeighborOffno = InvalidOffsetNumber;
	BlockNumber newInsertPage = InvalidBlockNumber;
	uint8		tupleVersion;
	char	   *base = NULL;

	/* Calculate sizes */
	etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(HnswPtrAccess(base, e->value)));
	ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(e->level, m);
	combinedSize = etupSize + ntupSize + sizeof(ItemIdData);
	maxSize = HNSW_MAX_SIZE;
	minCombinedSize = etupSize + HNSW_NEIGHBOR_TUPLE_SIZE(0, m) + sizeof(ItemIdData);

	/* Prepare element tuple */
	etup = palloc0(etupSize);
	HnswSetElementTuple(base, etup, e);

	/* Prepare neighbor tuple */
	ntup = palloc0(ntupSize);
	HnswSetNeighborTuple(base, ntup, e, m);

	/* Find a page (or two if needed) to insert the tuples */
	for (;;)
	{
		buf = ReadBuffer(index, currentPage);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

		if (building)
		{
			state = NULL;
			page = BufferGetPage(buf);
		}
		else
		{
			state = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(state, buf, 0);
		}

		/* Keep track of first page where element at level 0 can fit */
		if (!BlockNumberIsValid(newInsertPage) && PageGetFreeSpace(page) >= minCombinedSize)
			newInsertPage = currentPage;

		/* First, try the fastest path */
		/* Space for both tuples on the current page */
		/* This can split existing tuples in rare cases */
		if (PageGetFreeSpace(page) >= combinedSize)
		{
			nbuf = buf;
			npage = page;
			break;
		}

		/* Next, try space from a deleted element */
		if (HnswFreeOffset(index, buf, page, e, etupSize, ntupSize, &nbuf, &npage, &freeOffno, &freeNeighborOffno, &newInsertPage, &tupleVersion))
		{
			if (nbuf != buf)
			{
				if (building)
					npage = BufferGetPage(nbuf);
				else
					npage = GenericXLogRegisterBuffer(state, nbuf, 0);
			}

			/* Set tuple version */
			etup->version = tupleVersion;
			ntup->version = tupleVersion;

			break;
		}

		/* Finally, try space for element only if last page */
		/* Skip if both tuples can fit on the same page */
		if (combinedSize > maxSize && PageGetFreeSpace(page) >= etupSize && !BlockNumberIsValid(HnswPageGetOpaque(page)->nextblkno))
		{
			HnswInsertAppendPage(index, &nbuf, &npage, state, page, building);
			break;
		}

		currentPage = HnswPageGetOpaque(page)->nextblkno;

		if (BlockNumberIsValid(currentPage))
		{
			/* Move to next page */
			if (!building)
				GenericXLogAbort(state);
			UnlockReleaseBuffer(buf);
		}
		else
		{
			Buffer		newbuf;
			Page		newpage;

			HnswInsertAppendPage(index, &newbuf, &newpage, state, page, building);

			/* Commit */
			if (building)
				MarkBufferDirty(buf);
			else
				GenericXLogFinish(state);

			/* Unlock previous buffer */
			UnlockReleaseBuffer(buf);

			/* Prepare new buffer */
			buf = newbuf;
			if (building)
			{
				state = NULL;
				page = BufferGetPage(buf);
			}
			else
			{
				state = GenericXLogStart(index);
				page = GenericXLogRegisterBuffer(state, buf, 0);
			}

			/* Create new page for neighbors if needed */
			if (PageGetFreeSpace(page) < combinedSize)
				HnswInsertAppendPage(index, &nbuf, &npage, state, page, building);
			else
			{
				nbuf = buf;
				npage = page;
			}

			break;
		}
	}

	e->blkno = BufferGetBlockNumber(buf);
	e->neighborPage = BufferGetBlockNumber(nbuf);

	/* Added tuple to new page if newInsertPage is not set */
	/* So can set to neighbor page instead of element page */
	if (!BlockNumberIsValid(newInsertPage))
		newInsertPage = e->neighborPage;

	if (OffsetNumberIsValid(freeOffno))
	{
		e->offno = freeOffno;
		e->neighborOffno = freeNeighborOffno;
	}
	else
	{
		e->offno = OffsetNumberNext(PageGetMaxOffsetNumber(page));
		if (nbuf == buf)
			e->neighborOffno = OffsetNumberNext(e->offno);
		else
			e->neighborOffno = FirstOffsetNumber;
	}

	ItemPointerSet(&etup->neighbortid, e->neighborPage, e->neighborOffno);

	/* Add element and neighbors */
	if (OffsetNumberIsValid(freeOffno))
	{
		if (!PageIndexTupleOverwrite(page, e->offno, (Item) etup, etupSize))
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

		if (!PageIndexTupleOverwrite(npage, e->neighborOffno, (Item) ntup, ntupSize))
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
	}
	else
	{
		if (PageAddItem(page, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != e->offno)
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

		if (PageAddItem(npage, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != e->neighborOffno)
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
	}

	/* Commit */
	if (building)
	{
		MarkBufferDirty(buf);
		if (nbuf != buf)
			MarkBufferDirty(nbuf);
	}
	else
		GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
	if (nbuf != buf)
		UnlockReleaseBuffer(nbuf);

	/* Update the insert page */
	if (BlockNumberIsValid(newInsertPage) && newInsertPage != insertPage)
		*updatedInsertPage = newInsertPage;
}


static int
ComparePageNeighbors(const void *a, const void *b)
{

//    HnswPageNeighborCount *pageA = (HnswPageNeighborCount *)a;
//    HnswPageNeighborCount *pageB = (HnswPageNeighborCount *)b;
//
//    // 1. neighborCount 비교 (내림차순)
//    if (pageB->neighborCount != pageA->neighborCount)
//        return pageB->neighborCount - pageA->neighborCount;
//
//    // 2. neighborCount가 동일할 경우 blkno 비교 (오름차순)
//    return pageA->blkno - pageB->blkno;
    return ((HnswPageNeighborCount *)b)->neighborCount - ((HnswPageNeighborCount *)a)->neighborCount;
}


/* 페이지 neighbor count 분석 */
HnswInsertPageCandidate
CalculatePartitionNeighborCount(HnswElement element)
{
//    HnswInsertPageCandidate *pageCandidates;
//    int maxPages = HNSW_DEFAULT_M * 2 + MAX_INSERT_POOL_SIZE;  /* Assume max pages for now */ // m*2 + MAX_INSERT_POOL
//    MemoryContext oldCtx;
    char *base = NULL;

    HnswInsertPageCandidate pageCandidates = palloc(sizeof(HnswInsertPageCandidateData));
    pageCandidates->length = 0;

    HnswNeighborArray *neighbors = HnswGetNeighbors(base, element, 0);

    // by partitionid
    for (int i = 0; i < neighbors->length; i++) {

        HnswCandidate *hc = &neighbors->items[i];
        HnswElement neighborElement = HnswPtrAccess(base, hc->element);
        BlockNumber neighborPage = ItemPointerGetBlockNumber(&neighborElement->heaptids[0]);
        bool found = false;

        int pid = neighborElement->pid;

        /* Check if page already counted */
        for (int j = 0; j < pageCandidates->length; j++) {
            if (pageCandidates->items[j].pid == pid)
            {
                pageCandidates->items[j].neighborCount++;
                found = true;
                break;
            }
        }
        /* If page not found, add new entry */
        if (!found)
        {
            pageCandidates->items[pageCandidates->length].pid = pid;
            pageCandidates->items[pageCandidates->length].neighborCount = 1;
            pageCandidates->items[pageCandidates->length].pageType = ORIGINAL_PAGE;
            pageCandidates->items[pageCandidates->length].blkno = InvalidBlockNumber;
            pageCandidates->length++;
        }
    }


    return pageCandidates;
}


/*
 * Add to element and neighbor pages
 */
static void
AddElementOnDiskWithPartition(Relation index, HnswElement e, int m, BlockNumber insertPage, BlockNumber *updatedInsertPage,
                            bool building, HnswInsertPagePool insertPagePool,  HnswInsertPagePool *updatedInsertPagePool)
{
    Buffer		buf;
    Page		page;
    GenericXLogState *state;
    Size		etupSize;
    Size		ntupSize;
    Size		combinedSize;
    Size		maxSize;
    Size		minCombinedSize;
    HnswElementTuple etup;
    BlockNumber currentPage = insertPage;
    HnswNeighborTuple ntup;
    Buffer		nbuf;
    Page		npage;
    OffsetNumber freeOffno = InvalidOffsetNumber;
    OffsetNumber freeNeighborOffno = InvalidOffsetNumber;
    BlockNumber newInsertPage = InvalidBlockNumber;
    uint8		tupleVersion;
    char	   *base = NULL;

    BlockNumber tempPageNo;
    Buffer tempBuf;
    Page tempPage;
    GenericXLogState *tempState;
    BlockNumber newExtendedPage;
    int tempPid;

    bool pidExist, extendedExist;
    int pidE, extendId;

    /* Calculate sizes */
    etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(HnswPtrAccess(base, e->value)));
    ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(e->level, m);
    combinedSize = etupSize + ntupSize + sizeof(ItemIdData);
    maxSize = HNSW_MAX_SIZE;
    minCombinedSize = etupSize + HNSW_NEIGHBOR_TUPLE_SIZE(0, m) + sizeof(ItemIdData);

    /* Prepare element tuple */
    etup = palloc0(etupSize);
    HnswSetElementTuple(base, etup, e);

    /* Prepare neighbor tuple */
    ntup = palloc0(ntupSize);
    HnswSetNeighborTuple(base, ntup, e, m);

    //// neighbor 정보 받아옴 (original, blkno-cnt)
    HnswInsertPageCandidate pageCandidates = CalculatePartitionNeighborCount(e);
    // insert되는 애들에 대해서는 고려하지 않는 편이 나은 걸로 보여지고 있음. 근데 .. neighbor가 insert된 애들 중에도 있을텐데 ..? .. 왜지
//    qsort(pageCandidates->items, pageCandidates->length, sizeof(HnswPageNeighborCount), ComparePageNeighbors);
//    e->pid = pageCandidates->items[0].pid;
//    HnswSetElementTuple(base, etup, e);

//    //// 기존의 insertpagepool과 pageCandidates 를 통합
//    GetInsertPagePoolInfo(index, &newPoolSize, newInsertPagePool);




    // 각 insertpage entry에 대해 pagecandidate 확인해서 pageCandidate 업데이트
    for (int i = 0; i < insertPagePool->poolSize; i++)
    {
        int pid = insertPagePool->items[i].pid;
        BlockNumber extendedPage = insertPagePool->items[i].extendedPage;

//        elog(WARNING, "INVALID: original %d, extended %d", originalPage, extendedPage);

        // 만약에 pool에 이미 extend page가 존재한다면,
        if (pid != -2) {

            pidExist = false;
            pidE = -1;

            /* Replace classic page with its inserted page */
            for (int j = 0; j < pageCandidates->length; j++) {
                // page candidate의 originalPage, extendedPage 존재 여부 확인
                if (pageCandidates->items[j].pid == pid) {
                    pidExist = true;
                    pidE = j;
                }
            }

            if (pidExist){
                pageCandidates->items[pidE].pageType = EXTENDED_PAGE;
                pageCandidates->items[pidE].blkno = extendedPage;
            }
        }


        if (pid == -2 && extendedPage != InvalidBlockNumber){
            pageCandidates->items[pageCandidates->length].pid = pid;
            pageCandidates->items[pageCandidates->length].neighborCount = 0;
            pageCandidates->items[pageCandidates->length].pageType = SPARE_PAGE;
            pageCandidates->items[pageCandidates->length].blkno = extendedPage;
            pageCandidates->length++;
        }
    }

    qsort(pageCandidates->items, pageCandidates->length, sizeof(HnswPageNeighborCount), ComparePageNeighbors);


    // 현재 insert page 읽어옴 .. -> index 유지 용도
    buf = ReadBuffer(index, currentPage);
    LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
//    elog(WARNING, "                          ****                          ");
//    elog(WARNING, "start insert ~");
//    elog(WARNING, "current insert page: BlockNumber = %u", currentPage);

    if (building)
    {
        state = NULL;
        page = BufferGetPage(buf);
    }
    else
    {
        state = GenericXLogStart(index);
//        elog(WARNING, "GenericXLogStart called for main page: BlockNumber = %u", BufferGetBlockNumber(buf));
        page = GenericXLogRegisterBuffer(state, buf, 0);
    }
//    elog(WARNING, "          ****             ");


//
//// insertPagePool 정보 로그 출력
//    elog(WARNING, "Insert Page Pool (Size: %d)", insertPagePool->poolSize);
//    for (int p = 0; p < insertPagePool->poolSize; p++) {
//        int pid = insertPagePool->items[p].pid;
//        BlockNumber ep = insertPagePool->items[p].extendedPage;
//
//        elog(WARNING, "[Pool Entry %d] pid: %d, extendedPage: %u",
//             p,
//             pid,
//             ep);
//    }
//
//    // pageCandidates 정보 로그 출력
//    elog(WARNING, "Page Candidates (Total: %d)", pageCandidates->length);
//    for (int i = 0; i < pageCandidates->length; i++) {
//        elog(WARNING, "[Candidate %d] pid: %d, neighborCount: %d, pageType: %d, extendedPage: %u",
//             i,
//             pageCandidates->items[i].pid,
//             pageCandidates->items[i].neighborCount,
//             pageCandidates->items[i].pageType,
//             pageCandidates->items[i].blkno);
//    }


    bool useTempBuf = false;
    bool found = false;


//    elog(WARNING, "insertPagePool->poolSize: %d", insertPagePool->poolSize);

    // 나머지 candidate 페이지에 대해 ..
    for (int i = 0; i < pageCandidates->length; i++) {
        // 페이지 불러옴
        tempPid = pageCandidates->items[i].pid;
        tempPageNo = pageCandidates->items[i].blkno;
//        elog(WARNING, "Pid = %d", tempPid);


        if (pageCandidates->items[i].pageType == ORIGINAL_PAGE) {
            if (tempPid == -2){
                continue;
            }

            if (insertPagePool->poolSize < MAX_INSERT_POOL_SIZE) {
                // tempPageNo 가 invalid일거임 -> 하나 추가해줘야 됨..
//                elog(WARNING, "***** 1. [ORIGINAL - EXTEND]");

                Buffer newbuf;
                Page newpage;

                // 기존의 insert page 뒤에 페이지 하나 추가함 -> 여기에 insert할거
                HnswInsertAppendPage(index, &newbuf, &newpage, state, page, building);

                /* Commit */
                if (building)
                    MarkBufferDirty(buf);
                else
                    GenericXLogFinish(state);

                /* Unlock previous buffer */
                UnlockReleaseBuffer(buf);


                /* Prepare new buffer */
                buf = newbuf;
                if (building) {
                    state = NULL;
                    page = BufferGetPage(buf);
                } else {
                    state = GenericXLogStart(index);
                    page = GenericXLogRegisterBuffer(state, buf, 0);
                }

                nbuf = buf;
                npage = page;

                // 새로운 페이지 추가 -> origin-page 와 extended-page 연결해줘야됨
                newExtendedPage = BufferGetBlockNumber(newbuf);

                // insert page pool의 어느 자리가 비어있을 지 어케 앎? -> 이거는 앞에서부터 차곡차곡 .. 쌓는다 가정 ..

                insertPagePool->items[insertPagePool->poolSize].pid = tempPid;
                insertPagePool->items[insertPagePool->poolSize].extendedPage = newExtendedPage;
                insertPagePool->poolSize++;

//                found = true; // newextendedpage에 저장 -> buf, page 제대로 반영됨.
                break;
            } else {
                continue;
//                elog(WARNING, "moving to next ...");
//
//                if (currentPage != tempPageNo) {
//
//                    // max_pool 상태라면 buffer 놓고 continue
//                    if (building)
//                        MarkBufferDirty(tempBuf);
//                    else
//                        GenericXLogFinish(tempState);
//                    UnlockReleaseBuffer(tempBuf);
//
//                    elog(WARNING, "unlock buf ...");
//                }
            }

        } else {
//            elog(WARNING, "Block Number = %u", tempPageNo);

            if (currentPage == tempPageNo) {
                tempBuf = buf;
                tempPage = page;
                tempState = state;
            } else {
                useTempBuf = true;
                tempBuf = ReadBuffer(index, tempPageNo);
                LockBuffer(tempBuf, BUFFER_LOCK_EXCLUSIVE);

//                elog(WARNING, "candidate page: BlockNumber = %u (PageType = %d), neighbor count = %d",
//                     tempPageNo, pageCandidates->items[i].pageType, pageCandidates->items[i].neighborCount);

                if (building) {
                    tempState = NULL;
                    tempPage = BufferGetPage(tempBuf);
                } else {
                    tempState = GenericXLogStart(index);
//                    elog(WARNING, "GenericXLogStart called for candidate page: BlockNumber = %u",
//                         BufferGetBlockNumber(tempBuf));
                    tempPage = GenericXLogRegisterBuffer(tempState, tempBuf, 0);
                }
            }


            if (pageCandidates->items[i].pageType == EXTENDED_PAGE) {
                // 페이지 불러옴

//                elog(WARNING, "Page type: Extended");

                if (PageGetFreeSpace(tempPage) >= combinedSize) {
//                    elog(WARNING, "********** 2-1. [EXTENDED - INSERT]");

//                    elog(WARNING, "[INSERT] free space (%zu) >= combined size (%zu)",
//                         PageGetFreeSpace(tempPage), combinedSize);

                    // 자리 남아있으면 extended page에 넣고, 남은 자리 확인, 꽉 차면 바로 pool에서 제거
                    nbuf = tempBuf;
                    npage = tempPage;
                    found = true;

                    /* 페이지가 꽉 차면 Insert Page Pool에서 제거 */
                    if (PageGetFreeSpace(tempPage) < combinedSize * 2) {
//                        elog(WARNING, "    [REMOVED]");
//                        elog(WARNING, "Insert Page Pool (Size: %d)", insertPagePool->poolSize);
//                        for (int p = 0; p < insertPagePool->poolSize; p++) {
//                            int pid = insertPagePool->items[p].pid;
//                            BlockNumber ep = insertPagePool->items[p].extendedPage;
//
//                            elog(WARNING, "[Pool Entry %d] pid: %d, extendedPage: %u",
//                                 p,
//                                 pid,
//                                 ep);
//                        }
                        for (int j = 0; j < insertPagePool->poolSize; j++) {
                            if (insertPagePool->items[j].extendedPage == tempPageNo) {

                                /* 마지막 페이지와 바꿔서 pool 크기 줄임 */ // poolsize 같은걸 meta에 저장하는 함수 만들어야됨

                                insertPagePool->items[j] = insertPagePool->items[insertPagePool->poolSize - 1];
                                insertPagePool->poolSize--; //
//                                elog(WARNING, "    [REMOVED DONE] id: %d, tempNo: %d, pool size: %d", j, (int)tempPageNo,  insertPagePool->poolSize);
//                                elog(WARNING, "Insert Page Pool (Size: %d)", insertPagePool->poolSize);
//                                for (int p = 0; p < insertPagePool->poolSize; p++) {
//                                    int pid = insertPagePool->items[p].pid;
//                                    BlockNumber ep = insertPagePool->items[p].extendedPage;
//
//                                    elog(WARNING, "[Pool Entry %d] pid: %d, extendedPage: %u",
//                                         p,
//                                         pid,
//                                         ep);
//                                }
                                break;
                            }
                        }
                    }
                    break;

                } else {
//                    elog(WARNING, "[EXTEND] free space (%zu) < combined size (%zu)",
//                         PageGetFreeSpace(tempPage), combinedSize);
//                    elog(WARNING, "********** 2-2. [EXTENDED - EXTEND]");
                    // 자리 없다는 의미니까 ,, pool에서 없애버려야 됨.. -> 업데이트만 해주면 됨.
                    // 새로 하나 만들어서,,
                    // max_pool 상태 아니라면 extended page 만들고 metap 업데이트

                    Buffer newbuf;
                    Page newpage;

                    // 기존의 insert page 뒤에 페이지 하나 추가함 -> 여기에 insert할거
                    HnswInsertAppendPage(index, &newbuf, &newpage, state, page, building);

                    /* Commit */
                    if (building)
                        MarkBufferDirty(buf);
                    else
                        GenericXLogFinish(state);

                    /* Unlock previous buffer */
                    UnlockReleaseBuffer(buf);

                    /* Prepare new buffer */
                    buf = newbuf;
                    if (building) {
                        state = NULL;
                        page = BufferGetPage(buf);
                    } else {
                        state = GenericXLogStart(index);
                        page = GenericXLogRegisterBuffer(state, buf, 0);
                    }

                    nbuf = buf;
                    npage = page;

                    // 새로운 페이지 추가 -> origin-page 와 extended-page 연결해줘야됨
                    newExtendedPage = BufferGetBlockNumber(newbuf);

                    // 기존 pool 에서 찾아서 업데이트
                    for (int j = 0; j < insertPagePool->poolSize; j++) {
                        if (insertPagePool->items[j].extendedPage == tempPageNo) {
                            //                        insertPagePool->items[j].originalPage = tempPageNo;
                            insertPagePool->items[j].extendedPage = newExtendedPage;
                            break;
                        }
                    }
                    //                found = true;
                    break;
                }
            } else if (pageCandidates->items[i].pageType == SPARE_PAGE) {
//                elog(WARNING, "Page type: SPARE");

                if (PageGetFreeSpace(tempPage) >= combinedSize) {
//                    elog(WARNING, "*************** 3-1. [SPARE - INSERT]");
//                    elog(WARNING, "[INSERT] free space (%zu) >= combined size (%zu)",
//                         PageGetFreeSpace(tempPage), combinedSize);

                    nbuf = tempBuf;
                    npage = tempPage;
                    found = true; //// tempBuf에 저장한다고 알려줘야됨 !!!!!! -> buf, page 반영X

                    break;

                } else {
                    // max_pool 상태 아니라면 extended page 만들고 metap에 추가
//                    elog(WARNING, "*************** 3-2. [SPARE - EXTEND]");
//                    elog(WARNING, "[EXTEND] free space (%zu) < combined size (%zu)",
//                         PageGetFreeSpace(tempPage), combinedSize);

                    Buffer newbuf;
                    Page newpage;


                    // 기존의 insert page 뒤에 페이지 하나 추가함 -> 여기에 insert할거
                    HnswInsertAppendPage(index, &newbuf, &newpage, state, page, building);

                    /* Commit */
                    if (building)
                        MarkBufferDirty(buf);
                    else
                        GenericXLogFinish(state);

                    /* Unlock previous buffer */
                    UnlockReleaseBuffer(buf);


                    /* Prepare new buffer */
                    buf = newbuf;
                    if (building) {
                        state = NULL;
                        page = BufferGetPage(buf);
                    } else {
                        state = GenericXLogStart(index);
                        page = GenericXLogRegisterBuffer(state, buf, 0);
                    }

                    nbuf = buf;
                    npage = page;

                    // 새로운 페이지 추가 -> origin-page 와 extended-page 연결해줘야됨
                    newExtendedPage = BufferGetBlockNumber(newbuf);

                    // 기존 pool 에서 찾아서 업데이트
                    for (int j = 0; j < insertPagePool->poolSize; j++) {
                        if (insertPagePool->items[j].extendedPage == tempPageNo) {
                            insertPagePool->items[j].pid = -2;
                            insertPagePool->items[j].extendedPage = newExtendedPage;
                            break;
                        }
                    }


                    //                found = true; // newextendedpage에 저장 -> buf, page 제대로 반영됨.
                    break;
                }
            }
        }
    }


//    elog(WARNING, "                          ****                          ");

    e->pid = tempPid;
    HnswSetElementTuple(base, etup, e); // 왜 오히려 안좋아지는지 .. 의문 ... 왜일까 ..
//    elog(WARNING, "e->pid: %d", e->pid);

    if (found)
    {
        // tempBuf에 저장
        e->blkno = BufferGetBlockNumber(tempBuf);
        e->neighborPage = BufferGetBlockNumber(tempBuf);
        newInsertPage = currentPage;

        e->offno = OffsetNumberNext(PageGetMaxOffsetNumber(tempPage));
//        if (nbuf == buf)
        e->neighborOffno = OffsetNumberNext(e->offno);
//        else
//            e->neighborOffno = FirstOffsetNumber;


    } else
    {
        // 새로운 블록에 저장
        e->blkno = BufferGetBlockNumber(buf);
        e->neighborPage = BufferGetBlockNumber(nbuf);
        newInsertPage = e->neighborPage;


        e->offno = OffsetNumberNext(PageGetMaxOffsetNumber(page));
//        if (nbuf == buf)
        e->neighborOffno = OffsetNumberNext(e->offno);
//        else
//            e->neighborOffno = FirstOffsetNumber;

    }

//    elog(WARNING, "Attempting to add element at offno: %u on page: %u",
//         e->offno, e->blkno);
//
//    elog(WARNING, "Attempting to add neighbor at neighborOffno: %u on page: %u",
//         e->neighborOffno, e->neighborPage);


    ItemPointerSet(&etup->neighbortid, e->neighborPage, e->neighborOffno);

    if (found)
    {
        if (PageAddItem(tempPage, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != e->offno){
            elog(WARNING, "so sad ...");
            elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

        }

        if (PageAddItem(tempPage, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != e->neighborOffno){
            elog(WARNING, "OMG ....");
            elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

        }
    } else
    {
        if (PageAddItem(page, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != e->offno){
            elog(WARNING, "so sad ...");
            elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

        }


        if (PageAddItem(npage, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != e->neighborOffno){
            elog(WARNING, "OMG ....");
            elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

        }
    }


//    elog(WARNING, "                          ****                          ");
//    elog(WARNING, "                          ****                          ");
//    elog(WARNING, "                          ****                          ");
//    elog(WARNING, "                          ****                          ");


//    /* Commit */
    // lock 여부 .. 확인해야 되는데 귀찮아서 일단 그냥 rule-base로 구현 ..

//    elog(WARNING, "log1");
//    // Commit changes
//    if (state != NULL) {
//        GenericXLogFinish(state);
//    }
//    if (useTempBuf){
//        elog(WARNING, "log2");
//        if (tempState != NULL && tempState != state) {
//            GenericXLogFinish(tempState);
//        }
//    }
//
//    if (buf != InvalidBuffer) {
//        UnlockReleaseBuffer(buf);
//    }
//    elog(WARNING, "log4");
//    if (nbuf != InvalidBuffer && nbuf != buf) {
//        UnlockReleaseBuffer(nbuf);
//    }
//    elog(WARNING, "log5");
//
//    if (useTempBuf){
//
//        if (tempBuf != InvalidBuffer){
//            elog(WARNING, "invalid buffer");
//            if (tempBuf != buf && tempBuf != nbuf) {
//                elog(WARNING, "Releasing tempBuf: BlockNumber = %u", BufferGetBlockNumber(tempBuf));
//                UnlockReleaseBuffer(tempBuf);
//            }
//        }
//
//    }
//
//    elog(WARNING, "log6");

    GenericXLogFinish(state);
    if (useTempBuf){
        GenericXLogFinish(tempState);
    }

    UnlockReleaseBuffer(buf); // nbuf==buf인 상황만 고려
    if (useTempBuf){
        UnlockReleaseBuffer(tempBuf);
    }



        /* Update the insert page */
    if (BlockNumberIsValid(newInsertPage) && newInsertPage != insertPage)
        *updatedInsertPage = newInsertPage;


    *updatedInsertPagePool = insertPagePool;


}



/*
 * Add to element and neighbor pages
 */
static void
AddElementOnDiskWithPartitionCount(Relation index, HnswElement e, int m, BlockNumber insertPage, BlockNumber *updatedInsertPage,
                              bool building, HnswInsertPagePool insertPagePool,  HnswInsertPagePool *updatedInsertPagePool)
{
    Buffer		buf;
    Page		page;
    GenericXLogState *state;
    Size		etupSize;
    Size		ntupSize;
    Size		combinedSize;
    Size		maxSize;
    Size		minCombinedSize;
    HnswElementTuple etup;
    BlockNumber currentPage = insertPage;
    HnswNeighborTuple ntup;
    Buffer		nbuf;
    Page		npage;
    OffsetNumber freeOffno = InvalidOffsetNumber;
    OffsetNumber freeNeighborOffno = InvalidOffsetNumber;
    BlockNumber newInsertPage = InvalidBlockNumber;
    uint8		tupleVersion;
    char	   *base = NULL;

    BlockNumber tempPageNo;
    Buffer tempBuf;
    Page tempPage;
    GenericXLogState *tempState;
    BlockNumber newExtendedPage;
    int tempPid;

    bool pidExist, extendedExist;
    int pidE, extendId;

    /* Calculate sizes */
    etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(HnswPtrAccess(base, e->value)));
    ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(e->level, m);
    combinedSize = etupSize + ntupSize + sizeof(ItemIdData);
    maxSize = HNSW_MAX_SIZE;
    minCombinedSize = etupSize + HNSW_NEIGHBOR_TUPLE_SIZE(0, m) + sizeof(ItemIdData);

    /* Prepare element tuple */
    etup = palloc0(etupSize);
    HnswSetElementTuple(base, etup, e);

    /* Prepare neighbor tuple */
    ntup = palloc0(ntupSize);
    HnswSetNeighborTuple(base, ntup, e, m);

    //// neighbor 정보 받아옴 (original, blkno-cnt)
    HnswInsertPageCandidate pageCandidates = CalculatePartitionNeighborCount(e);
    // insert되는 애들에 대해서는 고려하지 않는 편이 나은 걸로 보여지고 있음. 근데 .. neighbor가 insert된 애들 중에도 있을텐데 ..? .. 왜지
//    qsort(pageCandidates->items, pageCandidates->length, sizeof(HnswPageNeighborCount), ComparePageNeighbors);
//    e->pid = pageCandidates->items[0].pid;
//    HnswSetElementTuple(base, etup, e);

//    //// 기존의 insertpagepool과 pageCandidates 를 통합
//    GetInsertPagePoolInfo(index, &newPoolSize, newInsertPagePool);




    // 각 insertpage entry에 대해 pagecandidate 확인해서 pageCandidate 업데이트
    for (int i = 0; i < insertPagePool->poolSize; i++)
    {
        int pid = insertPagePool->items[i].pid;
        BlockNumber extendedPage = insertPagePool->items[i].extendedPage;

//        elog(WARNING, "INVALID: original %d, extended %d", originalPage, extendedPage);

        // 만약에 pool에 이미 extend page가 존재한다면,
        if (pid != -2) {

            pidExist = false;
            pidE = -1;

            /* Replace classic page with its inserted page */
            for (int j = 0; j < pageCandidates->length; j++) {
                // page candidate의 originalPage, extendedPage 존재 여부 확인
                if (pageCandidates->items[j].pid == pid) {
                    pidExist = true;
                    pidE = j;
                }
            }

            if (pidExist){
                pageCandidates->items[pidE].pageType = EXTENDED_PAGE;
                pageCandidates->items[pidE].blkno = extendedPage;
            }
        }


        if (pid == -2 && extendedPage != InvalidBlockNumber){
            pageCandidates->items[pageCandidates->length].pid = pid;
            pageCandidates->items[pageCandidates->length].neighborCount = 0;
            pageCandidates->items[pageCandidates->length].pageType = SPARE_PAGE;
            pageCandidates->items[pageCandidates->length].blkno = extendedPage;
            pageCandidates->length++;
        }
    }

    qsort(pageCandidates->items, pageCandidates->length, sizeof(HnswPageNeighborCount), ComparePageNeighbors);


    // 현재 insert page 읽어옴 .. -> index 유지 용도
    buf = ReadBuffer(index, currentPage);
    LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
//    elog(WARNING, "                          ****                          ");
//    elog(WARNING, "start insert ~");
//    elog(WARNING, "current insert page: BlockNumber = %u", currentPage);

    if (building)
    {
        state = NULL;
        page = BufferGetPage(buf);
    }
    else
    {
        state = GenericXLogStart(index);
//        elog(WARNING, "GenericXLogStart called for main page: BlockNumber = %u", BufferGetBlockNumber(buf));
        page = GenericXLogRegisterBuffer(state, buf, 0);
    }
//    elog(WARNING, "          ****             ");


////
//// insertPagePool 정보 로그 출력
//    elog(WARNING, "Insert Page Pool (Size: %d)", insertPagePool->poolSize);
//    for (int p = 0; p < insertPagePool->poolSize; p++) {
//        int pid = insertPagePool->items[p].pid;
//        BlockNumber ep = insertPagePool->items[p].extendedPage;
//
//        elog(WARNING, "[Pool Entry %d] pid: %d, extendedPage: %u",
//             p,
//             pid,
//             ep);
//    }
//
//    // pageCandidates 정보 로그 출력
//    elog(WARNING, "Page Candidates (Total: %d)", pageCandidates->length);
//    for (int i = 0; i < pageCandidates->length; i++) {
//        elog(WARNING, "[Candidate %d] pid: %d, neighborCount: %d, pageType: %d, extendedPage: %u",
//             i,
//             pageCandidates->items[i].pid,
//             pageCandidates->items[i].neighborCount,
//             pageCandidates->items[i].pageType,
//             pageCandidates->items[i].blkno);
//    }


    bool useTempBuf = false;
    bool found = false;


//    elog(WARNING, "insertPagePool->poolSize: %d", insertPagePool->poolSize);

    // 나머지 candidate 페이지에 대해 ..
    for (int i = 0; i < pageCandidates->length; i++) {
        // 페이지 불러옴
        tempPid = pageCandidates->items[i].pid;
        tempPageNo = pageCandidates->items[i].blkno;
//        elog(WARNING, "Pid = %d", tempPid);


        if (pageCandidates->items[i].pageType == ORIGINAL_PAGE) {
            continue;

        }

        if (tempPageNo == InvalidBlockNumber) {
//            elog(WARNING, "Page type: EXTENDED (new)");


            Buffer newbuf;
            Page newpage;

            // 기존의 insert page 뒤에 페이지 하나 추가함 -> 여기에 insert할거
            HnswInsertAppendPage(index, &newbuf, &newpage, state, page, building);

            /* Commit */
            if (building)
                MarkBufferDirty(buf);
            else
                GenericXLogFinish(state);

            /* Unlock previous buffer */
            UnlockReleaseBuffer(buf);

            /* Prepare new buffer */
            buf = newbuf;
            if (building) {
                state = NULL;
                page = BufferGetPage(buf);
            } else {
                state = GenericXLogStart(index);
                page = GenericXLogRegisterBuffer(state, buf, 0);
            }

            nbuf = buf;
            npage = page;

            // 새로운 페이지 추가 -> origin-page 와 extended-page 연결해줘야됨
            newExtendedPage = BufferGetBlockNumber(newbuf);

            // 기존 pool 에서 찾아서 업데이트
            for (int j = 0; j < insertPagePool->poolSize; j++) {
                if (insertPagePool->items[j].pid == tempPid) {
                    //                        insertPagePool->items[j].originalPage = tempPageNo;
                    insertPagePool->items[j].extendedPage = newExtendedPage;
                    break;
                }
            }
            //                found = true;
            break;

        }

        // tempPageNo != invalid -> 페이지 존재

        if (currentPage == tempPageNo) {
            tempBuf = buf;
            tempPage = page;
            tempState = state;
        } else {
            useTempBuf = true;
            tempBuf = ReadBuffer(index, tempPageNo);
            LockBuffer(tempBuf, BUFFER_LOCK_EXCLUSIVE);

//                elog(WARNING, "candidate page: BlockNumber = %u (PageType = %d), neighbor count = %d",
//                     tempPageNo, pageCandidates->items[i].pageType, pageCandidates->items[i].neighborCount);

            if (building) {
                tempState = NULL;
                tempPage = BufferGetPage(tempBuf);
            } else {
                tempState = GenericXLogStart(index);
//                    elog(WARNING, "GenericXLogStart called for candidate page: BlockNumber = %u",
//                         BufferGetBlockNumber(tempBuf));
                tempPage = GenericXLogRegisterBuffer(tempState, tempBuf, 0);
            }
        }

        if (pageCandidates->items[i].pageType == EXTENDED_PAGE){
//            elog(WARNING, "Page type: EXTENDED ");
        } else{
//            elog(WARNING, "Page type: SPARE ");
        }

        if (PageGetFreeSpace(tempPage) >= combinedSize) {
//            elog(WARNING, "[INSERT] free space (%zu) > combined size (%zu)",
//                 PageGetFreeSpace(tempPage), combinedSize);

//            elog(WARNING, "INSERT");
            nbuf = tempBuf;
            npage = tempPage;
            found = true;
            break;

        } else {
//            elog(WARNING, "[EXTEND] free space (%zu) < combined size (%zu)",
//                 PageGetFreeSpace(tempPage), combinedSize);

            // 자리 없다는 의미니까 ,, pool에서 없애버려야 됨.. -> 업데이트만 해주면 됨.
            // 새로 하나 만들어서,,
            // max_pool 상태 아니라면 extended page 만들고 metap 업데이트
//            elog(WARNING, "EXTEND");

            Buffer newbuf;
            Page newpage;

            // 기존의 insert page 뒤에 페이지 하나 추가함 -> 여기에 insert할거
            HnswInsertAppendPage(index, &newbuf, &newpage, state, page, building);

            /* Commit */
            if (building)
                MarkBufferDirty(buf);
            else
                GenericXLogFinish(state);

            /* Unlock previous buffer */
            UnlockReleaseBuffer(buf);

            /* Prepare new buffer */
            buf = newbuf;
            if (building) {
                state = NULL;
                page = BufferGetPage(buf);
            } else {
                state = GenericXLogStart(index);
                page = GenericXLogRegisterBuffer(state, buf, 0);
            }

            nbuf = buf;
            npage = page;

            // 새로운 페이지 추가 -> origin-page 와 extended-page 연결해줘야됨
            newExtendedPage = BufferGetBlockNumber(newbuf);

            // 기존 pool 에서 찾아서 업데이트
            for (int j = 0; j < insertPagePool->poolSize; j++) {
                if (insertPagePool->items[j].extendedPage == tempPageNo) {
                    //                        insertPagePool->items[j].originalPage = tempPageNo;
                    insertPagePool->items[j].extendedPage = newExtendedPage;
                    break;
                }
            }
            //                found = true;
            break;
        }
    }



//    elog(WARNING, "                          ****                          ");

    e->pid = tempPid;
    HnswSetElementTuple(base, etup, e); // 왜 오히려 안좋아지는지 .. 의문 ... 왜일까 ..
//    elog(WARNING, "e->pid: %d", e->pid);

    if (found)
    {
        // tempBuf에 저장
        e->blkno = BufferGetBlockNumber(tempBuf);
        e->neighborPage = BufferGetBlockNumber(tempBuf);
        newInsertPage = currentPage;

        e->offno = OffsetNumberNext(PageGetMaxOffsetNumber(tempPage));
//        if (nbuf == buf)
        e->neighborOffno = OffsetNumberNext(e->offno);
//        else
//            e->neighborOffno = FirstOffsetNumber;


    } else
    {
        // 새로운 블록에 저장
        e->blkno = BufferGetBlockNumber(buf);
        e->neighborPage = BufferGetBlockNumber(nbuf);
        newInsertPage = e->neighborPage;


        e->offno = OffsetNumberNext(PageGetMaxOffsetNumber(page));
//        if (nbuf == buf)
        e->neighborOffno = OffsetNumberNext(e->offno);
//        else
//            e->neighborOffno = FirstOffsetNumber;

    }

//    elog(WARNING, "Attempting to add element at offno: %u on page: %u",
//         e->offno, e->blkno);
//
//    elog(WARNING, "Attempting to add neighbor at neighborOffno: %u on page: %u",
//         e->neighborOffno, e->neighborPage);


    ItemPointerSet(&etup->neighbortid, e->neighborPage, e->neighborOffno);

    if (found)
    {
        if (PageAddItem(tempPage, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != e->offno){
            elog(WARNING, "so sad ...");
            elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

        }

        if (PageAddItem(tempPage, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != e->neighborOffno){
            elog(WARNING, "OMG ....");
            elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

        }
    } else
    {
        if (PageAddItem(page, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != e->offno){
            elog(WARNING, "so sad ...");
            elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

        }


        if (PageAddItem(npage, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != e->neighborOffno){
            elog(WARNING, "OMG ....");
            elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

        }
    }

    GenericXLogFinish(state);
    if (useTempBuf){
        GenericXLogFinish(tempState);
    }

    UnlockReleaseBuffer(buf); // nbuf==buf인 상황만 고려
    if (useTempBuf){
        UnlockReleaseBuffer(tempBuf);
    }



    /* Update the insert page */
    if (BlockNumberIsValid(newInsertPage) && newInsertPage != insertPage)
        *updatedInsertPage = newInsertPage;


    *updatedInsertPagePool = insertPagePool;


}


/*
 * Load neighbors
 */
static HnswNeighborArray *
HnswLoadNeighbors(HnswElement element, Relation index, int m, int lm, int lc)
{
	char	   *base = NULL;
	HnswNeighborArray *neighbors = HnswInitNeighborArray(lm, NULL);
	ItemPointerData indextids[HNSW_MAX_M * 2];

	if (!HnswLoadNeighborTids(element, indextids, index, m, lm, lc))
		return neighbors;

	for (int i = 0; i < lm; i++)
	{
		ItemPointer indextid = &indextids[i];
		HnswElement e;
		HnswCandidate *hc;

		if (!ItemPointerIsValid(indextid))
			break;

		e = HnswInitElementFromBlock(ItemPointerGetBlockNumber(indextid), ItemPointerGetOffsetNumber(indextid));
		hc = &neighbors->items[neighbors->length++];
		HnswPtrStore(base, hc->element, e);
	}

	return neighbors;
}

/*
 * Load elements for insert
 */
static void
LoadElementsForInsert(HnswNeighborArray * neighbors, HnswQuery * q, int *idx, Relation index, HnswSupport * support)
{
	char	   *base = NULL;

	for (int i = 0; i < neighbors->length; i++)
	{
		HnswCandidate *hc = &neighbors->items[i];
		HnswElement element = HnswPtrAccess(base, hc->element);
		double		distance;

		HnswLoadElement(element, &distance, q, index, support, true, NULL);
		hc->distance = distance;

		/* Prune element if being deleted */
		if (element->heaptidsLength == 0)
		{
			*idx = i;
			break;
		}
	}
}

/*
 * Get update index
 */
static int
GetUpdateIndex(HnswElement element, HnswElement newElement, float distance, int m, int lm, int lc, Relation index, HnswSupport * support, MemoryContext updateCtx)
{
	char	   *base = NULL;
	int			idx = -1;
	HnswNeighborArray *neighbors;
	MemoryContext oldCtx = MemoryContextSwitchTo(updateCtx);

	/*
	 * Get latest neighbors since they may have changed. Do not lock yet since
	 * selecting neighbors can take time. Could use optimistic locking to
	 * retry if another update occurs before getting exclusive lock.
	 */
	neighbors = HnswLoadNeighbors(element, index, m, lm, lc);

	/*
	 * Could improve performance for vacuuming by checking neighbors against
	 * list of elements being deleted to find index. It's important to exclude
	 * already deleted elements for this since they can be replaced at any
	 * time.
	 */

	if (neighbors->length < lm)
		idx = -2;
	else
	{
		HnswQuery	q;

		q.value = HnswGetValue(base, element);

		LoadElementsForInsert(neighbors, &q, &idx, index, support);

		if (idx == -1)
			HnswUpdateConnection(base, neighbors, newElement, distance, lm, &idx, index, support);
	}

	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(updateCtx);

	return idx;
}

/*
 * Check if connection already exists
 */
static bool
ConnectionExists(HnswElement e, HnswNeighborTuple ntup, int startIdx, int lm)
{
	for (int i = 0; i < lm; i++)
	{
		ItemPointer indextid = &ntup->indextids[startIdx + i];

		if (!ItemPointerIsValid(indextid))
			break;

		if (ItemPointerGetBlockNumber(indextid) == e->blkno && ItemPointerGetOffsetNumber(indextid) == e->offno)
			return true;
	}

	return false;
}

/*
 * Update neighbor
 */
static void
UpdateNeighborOnDisk(HnswElement element, HnswElement newElement, int idx, int m, int lm, int lc, Relation index, bool checkExisting, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	HnswNeighborTuple ntup;
	int			startIdx;
	OffsetNumber offno = element->neighborOffno;

	/* Register page */
	buf = ReadBuffer(index, element->neighborPage);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
	{
		state = NULL;
		page = BufferGetPage(buf);
	}
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	/* Get tuple */
	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, offno));

	/* Calculate index for update */
	startIdx = (element->level - lc) * m;

	/* Check for existing connection */
	if (checkExisting && ConnectionExists(newElement, ntup, startIdx, lm))
		idx = -1;
	else if (idx == -2)
	{
		/* Find free offset if still exists */
		/* TODO Retry updating connections if not */
		for (int j = 0; j < lm; j++)
		{
			if (!ItemPointerIsValid(&ntup->indextids[startIdx + j]))
			{
				idx = startIdx + j;
				break;
			}
		}
	}
	else
		idx += startIdx;

	/* Make robust to issues */
	if (idx >= 0 && idx < ntup->count)
	{
		ItemPointer indextid = &ntup->indextids[idx];

		/* Update neighbor on the buffer */
		ItemPointerSet(indextid, newElement->blkno, newElement->offno);

		/* Commit */
		if (building)
			MarkBufferDirty(buf);
		else
			GenericXLogFinish(state);
	}
	else if (!building)
		GenericXLogAbort(state);

	UnlockReleaseBuffer(buf);
}

/*
 * Update neighbors
 */
void
HnswUpdateNeighborsOnDisk(Relation index, HnswSupport * support, HnswElement e, int m, bool checkExisting, bool building)
{
	char	   *base = NULL;

	/* Use separate memory context to improve performance for larger vectors */
	MemoryContext updateCtx = GenerationContextCreate(CurrentMemoryContext,
													  "Hnsw insert update context",
#if PG_VERSION_NUM >= 150000
													  128 * 1024, 128 * 1024,
#endif
													  128 * 1024);

	for (int lc = e->level; lc >= 0; lc--)
	{
		int			lm = HnswGetLayerM(m, lc);
		HnswNeighborArray *neighbors = HnswGetNeighbors(base, e, lc);

		for (int i = 0; i < neighbors->length; i++)
		{
			HnswCandidate *hc = &neighbors->items[i];
			HnswElement neighborElement = HnswPtrAccess(base, hc->element);
			int			idx;

			idx = GetUpdateIndex(neighborElement, e, hc->distance, m, lm, lc, index, support, updateCtx);

			/* New element was not selected as a neighbor */
			if (idx == -1)
				continue;

			UpdateNeighborOnDisk(neighborElement, e, idx, m, lm, lc, index, checkExisting, building);
		}
	}

	MemoryContextDelete(updateCtx);
}

/*
 * Add a heap TID to an existing element
 */
static bool
AddDuplicateOnDisk(Relation index, HnswElement element, HnswElement dup, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	HnswElementTuple etup;
	int			i;

	/* Read page */
	buf = ReadBuffer(index, dup->blkno);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
	{
		state = NULL;
		page = BufferGetPage(buf);
	}
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	/* Find space */
	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, dup->offno));
	for (i = 0; i < HNSW_HEAPTIDS; i++)
	{
		if (!ItemPointerIsValid(&etup->heaptids[i]))
			break;
	}

	/* Either being deleted or we lost our chance to another backend */
	if (i == 0 || i == HNSW_HEAPTIDS)
	{
		if (!building)
			GenericXLogAbort(state);
		UnlockReleaseBuffer(buf);
		return false;
	}

	/* Add heap TID, modifying the tuple on the page directly */
	etup->heaptids[i] = element->heaptids[0];

	/* Commit */
	if (building)
		MarkBufferDirty(buf);
	else
		GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);

	return true;
}

/*
 * Find duplicate element
 */
static bool
FindDuplicateOnDisk(Relation index, HnswElement element, bool building)
{
	char	   *base = NULL;
	HnswNeighborArray *neighbors = HnswGetNeighbors(base, element, 0);
	Datum		value = HnswGetValue(base, element);

	for (int i = 0; i < neighbors->length; i++)
	{
		HnswCandidate *neighbor = &neighbors->items[i];
		HnswElement neighborElement = HnswPtrAccess(base, neighbor->element);
		Datum		neighborValue = HnswGetValue(base, neighborElement);

		/* Exit early since ordered by distance */
		if (!datumIsEqual(value, neighborValue, false, -1))
			return false;

		if (AddDuplicateOnDisk(index, element, neighborElement, building))
			return true;
	}

	return false;
}

/*
 * Update graph on disk
 */
static void
UpdateGraphOnDisk(Relation index, HnswSupport * support, HnswElement element, int m, int efConstruction, HnswElement entryPoint, bool building)
{
	BlockNumber newInsertPage = InvalidBlockNumber;

	/* Look for duplicate */
	if (FindDuplicateOnDisk(index, element, building))
		return;

	/* Add element */
	AddElementOnDisk(index, element, m, GetInsertPage(index), &newInsertPage, building);

	/* Update insert page if needed */
	if (BlockNumberIsValid(newInsertPage))
		HnswUpdateMetaPage(index, 0, NULL, newInsertPage, MAIN_FORKNUM, building);

	/* Update neighbors */
	HnswUpdateNeighborsOnDisk(index, support, element, m, false, building);

	/* Update entry point if needed */
	if (entryPoint == NULL || element->level > entryPoint->level)
		HnswUpdateMetaPage(index, HNSW_UPDATE_ENTRY_GREATER, element, InvalidBlockNumber, MAIN_FORKNUM, building);
}


/*
 * Update graph on disk
 */
static void
UpdateGraphOnDiskWithPartition(Relation index, HnswSupport * support, HnswElement element, int m, int efConstruction, HnswElement entryPoint, bool building, HnswInsertPagePool insertPagePool)
{
    BlockNumber newInsertPage = InvalidBlockNumber;
    HnswInsertPagePool newInsertPagePool = NULL;

    /* Look for duplicate */
    if (FindDuplicateOnDisk(index, element, building)){
        elog(WARNING, "duplicate !!!");
        return;
    }


//    AddElementOnDisk(index, element, m, GetInsertPage(index), &newInsertPage, building);
//    AddElementOnDiskWithPool(index, element, m, GetInsertPage(index), &newInsertPage, building, insertPagePool);

//    AddElementOnDiskWithPartition(index, element, m, GetInsertPage(index), &newInsertPage, building, insertPagePool, &newInsertPagePool);
    AddElementOnDiskWithPartitionCount(index, element, m, GetInsertPage(index), &newInsertPage, building, insertPagePool, &newInsertPagePool);

//    elog(WARNING, "poolSize: %d", newInsertPagePool->poolSize);

    /* Update insert page if needed */
    if (BlockNumberIsValid(newInsertPage))
        HnswUpdateMetaPage(index, 0, NULL, newInsertPage, MAIN_FORKNUM, building);


    HnswUpdateMetaPageWithPartition(index, 0, NULL, InvalidBlockNumber, MAIN_FORKNUM, building, newInsertPagePool);

    /* Update neighbors */
    HnswUpdateNeighborsOnDisk(index, support, element, m, false, building);

//    UpdateInsertPagePool(index, insertPagePool, MAIN_FORKNUM, building);

    /* Update entry point if needed */
    if (entryPoint == NULL || element->level > entryPoint->level)
        HnswUpdateMetaPage(index, HNSW_UPDATE_ENTRY_GREATER, element, InvalidBlockNumber, MAIN_FORKNUM, building);

}


/*
 * Insert a tuple into the index
 */
bool
HnswInsertTupleOnDisk(Relation index, HnswSupport * support, Datum value, ItemPointer heaptid, bool building)
{
	HnswElement entryPoint;
	HnswElement element;
	int			m;
	int			efConstruction = HnswGetEfConstruction(index);
	LOCKMODE	lockmode = ShareLock;
	char	   *base = NULL;

	/*
	 * Get a shared lock. This allows vacuum to ensure no in-flight inserts
	 * before repairing graph. Use a page lock so it does not interfere with
	 * buffer lock (or reads when vacuuming).
	 */
	LockPage(index, HNSW_UPDATE_LOCK, lockmode);

	/* Get m and entry point */
	HnswGetMetaPageInfo(index, &m, &entryPoint);

	/* Create an element */
	element = HnswInitElement(base, heaptid, m, HnswGetMl(m), HnswGetMaxLevel(m), NULL);
	HnswPtrStore(base, element->value, DatumGetPointer(value));

	/* Prevent concurrent inserts when likely updating entry point */
	if (entryPoint == NULL || element->level > entryPoint->level)
	{
		/* Release shared lock */
		UnlockPage(index, HNSW_UPDATE_LOCK, lockmode);

		/* Get exclusive lock */
		lockmode = ExclusiveLock;
		LockPage(index, HNSW_UPDATE_LOCK, lockmode);

		/* Get latest entry point after lock is acquired */
		entryPoint = HnswGetEntryPoint(index);
	}

	/* Find neighbors for element */
	HnswFindElementNeighbors(base, element, entryPoint, index, support, m, efConstruction, false);

	/* Update graph on disk */
	UpdateGraphOnDisk(index, support, element, m, efConstruction, entryPoint, building);

	/* Release lock */
	UnlockPage(index, HNSW_UPDATE_LOCK, lockmode);

	return true;
}


bool
HnswInsertTupleOnDiskWithPartition(Relation index, HnswSupport * support, Datum value, ItemPointer heaptid, bool building)
{
    HnswElement entryPoint;
    HnswElement element;
    int			m;
    int			efConstruction = HnswGetEfConstruction(index);
    LOCKMODE	lockmode = ShareLock;
    char	   *base = NULL;

    HnswInsertPagePool insertPagePool;

    /*
     * Get a shared lock. This allows vacuum to ensure no in-flight inserts
     * before repairing graph. Use a page lock so it does not interfere with
     * buffer lock (or reads when vacuuming).
     */
    LockPage(index, HNSW_UPDATE_LOCK, lockmode);

    /* Get m and entry point */
    HnswGetMetaPageInfoWithPartition(index, &m, &entryPoint, &insertPagePool);

    /* Create an element */
    element = HnswInitElement(base, heaptid, m, HnswGetMl(m), HnswGetMaxLevel(m), NULL);
    HnswPtrStore(base, element->value, DatumGetPointer(value));

    /* Prevent concurrent inserts when likely updating entry point */
    if (entryPoint == NULL || element->level > entryPoint->level)
    {
        /* Release shared lock */
        UnlockPage(index, HNSW_UPDATE_LOCK, lockmode);

        /* Get exclusive lock */
        lockmode = ExclusiveLock;
        LockPage(index, HNSW_UPDATE_LOCK, lockmode);

        /* Get latest entry point after lock is acquired */
        entryPoint = HnswGetEntryPoint(index);
    }

    /* Find neighbors for element */
    HnswFindElementNeighbors(base, element, entryPoint, index, support, m, efConstruction, false);

    /* Update graph on disk */
    UpdateGraphOnDiskWithPartition(index, support, element, m, efConstruction, entryPoint, building, insertPagePool);

    /* Release lock */
    UnlockPage(index, HNSW_UPDATE_LOCK, lockmode);

//    elog(INFO, "insert done (on-disk)");

    return true;
}


/*
 * Insert a tuple into the index
 */
static void
HnswInsertTuple(Relation index, Datum *values, bool *isnull, ItemPointer heaptid)
{
	Datum		value;
	const		HnswTypeInfo *typeInfo = HnswGetTypeInfo(index);
	HnswSupport support;

	HnswInitSupport(&support, index);

	/* Form index value */
	if (!HnswFormIndexValue(&value, values, isnull, typeInfo, &support))
		return;

//	HnswInsertTupleOnDisk(index, &support, value, heaptid, false);
    HnswInsertTupleOnDiskWithPartition(index, &support, value, heaptid, false);
}

/*
 * Insert a tuple into the index
 */
bool
hnswinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
		   Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
		   ,bool indexUnchanged
#endif
		   ,IndexInfo *indexInfo
)
{
	MemoryContext oldCtx;
	MemoryContext insertCtx;

	/* Skip nulls */
	if (isnull[0])
		return false;

	/* Create memory context */
	insertCtx = AllocSetContextCreate(CurrentMemoryContext,
									  "Hnsw insert temporary context",
									  ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(insertCtx);

	/* Insert tuple */
	HnswInsertTuple(index, values, isnull, heap_tid);

	/* Delete memory context */
	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(insertCtx);

	return false;
}
