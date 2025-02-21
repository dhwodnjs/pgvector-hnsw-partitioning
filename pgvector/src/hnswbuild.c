/*
 * The HNSW build happens in two phases:
 *
 * 1. In-memory phase
 *
 * In this first phase, the graph is held completely in memory. When the graph
 * is fully built, or we run out of memory reserved for the build (determined
 * by maintenance_work_mem), we materialize the graph to disk (see
 * FlushPages()), and switch to the on-disk phase.
 *
 * In a parallel build, a large contiguous chunk of shared memory is allocated
 * to hold the graph. Each worker process has its own HnswBuildState struct in
 * private memory, which contains information that doesn't change throughout
 * the build, and pointers to the shared structs in shared memory. The shared
 * memory area is mapped to a different address in each worker process, and
 * 'HnswBuildState.hnswarea' points to the beginning of the shared area in the
 * worker process's address space. All pointers used in the graph are
 * "relative pointers", stored as an offset from 'hnswarea'.
 *
 * Each element is protected by an LWLock. It must be held when reading or
 * modifying the element's neighbors or 'heaptids'.
 *
 * In a non-parallel build, the graph is held in backend-private memory. All
 * the elements are allocated in a dedicated memory context, 'graphCtx', and
 * the pointers used in the graph are regular pointers.
 *
 * 2. On-disk phase
 *
 * In the on-disk phase, the index is built by inserting each vector to the
 * index one by one, just like on INSERT. The only difference is that we don't
 * WAL-log the individual inserts. If the graph fit completely in memory and
 * was fully built in the in-memory phase, the on-disk phase is skipped.
 *
 * After we have finished building the graph, we perform one more scan through
 * the index and write all the pages to the WAL.
 */
#include "postgres.h"

#include <math.h>

#include "access/parallel.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/xact.h"
#include "access/xloginsert.h"
#include "catalog/index.h"
#include "catalog/pg_type_d.h"
#include "commands/progress.h"
#include "hnsw.h"
#include "miscadmin.h"
#include "optimizer/optimizer.h"
#include "storage/bufmgr.h"
#include "tcop/tcopprot.h"
#include "utils/datum.h"
#include "utils/memutils.h"

#if PG_VERSION_NUM >= 140000
#include "utils/backend_progress.h"
#else
#include "pgstat.h"
#endif

#if PG_VERSION_NUM >= 140000
#include "utils/backend_status.h"
#include "utils/wait_event.h"
#endif

#define PARALLEL_KEY_HNSW_SHARED		UINT64CONST(0xA000000000000001)
#define PARALLEL_KEY_HNSW_AREA			UINT64CONST(0xA000000000000002)
#define PARALLEL_KEY_QUERY_TEXT			UINT64CONST(0xA000000000000003)

#include "miscadmin.h"  // GetCurrentTimestamp()

/*
 * Create the metapage
 */
static void
CreateMetaPage(HnswBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	buf = HnswNewBuffer(index, forkNum);
	page = BufferGetPage(buf);
	HnswInitPage(buf, page);

	/* Set metapage data */
	metap = HnswPageGetMeta(page);
	metap->magicNumber = HNSW_MAGIC_NUMBER;
	metap->version = HNSW_VERSION;
	metap->dimensions = buildstate->dimensions;
	metap->m = buildstate->m;
	metap->efConstruction = buildstate->efConstruction;
	metap->entryBlkno = InvalidBlockNumber;
	metap->entryOffno = InvalidOffsetNumber;
	metap->entryLevel = -1;
	metap->insertPage = InvalidBlockNumber;


    metap->poolSize = 0;

    for (int i = 0; i < MAX_INSERT_POOL_SIZE; i++){
//        metap->items[i].originalPage = InvalidBlockNumber;
        metap->items[i].extendedPage = InvalidBlockNumber;
        metap->items[i].pid = -2;
    }


    ((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(HnswMetaPageData)) - (char *) page;

	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}


static void
CreateMetaPageWithPartition(HnswBuildState * buildstate, HnswPartitionState *countPartitionstate)
{
    Relation	index = buildstate->index;
    ForkNumber	forkNum = buildstate->forkNum;
    Buffer		buf;
    Page		page;
    HnswMetaPage metap;

    buf = HnswNewBuffer(index, forkNum);
    page = BufferGetPage(buf);
    HnswInitPage(buf, page);

    /* Set metapage data */
    metap = HnswPageGetMeta(page);
    metap->magicNumber = HNSW_MAGIC_NUMBER;
    metap->version = HNSW_VERSION;
    metap->dimensions = buildstate->dimensions;
    metap->m = buildstate->m;
    metap->efConstruction = buildstate->efConstruction;
    metap->entryBlkno = InvalidBlockNumber;
    metap->entryOffno = InvalidOffsetNumber;
    metap->entryLevel = -1;
    metap->insertPage = InvalidBlockNumber;


    metap->poolSize = MAX_INSERT_POOL_SIZE - 1;

    for (int i = 1; i < MAX_INSERT_POOL_SIZE; i++){
        metap->items[i].extendedPage = InvalidBlockNumber;
        metap->items[i].pid = countPartitionstate->partitions[i-1].pid;
    }

    ((PageHeader) page)->pd_lower =
            ((char *) metap + sizeof(HnswMetaPageData)) - (char *) page;

    MarkBufferDirty(buf);
    UnlockReleaseBuffer(buf);
}

/*
 * Add a new page
 */
static void
HnswBuildAppendPage(Relation index, Buffer *buf, Page *page, ForkNumber forkNum)
{
	/* Add a new page */
	Buffer		newbuf = HnswNewBuffer(index, forkNum);

	/* Update previous page */
	HnswPageGetOpaque(*page)->nextblkno = BufferGetBlockNumber(newbuf);

	/* Commit */
	MarkBufferDirty(*buf);
	UnlockReleaseBuffer(*buf);

	/* Can take a while, so ensure we can interrupt */
	/* Needs to be called when no buffer locks are held */
	LockBuffer(newbuf, BUFFER_LOCK_UNLOCK);
	CHECK_FOR_INTERRUPTS();
	LockBuffer(newbuf, BUFFER_LOCK_EXCLUSIVE);

	/* Prepare new page */
	*buf = newbuf;
	*page = BufferGetPage(*buf);
	HnswInitPage(*buf, *page);
}

/*
 * Create graph pages
 */
static void
CreateGraphPages(HnswBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	Size		maxSize;
	HnswElementTuple etup;
	HnswNeighborTuple ntup;
	BlockNumber insertPage;
	HnswElement entryPoint;
	Buffer		buf;
	Page		page;
	HnswElementPtr iter = buildstate->graph->head;
	char	   *base = buildstate->hnswarea;

	/* Calculate sizes */
	maxSize = HNSW_MAX_SIZE;

	/* Allocate once */
	etup = palloc0(HNSW_TUPLE_ALLOC_SIZE);
	ntup = palloc0(HNSW_TUPLE_ALLOC_SIZE);

	/* Prepare first page */
	buf = HnswNewBuffer(index, forkNum);
	page = BufferGetPage(buf);
	HnswInitPage(buf, page);

	while (!HnswPtrIsNull(base, iter))
	{
		HnswElement element = HnswPtrAccess(base, iter);
		Size		etupSize;
		Size		ntupSize;
		Size		combinedSize;
		Pointer		valuePtr = HnswPtrAccess(base, element->value);

		/* Update iterator */
		iter = element->next;

		/* Zero memory for each element */
		MemSet(etup, 0, HNSW_TUPLE_ALLOC_SIZE);

		/* Calculate sizes */
		etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(valuePtr));
		ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, buildstate->m);
		combinedSize = etupSize + ntupSize + sizeof(ItemIdData);

		/* Initial size check */
		if (etupSize > HNSW_TUPLE_ALLOC_SIZE)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("index tuple too large")));

		HnswSetElementTuple(base, etup, element);

		/* Keep element and neighbors on the same page if possible */
		if (PageGetFreeSpace(page) < etupSize || (combinedSize <= maxSize && PageGetFreeSpace(page) < combinedSize))
			HnswBuildAppendPage(index, &buf, &page, forkNum);

		/* Calculate offsets */
		element->blkno = BufferGetBlockNumber(buf);
		element->offno = OffsetNumberNext(PageGetMaxOffsetNumber(page));
		if (combinedSize <= maxSize)
		{
			element->neighborPage = element->blkno;
			element->neighborOffno = OffsetNumberNext(element->offno);
		}
		else
		{
			element->neighborPage = element->blkno + 1;
			element->neighborOffno = FirstOffsetNumber;
		}

		ItemPointerSet(&etup->neighbortid, element->neighborPage, element->neighborOffno);

		/* Add element */
		if (PageAddItem(page, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != element->offno)
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

		/* Add new page if needed */
		if (PageGetFreeSpace(page) < ntupSize)
			HnswBuildAppendPage(index, &buf, &page, forkNum);

		/* Add placeholder for neighbors */
		if (PageAddItem(page, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != element->neighborOffno)
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
	}

	insertPage = BufferGetBlockNumber(buf);

	/* Commit */
	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);

	entryPoint = HnswPtrAccess(base, buildstate->graph->entryPoint);
	HnswUpdateMetaPage(index, HNSW_UPDATE_ENTRY_ALWAYS, entryPoint, insertPage, forkNum, true);

	pfree(etup);
	pfree(ntup);
}

static void
CreateGraphPagesWithPartitions(HnswBuildState * buildstate, HnswPartitionState *partitionstate)
{
    Relation	index = buildstate->index;
    ForkNumber	forkNum = buildstate->forkNum;
    Size		maxSize;
    HnswElementTuple etup;
    HnswNeighborTuple ntup;
    BlockNumber insertPage;
    HnswElement entryPoint;
    Buffer		buf;
    Page		page;
//    HnswElementPtr iter = buildstate->graph->head;
    char	   *base = buildstate->hnswarea;

    /* Calculate sizes */
    maxSize = HNSW_MAX_SIZE;

    /* Allocate once */
    etup = palloc0(HNSW_TUPLE_ALLOC_SIZE);
    ntup = palloc0(HNSW_TUPLE_ALLOC_SIZE);

    /* Prepare first page */
    buf = HnswNewBuffer(index, forkNum);
    page = BufferGetPage(buf);
    HnswInitPage(buf, page);

    int element_per_page_counter = 0;

    /* Iterate through partitions */
    for (unsigned i = 0; i < partitionstate->numPartitions; i++) {
        HnswPartition *partition = &partitionstate->partitions[i];

        for (unsigned j = 0; j < partition->size; j++) {
            element_per_page_counter++;
            HnswElement element = HnswPtrAccess(base, partition->nodes[j]);
            Size etupSize;
            Size ntupSize;
            Size combinedSize;
            Pointer valuePtr = HnswPtrAccess(base, element->value);

//            /* Update iterator */
//            iter = element->next;

            /* Zero memory for each element */
            MemSet(etup, 0, HNSW_TUPLE_ALLOC_SIZE);

            /* Calculate sizes */
            etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(valuePtr));
            ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, buildstate->m);
            combinedSize = etupSize + ntupSize + sizeof(ItemIdData);

            /* Initial size check */
            if (etupSize > HNSW_TUPLE_ALLOC_SIZE)
                ereport(ERROR,
                        (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                                errmsg("index tuple too large")));

            HnswSetElementTuple(base, etup, element);

            /* Keep element and neighbors on the same page if possible */
            if (PageGetFreeSpace(page) < etupSize || (combinedSize <= maxSize && PageGetFreeSpace(page) < combinedSize)){
//                element_per_page_counter > 8 || element_per_page_counter > 10 ||
                element_per_page_counter = 1;
                HnswBuildAppendPage(index, &buf, &page, forkNum);
            }

            /* Calculate offsets */
            element->blkno = BufferGetBlockNumber(buf);
            element->offno = OffsetNumberNext(PageGetMaxOffsetNumber(page));
            if (combinedSize <= maxSize) {
                element->neighborPage = element->blkno;
                element->neighborOffno = OffsetNumberNext(element->offno);
            } else {
                element->neighborPage = element->blkno + 1;
                element->neighborOffno = FirstOffsetNumber;
            }

            ItemPointerSet(&etup->neighbortid, element->neighborPage, element->neighborOffno);

            /* Add element */
            if (PageAddItem(page, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != element->offno)
                elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

            /* Add new page if needed */
            if (PageGetFreeSpace(page) < ntupSize)
                HnswBuildAppendPage(index, &buf, &page, forkNum);

            /* Add placeholder for neighbors */
            if (PageAddItem(page, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != element->neighborOffno)
                elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
        }
    }

    insertPage = BufferGetBlockNumber(buf);

    /* Commit */
    MarkBufferDirty(buf);
    UnlockReleaseBuffer(buf);

    entryPoint = HnswPtrAccess(base, buildstate->graph->entryPoint);
    HnswInsertPagePool insertPagePool = NULL;

    HnswUpdateMetaPageWithPartition(index, HNSW_UPDATE_ENTRY_ALWAYS, entryPoint, insertPage, forkNum, true, insertPagePool);
//    HnswUpdateMetaPage(index, HNSW_UPDATE_ENTRY_ALWAYS, entryPoint, insertPage, forkNum, true);

    elog(INFO, "CreateGraphPagesWithPartitions done");

    pfree(etup);
    pfree(ntup);
}


//static void
//CreateGraphPagesWithPartitionsLikeOnDisk(HnswBuildState * buildstate)
//{
//    elog(WARNING, "CreateGraphPagesWithPartitionsLikeOnDisk");
//
//    Relation	index = buildstate->index;
//    ForkNumber	forkNum = buildstate->forkNum;
//    Size		maxSize;
//    HnswElementTuple etup;
//    HnswNeighborTuple ntup;
//    BlockNumber insertPage;
//    HnswElement entryPoint;
//    Buffer		buf;
//    Page		page;
////    HnswElementPtr iter = buildstate->graph->head;
//    char	   *base = buildstate->hnswarea;
//    HnswPartitionState *partitionstate;
//
//    /* Calculate sizes */
//    maxSize = HNSW_MAX_SIZE;
//
//    /* Allocate once */
//    etup = palloc0(HNSW_TUPLE_ALLOC_SIZE);
//    ntup = palloc0(HNSW_TUPLE_ALLOC_SIZE);
//
//    /* Prepare first page */
//    buf = HnswNewBuffer(index, forkNum);
//    page = BufferGetPage(buf);
//    HnswInitPage(buf, page);
//
//    partitionstate = buildstate->partitionstate;
//
//
//    /* Iterate through partitions */
//    for (unsigned i = 0; i < partitionstate->numPartitions; i++) {
//        HnswPartition *partition = &partitionstate->partitions[i];
//
//        for (unsigned j = 0; j < partition->size; j++) {
//
//            HnswElement element = HnswPtrAccess(base, partition->nodes[j]);
//            Size etupSize;
//            Size ntupSize;
//            Size combinedSize;
//            Pointer valuePtr = HnswPtrAccess(base, element->value);
//
//            /* Zero memory for each element */
//            MemSet(etup, 0, HNSW_TUPLE_ALLOC_SIZE);
//
//            /* Calculate sizes */
//            etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(valuePtr));
//            ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, buildstate->m);
//            combinedSize = etupSize + ntupSize + sizeof(ItemIdData);
//
//            /* Initial size check */
//            if (etupSize > HNSW_TUPLE_ALLOC_SIZE)
//                ereport(ERROR,
//                        (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
//                                errmsg("index tuple too large")));
//
//            HnswSetElementTuple(base, etup, element);
//
//            /* Keep element and neighbors on the same page if possible */
//            if (PageGetFreeSpace(page) < etupSize || (combinedSize <= maxSize && PageGetFreeSpace(page) < combinedSize)){
////                element_per_page_counter > 8 || element_per_page_counter > 10 ||
//                HnswBuildAppendPage(index, &buf, &page, forkNum);
//            }
//
//            /* Calculate offsets */
//            element->blkno = BufferGetBlockNumber(buf);
//            element->offno = OffsetNumberNext(PageGetMaxOffsetNumber(page));
//            if (combinedSize <= maxSize) {
//                element->neighborPage = element->blkno;
//                element->neighborOffno = OffsetNumberNext(element->offno);
//            } else {
//                element->neighborPage = element->blkno + 1;
//                element->neighborOffno = FirstOffsetNumber;
//            }
//
//            ItemPointerSet(&etup->neighbortid, element->neighborPage, element->neighborOffno);
//
//            /* Add element */
//            if (PageAddItem(page, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != element->offno)
//                elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
//
//            /* Add new page if needed */
//            if (PageGetFreeSpace(page) < ntupSize)
//                HnswBuildAppendPage(index, &buf, &page, forkNum);
//
//            /* Add placeholder for neighbors */
//            if (PageAddItem(page, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != element->neighborOffno)
//                elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
//        }
//    }
//
//    if (buildstate->insertPartitionstate != NULL){
//
//        elog(WARNING, "buildstate->insertPartitionstate != NULL");
//
//        /* Iterate through partitions */
//        for (unsigned i = 0; i <= MAX_INSERT_POOL_SIZE; i++) {
//            HnswPartition *partition = &buildstate->insertPartitionstate->partitions[i];
//            HnswBuildAppendPage(index, &buf, &page, forkNum);
//
//            for (unsigned j = 0; j < partition->size; j++) {
//
//                HnswElement element = HnswPtrAccess(base, partition->nodes[j]);
//                Size etupSize;
//                Size ntupSize;
//                Size combinedSize;
//                Pointer valuePtr = HnswPtrAccess(base, element->value);
//
//                /* Zero memory for each element */
//                MemSet(etup, 0, HNSW_TUPLE_ALLOC_SIZE);
//
//                /* Calculate sizes */
//                etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(valuePtr));
//                ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, buildstate->m);
//                combinedSize = etupSize + ntupSize + sizeof(ItemIdData);
//
//                /* Initial size check */
//                if (etupSize > HNSW_TUPLE_ALLOC_SIZE)
//                    ereport(ERROR,
//                            (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
//                                    errmsg("index tuple too large")));
//
//                HnswSetElementTuple(base, etup, element);
//
//                /* Keep element and neighbors on the same page if possible */
//                if (PageGetFreeSpace(page) < etupSize || (combinedSize <= maxSize && PageGetFreeSpace(page) < combinedSize)){
////                element_per_page_counter > 8 || element_per_page_counter > 10 ||
//                    HnswBuildAppendPage(index, &buf, &page, forkNum);
//                }
//
//                /* Calculate offsets */
//                element->blkno = BufferGetBlockNumber(buf);
//                element->offno = OffsetNumberNext(PageGetMaxOffsetNumber(page));
//                if (combinedSize <= maxSize) {
//                    element->neighborPage = element->blkno;
//                    element->neighborOffno = OffsetNumberNext(element->offno);
//                } else {
//                    element->neighborPage = element->blkno + 1;
//                    element->neighborOffno = FirstOffsetNumber;
//                }
//
//                ItemPointerSet(&etup->neighbortid, element->neighborPage, element->neighborOffno);
//
//                /* Add element */
//                if (PageAddItem(page, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != element->offno)
//                    elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
//
//                /* Add new page if needed */
//                if (PageGetFreeSpace(page) < ntupSize)
//                    HnswBuildAppendPage(index, &buf, &page, forkNum);
//
//                /* Add placeholder for neighbors */
//                if (PageAddItem(page, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != element->neighborOffno)
//                    elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
//            }
//        }
//
//    }
//
//
//    insertPage = BufferGetBlockNumber(buf);
//
//    /* Commit */
//    MarkBufferDirty(buf);
//    UnlockReleaseBuffer(buf);
//
//    entryPoint = HnswPtrAccess(base, buildstate->graph->entryPoint);
////    HnswInsertPagePool insertPagePool = NULL;
//
////    HnswUpdateMetaPageWithPartition(index, HNSW_UPDATE_ENTRY_ALWAYS, entryPoint, insertPage, forkNum, true, insertPagePool);
//    HnswUpdateMetaPage(index, HNSW_UPDATE_ENTRY_ALWAYS, entryPoint, insertPage, forkNum, true);
//
//    elog(INFO, "CreateGraphPagesWithPartitions done");
//
//    pfree(etup);
//    pfree(ntup);
//}


/*
 * Write neighbor tuples
 */
static void
WriteNeighborTuples(HnswBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	int			m = buildstate->m;
	HnswElementPtr iter = buildstate->graph->head;
	char	   *base = buildstate->hnswarea;
	HnswNeighborTuple ntup;

	/* Allocate once */
	ntup = palloc0(HNSW_TUPLE_ALLOC_SIZE);

	while (!HnswPtrIsNull(base, iter))
	{
		HnswElement element = HnswPtrAccess(base, iter);
		Buffer		buf;
		Page		page;
		Size		ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, m);

		/* Update iterator */
		iter = element->next;

		/* Zero memory for each element */
		MemSet(ntup, 0, HNSW_TUPLE_ALLOC_SIZE);

		/* Can take a while, so ensure we can interrupt */
		/* Needs to be called when no buffer locks are held */
		CHECK_FOR_INTERRUPTS();

		buf = ReadBufferExtended(index, forkNum, element->neighborPage, RBM_NORMAL, NULL);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);

		HnswSetNeighborTuple(base, ntup, element, m);

		if (!PageIndexTupleOverwrite(page, element->neighborOffno, (Item) ntup, ntupSize))
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

		/* Commit */
		MarkBufferDirty(buf);
		UnlockReleaseBuffer(buf);
	}

	pfree(ntup);
}


/*
 * Write neighbor tuples
 */
static void
WriteNeighborTuplesWithPartitions(HnswBuildState * buildstate, HnswPartitionState *partitionstate)
{
    Relation	index = buildstate->index;
    ForkNumber	forkNum = buildstate->forkNum;
    int			m = buildstate->m;
//    HnswElementPtr iter = buildstate->graph->head;
    char	   *base = buildstate->hnswarea;
    HnswNeighborTuple ntup;

    /* Allocate once */
    ntup = palloc0(HNSW_TUPLE_ALLOC_SIZE);

    /* Iterate through partitions */
    for (unsigned i = 0; i < partitionstate->numPartitions; i++)
    {
        HnswPartition *partition = &partitionstate->partitions[i];

        for (unsigned j = 0; j < partition->size; j++)
        {
            HnswElement element = HnswPtrAccess(base, partition->nodes[j]);
            Buffer buf;
            Page page;
            Size ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, m);

            /* Update iterator */
//            iter = element->next;

            /* Zero memory for each element */
            MemSet(ntup, 0, HNSW_TUPLE_ALLOC_SIZE);

            /* Can take a while, so ensure we can interrupt */
            /* Needs to be called when no buffer locks are held */
            CHECK_FOR_INTERRUPTS();

            buf = ReadBufferExtended(index, forkNum, element->neighborPage, RBM_NORMAL, NULL);
            LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
            page = BufferGetPage(buf);

            HnswSetNeighborTuple(base, ntup, element, m);

            if (!PageIndexTupleOverwrite(page, element->neighborOffno, (Item) ntup, ntupSize))
                elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

            /* Commit */
            MarkBufferDirty(buf);
            UnlockReleaseBuffer(buf);
        }
    }


    pfree(ntup);

    elog(INFO, "WriteNeighborTuplesWithPartitions done");
}

//
///*
// * Write neighbor tuples
// */
//static void
//WriteNeighborTuplesWithPartitionsLikeOnDisk(HnswBuildState * buildstate)
//{
//    elog(WARNING, "CreateGraphPagesWithPartitionsLikeOnDisk");
//
//    Relation	index = buildstate->index;
//    ForkNumber	forkNum = buildstate->forkNum;
//    int			m = buildstate->m;
////    HnswElementPtr iter = buildstate->graph->head;
//    char	   *base = buildstate->hnswarea;
//    HnswNeighborTuple ntup;
//    HnswPartitionState *partitionstate;
//
//    /* Allocate once */
//    ntup = palloc0(HNSW_TUPLE_ALLOC_SIZE);
//
//    partitionstate = buildstate->partitionstate;
//
//    /* Iterate through partitions */
//    for (unsigned i = 0; i < partitionstate->numPartitions; i++)
//    {
//        HnswPartition *partition = &partitionstate->partitions[i];
//
//        for (unsigned j = 0; j < partition->size; j++)
//        {
//            HnswElement element = HnswPtrAccess(base, partition->nodes[j]);
//            Buffer buf;
//            Page page;
//            Size ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, m);
//
//            /* Update iterator */
////            iter = element->next;
//
//            /* Zero memory for each element */
//            MemSet(ntup, 0, HNSW_TUPLE_ALLOC_SIZE);
//
//            /* Can take a while, so ensure we can interrupt */
//            /* Needs to be called when no buffer locks are held */
//            CHECK_FOR_INTERRUPTS();
//
//            buf = ReadBufferExtended(index, forkNum, element->neighborPage, RBM_NORMAL, NULL);
//            LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
//            page = BufferGetPage(buf);
//
//            HnswSetNeighborTuple(base, ntup, element, m);
//
//            if (!PageIndexTupleOverwrite(page, element->neighborOffno, (Item) ntup, ntupSize))
//                elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
//
//            /* Commit */
//            MarkBufferDirty(buf);
//            UnlockReleaseBuffer(buf);
//        }
//    }
//
//    if (buildstate->insertPartitionstate != NULL){
//        /* Iterate through partitions */
//        elog(WARNING, "buildstate->insertPartitionstate != NULL");
//
//        for (unsigned i = 0; i <= MAX_INSERT_POOL_SIZE; i++){
//            HnswPartition *partition = &buildstate->insertPartitionstate->partitions[i];
//
//            for (unsigned j = 0; j < partition->size; j++)
//            {
//                HnswElement element = HnswPtrAccess(base, partition->nodes[j]);
//                Buffer buf;
//                Page page;
//                Size ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, m);
//
//                /* Update iterator */
////            iter = element->next;
//
//                /* Zero memory for each element */
//                MemSet(ntup, 0, HNSW_TUPLE_ALLOC_SIZE);
//
//                /* Can take a while, so ensure we can interrupt */
//                /* Needs to be called when no buffer locks are held */
//                CHECK_FOR_INTERRUPTS();
//
//                buf = ReadBufferExtended(index, forkNum, element->neighborPage, RBM_NORMAL, NULL);
//                LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
//                page = BufferGetPage(buf);
//
//                HnswSetNeighborTuple(base, ntup, element, m);
//
//                if (!PageIndexTupleOverwrite(page, element->neighborOffno, (Item) ntup, ntupSize))
//                    elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
//
//                /* Commit */
//                MarkBufferDirty(buf);
//                UnlockReleaseBuffer(buf);
//            }
//        }
//    }
//
//    pfree(ntup);
//
//    elog(INFO, "WriteNeighborTuplesWithPartitions done");
//}



/*
 * Flush pages
 */
static void
FlushPages(HnswBuildState * buildstate)
{
#ifdef HNSW_MEMORY
	elog(INFO, "memory: %zu MB", buildstate->graph->memoryUsed / (1024 * 1024));
#endif

	CreateMetaPage(buildstate);
	CreateGraphPages(buildstate);
	WriteNeighborTuples(buildstate);

	buildstate->graph->flushed = true;
	MemoryContextReset(buildstate->graphCtx);
}

static void
FlushPagesWithPartitions(HnswBuildState * buildstate, HnswPartitionState *partitionstate, HnswPartitionState *countPartitionstate)
{
#ifdef HNSW_MEMORY
    elog(INFO, "memory: %zu MB", buildstate->graph->memoryUsed / (1024 * 1024));
#endif

//    CreateMetaPage(buildstate); // insert pool !!!
    CreateMetaPageWithPartition(buildstate, countPartitionstate);
    CreateGraphPagesWithPartitions(buildstate, partitionstate);
    WriteNeighborTuplesWithPartitions(buildstate, partitionstate);

    buildstate->graph->flushed = true;
    MemoryContextReset(buildstate->graphCtx);

    elog(INFO, "FlushPagesWithPartitions done");
}

//
//static void
//FlushPagesWithPartitionsLikeOnDisk(HnswBuildState * buildstate)
//{
//#ifdef HNSW_MEMORY
//    elog(INFO, "memory: %zu MB", buildstate->graph->memoryUsed / (1024 * 1024));
//#endif
//
//    CreateMetaPage(buildstate); // insert pool !!!
////    CreateMetaPageWithPartition(buildstate, countPartitionstate);
////    CreateGraphPagesWithPartitionsLikeOnDisk(buildstate);
////    WriteNeighborTuplesWithPartitionsLikeOnDisk(buildstate);
//
//    buildstate->graph->flushed = true;
//    MemoryContextReset(buildstate->graphCtx);
//
//    elog(INFO, "FlushPagesWithPartitions done");
//}
//






static HnswPartitionState *
InitPartitionState(HnswBuildState *buildstate, int maxNodesPerPartition, HnswAllocator *allocator)
{
    int numNodes = (int)buildstate->graph->indtuples;
    int numPartitions = (numNodes + maxNodesPerPartition - 1) / maxNodesPerPartition;

    /* 전체 메모리 한 번에 할당 (HnswPartitionState + HnswPartition * numPartitions) */
    Size totalSize = sizeof(HnswPartitionState) + numPartitions * sizeof(HnswPartition);
    HnswPartitionState *partitionstate = HnswAlloc(allocator, totalSize);
    partitionstate->numPartitions = numPartitions;

    /* HnswPartition 배열을 슬라이싱하여 사용 */
    partitionstate->partitions = (HnswPartition *)((char *)partitionstate + sizeof(HnswPartitionState));

    /* 각 파티션 초기화 */
    for (int i = 0; i < numPartitions; i++)
    {
        HnswPartition *partition = &partitionstate->partitions[i];
        partition->capacity = maxNodesPerPartition;
        partition->size = 0;
        partition->pid = i;

        /* 각 파티션 내부의 nodes 배열을 상대 포인터로 할당 */
//        partition->nodes = HnswAlloc(allocator, sizeof(HnswElementPtr) * maxNodesPerPartition);
        partition->nodes = malloc(sizeof(HnswElementPtr) * maxNodesPerPartition);
    }

    return partitionstate;
}


/* 가득 차지 않은 파티션을 찾는 함수 */
static int
GetUnfilledPartition(HnswPartitionState *partitionstate, pairingheap *heap)
{

    if (pairingheap_is_empty(heap))
        return partitionstate->numPartitions;

    HnswPartition *unfilledPartition = HnswGetPartition(heapNode, pairingheap_remove_first(heap));
    return unfilledPartition->pid;
}



static int
SelectPartition(HnswPartitionState *oldPartitionstate, HnswPartitionState *newPartitionstate, HnswElementPtr elementPtr, char *base, pairingheap *heap)
{
    int bestPartition = oldPartitionstate->numPartitions; /* 초기화: 유효하지 않은 값 */
    int maxScore = 0;

    /* 각 파티션에 대한 이웃 노드 빈도 계산 */
    int *partitionScores = palloc0(sizeof(int) * oldPartitionstate->numPartitions);

    HnswElement element = HnswPtrAccess(base, elementPtr);
    HnswNeighborArray *neighbors = HnswGetNeighbors(base, element, 0);
    for (int j = 0; j < neighbors->length; j++)
    {
        HnswElementPtr neighborPtr = neighbors->items[j].element;
        HnswElement neighborElement = HnswPtrAccess(base, neighborPtr);
        int pid = neighborElement->pid;
        partitionScores[pid]++;

//        elog(INFO, "Neighbor %d (ptr: %p) belongs to partition %d, count in partition: %d, (%d/%d)",
//             j, neighborPtr, pid, partitionScores[pid],
//             newPartitionstate->partitions[pid].size, newPartitionstate->partitions[pid].capacity);

        int score = partitionScores[pid];

//        if (partitionScores[pid] > maxScore &&
        if (score > maxScore &&
            newPartitionstate->partitions[pid].size < newPartitionstate->partitions[pid].capacity)
        {
            bestPartition = pid;
            maxScore = score;
        }

    }

    /* 모든 파티션이 가득 찬 경우 처리 */

    if (bestPartition < newPartitionstate->numPartitions){
        pairingheap_remove(heap, &newPartitionstate->partitions[bestPartition].heapNode);
//        elog(INFO, "Selected bestPartition: %d with score %d", bestPartition, maxScore);
    }
    if (bestPartition == newPartitionstate->numPartitions){
        bestPartition = GetUnfilledPartition(newPartitionstate, heap);
//        elog(INFO, "GetUnfilledPartition returned: %d", bestPartition);
    }


    pfree(partitionScores);
    return bestPartition;
}

static void
AddNodeToPartition(HnswPartitionState *partitionstate, HnswElementPtr nodePtr, int partitionId, char *base, pairingheap *heap)
{
    HnswPartition *partition = &partitionstate->partitions[partitionId];
    HnswElement element = HnswPtrAccess(base, nodePtr);

    if (partition->size >= partition->capacity)
    {
        elog(ERROR, "Partition %d is full!", partitionId);
        return;
    }

    partition->nodes[partition->size] = nodePtr;
    partition->size++;

    element->pid = partitionId;

    if (partition->size < partition->capacity)
    {
        pairingheap_add(heap, &partition->heapNode);
    }
}

static void
SyncNodeToBestPartition(HnswPartitionState *oldPartitionstate, HnswPartitionState *newPartitionstate, HnswElementPtr elementPtr, char *base, pairingheap *heap)
{
    /* 최적의 파티션을 선택 */
    int bestPartition = SelectPartition(oldPartitionstate, newPartitionstate, elementPtr, base, heap);

    /* 최적의 파티션에 노드를 추가 */
    AddNodeToPartition(newPartitionstate, elementPtr, bestPartition, base, heap);
}


static int
ComparePartitionSize(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
    if (HnswGetPartitionConst(heapNode, a)->size < HnswGetPartitionConst(heapNode, b)->size)
        return -1;

    if (HnswGetPartitionConst(heapNode, a)->size > HnswGetPartitionConst(heapNode, b)->size)
        return 1;

    return 0;
}

/* 단일 LDG 수행 */
static void
HnswPartitionGraphLDG(HnswBuildState *buildstate, HnswPartitionState *oldPartitionstate, HnswPartitionState *newPartitionstate)
{
    HnswGraph *graph = buildstate->graph;
    char *base = buildstate->hnswarea;

    pairingheap *heap = pairingheap_allocate(ComparePartitionSize, NULL);
    for (int i = 0; i < newPartitionstate->numPartitions; i++)
    {
        pairingheap_add(heap, &newPartitionstate->partitions[i].heapNode);
    }

    /* 그래프 순회 */
    HnswElementPtr iter = graph->head;
    while (!HnswPtrIsNull(base, iter))
    {
        /* 노드를 최적의 파티션에 동기화 */
        SyncNodeToBestPartition(oldPartitionstate, newPartitionstate, iter, base, heap);
        iter = HnswPtrAccess(base, iter)->next;
    }
}

static HnswPartitionState *
HnswPartitionGraph(HnswBuildState *buildstate)
{
    int maxNodesPerPartition = 64;
    int numIterations = 61;
    HnswAllocator *allocator = &buildstate->allocator;
    elog(INFO, "maxNodesPerPartition: %d", maxNodesPerPartition);

    /* 파티션 상태 초기화 */
    HnswPartitionState *partitionstate = InitPartitionState(buildstate, maxNodesPerPartition, allocator);

    HnswGraph *graph = buildstate->graph;
    char *base = buildstate->hnswarea;
    int partitionIdx = 0; /* 현재 파티션 인덱스 */
    int nodesPerPartition = partitionstate->partitions[0].capacity;
    HnswElementPtr iter = graph->head;

    int cnt = 0;
    int enterCnt = 0;

    elog(INFO, "Starting initial partitioning...");

    /* 그래프 순회 및 초기 배정 */
    while (!HnswPtrIsNull(base, iter))
    {
        cnt++;
        HnswElement element = HnswPtrAccess(base, iter);

        /* 이미 파티션에 배정된 노드인지 확인 */
        int assignedPartition = element->pid;
        if (assignedPartition != -1)
        {
            iter = element->next;
            continue;
        }

        /* 현재 노드를 파티션에 추가 */
        HnswPartition *currentPartition = &partitionstate->partitions[partitionIdx];
        currentPartition->nodes[currentPartition->size++] = iter;
        element->pid = partitionIdx;
        enterCnt++;

        /* 연결된 노드 추가 */
        HnswNeighborArray *neighbors = HnswGetNeighbors(base, element, 0);
        for (int i = 0; i < neighbors->length; i++)
        {
            HnswElementPtr neighborPtr = neighbors->items[i].element;
            HnswElement neighborElement = HnswPtrAccess(base, neighborPtr);

            /* 이웃 노드가 이미 배정된 경우 스킵 */
            if (neighborElement->pid != -1)
                continue;

            /* 파티션이 가득 차면 다음 파티션으로 이동 */
            if (currentPartition->size >= nodesPerPartition)
            {
                break;
            }

            /* 이웃 노드를 파티션에 추가 */
            currentPartition->nodes[currentPartition->size++] = neighborPtr;
            neighborElement->pid = partitionIdx;
            enterCnt++;
        }

        /* 현재 파티션이 가득 차면 다음 파티션으로 이동 */
        if (currentPartition->size >= nodesPerPartition && partitionIdx < partitionstate->numPartitions - 1)
        {
            partitionIdx++;
        }

        iter = element->next;
    }

    elog(INFO, "Initial partitioning completed. Total nodes: %d, Assigned nodes: %d", cnt, enterCnt);

    HnswPartitionState *newPartitionstate;

    /* LDG 반복 수행 */
    for (unsigned iteration = 0; iteration < numIterations; iteration++)
    {
        TimestampTz start_time = GetCurrentTimestamp();

//        PartitionStatistic(partitionstate, buildstate);
        newPartitionstate = InitPartitionState(buildstate, maxNodesPerPartition, allocator);
        HnswPartitionGraphLDG(buildstate, partitionstate, newPartitionstate);

        TimestampTz end_time = GetCurrentTimestamp();
        double elapsed_ms = (end_time - start_time) / 1000.0;

        elog(WARNING, "LDG iteration %u completed in %.3f ms", iteration + 1, elapsed_ms);

        partitionstate = newPartitionstate;
    }

    graph->partitioned = true;

    return partitionstate;
}


static int
SelectPartitionForInsert(HnswPartitionState *partitionstate, HnswElementPtr elementPtr, char *base)
{
    int bestPartition = partitionstate->numPartitions;  /* 초기화: 유효하지 않은 값 */
    int maxScore = -1;

    int *partitionScores = palloc0(sizeof(int) * partitionstate->numPartitions);

    HnswElement element = HnswPtrAccess(base, elementPtr);
    HnswNeighborArray *neighbors = HnswGetNeighbors(base, element, 0);

    /* 이웃 노드를 기반으로 파티션 점수 계산 */
    for (int j = 0; j < neighbors->length; j++) {
        HnswElementPtr neighborPtr = neighbors->items[j].element;
        HnswElement neighborElement = HnswPtrAccess(base, neighborPtr);
        int pid = neighborElement->pid;

        partitionScores[pid]++;

        if (partitionScores[pid] > maxScore) {
            bestPartition = pid;
            maxScore = partitionScores[pid];
        }
    }

    pfree(partitionScores);
    return bestPartition;
}

static int
ComparePartitionSizeDesc(const void *a, const void *b)
{
    const HnswPartition *partitionA = (const HnswPartition *)a;
    const HnswPartition *partitionB = (const HnswPartition *)b;

    return partitionB->size - partitionA->size;
}

static HnswPartitionState *
CountPartitionSizeForInsert(HnswBuildState *buildstate, HnswPartitionState *partitionstate)
{
    HnswAllocator *allocator = &buildstate->allocator;
    HnswGraph *graph = buildstate->graph;
    char *base = buildstate->hnswarea;
    int maxNodesPerPartition = partitionstate->partitions[0].capacity;

    HnswPartitionState *countPartitionstate;

    countPartitionstate = InitPartitionState(buildstate, maxNodesPerPartition, allocator);

    /* 그래프 순회 */
    HnswElementPtr iter = graph->head;
    while (!HnswPtrIsNull(base, iter))
    {

        int bestPartition = SelectPartitionForInsert(partitionstate, iter, base);
        if (bestPartition < countPartitionstate->numPartitions) {
            countPartitionstate->partitions[bestPartition].size++;
        }
        iter = HnswPtrAccess(base, iter)->next;
    }

    /* 파티션별 노드 수 내림차순 정렬 */
    qsort(countPartitionstate->partitions, countPartitionstate->numPartitions, sizeof(HnswPartition), ComparePartitionSizeDesc);

//    /* 결과 출력 */
//    elog(INFO, "Partition Count results:");
//    for (int i = 0; i < countPartitionstate->numPartitions; i++) {
//        elog(INFO, "Partition %d: %d nodes", countPartitionstate->partitions[i].pid, countPartitionstate->partitions[i].size);
//    }

    return countPartitionstate;
}


static HnswPartitionState *
CountOverlapRatioForInsert(HnswBuildState *buildstate, HnswPartitionState *partitionstate)
{
    HnswAllocator *allocator = &buildstate->allocator;
    HnswGraph *graph = buildstate->graph;
    char *base = buildstate->hnswarea;
    int maxNodesPerPartition = partitionstate->partitions[0].capacity;

    HnswPartitionState *countPartitionstate;

    countPartitionstate = InitPartitionState(buildstate, maxNodesPerPartition, allocator);

    for (int i = 0; i < partitionstate->numPartitions; i++) {
        HnswPartition *currentPartition = &partitionstate->partitions[i];

        for (int j = 0; j < currentPartition->size; j++) {
            HnswElementPtr elementPtr = currentPartition->nodes[j];
            HnswElement element = HnswPtrAccess(base, elementPtr);
            HnswNeighborArray *neighbors = HnswGetNeighbors(base, element, 0);

            for (int k = 0; k < neighbors->length; k++) {
                HnswElementPtr neighborPtr = neighbors->items[k].element;
                HnswElement neighborElement = HnswPtrAccess(base, neighborPtr);

                if (element->pid == neighborElement->pid) {
                    countPartitionstate->partitions[i].size++;
                }
            }
        }

        countPartitionstate->partitions[i].size = (currentPartition->size > 0) ?
                                                  countPartitionstate->partitions[i].size / currentPartition->size : 0;

    }

    qsort(countPartitionstate->partitions, countPartitionstate->numPartitions, sizeof(HnswPartition), ComparePartitionSizeDesc);

//    /* 결과 출력 */
//    elog(INFO, "Overlap Cnt results:");
//    for (int i = 0; i < countPartitionstate->numPartitions; i++) {
//        elog(INFO, "Partition %d OR: %d ", countPartitionstate->partitions[i].pid, countPartitionstate->partitions[i].size);
//    }

    return countPartitionstate;
}





/*
 * Add a heap TID to an existing element
 */
static bool
AddDuplicateInMemory(HnswElement element, HnswElement dup)
{
	LWLockAcquire(&dup->lock, LW_EXCLUSIVE);

	if (dup->heaptidsLength == HNSW_HEAPTIDS)
	{
		LWLockRelease(&dup->lock);
		return false;
	}

	HnswAddHeapTid(dup, &element->heaptids[0]);

	LWLockRelease(&dup->lock);

	return true;
}

/*
 * Find duplicate element
 */
static bool
FindDuplicateInMemory(char *base, HnswElement element)
{
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

		/* Check for space */
		if (AddDuplicateInMemory(element, neighborElement))
			return true;
	}

	return false;
}

/*
 * Add to element list
 */
static void
AddElementInMemory(char *base, HnswGraph * graph, HnswElement element)
{
	SpinLockAcquire(&graph->lock);
	element->next = graph->head;
	HnswPtrStore(base, graph->head, element);
	SpinLockRelease(&graph->lock);
}

/*
 * Update neighbors
 */
static void
UpdateNeighborsInMemory(char *base, HnswSupport * support, HnswElement e, int m)
{
	for (int lc = e->level; lc >= 0; lc--)
	{
		int			lm = HnswGetLayerM(m, lc);
		Size		neighborsSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		HnswNeighborArray *neighbors = palloc(neighborsSize);

		/* Copy neighbors to local memory */
		LWLockAcquire(&e->lock, LW_SHARED);
		memcpy(neighbors, HnswGetNeighbors(base, e, lc), neighborsSize);
		LWLockRelease(&e->lock);

		for (int i = 0; i < neighbors->length; i++)
		{
			HnswCandidate *hc = &neighbors->items[i];
			HnswElement neighborElement = HnswPtrAccess(base, hc->element);

			/* Keep scan-build happy on Mac x86-64 */
			Assert(neighborElement);

			LWLockAcquire(&neighborElement->lock, LW_EXCLUSIVE);
			HnswUpdateConnection(base, HnswGetNeighbors(base, neighborElement, lc), e, hc->distance, lm, NULL, NULL, support);
			LWLockRelease(&neighborElement->lock);
		}
	}
}

/*
 * Update graph in memory
 */
static void
UpdateGraphInMemory(HnswSupport * support, HnswElement element, int m, int efConstruction, HnswElement entryPoint, HnswBuildState * buildstate)
{
	HnswGraph  *graph = buildstate->graph;
	char	   *base = buildstate->hnswarea;

	/* Look for duplicate */
	if (FindDuplicateInMemory(base, element))
		return;

	/* Add element */
	AddElementInMemory(base, graph, element);

	/* Update neighbors */
	UpdateNeighborsInMemory(base, support, element, m);

	/* Update entry point if needed (already have lock) */
	if (entryPoint == NULL || element->level > entryPoint->level)
		HnswPtrStore(base, graph->entryPoint, element);
}

/*
 * Insert tuple in memory
 */
static void
InsertTupleInMemory(HnswBuildState * buildstate, HnswElement element)
{
	HnswGraph  *graph = buildstate->graph;
	HnswSupport *support = &buildstate->support;
	HnswElement entryPoint;
	LWLock	   *entryLock = &graph->entryLock;
	LWLock	   *entryWaitLock = &graph->entryWaitLock;
	int			efConstruction = buildstate->efConstruction;
	int			m = buildstate->m;
	char	   *base = buildstate->hnswarea;

	/* Wait if another process needs exclusive lock on entry lock */
	LWLockAcquire(entryWaitLock, LW_EXCLUSIVE);
	LWLockRelease(entryWaitLock);

	/* Get entry point */
	LWLockAcquire(entryLock, LW_SHARED);
	entryPoint = HnswPtrAccess(base, graph->entryPoint);

	/* Prevent concurrent inserts when likely updating entry point */
	if (entryPoint == NULL || element->level > entryPoint->level)
	{
		/* Release shared lock */
		LWLockRelease(entryLock);

		/* Tell other processes to wait and get exclusive lock */
		LWLockAcquire(entryWaitLock, LW_EXCLUSIVE);
		LWLockAcquire(entryLock, LW_EXCLUSIVE);
		LWLockRelease(entryWaitLock);

		/* Get latest entry point after lock is acquired */
		entryPoint = HnswPtrAccess(base, graph->entryPoint);
	}

	/* Find neighbors for element */
	HnswFindElementNeighbors(base, element, entryPoint, NULL, support, m, efConstruction, false);

	/* Update graph in memory */
	UpdateGraphInMemory(support, element, m, efConstruction, entryPoint, buildstate);

	/* Release entry lock */
	LWLockRelease(entryLock);
}


static int
ComparePageNeighbors(const void *a, const void *b)
{
    return ((HnswPageNeighborCount *)b)->neighborCount - ((HnswPageNeighborCount *)a)->neighborCount;
}
//
//
//static void
//InsertTupleInMemoryLikeOnDisk(HnswBuildState * buildstate, HnswElement element)
//{
//    HnswGraph  *graph = buildstate->graph;
//    HnswSupport *support = &buildstate->support;
//    HnswElement entryPoint;
//    LWLock	   *entryLock = &graph->entryLock;
//    LWLock	   *entryWaitLock = &graph->entryWaitLock;
//    int			efConstruction = buildstate->efConstruction;
//    int			m = buildstate->m;
//    char	   *base = buildstate->hnswarea;
//
//    int pidE;
//    HnswPartition *partition;
//
//    /* Wait if another process needs exclusive lock on entry lock */
//    LWLockAcquire(entryWaitLock, LW_EXCLUSIVE);
//    LWLockRelease(entryWaitLock);
//
//    /* Get entry point */
//    LWLockAcquire(entryLock, LW_SHARED);
//    entryPoint = HnswPtrAccess(base, graph->entryPoint);
//
//    /* Prevent concurrent inserts when likely updating entry point */
//    if (entryPoint == NULL || element->level > entryPoint->level)
//    {
//        /* Release shared lock */
//        LWLockRelease(entryLock);
//
//        /* Tell other processes to wait and get exclusive lock */
//        LWLockAcquire(entryWaitLock, LW_EXCLUSIVE);
//        LWLockAcquire(entryLock, LW_EXCLUSIVE);
//        LWLockRelease(entryWaitLock);
//
//        /* Get latest entry point after lock is acquired */
//        entryPoint = HnswPtrAccess(base, graph->entryPoint);
//    }
//
//    /* Find neighbors for element */
//    HnswFindElementNeighbors(base, element, entryPoint, NULL, support, m, efConstruction, false);
//
//    /* Update graph in memory */
//    UpdateGraphInMemory(support, element, m, efConstruction, entryPoint, buildstate);
//
//    /* Release entry lock */
//    LWLockRelease(entryLock);
//
//
//    // 여기서 그냥 partition 배정만 해주면 될듯??? -> on-disk와 동일하게, neighbor count 하고, 복잡하게 할 필요 있나 .>?
//    HnswInsertPageCandidate pageCandidates = CalculatePartitionNeighborCount(element);
//
//    // spare page 하나 추가
//    pageCandidates->items[pageCandidates->length].pid = -2;
//    pageCandidates->items[pageCandidates->length].neighborCount = 0;
//    pageCandidates->items[pageCandidates->length].pageType = SPARE_PAGE;
//    pageCandidates->length++;
//
//    // 각 insertpage entry에 대해 pagecandidate 확인해서 pageCandidate 업데이트
//    for (int i = 0; i < MAX_INSERT_POOL_SIZE - 1; i++)
//    {
//        // i번째 partition
//        int pid = buildstate->countPartitionstate->partitions[i].pid;
//
//        /* Replace classic page with its inserted page */
//        for (int j = 0; j < pageCandidates->length; j++) {
//            // page candidate의 originalPage, extendedPage 존재 여부 확인
//            if (pageCandidates->items[j].pid == pid) {
//                pageCandidates->items[j].pageType = EXTENDED_PAGE;
//                pageCandidates->items[j].blkno = (BlockNumber) i;
//            }
//        }
//    }
//
//    qsort(pageCandidates->items, pageCandidates->length, sizeof(HnswPageNeighborCount), ComparePageNeighbors);
//
////    elog(WARNING, "Insert Page Pool (Size: %d)");
////    for (int i = 0; i < MAX_INSERT_POOL_SIZE; i++) {
////        elog(WARNING, "[Pool Entry %d] pid: %d",
////             i,
////             buildstate->countPartitionstate->partitions[i].pid);
////    }
////
////    elog(WARNING, "Page Candidates (Total: %d)", pageCandidates->length);
////    for (int i = 0; i < pageCandidates->length; i++) {
////        elog(WARNING, "[Candidate %d] pid: %d, neighborCount: %d, pageType: %d",
////             i,
////             pageCandidates->items[i].pid,
////             pageCandidates->items[i].neighborCount,
////             pageCandidates->items[i].pageType);
////    }
//
//    for (int i = 0; i < pageCandidates->length; i++) {
//
//        int tempPid = pageCandidates->items[i].pid;
//
//        if (pageCandidates->items[i].pageType == ORIGINAL_PAGE) {
//            continue;
//        } else if (pageCandidates->items[i].pageType == EXTENDED_PAGE) {
////            elog(WARNING, "Page type: EXTENDED ");
//            // pool에 존재 -> 해당 파티션에 배정
//            // insert 되는 애들 담아둘 자료구조 필요
//            partition = &buildstate->insertPartitionstate->partitions[pageCandidates->items[i].blkno];
//
//            if (partition->size == partition->capacity){
//                partition->capacity *= 2;
//
//                partition->nodes = realloc(partition->nodes, sizeof(HnswElementPtr) * partition->capacity);
//
////                HnswElementPtr *tempPartition = malloc(sizeof(HnswElementPtr) * partition->capacity);
//
////                memcpy(tempPartition, partition->nodes, sizeof(HnswElementPtr) * partition->capacity/2);
////                for (int k=0; k<partition->size; k++){
////                    tempPartition[k] = partition->nodes[k];
////                }
//
////                partition->nodes = tempPartition;
//            }
//
//            HnswPtrStore(base, partition->nodes[partition->size], element);
//            partition->size++;
//            break;
//
//        } else {
////            elog(WARNING, "Page type: SPARE ");
//            // pool에 존재X -> spare에 배정
//            partition = &buildstate->insertPartitionstate->partitions[MAX_INSERT_POOL_SIZE];
//            HnswPtrStore(base, partition->nodes[partition->size], element);
//            partition->size++;
//            break;
//        }
//    }
//
//}

/*
 * Insert tuple
 */
static bool
InsertTuple(Relation index, Datum *values, bool *isnull, ItemPointer heaptid, HnswBuildState * buildstate)
{
	HnswGraph  *graph = buildstate->graph;
	HnswElement element;
	HnswAllocator *allocator = &buildstate->allocator;
	HnswSupport *support = &buildstate->support;
	Size		valueSize;
	Pointer		valuePtr;
	LWLock	   *flushLock = &graph->flushLock;
	char	   *base = buildstate->hnswarea;
	Datum		value;

	/* Form index value */
	if (!HnswFormIndexValue(&value, values, isnull, buildstate->typeInfo, support))
		return false;

	/* Get datum size */
	valueSize = VARSIZE_ANY(DatumGetPointer(value));

	/* Ensure graph not flushed when inserting */
	LWLockAcquire(flushLock, LW_SHARED);


	/* Are we in the on-disk phase? */
	if (graph->flushed)
	{
		LWLockRelease(flushLock);

		return HnswInsertTupleOnDisk(index, support, value, heaptid, true);
	}

	/*
	 * In a parallel build, the HnswElement is allocated from the shared
	 * memory area, so we need to coordinate with other processes.
	 */
	LWLockAcquire(&graph->allocatorLock, LW_EXCLUSIVE);

	/*
	 * Check that we have enough memory available for the new element now that
	 * we have the allocator lock, and flush pages if needed.
	 */
	if (graph->memoryUsed >= graph->memoryTotal)
	{
		LWLockRelease(&graph->allocatorLock);

		LWLockRelease(flushLock);
		LWLockAcquire(flushLock, LW_EXCLUSIVE);

		if (!graph->flushed)
		{
			ereport(NOTICE,
					(errmsg("hnsw graph no longer fits into maintenance_work_mem after " INT64_FORMAT " tuples", (int64) graph->indtuples),
					 errdetail("Building will take significantly more time."),
					 errhint("Increase maintenance_work_mem to speed up builds.")));


////    /* LDG 기반 그래프 파티셔닝 */
//            HnswPartitionState *partitionstate;
//            partitionstate = HnswPartitionGraph(buildstate);
//
//            // insert pool 구성?
//            HnswPartitionState *countPartitionstate;
//            countPartitionstate = CountPartitionSizeForInsert(buildstate, partitionstate);

            // countPartitionstate -> 이 결과를 ... 넣어보자 ..!!


            FlushPages(buildstate);
		}

		LWLockRelease(flushLock);

		return HnswInsertTupleOnDisk(index, support, value, heaptid, true);
	}

    if ((int)graph->indtuples % 100 == 0){
        elog(WARNING, "graph->indtuples: %d", (int) graph->indtuples);
    }

	/* Ok, we can proceed to allocate the element */
	element = HnswInitElement(base, heaptid, buildstate->m, buildstate->ml, buildstate->maxLevel, allocator);
	valuePtr = HnswAlloc(allocator, valueSize);

	/*
	 * We have now allocated the space needed for the element, so we don't
	 * need the allocator lock anymore. Release it and initialize the rest of
	 * the element.
	 */
	LWLockRelease(&graph->allocatorLock);

	/* Copy the datum */
	memcpy(valuePtr, DatumGetPointer(value), valueSize);
	HnswPtrStore(base, element->value, valuePtr);

	/* Create a lock for the element */
	LWLockInitialize(&element->lock, hnsw_lock_tranche_id);


//    /* Are we in the in-memory insert phase? */
//    if (graph->partitioned)
//    {
//        InsertTupleInMemoryLikeOnDisk(buildstate, element);
//
//        /* Release flush lock */
//        LWLockRelease(flushLock);
//        return true;
//    }
//
//    if (graph->indtuples > 10000)
//    {
//        if (!graph->partitioned)
//        {
////            LWLockAcquire(&graph->partitionLock, LW_EXCLUSIVE);
//
//            /* LDG 기반 그래프 파티셔닝 */ // partition state를 build state에 넣어주면 되겠지 ..?
//            HnswPartitionState *partitionstate;
//            partitionstate = HnswPartitionGraph(buildstate);
//
//            HnswPartitionState *countPartitionstate;
//            countPartitionstate = CountOverlapRatioForInsert(buildstate, partitionstate);
//
//            HnswPartitionState *insertPartitionState;
//            insertPartitionState = InitPartitionState(buildstate, partitionstate->partitions[0].capacity, allocator);
//
//            for (int i=0; i<MAX_INSERT_POOL_SIZE; i++){
//                insertPartitionState->partitions[i].pid = countPartitionstate->partitions[i].pid;
//            }
//
//            buildstate->partitionstate = partitionstate;
//            buildstate->countPartitionstate = countPartitionstate;
//            buildstate->insertPartitionstate = insertPartitionState;
//
//            elog(WARNING, "set partitions done");
//
////            LWLockRelease(&graph->partitionLock);
//        }
//        InsertTupleInMemoryLikeOnDisk(buildstate, element);
//
//        /* Release flush lock */
//        LWLockRelease(flushLock);
//
//        return true;
//    }

	/* Insert tuple */
	InsertTupleInMemory(buildstate, element);

	/* Release flush lock */
	LWLockRelease(flushLock);

	return true;
}

/*
 * Callback for table_index_build_scan
 */
static void
BuildCallback(Relation index, ItemPointer tid, Datum *values,
			  bool *isnull, bool tupleIsAlive, void *state)
{
	HnswBuildState *buildstate = (HnswBuildState *) state;
	HnswGraph  *graph = buildstate->graph;
	MemoryContext oldCtx;

	/* Skip nulls */
	if (isnull[0])
		return;

	/* Use memory context */
	oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

	/* Insert tuple */
	if (InsertTuple(index, values, isnull, tid, buildstate))
	{
		/* Update progress */
		SpinLockAcquire(&graph->lock);
		pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, ++graph->indtuples);
		SpinLockRelease(&graph->lock);
	}

	/* Reset memory context */
	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(buildstate->tmpCtx);
}

/*
 * Initialize the graph
 */
static void
InitGraph(HnswGraph * graph, char *base, Size memoryTotal)
{
	/* Initialize the lock tranche if needed */
	HnswInitLockTranche();

	HnswPtrStore(base, graph->head, (HnswElement) NULL);
	HnswPtrStore(base, graph->entryPoint, (HnswElement) NULL);
	graph->memoryUsed = 0;
	graph->memoryTotal = memoryTotal;
	graph->flushed = false;
	graph->indtuples = 0;
	SpinLockInit(&graph->lock);
	LWLockInitialize(&graph->entryLock, hnsw_lock_tranche_id);
	LWLockInitialize(&graph->entryWaitLock, hnsw_lock_tranche_id);
	LWLockInitialize(&graph->allocatorLock, hnsw_lock_tranche_id);
	LWLockInitialize(&graph->flushLock, hnsw_lock_tranche_id);
    LWLockInitialize(&graph->partitionLock, hnsw_lock_tranche_id);

    graph->partitioned = false;
}

/*
 * Initialize an allocator
 */
static void
InitAllocator(HnswAllocator * allocator, void *(*alloc) (Size size, void *state), void *state)
{
	allocator->alloc = alloc;
	allocator->state = state;
}

/*
 * Memory context allocator
 */
static void *
HnswMemoryContextAlloc(Size size, void *state)
{
	HnswBuildState *buildstate = (HnswBuildState *) state;
	void	   *chunk = MemoryContextAlloc(buildstate->graphCtx, size);

	buildstate->graphData.memoryUsed = MemoryContextMemAllocated(buildstate->graphCtx, false);

	return chunk;
}

/*
 * Shared memory allocator
 */
static void *
HnswSharedMemoryAlloc(Size size, void *state)
{
	HnswBuildState *buildstate = (HnswBuildState *) state;
	void	   *chunk = buildstate->hnswarea + buildstate->graph->memoryUsed;

	buildstate->graph->memoryUsed += MAXALIGN(size);
	return chunk;
}

/*
 * Initialize the build state
 */
static void
InitBuildState(HnswBuildState * buildstate, Relation heap, Relation index, IndexInfo *indexInfo, ForkNumber forkNum)
{
	buildstate->heap = heap;
	buildstate->index = index;
	buildstate->indexInfo = indexInfo;
	buildstate->forkNum = forkNum;
	buildstate->typeInfo = HnswGetTypeInfo(index);

	buildstate->m = HnswGetM(index);
	buildstate->efConstruction = HnswGetEfConstruction(index);
	buildstate->dimensions = TupleDescAttr(index->rd_att, 0)->atttypmod;

	/* Disallow varbit since require fixed dimensions */
	if (TupleDescAttr(index->rd_att, 0)->atttypid == VARBITOID)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("type not supported for hnsw index")));

	/* Require column to have dimensions to be indexed */
	if (buildstate->dimensions < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("column does not have dimensions")));

	if (buildstate->dimensions > buildstate->typeInfo->maxDimensions)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("column cannot have more than %d dimensions for hnsw index", buildstate->typeInfo->maxDimensions)));

	if (buildstate->efConstruction < 2 * buildstate->m)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ef_construction must be greater than or equal to 2 * m")));

	buildstate->reltuples = 0;
	buildstate->indtuples = 0;

	/* Get support functions */
	HnswInitSupport(&buildstate->support, index);

	InitGraph(&buildstate->graphData, NULL, (Size) maintenance_work_mem * 1024L);
	buildstate->graph = &buildstate->graphData;
	buildstate->ml = HnswGetMl(buildstate->m);
	buildstate->maxLevel = HnswGetMaxLevel(buildstate->m);

	buildstate->graphCtx = GenerationContextCreate(CurrentMemoryContext,
												   "Hnsw build graph context",
#if PG_VERSION_NUM >= 150000
												   1024 * 1024, 1024 * 1024,
#endif
												   1024 * 1024);
	buildstate->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											   "Hnsw build temporary context",
											   ALLOCSET_DEFAULT_SIZES);

	InitAllocator(&buildstate->allocator, &HnswMemoryContextAlloc, buildstate);

	buildstate->hnswleader = NULL;
	buildstate->hnswshared = NULL;
	buildstate->hnswarea = NULL;

    buildstate->partitionstate = NULL;
    buildstate->countPartitionstate = NULL;
    buildstate->insertPartitionstate = NULL;
}

/*
 * Free resources
 */
static void
FreeBuildState(HnswBuildState * buildstate)
{
	MemoryContextDelete(buildstate->graphCtx);
	MemoryContextDelete(buildstate->tmpCtx);
}

/*
 * Within leader, wait for end of heap scan
 */
static double
ParallelHeapScan(HnswBuildState * buildstate)
{
	HnswShared *hnswshared = buildstate->hnswleader->hnswshared;
	int			nparticipanttuplesorts;
	double		reltuples;

	nparticipanttuplesorts = buildstate->hnswleader->nparticipanttuplesorts;
	for (;;)
	{
		SpinLockAcquire(&hnswshared->mutex);
		if (hnswshared->nparticipantsdone == nparticipanttuplesorts)
		{
			buildstate->graph = &hnswshared->graphData;
			buildstate->hnswarea = buildstate->hnswleader->hnswarea;
			reltuples = hnswshared->reltuples;
			SpinLockRelease(&hnswshared->mutex);
			break;
		}
		SpinLockRelease(&hnswshared->mutex);

		ConditionVariableSleep(&hnswshared->workersdonecv,
							   WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	}

	ConditionVariableCancelSleep();

	return reltuples;
}

/*
 * Perform a worker's portion of a parallel insert
 */
static void
HnswParallelScanAndInsert(Relation heapRel, Relation indexRel, HnswShared * hnswshared, char *hnswarea, bool progress)
{
	HnswBuildState buildstate;
	TableScanDesc scan;
	double		reltuples;
	IndexInfo  *indexInfo;

	/* Join parallel scan */
	indexInfo = BuildIndexInfo(indexRel);
	indexInfo->ii_Concurrent = hnswshared->isconcurrent;
	InitBuildState(&buildstate, heapRel, indexRel, indexInfo, MAIN_FORKNUM);
	buildstate.graph = &hnswshared->graphData;
	buildstate.hnswarea = hnswarea;
	InitAllocator(&buildstate.allocator, &HnswSharedMemoryAlloc, &buildstate);
	scan = table_beginscan_parallel(heapRel,
									ParallelTableScanFromHnswShared(hnswshared));
	reltuples = table_index_build_scan(heapRel, indexRel, indexInfo,
									   true, progress, BuildCallback,
									   (void *) &buildstate, scan);

	/* Record statistics */
	SpinLockAcquire(&hnswshared->mutex);
	hnswshared->nparticipantsdone++;
	hnswshared->reltuples += reltuples;
	SpinLockRelease(&hnswshared->mutex);

	/* Log statistics */
	if (progress)
		ereport(DEBUG1, (errmsg("leader processed " INT64_FORMAT " tuples", (int64) reltuples)));
	else
		ereport(DEBUG1, (errmsg("worker processed " INT64_FORMAT " tuples", (int64) reltuples)));

	/* Notify leader */
	ConditionVariableSignal(&hnswshared->workersdonecv);

	FreeBuildState(&buildstate);
}

/*
 * Perform work within a launched parallel process
 */
void
HnswParallelBuildMain(dsm_segment *seg, shm_toc *toc)
{
	char	   *sharedquery;
	HnswShared *hnswshared;
	char	   *hnswarea;
	Relation	heapRel;
	Relation	indexRel;
	LOCKMODE	heapLockmode;
	LOCKMODE	indexLockmode;

	/* Set debug_query_string for individual workers first */
	sharedquery = shm_toc_lookup(toc, PARALLEL_KEY_QUERY_TEXT, true);
	debug_query_string = sharedquery;

	/* Report the query string from leader */
	pgstat_report_activity(STATE_RUNNING, debug_query_string);

	/* Look up shared state */
	hnswshared = shm_toc_lookup(toc, PARALLEL_KEY_HNSW_SHARED, false);

	/* Open relations using lock modes known to be obtained by index.c */
	if (!hnswshared->isconcurrent)
	{
		heapLockmode = ShareLock;
		indexLockmode = AccessExclusiveLock;
	}
	else
	{
		heapLockmode = ShareUpdateExclusiveLock;
		indexLockmode = RowExclusiveLock;
	}

	/* Open relations within worker */
	heapRel = table_open(hnswshared->heaprelid, heapLockmode);
	indexRel = index_open(hnswshared->indexrelid, indexLockmode);

	hnswarea = shm_toc_lookup(toc, PARALLEL_KEY_HNSW_AREA, false);

	/* Perform inserts */
	HnswParallelScanAndInsert(heapRel, indexRel, hnswshared, hnswarea, false);

	/* Close relations within worker */
	index_close(indexRel, indexLockmode);
	table_close(heapRel, heapLockmode);
}

/*
 * End parallel build
 */
static void
HnswEndParallel(HnswLeader * hnswleader)
{
	/* Shutdown worker processes */
	WaitForParallelWorkersToFinish(hnswleader->pcxt);

	/* Free last reference to MVCC snapshot, if one was used */
	if (IsMVCCSnapshot(hnswleader->snapshot))
		UnregisterSnapshot(hnswleader->snapshot);
	DestroyParallelContext(hnswleader->pcxt);
	ExitParallelMode();
}

/*
 * Return size of shared memory required for parallel index build
 */
static Size
ParallelEstimateShared(Relation heap, Snapshot snapshot)
{
	return add_size(BUFFERALIGN(sizeof(HnswShared)), table_parallelscan_estimate(heap, snapshot));
}

/*
 * Within leader, participate as a parallel worker
 */
static void
HnswLeaderParticipateAsWorker(HnswBuildState * buildstate)
{
	HnswLeader *hnswleader = buildstate->hnswleader;

	/* Perform work common to all participants */
	HnswParallelScanAndInsert(buildstate->heap, buildstate->index, hnswleader->hnswshared, hnswleader->hnswarea, true);
}

/*
 * Begin parallel build
 */
static void
HnswBeginParallel(HnswBuildState * buildstate, bool isconcurrent, int request)
{
	ParallelContext *pcxt;
	Snapshot	snapshot;
	Size		esthnswshared;
	Size		esthnswarea;
	Size		estother;
	HnswShared *hnswshared;
	char	   *hnswarea;
	HnswLeader *hnswleader = (HnswLeader *) palloc0(sizeof(HnswLeader));
	bool		leaderparticipates = true;
	int			querylen;

#ifdef DISABLE_LEADER_PARTICIPATION
	leaderparticipates = false;
#endif

	/* Enter parallel mode and create context */
	EnterParallelMode();
	Assert(request > 0);
	pcxt = CreateParallelContext("vector", "HnswParallelBuildMain", request);

	/* Get snapshot for table scan */
	if (!isconcurrent)
		snapshot = SnapshotAny;
	else
		snapshot = RegisterSnapshot(GetTransactionSnapshot());

	/* Estimate size of workspaces */
	esthnswshared = ParallelEstimateShared(buildstate->heap, snapshot);
	shm_toc_estimate_chunk(&pcxt->estimator, esthnswshared);

	/* Leave space for other objects in shared memory */
	/* Docker has a default limit of 64 MB for shm_size */
	/* which happens to be the default value of maintenance_work_mem */
	esthnswarea = maintenance_work_mem * 1024L;
	estother = 3 * 1024 * 1024;
	if (esthnswarea > estother)
		esthnswarea -= estother;

	shm_toc_estimate_chunk(&pcxt->estimator, esthnswarea);
	shm_toc_estimate_keys(&pcxt->estimator, 2);

	/* Finally, estimate PARALLEL_KEY_QUERY_TEXT space */
	if (debug_query_string)
	{
		querylen = strlen(debug_query_string);
		shm_toc_estimate_chunk(&pcxt->estimator, querylen + 1);
		shm_toc_estimate_keys(&pcxt->estimator, 1);
	}
	else
		querylen = 0;			/* keep compiler quiet */

	/* Everyone's had a chance to ask for space, so now create the DSM */
	InitializeParallelDSM(pcxt);

	/* If no DSM segment was available, back out (do serial build) */
	if (pcxt->seg == NULL)
	{
		if (IsMVCCSnapshot(snapshot))
			UnregisterSnapshot(snapshot);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return;
	}

	/* Store shared build state, for which we reserved space */
	hnswshared = (HnswShared *) shm_toc_allocate(pcxt->toc, esthnswshared);
	/* Initialize immutable state */
	hnswshared->heaprelid = RelationGetRelid(buildstate->heap);
	hnswshared->indexrelid = RelationGetRelid(buildstate->index);
	hnswshared->isconcurrent = isconcurrent;
	ConditionVariableInit(&hnswshared->workersdonecv);
	SpinLockInit(&hnswshared->mutex);
	/* Initialize mutable state */
	hnswshared->nparticipantsdone = 0;
	hnswshared->reltuples = 0;
	table_parallelscan_initialize(buildstate->heap,
								  ParallelTableScanFromHnswShared(hnswshared),
								  snapshot);

	hnswarea = (char *) shm_toc_allocate(pcxt->toc, esthnswarea);
	/* Report less than allocated so never fails */
	InitGraph(&hnswshared->graphData, hnswarea, esthnswarea - 1024 * 1024);

	/*
	 * Avoid base address for relptr for Postgres < 14.5
	 * https://github.com/postgres/postgres/commit/7201cd18627afc64850537806da7f22150d1a83b
	 */
#if PG_VERSION_NUM < 140005
	hnswshared->graphData.memoryUsed += MAXALIGN(1);
#endif

	shm_toc_insert(pcxt->toc, PARALLEL_KEY_HNSW_SHARED, hnswshared);
	shm_toc_insert(pcxt->toc, PARALLEL_KEY_HNSW_AREA, hnswarea);

	/* Store query string for workers */
	if (debug_query_string)
	{
		char	   *sharedquery;

		sharedquery = (char *) shm_toc_allocate(pcxt->toc, querylen + 1);
		memcpy(sharedquery, debug_query_string, querylen + 1);
		shm_toc_insert(pcxt->toc, PARALLEL_KEY_QUERY_TEXT, sharedquery);
	}

	/* Launch workers, saving status for leader/caller */
	LaunchParallelWorkers(pcxt);
	hnswleader->pcxt = pcxt;
	hnswleader->nparticipanttuplesorts = pcxt->nworkers_launched;
	if (leaderparticipates)
		hnswleader->nparticipanttuplesorts++;
	hnswleader->hnswshared = hnswshared;
	hnswleader->snapshot = snapshot;
	hnswleader->hnswarea = hnswarea;

	/* If no workers were successfully launched, back out (do serial build) */
	if (pcxt->nworkers_launched == 0)
	{
		HnswEndParallel(hnswleader);
		return;
	}

	/* Log participants */
	ereport(DEBUG1, (errmsg("using %d parallel workers", pcxt->nworkers_launched)));

	/* Save leader state now that it's clear build will be parallel */
	buildstate->hnswleader = hnswleader;

	/* Join heap scan ourselves */
	if (leaderparticipates)
		HnswLeaderParticipateAsWorker(buildstate);

	/* Wait for all launched workers */
	WaitForParallelWorkersToAttach(pcxt);
}

/*
 * Compute parallel workers
 */
static int
ComputeParallelWorkers(Relation heap, Relation index)
{
	int			parallel_workers;

	/* Make sure it's safe to use parallel workers */
	parallel_workers = plan_create_index_workers(RelationGetRelid(heap), RelationGetRelid(index));
	if (parallel_workers == 0)
		return 0;

	/* Use parallel_workers storage parameter on table if set */
	parallel_workers = RelationGetParallelWorkers(heap, -1);
	if (parallel_workers != -1)
		return Min(parallel_workers, max_parallel_maintenance_workers);

	return max_parallel_maintenance_workers;
}



/*
 * Build graph
 */
static void
BuildGraph(HnswBuildState * buildstate, ForkNumber forkNum)
{
	int			parallel_workers = 0;

	pgstat_progress_update_param(PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_HNSW_PHASE_LOAD);

	/* Calculate parallel workers */
	if (buildstate->heap != NULL)
		parallel_workers = ComputeParallelWorkers(buildstate->heap, buildstate->index);

	/* Attempt to launch parallel worker scan when required */
	if (parallel_workers > 0)
		HnswBeginParallel(buildstate, buildstate->indexInfo->ii_Concurrent, parallel_workers);

	/* Add tuples to graph */
	if (buildstate->heap != NULL)
	{
		if (buildstate->hnswleader)
			buildstate->reltuples = ParallelHeapScan(buildstate);
		else
			buildstate->reltuples = table_index_build_scan(buildstate->heap, buildstate->index, buildstate->indexInfo,
														   true, true, BuildCallback, (void *) buildstate, NULL);

		buildstate->indtuples = buildstate->graph->indtuples;
	}

	/* Flush pages */
	if (!buildstate->graph->flushed)
		FlushPages(buildstate);

	/* End parallel build */
	if (buildstate->hnswleader)
		HnswEndParallel(buildstate->hnswleader);
}

static void
BuildGraphWithPartition(HnswBuildState * buildstate, ForkNumber forkNum)
{
    int			parallel_workers = 0;

    elog(WARNING, "build with partition");

    pgstat_progress_update_param(PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_HNSW_PHASE_LOAD);

    /* Calculate parallel workers */
    if (buildstate->heap != NULL)
        parallel_workers = ComputeParallelWorkers(buildstate->heap, buildstate->index);

    /* Attempt to launch parallel worker scan when required */
    if (parallel_workers > 0)
        HnswBeginParallel(buildstate, buildstate->indexInfo->ii_Concurrent, parallel_workers);

    elog(WARNING, "workers: %d", parallel_workers);
    /* Add tuples to graph */
    if (buildstate->heap != NULL)
    {
        if (buildstate->hnswleader)
            buildstate->reltuples = ParallelHeapScan(buildstate);
        else
            buildstate->reltuples = table_index_build_scan(buildstate->heap, buildstate->index, buildstate->indexInfo,
                                                           true, true, BuildCallback, (void *) buildstate, NULL);

        buildstate->indtuples = buildstate->graph->indtuples;
    }

    elog(WARNING, "start flush");
    /* Flush pages */
    if (!buildstate->graph->flushed)
    {

        elog(WARNING, "start flush");
        /* LDG 기반 그래프 파티셔닝 */ // partition state를 build state에 넣어주면 되겠지 ..?
        HnswPartitionState *partitionstate;
        partitionstate = HnswPartitionGraph(buildstate);

        HnswPartitionState *countPartitionstate;
        countPartitionstate = CountOverlapRatioForInsert(buildstate, partitionstate);
    //
    //            HnswPartitionState *insertPartitionState;
    //            insertPartitionState = InitPartitionState(buildstate, partitionstate->partitions[0].capacity, &buildstate->allocator);
    //
    //            for (int i=0; i<MAX_INSERT_POOL_SIZE; i++){
    //                insertPartitionState->partitions[i].pid = countPartitionstate->partitions[i].pid;
    //            }

        buildstate->partitionstate = partitionstate;
        buildstate->countPartitionstate = countPartitionstate;
        buildstate->insertPartitionstate = NULL;

        /* Partition 기반으로 FlushPages 호출 */
        FlushPagesWithPartitions(buildstate, buildstate->partitionstate, buildstate->countPartitionstate);
//        FlushPagesWithPartitionsLikeOnDisk(buildstate);
    }

    /* End parallel build */
    if (buildstate->hnswleader)
        HnswEndParallel(buildstate->hnswleader);
}

/*
 * Build the index
 */
static void
BuildIndex(Relation heap, Relation index, IndexInfo *indexInfo,
		   HnswBuildState * buildstate, ForkNumber forkNum)
{
#ifdef HNSW_MEMORY
	SeedRandom(42);
#endif

	InitBuildState(buildstate, heap, index, indexInfo, forkNum);

//	BuildGraph(buildstate, forkNum);
    BuildGraphWithPartition(buildstate, forkNum);

	if (RelationNeedsWAL(index) || forkNum == INIT_FORKNUM)
		log_newpage_range(index, forkNum, 0, RelationGetNumberOfBlocksInFork(index, forkNum), true);

	FreeBuildState(buildstate);
}

/*
 * Build the index for a logged table
 */
IndexBuildResult *
hnswbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result;
	HnswBuildState buildstate;

	BuildIndex(heap, index, indexInfo, &buildstate, MAIN_FORKNUM);

	result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
	result->heap_tuples = buildstate.reltuples;
	result->index_tuples = buildstate.indtuples;

	return result;
}

/*
 * Build the index for an unlogged table
 */
void
hnswbuildempty(Relation index)
{
	IndexInfo  *indexInfo = BuildIndexInfo(index);
	HnswBuildState buildstate;

	BuildIndex(NULL, index, indexInfo, &buildstate, INIT_FORKNUM);
}
