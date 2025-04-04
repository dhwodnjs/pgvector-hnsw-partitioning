////
//// Created by jaewonoh on 4/3/25.
////
//
//// ivf kmeans 코드 필요한 부분만 가져오기 .. index를 다 빼는게 맞나? .,
//
//#include "postgres.h"
//
//#include <float.h>
//
//#include "access/table.h"
//#include "access/tableam.h"
//#include "access/parallel.h"
//#include "access/xact.h"
//#include "bitvec.h"
//#include "catalog/index.h"
//#include "catalog/pg_operator_d.h"
//#include "catalog/pg_type_d.h"
//#include "commands/progress.h"
//#include "halfvec.h"
//#include "ivfflat.h"
//#include "miscadmin.h"
//#include "optimizer/optimizer.h"
//#include "storage/bufmgr.h"
//#include "tcop/tcopprot.h"
//#include "utils/memutils.h"
//#include "vector.h"
//
//#if PG_VERSION_NUM >= 140000
//#include "utils/backend_progress.h"
//#else
//#include "pgstat.h"
//#endif
//
//#if PG_VERSION_NUM >= 140000
//#include "utils/backend_status.h"
//#include "utils/wait_event.h"
//#endif
//
//#include "postgres.h"
//
//#include <float.h>
//#include <math.h>
//
//#include "bitvec.h"
//#include "halfutils.h"
//#include "halfvec.h"
//#include "ivfflat.h"
//#include "miscadmin.h"
//#include "utils/builtins.h"
//#include "utils/datum.h"
//#include "utils/memutils.h"
//#include "vector.h"
//
//
//// init buildstate (build) -> compute centers (build) -> ... (kmeans) 가져와보자 !
//// index 대신에 hnsw buildstate에서 필요한 정보 가져다 쓸거임 -> input으로 이거만 받아도 되나? heap이랑 ,,, .,?
//// index도 기존의 index를 써도 되나? 그냥 buildstate만 다른거 써도 잘 돌아가려나
//// 기존 index, hnswbuildstate를 넣어주는걸로 ., 해보자
//
//
///*
// * Initialize the build state
// */
//static void
//InitBuildState(IvfflatBuildState * buildstate, Relation heap, Relation index, IndexInfo *indexInfo)
//{
//    buildstate->heap = heap;
//    buildstate->index = index;
//    buildstate->indexInfo = indexInfo;
//    buildstate->typeInfo = IvfflatGetTypeInfo(index);
//    buildstate->tupdesc = RelationGetDescr(index);
//
//    buildstate->lists = IvfflatGetLists(index);
//    buildstate->dimensions = TupleDescAttr(index->rd_att, 0)->atttypmod;
//
//    /* Disallow varbit since require fixed dimensions */
//    if (TupleDescAttr(index->rd_att, 0)->atttypid == VARBITOID)
//        ereport(ERROR,
//                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
//                        errmsg("type not supported for ivfflat index")));
//
//    /* Require column to have dimensions to be indexed */
//    if (buildstate->dimensions < 0)
//        ereport(ERROR,
//                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
//                        errmsg("column does not have dimensions")));
//
//    if (buildstate->dimensions > buildstate->typeInfo->maxDimensions)
//        ereport(ERROR,
//                (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
//                        errmsg("column cannot have more than %d dimensions for ivfflat index", buildstate->typeInfo->maxDimensions)));
//
//    buildstate->reltuples = 0;
//    buildstate->indtuples = 0;
//
//    /* Get support functions */
//    buildstate->procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
//    buildstate->normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
//    buildstate->kmeansnormprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_KMEANS_NORM_PROC);
//    buildstate->collation = index->rd_indcollation[0];
//
//    /* Require more than one dimension for spherical k-means */
//    if (buildstate->kmeansnormprocinfo != NULL && buildstate->dimensions == 1)
//        ereport(ERROR,
//                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
//                        errmsg("dimensions must be greater than one for this opclass")));
//
//    /* Create tuple description for sorting */
//    buildstate->sortdesc = CreateTemplateTupleDesc(3);
//    TupleDescInitEntry(buildstate->sortdesc, (AttrNumber) 1, "list", INT4OID, -1, 0);
//    TupleDescInitEntry(buildstate->sortdesc, (AttrNumber) 2, "tid", TIDOID, -1, 0);
//    TupleDescInitEntry(buildstate->sortdesc, (AttrNumber) 3, "vector", buildstate->tupdesc->attrs[0].atttypid, -1, 0);
//
//    buildstate->slot = MakeSingleTupleTableSlot(buildstate->sortdesc, &TTSOpsVirtual);
//
//    buildstate->centers = VectorArrayInit(buildstate->lists, buildstate->dimensions, buildstate->typeInfo->itemSize(buildstate->dimensions));
//    buildstate->listInfo = palloc(sizeof(ListInfo) * buildstate->lists);
//
//    buildstate->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
//                                               "Ivfflat build temporary context",
//                                               ALLOCSET_DEFAULT_SIZES);
//
//#ifdef IVFFLAT_KMEANS_DEBUG
//    buildstate->inertia = 0;
//	buildstate->listSums = palloc0(sizeof(double) * buildstate->lists);
//	buildstate->listCounts = palloc0(sizeof(int) * buildstate->lists);
//#endif
//
//    buildstate->ivfleader = NULL;
//}
//
///*
// * Add sample
// */
//static void
//AddSample(Datum *values, IvfflatBuildState * buildstate)
//{
//    VectorArray samples = buildstate->samples;
//    int			targsamples = samples->maxlen;
//
//    /* Detoast once for all calls */
//    Datum		value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));
//
//    /*
//     * Normalize with KMEANS_NORM_PROC since spherical distance function
//     * expects unit vectors
//     */
//    if (buildstate->kmeansnormprocinfo != NULL)
//    {
//        if (!IvfflatCheckNorm(buildstate->kmeansnormprocinfo, buildstate->collation, value))
//            return;
//
//        value = IvfflatNormValue(buildstate->typeInfo, buildstate->collation, value);
//    }
//
//    if (samples->length < targsamples)
//    {
//        VectorArraySet(samples, samples->length, DatumGetPointer(value));
//        samples->length++;
//    }
//    else
//    {
//        if (buildstate->rowstoskip < 0)
//            buildstate->rowstoskip = reservoir_get_next_S(&buildstate->rstate, samples->length, targsamples);
//
//        if (buildstate->rowstoskip <= 0)
//        {
//#if PG_VERSION_NUM >= 150000
//            int			k = (int) (targsamples * sampler_random_fract(&buildstate->rstate.randstate));
//#else
//            int			k = (int) (targsamples * sampler_random_fract(buildstate->rstate.randstate));
//#endif
//
//            Assert(k >= 0 && k < targsamples);
//            VectorArraySet(samples, k, DatumGetPointer(value));
//        }
//
//        buildstate->rowstoskip -= 1;
//    }
//}
//
///*
// * Callback for sampling
// */
//static void
//SampleCallback(Relation index, ItemPointer tid, Datum *values,
//               bool *isnull, bool tupleIsAlive, void *state)
//{
//    IvfflatBuildState *buildstate = (IvfflatBuildState *) state;
//    MemoryContext oldCtx;
//
//    /* Skip nulls */
//    if (isnull[0])
//        return;
//
//    /* Use memory context since detoast can allocate */
//    oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);
//
//    /* Add sample */
//    AddSample(values, buildstate);
//
//    /* Reset memory context */
//    MemoryContextSwitchTo(oldCtx);
//    MemoryContextReset(buildstate->tmpCtx);
//}
//
//
///*
// * Sample rows with same logic as ANALYZE
// */
//static void
//SampleRows(IvfflatBuildState * buildstate)
//{
//    int			targsamples = buildstate->samples->maxlen;
//    BlockNumber totalblocks = RelationGetNumberOfBlocks(buildstate->heap);
//
//    buildstate->rowstoskip = -1;
//
//    BlockSampler_Init(&buildstate->bs, totalblocks, targsamples, RandomInt());
//
//    reservoir_init_selection_state(&buildstate->rstate, targsamples);
//    while (BlockSampler_HasMore(&buildstate->bs))
//    {
//        BlockNumber targblock = BlockSampler_Next(&buildstate->bs);
//
//        table_index_build_range_scan(buildstate->heap, buildstate->index, buildstate->indexInfo,
//                                     false, true, false, targblock, 1, SampleCallback, (void *) buildstate, NULL);
//    }
//}
//
//static void
//InitCenters(Relation index, VectorArray samples, VectorArray centers, float *lowerBound)
//{
//    FmgrInfo   *procinfo;
//    Oid			collation;
//    int64		j;
//    float	   *weight = palloc(samples->length * sizeof(float));
//    int			numCenters = centers->maxlen;
//    int			numSamples = samples->length;
//
//    procinfo = index_getprocinfo(index, 1, IVFFLAT_KMEANS_DISTANCE_PROC);
//    collation = index->rd_indcollation[0];
//
//    /* Choose an initial center uniformly at random */
//    VectorArraySet(centers, 0, VectorArrayGet(samples, RandomInt() % samples->length));
//    centers->length++;
//
//    for (j = 0; j < numSamples; j++)
//        weight[j] = FLT_MAX;
//
//    for (int i = 0; i < numCenters; i++)
//    {
//        double		sum;
//        double		choice;
//
//        CHECK_FOR_INTERRUPTS();
//
//        sum = 0.0;
//
//        for (j = 0; j < numSamples; j++)
//        {
//            Datum		vec = PointerGetDatum(VectorArrayGet(samples, j));
//            double		distance;
//
//            /* Only need to compute distance for new center */
//            /* TODO Use triangle inequality to reduce distance calculations */
//            distance = DatumGetFloat8(FunctionCall2Coll(procinfo, collation, vec, PointerGetDatum(VectorArrayGet(centers, i))));
//
//            /* Set lower bound */
//            lowerBound[j * numCenters + i] = distance;
//
//            /* Use distance squared for weighted probability distribution */
//            distance *= distance;
//
//            if (distance < weight[j])
//                weight[j] = distance;
//
//            sum += weight[j];
//        }
//
//        /* Only compute lower bound on last iteration */
//        if (i + 1 == numCenters)
//            break;
//
//        /* Choose new center using weighted probability distribution. */
//        choice = sum * RandomDouble();
//        for (j = 0; j < numSamples - 1; j++)
//        {
//            choice -= weight[j];
//            if (choice <= 0)
//                break;
//        }
//
//        VectorArraySet(centers, i + 1, VectorArrayGet(samples, j));
//        centers->length++;
//    }
//
//    pfree(weight);
//}
//
//
///*
// * Sum centers
// */
//static void
//SumCenters(VectorArray samples, float *agg, int *closestCenters, const IvfflatTypeInfo * typeInfo)
//{
//    for (int j = 0; j < samples->length; j++)
//    {
//        float	   *x = agg + ((int64) closestCenters[j] * samples->dim);
//
//        typeInfo->sumCenter(VectorArrayGet(samples, j), x);
//    }
//}
//
///*
// * Update centers
// */
//static void
//UpdateCenters(float *agg, VectorArray centers, const IvfflatTypeInfo * typeInfo)
//{
//    for (int j = 0; j < centers->length; j++)
//    {
//        float	   *x = agg + ((int64) j * centers->dim);
//
//        typeInfo->updateCenter(VectorArrayGet(centers, j), centers->dim, x);
//    }
//}
//
//
///*
// * Norm centers
// */
//static void
//NormCenters(const IvfflatTypeInfo * typeInfo, Oid collation, VectorArray centers)
//{
//    MemoryContext normCtx = AllocSetContextCreate(CurrentMemoryContext,
//                                                  "Ivfflat norm temporary context",
//                                                  ALLOCSET_DEFAULT_SIZES);
//    MemoryContext oldCtx = MemoryContextSwitchTo(normCtx);
//
//    for (int j = 0; j < centers->length; j++)
//    {
//        Datum		center = PointerGetDatum(VectorArrayGet(centers, j));
//        Datum		newCenter = IvfflatNormValue(typeInfo, collation, center);
//        Size		size = VARSIZE_ANY(DatumGetPointer(newCenter));
//
//        if (size > centers->itemsize)
//            elog(ERROR, "safety check failed");
//
//        memcpy(DatumGetPointer(center), DatumGetPointer(newCenter), size);
//        MemoryContextReset(normCtx);
//    }
//
//    MemoryContextSwitchTo(oldCtx);
//    MemoryContextDelete(normCtx);
//}
//
//
///*
// * Compute new centers
// */
//static void
//ComputeNewCenters(VectorArray samples, float *agg, VectorArray newCenters, int *centerCounts, int *closestCenters, FmgrInfo *normprocinfo, Oid collation, const IvfflatTypeInfo * typeInfo)
//{
//    int			dimensions = newCenters->dim;
//    int			numCenters = newCenters->length;
//    int			numSamples = samples->length;
//
//    /* Reset sum and count */
//    for (int j = 0; j < numCenters; j++)
//    {
//        float	   *x = agg + ((int64) j * dimensions);
//
//        for (int k = 0; k < dimensions; k++)
//            x[k] = 0.0;
//
//        centerCounts[j] = 0;
//    }
//
//    /* Increment sum of closest center */
//    SumCenters(samples, agg, closestCenters, typeInfo);
//
//    /* Increment count of closest center */
//    for (int j = 0; j < numSamples; j++)
//        centerCounts[closestCenters[j]] += 1;
//
//    /* Divide sum by count */
//    for (int j = 0; j < numCenters; j++)
//    {
//        float	   *x = agg + ((int64) j * dimensions);
//
//        if (centerCounts[j] > 0)
//        {
//            /* Double avoids overflow, but requires more memory */
//            /* TODO Update bounds */
//            for (int k = 0; k < dimensions; k++)
//            {
//                if (isinf(x[k]))
//                    x[k] = x[k] > 0 ? FLT_MAX : -FLT_MAX;
//            }
//
//            for (int k = 0; k < dimensions; k++)
//                x[k] /= centerCounts[j];
//        }
//        else
//        {
//            /* TODO Handle empty centers properly */
//            for (int k = 0; k < dimensions; k++)
//                x[k] = RandomDouble();
//        }
//    }
//
//    /* Set new centers */
//    UpdateCenters(agg, newCenters, typeInfo);
//
//    /* Normalize if needed */
//    if (normprocinfo != NULL)
//        NormCenters(typeInfo, collation, newCenters);
//}
//
//
//
//static void
//ElkanKmeans(Relation index, VectorArray samples, VectorArray centers, const IvfflatTypeInfo * typeInfo)
//{
//    FmgrInfo   *procinfo;
//    FmgrInfo   *normprocinfo;
//    Oid			collation;
//    int			dimensions = centers->dim;
//    int			numCenters = centers->maxlen;
//    int			numSamples = samples->length;
//    VectorArray newCenters;
//    float	   *agg;
//    int		   *centerCounts;
//    int		   *closestCenters;
//    float	   *lowerBound;
//    float	   *upperBound;
//    float	   *s;
//    float	   *halfcdist;
//    float	   *newcdist;
//
//    /* Calculate allocation sizes */
//    Size		samplesSize = VECTOR_ARRAY_SIZE(samples->maxlen, samples->itemsize);
//    Size		centersSize = VECTOR_ARRAY_SIZE(centers->maxlen, centers->itemsize);
//    Size		newCentersSize = VECTOR_ARRAY_SIZE(numCenters, centers->itemsize);
//    Size		aggSize = sizeof(float) * (int64) numCenters * dimensions;
//    Size		centerCountsSize = sizeof(int) * numCenters;
//    Size		closestCentersSize = sizeof(int) * numSamples;
//    Size		lowerBoundSize = sizeof(float) * numSamples * numCenters;
//    Size		upperBoundSize = sizeof(float) * numSamples;
//    Size		sSize = sizeof(float) * numCenters;
//    Size		halfcdistSize = sizeof(float) * numCenters * numCenters;
//    Size		newcdistSize = sizeof(float) * numCenters;
//
//    /* Calculate total size */
//    Size		totalSize = samplesSize + centersSize + newCentersSize + aggSize + centerCountsSize + closestCentersSize + lowerBoundSize + upperBoundSize + sSize + halfcdistSize + newcdistSize;
//
//    /* Check memory requirements */
//    /* Add one to error message to ceil */
//    if (totalSize > (Size) maintenance_work_mem * 1024L)
//        ereport(ERROR,
//                (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
//                        errmsg("memory required is %zu MB, maintenance_work_mem is %d MB",
//                               totalSize / (1024 * 1024) + 1, maintenance_work_mem / 1024)));
//
//    /* Ensure indexing does not overflow */
//    if (numCenters * numCenters > INT_MAX)
//        elog(ERROR, "Indexing overflow detected. Please report a bug.");
//
//    /* Set support functions */
//    procinfo = index_getprocinfo(index, 1, IVFFLAT_KMEANS_DISTANCE_PROC);
//    normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_KMEANS_NORM_PROC);
//    collation = index->rd_indcollation[0];
//
//    /* Allocate space */
//    /* Use float instead of double to save memory */
//    agg = palloc(aggSize);
//    centerCounts = palloc(centerCountsSize);
//    closestCenters = palloc(closestCentersSize);
//    lowerBound = palloc_extended(lowerBoundSize, MCXT_ALLOC_HUGE);
//    upperBound = palloc(upperBoundSize);
//    s = palloc(sSize);
//    halfcdist = palloc_extended(halfcdistSize, MCXT_ALLOC_HUGE);
//    newcdist = palloc(newcdistSize);
//
//    /* Initialize new centers */
//    newCenters = VectorArrayInit(numCenters, dimensions, centers->itemsize);
//    newCenters->length = numCenters;
//
//#ifdef IVFFLAT_MEMORY
//    ShowMemoryUsage(MemoryContextGetParent(CurrentMemoryContext), totalSize);
//#endif
//
//    /* Pick initial centers */
//    InitCenters(index, samples, centers, lowerBound);
//
//    /* Assign each x to its closest initial center c(x) = argmin d(x,c) */
//    for (int64 j = 0; j < numSamples; j++)
//    {
//        float		minDistance = FLT_MAX;
//        int			closestCenter = 0;
//
//        /* Find closest center */
//        for (int64 k = 0; k < numCenters; k++)
//        {
//            /* TODO Use Lemma 1 in k-means++ initialization */
//            float		distance = lowerBound[j * numCenters + k];
//
//            if (distance < minDistance)
//            {
//                minDistance = distance;
//                closestCenter = k;
//            }
//        }
//
//        upperBound[j] = minDistance;
//        closestCenters[j] = closestCenter;
//    }
//
//    /* Give 500 iterations to converge */
//    for (int iteration = 0; iteration < 500; iteration++)
//    {
//        int			changes = 0;
//        bool		rjreset;
//
//        /* Can take a while, so ensure we can interrupt */
//        CHECK_FOR_INTERRUPTS();
//
//        /* Step 1: For all centers, compute distance */
//        for (int64 j = 0; j < numCenters; j++)
//        {
//            Datum		vec = PointerGetDatum(VectorArrayGet(centers, j));
//
//            for (int64 k = j + 1; k < numCenters; k++)
//            {
//                float		distance = 0.5 * DatumGetFloat8(FunctionCall2Coll(procinfo, collation, vec, PointerGetDatum(VectorArrayGet(centers, k))));
//
//                halfcdist[j * numCenters + k] = distance;
//                halfcdist[k * numCenters + j] = distance;
//            }
//        }
//
//        /* For all centers c, compute s(c) */
//        for (int64 j = 0; j < numCenters; j++)
//        {
//            float		minDistance = FLT_MAX;
//
//            for (int64 k = 0; k < numCenters; k++)
//            {
//                float		distance;
//
//                if (j == k)
//                    continue;
//
//                distance = halfcdist[j * numCenters + k];
//                if (distance < minDistance)
//                    minDistance = distance;
//            }
//
//            s[j] = minDistance;
//        }
//
//        rjreset = iteration != 0;
//
//        for (int64 j = 0; j < numSamples; j++)
//        {
//            bool		rj;
//
//            /* Step 2: Identify all points x such that u(x) <= s(c(x)) */
//            if (upperBound[j] <= s[closestCenters[j]])
//                continue;
//
//            rj = rjreset;
//
//            for (int64 k = 0; k < numCenters; k++)
//            {
//                Datum		vec;
//                float		dxcx;
//
//                /* Step 3: For all remaining points x and centers c */
//                if (k == closestCenters[j])
//                    continue;
//
//                if (upperBound[j] <= lowerBound[j * numCenters + k])
//                    continue;
//
//                if (upperBound[j] <= halfcdist[closestCenters[j] * numCenters + k])
//                    continue;
//
//                vec = PointerGetDatum(VectorArrayGet(samples, j));
//
//                /* Step 3a */
//                if (rj)
//                {
//                    dxcx = DatumGetFloat8(FunctionCall2Coll(procinfo, collation, vec, PointerGetDatum(VectorArrayGet(centers, closestCenters[j]))));
//
//                    /* d(x,c(x)) computed, which is a form of d(x,c) */
//                    lowerBound[j * numCenters + closestCenters[j]] = dxcx;
//                    upperBound[j] = dxcx;
//
//                    rj = false;
//                }
//                else
//                    dxcx = upperBound[j];
//
//                /* Step 3b */
//                if (dxcx > lowerBound[j * numCenters + k] || dxcx > halfcdist[closestCenters[j] * numCenters + k])
//                {
//                    float		dxc = DatumGetFloat8(FunctionCall2Coll(procinfo, collation, vec, PointerGetDatum(VectorArrayGet(centers, k))));
//
//                    /* d(x,c) calculated */
//                    lowerBound[j * numCenters + k] = dxc;
//
//                    if (dxc < dxcx)
//                    {
//                        closestCenters[j] = k;
//
//                        /* c(x) changed */
//                        upperBound[j] = dxc;
//
//                        changes++;
//                    }
//                }
//            }
//        }
//
//        /* Step 4: For each center c, let m(c) be mean of all points assigned */
//        ComputeNewCenters(samples, agg, newCenters, centerCounts, closestCenters, normprocinfo, collation, typeInfo);
//
//        /* Step 5 */
//        for (int j = 0; j < numCenters; j++)
//            newcdist[j] = DatumGetFloat8(FunctionCall2Coll(procinfo, collation, PointerGetDatum(VectorArrayGet(centers, j)), PointerGetDatum(VectorArrayGet(newCenters, j))));
//
//        for (int64 j = 0; j < numSamples; j++)
//        {
//            for (int64 k = 0; k < numCenters; k++)
//            {
//                float		distance = lowerBound[j * numCenters + k] - newcdist[k];
//
//                if (distance < 0)
//                    distance = 0;
//
//                lowerBound[j * numCenters + k] = distance;
//            }
//        }
//
//        /* Step 6 */
//        /* We reset r(x) before Step 3 in the next iteration */
//        for (int j = 0; j < numSamples; j++)
//            upperBound[j] += newcdist[closestCenters[j]];
//
//        /* Step 7 */
//        for (int j = 0; j < numCenters; j++)
//            VectorArraySet(centers, j, VectorArrayGet(newCenters, j));
//
//        if (changes == 0 && iteration != 0)
//            break;
//    }
//}
//
//
///*
// * Perform naive k-means centering
// * We use spherical k-means for inner product and cosine
// */
//void
//IvfflatKmeans(Relation index, VectorArray samples, VectorArray centers, const IvfflatTypeInfo * typeInfo)
//{
//    MemoryContext kmeansCtx = AllocSetContextCreate(CurrentMemoryContext,
//                                                    "Ivfflat kmeans temporary context",
//                                                    ALLOCSET_DEFAULT_SIZES);
//    MemoryContext oldCtx = MemoryContextSwitchTo(kmeansCtx);
//
//    ElkanKmeans(index, samples, centers, typeInfo);
//
//    MemoryContextSwitchTo(oldCtx);
//    MemoryContextDelete(kmeansCtx);
//}
//
//
///*
// * Compute centers
// */
//static void
//ComputeCenters(IvfflatBuildState * buildstate)
//{
//    int			numSamples;
//
//    pgstat_progress_update_param(PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_IVFFLAT_PHASE_KMEANS);
//
//    /* Target 50 samples per list, with at least 10000 samples */
//    /* The number of samples has a large effect on index build time */
//    numSamples = buildstate->lists * 50;
//    if (numSamples < 10000)
//        numSamples = 10000;
//
//    /* Skip samples for unlogged table */
//    if (buildstate->heap == NULL)
//        numSamples = 1;
//
//    /* Sample rows */
//    /* TODO Ensure within maintenance_work_mem */
//    buildstate->samples = VectorArrayInit(numSamples, buildstate->dimensions, buildstate->centers->itemsize);
//    if (buildstate->heap != NULL)
//    {
//        SampleRows(buildstate);
//
//        if (buildstate->samples->length < buildstate->lists)
//        {
//            ereport(NOTICE,
//                    (errmsg("ivfflat index created with little data"),
//                            errdetail("This will cause low recall."),
//                            errhint("Drop the index until the table has more data.")));
//        }
//    }
//
//    /* Calculate centers */
//    IvfflatBench("k-means", IvfflatKmeans(buildstate->index, buildstate->samples, buildstate->centers, buildstate->typeInfo));
//
//    /* Free samples before we allocate more memory */
//    VectorArrayFree(buildstate->samples);
//}
