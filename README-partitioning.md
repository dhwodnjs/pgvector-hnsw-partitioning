# HNSW 파티셔닝 코드 한눈에 보기

## `hnsw.h` 
- 파티션 상수: `MAX_NODES_PER_PARTITION`, `INSERT_PAGE_PER_PARTITION`, `MAX_PARTITION_ENTRIES`.
- PID 확장: `HnswElementData.pid`, `HnswElementTuple.pid`.
- 상태 구조체: `HnswPartition`/`HnswPartitionState`.

## `pgvector/src/hnswbuild.c` (Build)
- `BuildIndex` 옵션
  - `BuildGraph`(vanilla): 병렬 스캔 후 `FlushPages`로 메타/그래프/이웃 기록.
  - `BuildGraphWithPartition`(partitioning): 병렬 스캔 후 partitioning(`HnswPartitionGraph`) 적용 → `FlushPagesWithPartitionsPage`로 메타/그래프/이웃 기록.
- partitioning
  - `InitPartitionState`: 노드 수 기반 파티션 배열(capacity/size/pid/nodes) 준비.
  - `HnswPartitionGraph`: 초기 배치 + LDG 반복(`HnswPartitionGraphLDG`)으로 pid 지정.
  - `SelectPartition`/`GetUnfilledPartition`/`AddNodeToPartition`: LDG에서 이웃 다수결+capacity로 배치.
  - `CountOverlapRatioForInsert`: 파티션 내부 연결 비율 계산·정렬(삽입 우선순위에 사용).
- flush
  - `CreateMetaPageWithPartition`: 메타에 파티션/삽입 풀/`partitionPageCount` 기록.
  - `CreatePartitionPages`: PID→확장 페이지 슬롯 테이블 작성.
  - `CreateGraphPagesWithPartitions`/`WriteNeighborTuplesWithPartitions`: 파티션 순서로 요소/이웃 튜플 기록.
  - `FlushPagesWithPartitionsPage`: 위 절차 묶음 호출.

## `pgvector/src/hnswinsert.c` (Insert)
- `HnswInsertTuple` 옵션
  - `HnswInsertTupleOnDisk`(vanilla)
  - `HnswInsertTupleOnDiskWithPartitionPage`(partitioning)
- partitioning 함수:
  - `CalculatePartitionNeighborCount`: 레벨0 이웃 PID별 빈도 집계.
  - `AddElementOnDiskWithPartitionPage`: 메타의 `partitionPageCount`와 파티션 페이지(`entries`)를 읽어 PID별 확장 페이지 선택/할당(`HnswUpdatePartitionPage`), 요소/이웃 튜플 기록, `pid` 설정.
  - `UpdateGraphOnDiskWithPartitionPage`: insertPage/entryPoint 갱신 후 이웃 업데이트.

## `pgvector/src/hnswutils.c`
- 메타/파티션 페이지: `HnswGetMetaPageInfoWithPartitionPage`, `HnswUpdateMetaPagePartitionPage`, `HnswUpdatePartitionPage`, `HnswUpdateMetaPageInfoWithPartition`.
- 튜플 입출력: `HnswSetElementTuple`, `HnswLoadElementFromTuple`에서 `pid` 직렬화/복원, `HnswInitElement` pid 초기값 -1.

## 빠른 실행 가이드
- vanilla
  - build: `BuildIndex`→`BuildGraph`
  - insert: `HnswInsertTuple`→`HnswInsertTupleOnDisk`
- partitioning
  - build: `BuildIndex`→ `BuildGraphWithPartition`
  - insert: `HnswInsertTuple`→ `HnswInsertTupleOnDiskWithPartitionPage`
