# D-18 Database retrieval comparison

Status: full run.
LitSearch revision `9573fb284a1026c998df47024b888a163f0f0e25`, D-03 corpus: 56,115 papers, 597 queries. Paper-level binary relevance; title + abstract.
Embedding model: `ibm-granite/granite-embedding-english-r2`.
Offline SQL timeout: 60 seconds per statement. The API's default 10-second cap is unchanged; this is a retrieval-quality benchmark, not a response-time or availability test.
Dev/test: D-22's 50/50 split stratified by `query_set`, seed 42. The split is assigned on all 597 queries before any smoke-run limit.
Dense and BM25 use the application's SQL and read-only database connections. BM25 title boost is 0.5; abstract boost is the default. Hybrid uses RRF k=60 on 100 candidates from each method. All returned rankings are capped at 100.
Hybrid top-20 matches the application's candidate budget at k=20; only this evaluation extends its output to 100 for MRR@100. No planner, LLM or chunk search.

## Dev (300 queries)

| Method | Recall@5 | Recall@20 | MRR@100 | NDCG@10 |
|---|---:|---:|---:|---:|
| dense | 0.580 | 0.672 | 0.476 | 0.505 |
| bm25 | 0.454 | 0.588 | 0.348 | 0.385 |
| hybrid | 0.561 | 0.688 | 0.445 | 0.480 |

## Test (297 queries)

| Method | Recall@5 | Recall@20 | MRR@100 | NDCG@10 |
|---|---:|---:|---:|---:|
| dense | 0.572 | 0.709 | 0.479 | 0.507 |
| bm25 | 0.471 | 0.626 | 0.378 | 0.415 |
| hybrid | 0.590 | 0.730 | 0.459 | 0.495 |

## All (597 queries)

| Method | Recall@5 | Recall@20 | MRR@100 | NDCG@10 |
|---|---:|---:|---:|---:|
| dense | 0.576 | 0.690 | 0.478 | 0.506 |
| bm25 | 0.463 | 0.607 | 0.363 | 0.400 |
| hybrid | 0.576 | 0.709 | 0.452 | 0.488 |

## Interpretation

Hybrid minus dense: paired percentile bootstrap over queries, separately on Dev and Test; 10,000 resamples, seed 42. Intervals are pointwise 95% intervals; all table values are in percentage points.

| Split | Metric | Difference | 95% interval |
|---|---|---:|---:|
| dev | Recall@20 | +1.60 | [-2.13, +5.37] |
| dev | MRR@100 | -3.09 | [-6.74, +0.44] |
| dev | NDCG@10 | -2.47 | [-5.91, +0.77] |
| test | Recall@20 | +2.08 | [-2.08, +6.29] |
| test | MRR@100 | -2.02 | [-5.72, +1.78] |
| test | NDCG@10 | -1.15 | [-4.67, +2.44] |

Hybrid has higher observed Recall@20, but lower MRR@100 and NDCG@10 on both splits: broader coverage within 20 results, with worse observed early ranking.

All six intervals include zero. This evaluation does not establish a statistically clear advantage for either method on these comparisons; it does not establish equivalence.

RRF k=60 and the existing weights remain unchanged. The team should agree on a primary metric before any tuning, use Dev only for that tuning, then freeze settings for the final Test evaluation.

Scores are macro-averaged over queries; no failures or empty rankings are dropped. The JSON key `mrr` follows D-22 and means MRR@100 here. Settings are fixed for this run, not tuned on test. This split does not undo any earlier use of test queries to choose settings. PostgreSQL dense retrieval is not D-22's exact cosine search.
NDCG@10 alone does not make these scores directly comparable with MTEB: this run uses the filtered D-03 corpus and the local database retrieval configuration.
