# D-18: database paper retrieval evaluation

`scripts/compare_retrieval.py` compares dense, BM25 and hybrid search on the configured
PostgreSQL database. It evaluates paper retrieval over title + abstract, independently
of the planner, generation, chunk search and guardrails. It calls the real search backend
even when `USE_MOCKS=1`.

D-22 compares embedding models with exact cosine search outside PostgreSQL. D-18 reuses
its query reader, stratified split and metric functions, but measures the application's
database retrieval with the model already stored in the database. The existing overall
Recall@20 measurements in D054/D064 are a historical reference, not a substitute for
this split-specific comparison. Do not interpret differences from D-22 as model gains:
the retrieval implementation and embedding input limits also differ (D050).

## Run

Requirements:

- `DATA_DIR/filtered/d03/{query.arrow,manifest.json}` at the pinned LitSearch revision.
- `DATABASE_URL` pointing to the full D-03 database, with search indexes and embeddings ready.
- The pinned embedding model available locally or downloadable. CPU execution is supported;
  rebuilding the database, recomputing stored embeddings and using an LLM are unnecessary.

Run from the repository root. Configuration is read from `.env` and the environment.

```powershell
# Trial on the first five queries, with the original full-set split assignments.
uv run --locked --group data --group rag --group embed python scripts/compare_retrieval.py --limit 5

# Full evaluation: all 597 queries.
uv run --locked --group data --group rag --group embed python scripts/compare_retrieval.py
```

Full runs write `eval/results/d18_retrieval_comparison.json` and
`docs/d18_retrieval_comparison.md`. Smoke runs default to
`eval/private/d18_retrieval_smoke.json` and `.md`, so a trial does not replace the full
results. `--output PATH` and `--report PATH` override these destinations. Existing output
files at the chosen paths are replaced after a successful evaluation.

Offline evaluation allows 60 seconds per SQL statement by default. Use
`--statement-timeout SECONDS` to choose another positive, finite budget; the value is
recorded in JSON and Markdown. The API keeps its existing request budget and 10-second
SQL cap. This comparison measures retrieval quality, not interactive response times or
availability: a query that takes more than 10 seconds may complete in this batch run.

The script verifies the dataset revision, all 597 queries, the database's paper IDs against
the D-03 manifest, target coverage and non-null paper embeddings. The normal backend also
checks schema, model and search indexes. A partial fixture is rejected. Database access is
read-only. A failed search aborts the evaluation; it is not silently removed from the mean.

## Protocol

| Setting | Value |
|---|---|
| Dataset | `princeton-nlp/LitSearch`, revision `9573fb284a1026c998df47024b888a163f0f0e25` |
| Queries | All 597, with the original `corpusids` as binary relevance labels |
| Split | D-22: 50/50 within each `query_set`, NumPy seed 42; 300 dev / 297 test |
| Paper input | Title + abstract; existing stored paper embeddings |
| Dense and BM25 | Application `SearchBackend.dense` / `.bm25`, 100 candidates each |
| BM25 title weight | Existing `pdb.boost(0.5)`; abstract keeps its default boost |
| Hybrid | Application reciprocal rank fusion, k=60, using those same candidates |
| Result cutoff | Up to 100 distinct paper IDs per method |
| Metrics | Recall@5, Recall@20, MRR@100, NDCG@10; macro-average over queries |
| Offline SQL timeout | 60 seconds per statement by default; recorded with results |

The application returns at most 20 hybrid results. At `k=20` it retrieves 100 candidates
per method before fusion. The evaluation keeps that candidate budget and extends only
the fused output to 100. Its first 20 results therefore match the application's ranking
for the same candidate lists. Results at smaller UI limits can differ because the
application then uses a smaller candidate budget. No production limits are changed.

MRR@100 is the reciprocal rank of the first relevant paper among the first 100 results,
or zero when absent. For compatibility with D-22, the JSON metric key is `mrr`; its cutoff
is explicit in `metric_cutoffs`. Empty rankings receive zero for every metric. Short
BM25 result lists are not padded. Dev/test group means are `null` when a smoke run has no
queries from that group; full runs contain both groups.

The JSON records per-query IDs, `query_set`, split and metrics,
plus the database metadata, query-file SHA-256, split seed and fixed search settings.
Query IDs are row indices (`q000` to `q596`) at the pinned revision; the file hash identifies
the exact input order. Target paper IDs and rankings are used only in memory, as in D-22.
Paper IDs, query texts, titles, abstracts, credentials and database URLs are not exported.
The Markdown report presents separate dev, test and overall tables.

## Interpretation and validation

Keep the title weight and fusion settings fixed for this comparison. Use dev for future
changes and test for final reporting; recording a split does not undo earlier tuning on
the test queries. NDCG@10 uses the same metric definition as D-22, but the filtered corpus
and local retrieval protocol prevent direct comparisons with MTEB scores.

Full runs also compute paired percentile bootstrap intervals for hybrid minus dense on
Recall@20, MRR@100 and NDCG@10, separately on Dev and Test: 10,000 query resamples, seed 42,
pointwise 95% intervals, using D-22's helper. The JSON stores these under `bootstrap` and
the report includes the intervals. Smoke runs omit bootstrap inference.
Recompute these statistics from the saved metrics without querying the database or
loading an embedding model:

```powershell
uv run --locked --group data --group rag python scripts/compare_retrieval.py --score-only eval/results/d18_retrieval_comparison.json
```

This preserves the original retrieval scores and regenerates the report. Alternative
destinations can be supplied with `--output` and `--report`.
Before writing, saved results are checked for query IDs and counts, the D-22 split,
finite metric values and agreement between per-query metrics and stored averages.
Incomplete or inconsistent files are rejected without replacing their contents.
Smoke files retain their original full-set split assignments; no bootstrap inference
is made from a partial run.

Run the unit tests with:

```powershell
uv run --locked --group rag --group embed python -m pytest tests/test_compare_retrieval.py tests/test_compare_embeddings.py -q
```

Tests use controlled candidate lists and cover rank-fusion equivalence at 20 results,
MRR beyond rank 20, split-specific averaging, empty results, failure handling, corpus
preflight, D-22 split reuse and score-only exports. They do not establish retrieval quality
on the live database. Commit a result table only after a successful full run.

Local validation on 2026-10-29: the database contained all 56,115 D-03 papers with
embeddings, and all 597 split assignments matched the saved D-22 split exactly.
A full run completed all 597 queries (300 dev, 297 test) with the 60-second offline SQL
budget. The exported averages were independently checked against the per-query metrics;
no paper IDs, rankings or query text are present. See the
[result tables](d18_retrieval_comparison.md) and
[score JSON](../eval/results/d18_retrieval_comparison.json).
The separate offline budget was introduced after an initial run hit the API's 10-second
cap at `q202`; the API's timeout and all ranking settings remain unchanged.
