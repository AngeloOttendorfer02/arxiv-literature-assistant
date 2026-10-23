# Retrieval, generation and citations

## Components and boundaries

| Component | Implementation | Responsibility |
| --- | --- | --- |
| Planner | `scilitrag.rag.planner` | Strict scenario JSON, explicit filter reconciliation and S1 fallback |
| Model access | `scilitrag.rag.llm` | Configured completions, bounded retries and cancellable token streaming |
| Search | `scilitrag.rag.search` | Parameterized database reads, index checks, entity linking and rank fusion |
| Scenario retrieval | `scilitrag.rag.retrieval`, `scilitrag.rag.scenarios` | Topic evidence, citation paths and author citation credits |
| SQL facts | `scilitrag.rag.sql_facts` | Deterministic ranking tables with parameters, revision and query identity |
| Evidence | `scilitrag.rag.evidence` | Request-local labels, context budgeting, source snapshots and citation rendering |
| Evaluation | `scilitrag.evaluation` | Validated reference sets, recorded pipeline runs and reviewed quality metrics |

Shared contracts remain in `scilitrag.contracts`. Evaluation has its own input models;
it does not change the API contracts or import executable scripts. Data ingestion owns
paper/author links, corpus embeddings, index contents and dataset metadata. The guardrail
layer owns semantic sentence support and input safety. The UI consumes shared event types
and does not import retrieval implementation modules.

## Scenario behavior

### S1: topic literature search

The planner extracts the topic, explicit publication-year bounds and result limit.
Dense pgvector search and BM25 search use the same bound filter parameters. Reciprocal
rank fusion uses k=60. Passage retrieval is limited to selected paper IDs and prioritizes
section coverage while suppressing near duplicates. Each result retains corpus, chunk,
year and author identity. Results without retrieved passages are explicitly abstract-only;
abstracts are not presented as proof of full-text experimental results.

### S2: citation paths and experimental baseline candidates

EntityLinker resolves a numeric corpus ID, full title, unambiguous title prefix or confident
trigram match. Ambiguous references produce a clarification with candidate IDs. SQL follows
incoming edges: `citations.cited_id` is the target and `citations.citing_id` is a result.
Year/author filters apply to citing papers, not the target's publication year.

Ordinary searches return up to k citing papers by descending year, then corpus ID. Baseline
searches inspect up to 100 citing papers and up to four retrieved passages per paper, returning
at most k results. The target title or colon-prefix alias must occur with an explicit
experimental-use expression in the same sentence. Negations and related-work sections are
excluded. A citation edge, mention, or substring such as BERT inside RoBERTa is insufficient.

This English lexical filter selects candidates. It can miss abbreviations and paraphrases,
and can still misread complex comparisons. Human baseline labels and semantic validation
are required before reporting its quality. A no-support response describes the bounded
search and possible absence of indexed full text; it does not claim global absence of use.

### S3: cited-author ranking

Topic retrieval selects up to k citing-paper seeds; SQL ranks up to k authors from their
outgoing citation edges. For every distinct (citing paper, cited paper) pair, every author
of the cited paper receives +1. Coauthors receive full credit; self-citations are included.
Ties sort by author name under deterministic PostgreSQL C collation, then author ID.
An author filter restricts credited authors, not the selected citing-paper set.

The pipeline passes its completed retrieval output to the SQL adapter's optional
`with_evidence` hook, so author credits use the exact selected seed set. Standalone
`sql_facts(plan)` remains available and retrieves its own seeds. There is no global request
cache. Each table records the selected IDs, counting rule, bound SQL parameters, dataset
revision, example cited papers and a content-derived query ID. These are subset counts,
not global bibliometric totals. Topic selection and ranking both use the plan's k value.

Ranks refer to stored author IDs. O2 normalization can merge namesakes or split one person's
work across spellings, and source extraction can create malformed names. See the existing
[author review](d08_author_review.md). Aggregation cannot repair author identity errors.
That author review is separate from experimental baseline-use annotation.

## Grounded generation and Citation Engine

Paper metadata and passages are untrusted data. The prompt requires short factual sentences
with trailing labels and forbids document instructions from overriding the system prompt.
Context assembly respects character and per-item budgets and preserves source identity.
SQL tables are excluded from the model context: the application renders their values and
supplies the counting-rule introduction, while the model summarizes paper evidence.

Evidence labels are unique positive integers allocated per request. SQL snapshots append
new labels without changing existing ones. Every complete sentence passes structural label
checks, the selected sentence validator, PII masking and deterministic citation rendering
before it is emitted. Unknown or malformed labels and unsupported sentences are removed.
Citation objects include source identity, title, year, section, text and a fingerprint;
SQL citations resolve to their exact table by query ID. Label validity establishes structure,
not factual entailment. See [execution flow](execution_flow.md).

The API translates controlled entity, no-result and provider failures into terminal SSE
errors. Cancellation closes the provider stream. The request timeout bounds the pipeline;
database reads are read-only and have a statement timeout. Separate database reads do not
provide a transaction-wide snapshot: evaluate against a frozen corpus and stable revision.

## Configuration and integration

Install dependencies with `uv sync --locked --group rag`. Configure DATABASE_URL with the
SELECT-only app login, and LLM_BASE_URL, LLM_MODEL and EMBED_MODEL using `.env.example`.
The embedding model must match the stored 768-dimensional vectors and schema metadata.

Before real retrieval:

```powershell
uv run --locked --group rag python scripts/check_database.py --require-search
```

The uploaded branch includes real planner, retrieval, SQL facts, generation and citation
components. The shared `scilitrag.data.embedding` adapter and corpus/index population are
still required from the data pipeline. Real guardrail modules remain stubs in this snapshot.
Use the selective development configuration until those integrations are ready:

```powershell
$env:USE_MOCKS = "1"
$env:REAL_COMPONENTS = "plan,retrieve,sql_facts,generate,render_citations"
uv run --locked --group rag uvicorn scilitrag.api.app:create_app --factory --reload
```

This requires the real data adapter and model service; input/semantic checks remain mocked.
Select real retrieve and sql_facts together for S3: mock corpus IDs cannot define a meaningful
real database ranking. Do not enable USE_MOCKS=0 while required real modules remain stubs.

## Acceptance checklist

| Area | Implemented | Remaining evidence |
| --- | --- | --- |
| Model client, JSON planning and filter reconciliation | Yes | Configured-model acceptance run and scenario routing |
| Hybrid retrieval, linking and S1 coverage | Yes | Frozen corpus, real embedder/index handoff and live query checks |
| Citation paths and baseline candidate filter | Yes | Representative human labels and precision/recall measurement |
| SQL author aggregation and provenance | Yes | Disposable-DB tests and independently checked real-corpus references |
| Guarded streaming and citation rendering | Yes | Real semantic-validator and browser integration |
| Reference export and quality evaluation | Yes | Real candidate-model runs, reviewed support labels and final model choice |
| Multi-hop implementation documentation | Yes | Measured timings/results for the final report |

Run local checks from the repository root:

```powershell
$env:USE_MOCKS = "1"
$env:REAL_COMPONENTS = ""
$env:LOG_DIR = ""
uv run --locked --group data --group rag --group ui ruff check .
uv run --locked --group data --group rag --group ui ruff format --check .
uv run --locked --group data --group rag --group ui pytest -q
uv run --locked --group data --group rag --group ui python scripts/smoke.py
```

Tests in `tests/rag/` cover retrieval/generation behavior. Tests in `tests/evaluation/`
cover reference validity, recorded runs, review scoring and failure cases. Optional SQL
tests require RETRIEVAL_TEST_DATABASE_URL for a disposable ParadeDB database with schema
creation privileges; they use isolated schemas, not the application corpus. Missing live
services and the known mock jailbreak failure are not evidence of semantic acceptance.
