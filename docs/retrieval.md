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
BM25 matches the title or abstract; chunk BM25 matches passage text within selected
papers. Dense pgvector search and BM25 search use the same bound filter parameters. Reciprocal
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
at most k results. The target title, colon-prefix alias or reviewed identity-specific alias
must occur with an explicit
experimental-use expression in the same sentence. The filter requires wording tied to the
paper's own experiments, comparison or baseline, including the passive forms below. Negations,
hypothetical comparisons and related-work sections are excluded. A comparison attributed
to earlier work, a citation edge, a mention, or BERT inside RoBERTa is insufficient.

`rag/search/paper_aliases.py` holds the shared registry. Initially it covers DPR (corpus ID
215737187), `DPR` and `Karpukhin et al.` with year 2020 (comma, parenthesized and plain-year
forms). Other short names and citation aliases need reviewed entries; the code does not
guess acronyms or first-author order. Baseline passage queries include the reviewed aliases.
Matching supports `we use DPR ... as its pretrained baseline`, `our comparisons with DPR`,
`DPR (Karpukhin et al., 2020) is our baseline` and `DPR [12] serves as our baseline`.
Passive use is accepted when directly linked to own-study context, for example `DPR was
used as a baseline in our experiments`, `in our evaluation, DPR was used as the baseline`,
`DPR was used as our baseline` or `our method was compared with DPR`. A study cue elsewhere
in the sentence does not suffice; generic passive mentions and descriptions of earlier
work remain excluded.
Recognized author-year aliases are normalized before sentence splitting, preventing their
internal punctuation from breaking the match. This is not a general fix for every `et al.`.
An `outperforms` statement alone does not qualify, even with `our experiments` or `we find`;
the explicit baseline/comparison patterns above are still required.
Reviewed short names also exclude adjoining hyphens: `DPR-PAQ`, `DPR-based` and `pre-DPR`
do not count as the standalone target `DPR`. Titles and colon-prefix aliases retain the
existing word boundaries, so `we use BERT-base as our baseline` and `we use BERT-large as
a baseline` can still match the BERT title.

Known conservative gaps remain: plural `baselines`, an intervening word in `we also use`,
en-dash citation ranges such as `[12–14]`, and parentheses containing multiple citations.
These forms need a separately evaluated extension and remain outside the current rules.

Offline baseline scoring passes the annotation's `target_id` into the same matching rule.

This English lexical filter selects candidates. Reviewed aliases improve abbreviation coverage,
but retrieval limits and restrictive patterns can still miss relevant passages and paraphrases.
The filter can also misread complex comparisons. Human baseline labels and semantic validation
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

Install dependencies with `uv sync --locked --all-groups`. Configure DATABASE_URL with the
SELECT-only app login, and LLM_BASE_URL, LLM_MODEL and EMBED_MODEL using `.env.example`.
The embedding model must match the stored 768-dimensional vectors and schema metadata.

Before real retrieval:

```powershell
uv run --locked --group rag python scripts/check_database.py --require-search
```

The current main includes real planner, retrieval, SQL facts, generation, citation,
embedding and guardrail components. Populate or restore the database and verify its
embedding fingerprint and indexes before real queries. Full validation also needs the
configured NLI/embedding weights and access to the gated Prompt Guard checkpoint.
For complete pipeline evaluation use `USE_MOCKS=0`. A selective diagnostic setup is:

```powershell
$env:USE_MOCKS = "1"
$env:REAL_COMPONENTS = "plan,retrieve,sql_facts,generate,render_citations,check_evidence,check_sentence,mask_pii"
uv run --locked --all-groups uvicorn scilitrag.api.app:create_app --factory --reload
```

This configuration mocks input checks while exercising the real evidence and output checks;
it needs their model assets. Reports record the mocked stages and remain development-only.
Select real retrieve and sql_facts together for S3: mock corpus IDs cannot define a meaningful
real database ranking. See [evaluation](evaluation.md) for complete runs and reviews.

## Acceptance checklist

| Requirement | Implemented | Remaining evidence |
| --- | --- | --- |
| R-09: citation paths and baseline candidates | Incoming citation SQL, scoped chunks, conservative experimental-use filter | Human passage labels and real-corpus retrieval acceptance |
| R-10: guarded sentence streaming | Bounded buffering, real component handoff tests, fragment rejection | Live inference and WP4 browser acceptance |
| R-13: pipeline model comparison | Fixed cases, model/scenario summaries, timings, failures and coverage | At least two real candidates and joint decision with WP4 |
| R-14: S2/S3 reference tooling | DB citing IDs, twenty unlabeled passages, independent counts from recorded S3 scope | Human baseline labels and SQL checks on the frozen real corpus |
| R-15: scenario answer-quality metrics | Per-reference precision, sentence support, year/count Exact Match reviews | Completed source-bound reviews and reported quality values |
| R-16: execution flow and Citation Engine | Updated flow documentation and report paragraphs | Incorporate measured live examples/results when available |

Run local checks from the repository root:

```powershell
$env:USE_MOCKS = "1"
$env:REAL_COMPONENTS = ""
$env:LOG_DIR = ""
$env:EMBED_MODEL = "ibm-granite/granite-embedding-english-r2"
uv run --locked --all-groups ruff check .
uv run --locked --all-groups ruff format --check .
uv run --locked --all-groups pytest -q -rs
uv run --locked --all-groups python scripts/smoke.py
```

Tests in `tests/rag/` cover retrieval/generation behavior. Tests in `tests/evaluation/`
cover reference validity, recorded runs, review scoring and failure cases. Optional SQL
tests require RETRIEVAL_TEST_DATABASE_URL for a disposable ParadeDB database with schema
creation privileges; they use isolated schemas, not the application corpus. Missing live
services and the known mock jailbreak failure are not evidence of semantic acceptance.
