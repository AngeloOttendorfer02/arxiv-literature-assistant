# Retrieval evaluation

## Layout and commands

| Path | Purpose |
| --- | --- |
| `eval/retrieval/cases.example.json` | Synthetic questions for offline development checks |
| `eval/private/` | Ignored real references, source passages, annotations and run outputs |
| `src/scilitrag/evaluation/datasets.py` | Validated reference inputs and unique case identities |
| `src/scilitrag/evaluation/reference.py` | Private reference export and independent citation-credit calculation |
| `src/scilitrag/evaluation/runner.py` | Pipeline runs, provenance capture and offline re-scoring |
| `src/scilitrag/evaluation/metrics.py` | Structural, reference and human-reviewed quality metrics |
| `src/scilitrag/evaluation/review.py` | Source-bound CSV export/import for human judgments |
| `scripts/build_reference_set.py` | Reference-export command |
| `scripts/evaluate_retrieval.py` | Run and re-score command |

The scripts are thin entry points. Tests and reusable callers import the package modules,
so evaluation is independent of script path configuration. Production modules and public
example directories use feature names for clarity.

## Offline development check

With the rag group installed, use the synthetic examples:

```powershell
$env:USE_MOCKS = "1"
$env:REAL_COMPONENTS = ""
$env:LOG_DIR = ""
uv run --locked --group rag python scripts/evaluate_retrieval.py eval/retrieval/cases.example.json --allow-mocks --output eval/private/development.json
```

The runner labels this development-only and records mocked guardrails. It does not contact
a database in fully mocked mode. Without --allow-mocks, it requires all core retrieval,
planning, SQL, generation and citation components to be selected as real. Guardrails that
remain mocked are explicitly recorded and keep the report development-only.

A successful execution requires a non-refused done event and at least one emitted sentence
or nonempty table, plus correct scenario routing. Empty answers, refusals, errors and timeouts
do not count as completed evaluations. `--fail-fast` saves a partial diagnostic report after
the first failure or routing mismatch, recording unrun cases for every requested candidate.
Omit it for final comparisons. A nonzero exit can still leave a useful failure report.
Exit zero establishes execution and routing, not a reviewed quality threshold.

## Export frozen references

Configure the SELECT-only DATABASE_URL and publish a real dataset revision in schema_metadata.
Use IDs inspected in your own corpus. The IDs below are the synthetic integration fixture,
not guaranteed real-corpus targets:

```powershell
uv run --locked --group rag python scripts/build_reference_set.py --target-paper 1 --seed-paper 2 --seed-paper 5 --topic retrieval --output eval/private/reference-set.json
```

The exporter rejects unknown/out-of-range IDs and missing revisions, deduplicates selected
IDs, preserves existing output files, and leaves baseline-use labels unset. For S2 it
exports the top 20 citing papers by publication year. It considers up to 200 indexed
passages per target, prioritizing experimental sections and a deterministic hash order.
The default sample is **20 passages total**, alternating targets and heuristic-positive
and negative candidates, with at most one passage per citing paper and target. Adjust
`--baseline-limit` if necessary. This stratified diagnostic sample is not a corpus-wide
precision or prevalence estimate. Review passages in context; a mention alone is false.
Source hashes bind the text, target and passage identity while labels and notes stay editable.

For S3, an independent reference enumerates distinct citation-pair/author triples, counts
credits in Python and applies the SQL tie rule. For an actual pipeline run use:

```powershell
uv run --locked --group rag python scripts/build_reference_set.py --from-run eval/private/model-runs.json --case-id s3-topic --model YOUR_FIRST_MODEL --target-paper 1 --output eval/private/scoped-reference.json
```

Replace the case, model and target with inspected values from your corpus. This preserves
the S3 query, case ID, SQL seed IDs, result limit and author filter. It rejects mismatched
paper evidence, SQL parameters and dataset revisions, and records a source-run fingerprint.
It independently recomputes counts rather than copying table values. A run with a later
generation error can still supply a valid SQL scope; this does not mark its answer successful.
Standalone `--seed-paper` export still requests k=20 and is exploratory until its seed set
matches an actual run. Freeze the corpus and regenerate references when changing scope.

The JSON root contains `cases`, optional `dataset_revision` and `baseline_annotations`.
Each case needs a unique nonblank id, a scenario S1/S2/S3 and a nonblank query. Optional
`expected_citing_ids`, `expected_years`, `seed_ids` and `expected_author_counts` supply the
applicable reference checks. Author-count references require nonempty S3 seed IDs. Unknown
fields, duplicate case/seed IDs and invalid label types are rejected before service access.
The reference schema is evaluation-specific and does not modify shared API contracts.

Keep all real excerpts and outputs in eval/private; they may reproduce LitSearch content
and must not be redistributed. Do not commit human annotation files containing real data.

## Compare installed model candidates

After the [real retrieval prerequisites](retrieval.md#configuration-and-integration) are
satisfied, use explicitly selected installed model names:

```powershell
$env:USE_MOCKS = "0"
$env:REAL_COMPONENTS = ""
uv run --locked --all-groups python scripts/check_database.py --require-search
uv run --locked --all-groups python scripts/evaluate_retrieval.py eval/private/reference-set.json --models YOUR_FIRST_MODEL YOUR_SECOND_MODEL --output eval/private/model-runs.json
```

Every model receives the same case queries and configured pipeline parameters; its planner
is part of the evaluated pipeline. Reports retain the actual plan, evidence, tables,
sentence events, errors, model identifier, selected settings and dataset revision. Secrets
and connection URLs are excluded from the settings snapshot.

The current main contains the real embedding adapter, input gate, evidence guard, PII
masking and sentence validator. Full runs need their dependencies/model weights, access
to the gated Prompt Guard checkpoint, and the ready frozen database. For diagnostics,
select only available components with `USE_MOCKS=1`; mocked checks stay explicitly marked
and those runs remain development-only. Do not present them as full guardrail acceptance.
Start with `cases.live.example.json` and `--fail-fast`, then freeze a corpus-specific case
set spanning S1, S2 (including reviewed baseline use) and S3 for the final comparison.

The runner checks the database revision before and after the batch. A supplied reference
revision must match before real inference begins. A changed revision invalidates reference
comparisons and gives a nonzero exit. This detects metadata changes, not unversioned row
mutations: the database must actually remain frozen. Offline reports lacking provenance
cannot establish agreement with a revision-pinned reference.

Record hardware, model digests, quantization, warm-up policy and engine measurements with
the benchmark owner. Application timing events and text-delta throughput do not replace
engine-reported token counts, VRAM or the quantization benchmark. This code supplies no
measured model recommendation until the real runs and reviews are complete.

`summary` groups each candidate by expected scenario, including failed and missing cases.
Elapsed-time distributions include failed runs; missing legacy timing stays null. P95 is
the nearest-rank percentile. Quality averages use the number of reviewed citations or
sentences as weights; their denominators are included. `component_ms` captures search
timings even when a later stage fails. Saved evidence reflects the evidence guard's output.
The legacy API `ttft_engine_ms` starts at pipeline entry and can include code-generated
framing; summaries call it `pipeline_first_delta_ms`. It is not engine-only model TTFT.

R-13's final decision remains a joint review with WP4: compare at least two real candidates
on the same frozen cases, corpus, thresholds and request settings. Record model digests,
the run/review file identities, scenario coverage, reviewed Citation Precision, year/count
Exact Match, failures, engine measurements from `bench/`, and the reason for the chosen
model in `docs/decisions.md`. The engine benchmark already controls prompt-only comparisons;
this runner evaluates the planner and full application pipeline. It does not choose a winner.

## Human review and offline scoring

Fill `baseline_annotations[].baseline_use` with true/false after reviewing the target and
passage. Leave uncertain entries null and add reviewer notes. For each saved run, fill its
`sentence_supported` array in emitted-sentence order with true/false/null. A sentence is
supported only if its cited sources support the complete claim; valid label syntax alone
is insufficient. Keep exactly one label per emitted sentence when supplying reviews.

For a worksheet instead of editing the report, export and later import CSV offline:

```powershell
uv run --locked --all-groups python scripts/evaluate_retrieval.py eval/private/reference-set.json --score-only eval/private/model-runs.json --output eval/private/rescored-runs.json --export-review eval/private/model-review.csv
# Fill the CSV, then use a new report filename:
uv run --locked --all-groups python scripts/evaluate_retrieval.py eval/private/reference-set.json --score-only eval/private/model-runs.json --import-review eval/private/model-review.csv --output eval/private/reviewed-runs.json
```

Each `sentence` row reviews the whole claim against all supplied citations. Each `citation`
row reviews that individual reference's support for the claim it is attached to. Edit only
`supported`, sentence-row `year_exact`/`count_exact`, and `note`; use true, false or blank.
Mark year/count exact only after comparing **every** applicable value in that sentence with
the identified paper year or independently checked SQL count; one wrong value makes the
sentence false. Leave non-applicable and uncertain fields blank. These are explicit human
Exact Match judgments, separate from automatic source-metadata and SQL-table comparisons.
Keep the static text, identities, hashes, columns and full row set; sorting rows is allowed.
Stale, changed, duplicate or missing rows are rejected. All blank reviews remain unknown.
This Citation Precision measures support, not the ALCE redundancy score in `bench/`.

```powershell
uv run --locked --group rag python scripts/evaluate_retrieval.py eval/private/reference-set.json --score-only eval/private/model-runs.json --output eval/private/reviewed-runs.json
```

Offline scoring does not load runtime settings or contact database/model services. It
rejects empty reports, unknown case IDs and malformed records rather than returning
success for an empty run list. Original outputs are preserved when a different output
path is supplied. Old reports can still be inspected, but pinned-reference metrics remain
unset when their dataset revision is missing or mismatched.

| Metric | Interpretation |
| --- | --- |
| completed / refused | Execution outcome; a refusal or empty answer is not successful completion |
| scenario_exact | Planned scenario versus the reference case |
| structural_citation_validity | Emitted sentences with valid request-local trailing labels |
| semantic_citation_precision | Fully supported reviewed cited sentences divided by reviewed sentences |
| semantic_review_coverage | Reviewed emitted sentences divided by all emitted sentences |
| citation_precision / citation_review_coverage | Supported individual references / reviewed references, plus review coverage |
| answer_year_exact / answer_count_exact | Human exact-value judgments on applicable sentences; reviewed denominators are reported |
| citing_precision / citing_recall | Returned citing-paper IDs versus the supplied bounded S2 reference |
| source_year_exact | Agreement for reference papers with known years; missing returned papers/years count as wrong |
| author_scope_exact | Retrieved S3 seed IDs match the reference IDs |
| author_count_exact | Count agreement only with matching seed scope and compatible dataset revision |
| dataset_revision_exact | Known actual revision agrees with the supplied reference revision |
| baseline_filter_metrics | Precision, recall and confusion counts on reviewed baseline passages |

`semantic_citation_precision` is a sentence-level support measure; `citation_precision` is
per reference. `source_year_exact` checks metadata, while `answer_year_exact` and
`answer_count_exact` explicitly review rendered numerical claims. The output validator's
number/NLI checks remain active and independent. Unreviewed quality, undefined ratios and
incompatible dataset-reference comparisons remain null. Existing outputs are never overwritten.

## Renamed interfaces

| Previous path | Current path |
| --- | --- |
| `eval/wp2/cases.mock.json` | `eval/retrieval/cases.example.json` |
| `scripts/build_wp2_golden.py` | `scripts/build_reference_set.py` |
| `scripts/evaluate_wp2.py` | `scripts/evaluate_retrieval.py` |
| `scilitrag.rag.evaluation` | `scilitrag.evaluation.metrics` |
| Script-level `build_goldens` | `scilitrag.evaluation.reference.build_reference_set` |

Update local commands and custom imports to these paths. No duplicate compatibility
wrappers or duplicate example folders are retained. Existing eval/private files remain
where they are and stay ignored.
