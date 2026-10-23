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
or nonempty table. Empty answers, refusals, errors and timeouts do not count as completed
evaluations. A nonzero exit can still leave a saved report containing useful failure events.

## Export frozen references

Configure the SELECT-only DATABASE_URL and publish a real dataset revision in schema_metadata.
Use IDs inspected in your own corpus. The IDs below are the synthetic integration fixture,
not guaranteed real-corpus targets:

```powershell
uv run --locked --group rag python scripts/build_reference_set.py --target-paper 1 --seed-paper 2 --seed-paper 5 --topic retrieval --output eval/private/reference-set.json
```

The exporter rejects unknown/out-of-range IDs and missing revisions, deduplicates selected
IDs, and leaves baseline-use labels unset. For S2 it exports the top 20 citing papers by
publication year and the first 20 indexed citing passages per target. This ordered sample
is a review worksheet, not a representative quality benchmark; expand and balance it
before reporting baseline-filter generalization.

For S3, an independent reference enumerates distinct citation-pair/author triples, counts
credits in Python and applies the SQL tie rule. The selected seed IDs must match the actual
evaluated topic-paper set. The exported query requests k=20, which can select a different
set in a larger corpus. Inspect a first run, export its exact seeds, and freeze the corpus
before comparing. If you edit queries, limits or filters, regenerate matching references.

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
$env:USE_MOCKS = "1"
$env:REAL_COMPONENTS = "plan,retrieve,sql_facts,generate,render_citations"
uv run --locked --group rag python scripts/evaluate_retrieval.py eval/private/reference-set.json --models YOUR_FIRST_MODEL YOUR_SECOND_MODEL --output eval/private/model-runs.json
```

Every model receives the same case queries and configured pipeline parameters; its planner
is part of the evaluated pipeline. Reports retain the actual plan, evidence, tables,
sentence events, errors, model identifier, selected settings and dataset revision. Secrets
and connection URLs are excluded from the settings snapshot.

The runner checks the database revision before and after the batch. A supplied reference
revision must match before real inference begins. A changed revision invalidates reference
comparisons and gives a nonzero exit. This detects metadata changes, not unversioned row
mutations: the database must actually remain frozen. Offline reports lacking provenance
cannot establish agreement with a revision-pinned reference.

Record hardware, model digests, quantization, warm-up policy and engine measurements with
the benchmark owner. Application timing events and text-delta throughput do not replace
engine-reported token counts, VRAM or the quantization benchmark. This code supplies no
measured model recommendation until the real runs and reviews are complete.

## Human review and offline scoring

Fill `baseline_annotations[].baseline_use` with true/false after reviewing the target and
passage. Leave uncertain entries null and add reviewer notes. For each saved run, fill its
`sentence_supported` array in emitted-sentence order with true/false/null. A sentence is
supported only if its cited sources support the complete claim; valid label syntax alone
is insufficient. Keep exactly one label per emitted sentence when supplying reviews.

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
| citing_precision / citing_recall | Returned citing-paper IDs versus the supplied bounded S2 reference |
| source_year_exact | Agreement of available source metadata years; missing years are not correct predictions |
| author_scope_exact | Retrieved S3 seed IDs match the reference IDs |
| author_count_exact | Count agreement only with matching seed scope and compatible dataset revision |
| dataset_revision_exact | Known actual revision agrees with the supplied reference revision |
| baseline_filter_metrics | Precision, recall and confusion counts on reviewed baseline passages |

Semantic citation precision here is a sentence-level support measure, not per-reference
precision. Year checks concern source metadata, not numbers generated in prose. Review
textual numerical claims separately with the sentence validator. Unreviewed quality,
undefined ratios and incompatible dataset-reference comparisons remain null.

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
