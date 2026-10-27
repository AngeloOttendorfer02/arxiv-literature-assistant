# Guardrail test sets (S-12)

All texts are our own wording and contain no LitSearch records. Each set is split into
`tuning` and `test`. Thresholds are chosen on `tuning` only (S-13); `test` is read once for
the reported numbers.

| File | Cases | Used against |
| --- | --- | --- |
| `input_attacks.jsonl` | Direct injection (overrides, system prompt exfiltration, jailbreaks), write requests, off-topic requests | `check_input`; `category` is the expected refusal category |
| `input_harmless.jsonl` | 100 legitimate questions with `category: ok`, including German, security topics and words such as delete or update | `check_input`; false refusals |
| `evidence_cases.jsonl` | Indirect injection: a paper passage with an instruction at the start, middle or end (`attack`), the same passage without it (`clean`), and paper abstracts that discuss attacks (`security_paper`) | `check_evidence`; `expected` is `removed` or `kept` |

Every row has `source` (`hand` or `ai`) and `reviewed`. The drafts are `ai` with
`reviewed: false` until a person has read them; set it to `true` or change the text. A person
checks every case before numbers go into the report.

LitSearch questions are not stored here. For the harmless set in the report, use the test
half of the D-22 split (`scripts/compare_embeddings.py`), read from `DATA_DIR` at
evaluation time.

Load the files with `scilitrag.evaluation.guardrail_sets`.
