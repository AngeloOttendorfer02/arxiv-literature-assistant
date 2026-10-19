# D-03 Quality filters

Dataset: `princeton-nlp/LitSearch`, revision `9573fb284a1026c998df47024b888a163f0f0e25`, split `full`.

Filtered Arrow files exported.


## Summary

| Measure | Papers / queries |
|---|---:|
| input_papers | 64,183 |
| retained_papers | 56,115 |
| rejected_papers | 8,068 |
| queries | 597 |
| target_papers | 574 |
| retained_target_papers | 574 |
| targets_with_rejection_criteria | 7 |

## Rejection rules

| Criterion | Affected | Rejected | Protected targets |
|---|---:|---:|---:|
| Empty/blank corpus_clean.full_paper | 10 | 10 | 0 |
| Nonblank full_paper of exactly 1 Unicode character | 2 | 2 | 0 |
| Empty/blank corpus_clean.title | 6,760 | 6,753 | 7 |
| Title longer than 500 Unicode characters | 75 | 75 | 0 |
| No nonblank S2ORC sectionheader annotation | 1,599 | 1,599 | 0 |
| Both S2ORC authorfirstname and authorlastname annotations absent | 1,720 | 1,720 | 0 |

Rule counts overlap. The summary counts each rejected paper once. All query targets are retained, even when a rejection criterion applies.

Lengths count Unicode characters including surrounding spaces. Blank text is counted as empty, not as a 1-character full text.

Author absence means that both first-name and last-name annotations lack nonblank text spans. Having only one of these annotations does not reject the paper. No author normalization or person identification is performed.

## Missing years

Missing years do not reject papers. Output years are nullable integers: missing values become Arrow null (`None` in Python, SQL `NULL` at import).

| Scope | Missing years |
|---|---:|
| input | 458 |
| retained | 451 |
| Rejected for other quality criteria | 7 |
| targets | 0 |

## Output and scope

Outputs: `DATA_DIR/filtered/d03/{query,corpus_clean,corpus_s2orc}.arrow` (Arrow IPC streams) and `manifest.json`. Both corpus subsets contain the same retained IDs; queries and their target lists are preserved. Read an output with `pyarrow.ipc.open_stream(path).read_all()`.

The D-01 cache is unchanged. Existing filtered output is never overwritten. Title repair/truncation, email masking, author normalization and citation filtering are not performed. Citation IDs may point outside the retained corpus; D-09 must handle this explicitly.

Nonblank text can still contain extraction errors; these rules do not certify text quality.

## Protected target papers

All 574 distinct target papers are retained. Listed below are targets with a rejection criterion or missing year. No replacement titles or author names are invented.

| corpusid | Issues |
|---|---|
| 218974489 | empty_title |
| 235097235 | empty_title |
| 236477489 | empty_title |
| 236478113 | empty_title |
| 236486171 | empty_title |
| 249062579 | empty_title |
| 254685611 | empty_title |
