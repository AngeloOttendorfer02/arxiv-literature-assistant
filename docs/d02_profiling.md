# D-02 Profiling LitSearch (revision 9573fb2, split "full")

Generated with `scripts/profile_data.py` from the prepared D-01 cache. The full report is written to `DATA_DIR/profiling/d02_profile.json`. Percentages and the total full_paper length are derived from that report.

## Rows

| Subset | Rows |
|---|---|
| query | 597 |
| corpus_clean | 64,183 |
| corpus_s2orc | 64,183 |

## Empty fields (corpus_clean)

| Field | Empty | Share | Kind |
|---|---|---|---|
| title | 6,760 | 10.5 % | empty string |
| abstract | 7,989 | 12.4 % | empty string |
| citations | 16,946 | 26.4 % | empty list |
| full_paper | 10 | 0.02 % | empty string |
| corpusid | 0 | 0 % | |

query: no empty fields.

## Length distributions (characters, nonblank values only)

| Field | n | Min | Median | Mean | p95 | Max |
|---|---|---|---|---|---|---|
| title | 57,423 | 1 | 70 | 75 | 119 | 8,560 |
| abstract | 56,194 | 1 | 880 | 961 | 1,681 | 18,356 |
| full_paper | 64,173 | 1 | 35,483 | 38,738 | 72,959 | 635,173 |

Total full_paper length: about 2.5 billion characters (mean x n).

## Very short full texts (corpus_clean)

Nonblank full_paper values only; the 10 completely empty ones are counted under empty fields.

| full_paper shorter than | Papers | of which without title and abstract |
|---|---|---|
| 10 characters | 21 | 21 |
| 100 characters | 104 | 99 |
| 2,000 characters | 345 | 195 |

The shortest texts are page numbers or stray characters (for example "1", "91 92"), which points to failed PDF text extraction.

## Missing years (corpus_s2orc, the only subset with a year)

| Field | Missing | Share |
|---|---|---|
| year | 458 | 0.7 % |

## corpusid and join corpus_clean / corpus_s2orc

| Check | corpus_clean | corpus_s2orc |
|---|---|---|
| Distinct IDs | 64,183 | 64,183 |
| Missing IDs | 0 | 0 |
| Duplicate IDs | 0 | 0 |
| Only in this subset | 0 | 0 |

Join: 64,183 common IDs, exactly one row pair per ID (1:1).

## Selected annotation gaps (corpus_s2orc, papers without this annotation)

| Annotation | Papers without | Relevant for |
|---|---|---|
| author | 1,338 | D-08 (missing author annotation) |
| title | 1,619 | D-03 (possible source for missing titles) |
| sectionheader | 1,599 | D-13 (no section boundaries) |
| content.text whitespace only | 10 | D-10 |

## Follow-up for D-03 / O4

- D-03 rules are recorded in D021 in [decisions.md](decisions.md). Non-target papers with empty or over-500-character titles are rejected; titles are not repaired or truncated. All 574 distinct query targets are protected.
- Empty full texts and nonblank full texts of exactly one character are rejected for non-target papers. Missing section headers and the absence of both first-name and last-name annotations are additional rejection criteria. Missing years become null values.
- Counts, including overlaps and protected exceptions, are generated in [d03_quality_filters.md](d03_quality_filters.md) and [d03_target_exceptions.md](d03_target_exceptions.md). D-02 measurements above still describe the original cache.
- No additional rejection rule has been selected for other full texts under 100 characters or for one-character titles/abstracts.
- O3: about 2.5 billion characters of full text, so a full-text subset is needed.
