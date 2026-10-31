# D-23 Data statistics

LitSearch revision `9573fb284a1026c998df47024b888a163f0f0e25`, D-03 corpus. Counts from
`scripts/data_statistics.py` ([JSON](../eval/results/d23_data_statistics.json)) unless the
source column names another decision. The script reads only the local D-03 and D-12 files
and exports aggregate counts, no IDs, names or text:

```powershell
uv run --locked --group data --group rag python scripts/data_statistics.py
```

## Sizes

| | Count | Source |
|---|---:|---|
| LitSearch papers (`corpus_clean`, `corpus_s2orc`) | 64,183 | D-02 |
| Kept by D-03 | 56,115 | D-03 manifest |
| Removed by D-03 | 8,068 | D-03 manifest |
| Queries / distinct target papers (all kept) | 597 / 574 | D-03 manifest |
| Authors / paper-author links | 70,171 / 196,079 | script, equal to the database (D064) |
| Citation pairs | 328,622 | script, equal to the database (D064) |
| Papers selected for chunking / chunks | 1,000 / 48,860 | D-12 selection, D064 |

D-03 removes a paper for an empty title (6,753), no author names (1,720), no section heading
(1,599), a title over 500 characters (75) or an empty or one-character full text (12); a paper
can meet several rules. The 7 query targets with an empty title are kept.

Missing metadata in the kept corpus (equal to the INFO checks of `validate_database.py`, D055):

| | Papers |
|---|---:|
| without title | 7 |
| without abstract | 1,230 |
| without title and abstract | 5 |
| without year | 451 |

All papers without a title are kept query targets, so the 5 without title and abstract are
query targets too. Paper search over title and abstract has no text for them; like every
target they have full-text chunks (D048).

## Publication years

Kept corpus, share of all 56,115 papers. Years range from 1956 to 2024, median 2016;
74.4 % of the papers are from 2010 or later, 32.8 % from 2020 or later. The per-year counts
are in the JSON under `papers.by_year`.

| Year | Papers | Share |
|---|---:|---:|
| 1956 to 1989 | 1,203 | 2.1 % |
| 1990 to 1999 | 3,143 | 5.6 % |
| 2000 to 2009 | 9,583 | 17.1 % |
| 2010 | 1,976 | 3.5 % |
| 2011 | 1,335 | 2.4 % |
| 2012 | 2,230 | 4.0 % |
| 2013 | 1,830 | 3.3 % |
| 2014 | 2,409 | 4.3 % |
| 2015 | 1,897 | 3.4 % |
| 2016 | 2,806 | 5.0 % |
| 2017 | 2,462 | 4.4 % |
| 2018 | 3,524 | 6.3 % |
| 2019 | 2,843 | 5.1 % |
| 2020 | 3,566 | 6.4 % |
| 2021 | 4,465 | 8.0 % |
| 2022 | 6,215 | 11.1 % |
| 2023 | 3,726 | 6.6 % |
| 2024 | 451 | 0.8 % |
| no year | 451 | 0.8 % |

Each even year from 2004 to 2018 has more papers than the odd years before and after it.
2024 is probably a partial year, since LitSearch itself was published in 2024.

## Citation coverage

| | References | Share |
|---|---:|---:|
| Bibliography entries in S2ORC | 1,437,622 | 100 % |
| matched by S2ORC to any Semantic Scholar paper | 980,745 | 68.2 % |
| matched to another paper of the D-03 corpus | 342,467 | **23.8 %** |

23.8 % of all references point to another corpus paper; this is the citation coverage
(the plan estimated about 22 %). A paper has a median of 22 bibliography entries; 1,409
papers have none. Citation counts (S3) and citation paths (S2) therefore count only citations
within LitSearch, not a paper's real citation count.

The database stores distinct pairs, not references. 2,536 papers list the same corpus paper
more than once in their bibliography (possibly a preprint and its proceedings version; the
cases were not inspected). Without these 13,845 repeated entries, 328,622 distinct pairs
remain (22.9 % of the references, a ratio of pairs to entries). They are exactly the database
pairs: the pairs come from `corpus_clean.citations` (D032: 340,484 list entries, 11,767
dropped outside the corpus, 95 self-citations), and all 328,622 are in the citing paper's
S2ORC bibliography.

## Papers without links

| | Papers |
|---|---:|
| cite no corpus paper | 10,037 |
| not cited by a corpus paper | 23,117 |
| neither (no citation link) | 6,233 |
| mutual citation pairs | 964 |

## Authors

| | Count |
|---|---:|
| Author blocks in S2ORC | 227,528 |
| dropped: no first or last name | 18,530 |
| dropped: affiliation or header | 2,471 |
| dropped: same name twice in one paper | 10,448 |
| Paper-author links | 196,079 |
| Distinct authors (equal `normalize_name`, O2) | 70,171 |
| Papers without an author (kept, no query target) | 255 |

D-08 reported 74,949 authors and 196,083 links before the doubled-span fix (`b2c3bb0`):
S2ORC repeats some name spans, which produced names such as "Han Han Shi Shi" next to
"Han Shi". Joining each span once removed 4,778 authors and 4 links and slightly shifted the
dropped counts (2,463 and 10,452 before).

Known open variants. The loaded data keeps these as separate authors; the numbers are
authors that would disappear if the names were merged:

| Variant | Example | Extra authors | Status |
|---|---|---:|---|
| Accents only | "Adam Liska" / "Adam Liška" | 464 | open; would change `normalize_name` (O2, WP2) |
| Middle initial only | "Aaron Li" / "Aaron W Li" | 1,108 | open; merging may join different people |
| Either of both | | 1,575 | |
| Same name, different people | | unknown | needs author IDs |

## Full text

1,000 papers (1.8 % of the corpus) have full-text chunks (D048): all 574 query targets,
36 RAG papers, the S2 demo paper (DPR), 192 papers citing it and 197 random papers.
Full-text cleaning, measured on all 56,115 papers, keeps 70.5 % of the characters as body
text and removes 4.0 % front matter and 24.7 % bibliography (D045). E-mail masking replaced
1,871 expressions in titles, abstracts and full texts of all papers (D046).
