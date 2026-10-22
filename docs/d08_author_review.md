# D-08 manual author review

Review date: 2026-10-22.

## Scope and result

A SQL query selected 30 imported paper-author pairs from the DB. The reviewer
then manually compared these pairs with the corresponding LitSearch source
records and reported the following results:

| Verdict | Count |
|---|---:|
| Correct | 29 |
| Incorrect | 1 |
| Uncertain | 0 |
| Total checked | 30 |

The dataset revision used by the local fixture is
`9573fb284a1026c998df47024b888a163f0f0e25`. These results are the reviewer's
reported assessment, not an automated verification of all 30 cases.

This report covers the SQL-selected sample from the current DB and its manual review.
The older generated worksheet is not used as evidence for the current DB
contents. The sample does not establish a full-corpus error rate.

## SQL sample selection

The review workflow selected paper-author pairs with this read-only query:

```sql
SELECT
    pa.corpus_id,
    a.author_id,
    a.first_name,
    a.last_name,
    a.name,
    a.normalized_name
FROM paper_authors pa
JOIN authors a ON a.author_id = pa.author_id
ORDER BY md5(pa.corpus_id::text || ':' || pa.author_id::text)
LIMIT 30;
```

Selection was performed by SQL; the subsequent source comparison was manual.
The sample contains 30 pairs, not necessarily 30 distinct papers or authors.
The ordering is repeatable for unchanged rows and IDs. Rebuilding or changing
the DB can change the selection, so retain the selected pairs in local notes.

## Incorrect case in the 30-sample review: fragmented name

| Field | Observation |
|---|---|
| Stored first name | `T N` |
| Stored last name | `N Gu Y En` |
| Verdict | Incorrect name extraction or first-name/surname assignment |
| Corpus ID | 15650224
| Author ID | 875

The source text displays a fragmented name. The stored name parts do not
provide a reliable first-name/surname assignment. The intended full spelling
has not been established here; no replacement name is inferred. O2
normalization changes the search representation but cannot reconstruct a
name from damaged source annotations.

## Author identity and relation limitations

O2 treats equal normalized full names as one author during ingestion. This
is a matching assumption, not evidence of a person's identity:

- Different people sharing a normalized name can be combined into one author
  row, with all their papers linked to that row.
- Different spellings of the same person's name can produce different keys
  and separate author rows. Normalization does not resolve initials or
  transliterate names into a verified common identity.
- Incorrect first-name/surname annotations can create malformed author rows.
  Excluding incomplete names or institution text can also remove real authors
  when the source parser mixes a person with an affiliation.

Primary and foreign keys keep the stored paper-author links structurally
consistent. They do not establish that each author row represents exactly one
real person. Author-level citation counts can therefore combine different
people or split one person's works across multiple IDs. The 29/1 sample result
does not validate global author identity or completeness of the relation.

## Coverage of author questions in LitSearch

On 2026-10-22, all 597 query texts were inspected in
`DATA_DIR/filtered/d03/query.arrow`, using full-text review supplemented by
keyword screening for author identification and citation queries. All 597
rows were also compared with the original local query cache and were equal.
The revision is the pinned revision stated above.

| Question type | Queries identified |
|---|---:|
| Find papers by a named author | 0 |
| Return the authors of a paper | 0 |
| Compare or rank authors by citation counts | 0 |

The keyword screen alone cannot establish absence: alternative wording can
escape its patterns. The findings above include review of every query text,
not just keyword matches. They apply to this dataset revision.

LitSearch evaluates literature retrieval using relevant paper IDs, as described
in the [dataset documentation](https://huggingface.co/datasets/princeton-nlp/LitSearch).
Its queries do not validate this project's author disambiguation or author-level
citation statistics. The project's author-related S3 scenario still requires
separate test cases for paper-author links, homonyms and citation aggregation.


