"""S3: deterministic author rankings over the exact retrieved topic-paper set."""

import hashlib
import json
import logging

from scilitrag.contracts import Evidence, Plan, SqlFacts
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.search.entities import EntityLinker

logger = logging.getLogger(__name__)
COUNT_RULE = (
    "Each distinct (citing paper, cited paper) pair gives every author of the cited paper +1. "
    "Ties use alphabetical author name, then author ID. Counts cover only the selected topic "
    "papers and citation edges present in this dataset, not global citation counts. "
    "Year filters apply to citing topic papers; example papers are cited papers. "
    "Ranks use stored author IDs: name normalization can merge namesakes or split one "
    "person across IDs, and source annotations can contain extraction errors."
)
AUTHOR_SQL = """
WITH seeds AS (
    SELECT value::bigint AS corpus_id FROM jsonb_array_elements_text(%(seed_ids)s::jsonb)
), edges AS (
    SELECT DISTINCT c.citing_id,c.cited_id FROM citations c
    JOIN seeds s ON s.corpus_id=c.citing_id
), credits AS (
    SELECT e.citing_id,e.cited_id,pa.author_id FROM edges e
    JOIN paper_authors pa ON pa.corpus_id=e.cited_id
), ranked AS (
    SELECT a.author_id,a.name AS author_name,count(*) AS citation_count
    FROM credits cr JOIN authors a ON a.author_id=cr.author_id
    WHERE (%(author_id)s::bigint IS NULL OR a.author_id=%(author_id)s)
    GROUP BY a.author_id,a.name
)
SELECT r.author_id,r.author_name,r.citation_count,
    (SELECT string_agg(ex.example,'; ' ORDER BY ex.corpus_id) FROM (
        SELECT DISTINCT p.corpus_id,p.corpus_id::text || ': ' ||
            COALESCE(p.title,'Untitled paper') AS example
        FROM credits cr JOIN papers p ON p.corpus_id=cr.cited_id
        WHERE cr.author_id=r.author_id ORDER BY p.corpus_id LIMIT 3
    ) ex) AS example_papers
FROM ranked r ORDER BY r.citation_count DESC,r.author_name COLLATE "C",r.author_id
LIMIT %(limit)s
"""
COLUMNS = ["author_id", "author_name", "citation_count", "example_papers"]


def author_facts(plan: Plan, evidence: list[Evidence], reader, *, dataset_revision=None):
    """Consume existing retrieval output, avoiding a second search with a drifting scope."""
    if plan.scenario != "S3":
        return []
    if not 1 <= plan.k <= 20:
        raise RagUserError("Choose a result limit between 1 and 20.")
    ids = sorted({e.corpus_id for e in evidence if e.type == "paper" and e.corpus_id is not None})
    if not ids:
        raise RagUserError("No topic papers were found for an author ranking. Try fewer filters.")
    if len(ids) > 20:
        raise RagUserError("Author ranking requires at most 20 selected topic papers.")
    params = {
        "seed_ids": json.dumps(ids),
        "limit": plan.k,
        "author_id": EntityLinker(reader).author(plan.author_ref) if plan.author_ref else None,
    }
    rows = reader(AUTHOR_SQL, params)
    if not rows:
        raise RagUserError(
            "No author citation credits were found for the selected topic papers and author filter. "
            "Citation or author coverage may be incomplete."
        )
    if dataset_revision is None:
        metadata = reader("SELECT dataset_revision FROM schema_metadata WHERE id=1")
        dataset_revision = metadata[0]["dataset_revision"] if metadata else None
    identity = json.dumps([AUTHOR_SQL, params, dataset_revision], sort_keys=True)
    query_id = "s3_authors_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    scope = COUNT_RULE + f" Selected citing paper IDs: {', '.join(map(str, ids))}."
    logger.info("sql_facts query_id=%s seed_count=%d rows=%d", query_id, len(ids), len(rows))
    return [
        SqlFacts(
            query_id=query_id,
            sql=AUTHOR_SQL,
            parameters=params,
            scope=scope,
            dataset_revision=dataset_revision,
            columns=COLUMNS,
            rows=[[row[column] for column in COLUMNS] for row in rows],
        )
    ]


def attach_sql_evidence(evidence: list[Evidence], facts: list[SqlFacts]) -> list[Evidence]:
    """Append SQL snapshots without changing existing labels or duplicating mock SQL evidence."""
    result = list(evidence)
    next_label = max((item.label for item in result), default=0) + 1
    known = {item.query_id for item in result if item.type == "sql"}
    for table in facts:
        if table.query_id in known:
            continue
        result.append(
            Evidence(
                label=next_label,
                type="sql",
                query_id=table.query_id,
                title="Author citation ranking",
                section="SQL aggregation",
                text=json.dumps(table.model_dump(), ensure_ascii=False),
            )
        )
        next_label += 1
        known.add(table.query_id)
    return result
