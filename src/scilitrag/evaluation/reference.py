"""Export private S2 ground truth, unlabeled baseline passages and an independent S3 count."""

import argparse
import json
from collections import Counter, deque
from pathlib import Path

from scilitrag.config import ConfigError, load_settings
from scilitrag.contracts import Evidence, Plan, SqlFacts
from scilitrag.evaluation.datasets import MAX_DATABASE_ID, ReferenceSet, run_fingerprint
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.evidence.provenance import digest
from scilitrag.rag.scenarios.citation_path import baseline_candidate
from scilitrag.rag.search.database import DatabaseReader
from scilitrag.rag.search.entities import EntityLinker


def _run_scope(record):
    """Use the SQL stage's actual seed scope, never a second semantic search."""
    plan = Plan.model_validate(record["plan"])
    if plan.scenario != "S3" or record.get("dataset_changed") or not 1 <= plan.k <= 20:
        raise ValueError("Select an S3 run with a stable dataset and valid result limit")
    tables = [
        SqlFacts.model_validate(e["data"])
        for e in record["events"]
        if e["event"] == "table" and "citation_count" in e["data"].get("columns", [])
    ]
    if len(tables) != 1:
        raise ValueError("The source run must contain exactly one author-count table")
    table = tables[0]
    seeds = json.loads(table.parameters["seed_ids"])
    evidence = [Evidence.model_validate(item) for item in record["evidence"]]
    paper_ids = sorted({e.corpus_id for e in evidence if e.type == "paper"})
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= MAX_DATABASE_ID
            for i in seeds
        )
        or len(seeds) != len(set(seeds))
        or sorted(seeds) != paper_ids
    ):
        raise ValueError("Source SQL seeds must match the recorded paper evidence")
    if table.parameters.get("limit") != plan.k:
        raise ValueError("Source SQL limit must match the recorded plan")
    if not record.get("dataset_revision") or table.dataset_revision != record["dataset_revision"]:
        raise ValueError("Source SQL and run dataset revisions must match")
    query = record.get("query") or plan.query
    if not query or not record.get("id"):
        raise ValueError("The source run needs its original query and case ID")
    return plan, table, seeds, query


def _baseline_sample(groups, limit):
    """Round-robin targets and heuristic strata; retain one passage per citing paper.

    This is a bounded diagnostic sample, not an unbiased corpus prevalence estimate.
    Human labels stay empty, including for heuristic-positive candidates.
    """
    queues = []
    for rows in groups:
        positive, negative = deque(), deque()
        for row in rows:
            bucket = (
                positive
                if baseline_candidate(
                    row["text"], row.get("target_title") or "", row.get("section")
                )
                else negative
            )
            bucket.append(row)
        queues.extend([positive, negative])
    selected, seen = [], set()
    while any(queues) and len(selected) < limit:
        for queue in queues:
            while queue:
                row = queue.popleft()
                key = row["target_id"], row["corpus_id"]
                if key in seen:
                    continue
                seen.add(key)
                selected.append(
                    {**row, "source_hash": digest(row), "baseline_use": None, "reviewer_note": ""}
                )
                break
            if len(selected) == limit:
                break
    return selected


def build_reference_set(reader, targets, seed_ids, topic, *, source_run=None, baseline_limit=20):
    source = _run_scope(source_run) if source_run is not None else None
    if source:
        if seed_ids:
            raise ValueError("Use either source-run scope or explicit seed papers")
        seed_ids = source[2]
    if isinstance(baseline_limit, bool) or not 1 <= baseline_limit <= 100:
        raise ValueError("baseline_limit must be between 1 and 100")
    if not targets and not seed_ids:
        raise ValueError("Provide at least one target or seed paper")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_DATABASE_ID
        for value in targets + seed_ids
    ):
        raise ValueError("Paper IDs must be positive database IDs")
    targets = list(dict.fromkeys(targets))
    seed_ids = sorted(set(seed_ids))
    if len(seed_ids) > 20 or not topic.strip():
        raise ValueError("Use at most 20 seed papers and a nonempty topic")
    selected_ids = sorted(set(targets + seed_ids))
    known = reader(
        "SELECT corpus_id FROM papers WHERE corpus_id=ANY(%(ids)s)", {"ids": selected_ids}
    )
    if {row["corpus_id"] for row in known} != set(selected_ids):
        raise RagUserError(
            "Reference export contains unknown paper IDs. Check the selected corpus."
        )
    metadata = reader("SELECT dataset_revision FROM schema_metadata WHERE id=1")
    revision = metadata[0]["dataset_revision"] if metadata else None
    if not revision:
        raise RagUserError("Publish the frozen dataset revision before exporting references.")
    if source and source_run["dataset_revision"] != revision:
        raise RagUserError("Source run dataset revision does not match the current database")
    cases, passage_groups = [], []
    for target in targets:
        papers = reader(
            "SELECT p.corpus_id,p.year FROM papers p WHERE EXISTS (SELECT 1 FROM citations c "
            "WHERE c.cited_id=%(target)s AND c.citing_id=p.corpus_id) "
            "ORDER BY p.year DESC NULLS LAST,p.corpus_id LIMIT 20",
            {"target": target},
        )
        cases.append(
            {
                "id": f"s2-{target}",
                "scenario": "S2",
                "query": f"Find up to 20 papers that cite the paper with corpus ID {target}.",
                "expected_citing_ids": [p["corpus_id"] for p in papers],
                "expected_years": {str(p["corpus_id"]): p["year"] for p in papers},
            }
        )
        passages = reader(
            "SELECT c.chunk_id,c.corpus_id,c.section,c.text,target.title AS target_title "
            "FROM chunks c CROSS JOIN papers target WHERE target.corpus_id=%(target)s "
            "AND EXISTS (SELECT 1 FROM citations edge WHERE edge.citing_id=c.corpus_id "
            "AND edge.cited_id=%(target)s) "
            "ORDER BY (COALESCE(c.section,'') ~* 'experiment|evaluation|result|baseline') DESC, "
            "md5(c.chunk_id || ':' || target.corpus_id::text),c.chunk_id LIMIT 200",
            {"target": target},
        )
        passage_groups.append([{**p, "target_id": target} for p in passages])
    if seed_ids:
        # Independent reference: enumerate credits rather than reusing AUTHOR_SQL.
        credits = reader(
            "SELECT DISTINCT c.citing_id,c.cited_id,pa.author_id,a.name FROM citations c "
            "JOIN paper_authors pa ON pa.corpus_id=c.cited_id "
            "JOIN authors a ON a.author_id=pa.author_id "
            "WHERE c.citing_id=ANY(%(seeds)s) ORDER BY c.citing_id,c.cited_id,pa.author_id",
            {"seeds": seed_ids},
        )
        distinct = {(row["citing_id"], row["cited_id"], row["author_id"]) for row in credits}
        counts = Counter(author for _, _, author in distinct)
        names = {row["author_id"]: row["name"] for row in credits}
        limit = source[0].k if source else 20
        if source:
            author_id = source[1].parameters.get("author_id")
            expected_author = (
                EntityLinker(reader).author(source[0].author_ref) if source[0].author_ref else None
            )
            if author_id != expected_author:
                raise ValueError("Source SQL author filter must match the recorded plan")
            if author_id is not None:
                counts = Counter({key: value for key, value in counts.items() if key == author_id})
        order = sorted(
            counts, key=lambda author: (-counts[author], names[author].encode(), author)
        )[:limit]
        cases.append(
            {
                "id": source_run["id"] if source else "s3-topic",
                "scenario": "S3",
                "query": source[3]
                if source
                else f"Rank up to 20 most cited authors for the research topic {topic}.",
                "seed_ids": sorted(set(seed_ids)),
                "expected_author_counts": {str(author): counts[author] for author in order},
                **({"source_run_sha256": run_fingerprint(source_run)} if source else {}),
            }
        )
    bundle = {
        "dataset_revision": revision,
        "cases": cases,
        "baseline_annotations": _baseline_sample(passage_groups, baseline_limit),
    }
    ReferenceSet.model_validate(bundle)
    ending = reader("SELECT dataset_revision FROM schema_metadata WHERE id=1")
    if len(ending) != 1 or ending[0]["dataset_revision"] != revision:
        raise RagUserError(
            "Dataset revision changed during reference export; retry against a frozen corpus."
        )
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-paper", type=int, action="append", default=[])
    parser.add_argument("--seed-paper", type=int, action="append", default=[])
    parser.add_argument("--topic", default="retrieval")
    parser.add_argument("--from-run", type=Path, help="Use the exact S3 scope in a saved report")
    parser.add_argument("--case-id", help="Source S3 case ID (required with --from-run)")
    parser.add_argument("--model", help="Source model when the report contains multiple candidates")
    parser.add_argument(
        "--baseline-limit", type=int, default=20, help="Total sample size across targets"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve references and labels")
    if not args.target_paper and not args.seed_paper and not args.from_run:
        parser.error("Provide --target-paper, --seed-paper or --from-run")
    if bool(args.from_run) != bool(args.case_id) or args.model and not args.from_run:
        parser.error("--from-run requires --case-id; --model selects a source run")
    if len(set(args.seed_paper)) > 20 or any(
        not 1 <= i <= MAX_DATABASE_ID for i in args.seed_paper + args.target_paper
    ):
        parser.error("IDs must be positive; use at most 20 seed papers")
    try:
        source_run = None
        if args.from_run:
            report = json.loads(args.from_run.read_text(encoding="utf-8"))
            runs = [
                r
                for r in report["runs"]
                if r["id"] == args.case_id and (args.model is None or r.get("model") == args.model)
            ]
            if len(runs) != 1:
                raise ValueError("Select exactly one source S3 run using --case-id and --model")
            source_run = runs[0]
        result = build_reference_set(
            DatabaseReader(load_settings()),
            args.target_paper,
            args.seed_paper,
            args.topic,
            source_run=source_run,
            baseline_limit=args.baseline_limit,
        )
    except (ConfigError, RagUserError, ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {len(result['cases'])} cases and {len(result['baseline_annotations'])} unlabeled passages"
    )


if __name__ == "__main__":
    main()
