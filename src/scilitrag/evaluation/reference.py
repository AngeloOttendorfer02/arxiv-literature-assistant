"""Export private S2 ground truth, unlabeled baseline passages and an independent S3 count."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scilitrag.config import ConfigError, load_settings
from scilitrag.evaluation.datasets import MAX_DATABASE_ID, ReferenceSet
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.search.database import DatabaseReader


def build_reference_set(reader, targets, seed_ids, topic):
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
    cases, annotations = [], []
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
            "AND edge.cited_id=%(target)s) ORDER BY c.corpus_id,c.seq_no LIMIT 20",
            {"target": target},
        )
        annotations.extend(
            {**p, "target_id": target, "baseline_use": None, "reviewer_note": ""} for p in passages
        )
    if seed_ids:
        # Independent reference: enumerate credits rather than reusing AUTHOR_SQL.
        credits = reader(
            "SELECT DISTINCT c.citing_id,c.cited_id,pa.author_id,a.name FROM citations c "
            "JOIN paper_authors pa ON pa.corpus_id=c.cited_id "
            "JOIN authors a ON a.author_id=pa.author_id "
            "WHERE c.citing_id=ANY(%(seeds)s) ORDER BY c.citing_id,c.cited_id,pa.author_id",
            {"seeds": seed_ids},
        )
        counts = Counter(row["author_id"] for row in credits)
        names = {row["author_id"]: row["name"] for row in credits}
        order = sorted(
            counts, key=lambda author: (-counts[author], names[author].encode(), author)
        )[:20]
        cases.append(
            {
                "id": "s3-topic",
                "scenario": "S3",
                "query": f"Rank up to 20 most cited authors for the research topic {topic}.",
                "seed_ids": sorted(set(seed_ids)),
                "expected_author_counts": {str(author): counts[author] for author in order},
            }
        )
    bundle = {"dataset_revision": revision, "cases": cases, "baseline_annotations": annotations}
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.target_paper and not args.seed_paper:
        parser.error("Provide at least one --target-paper or --seed-paper")
    if len(set(args.seed_paper)) > 20 or any(
        not 1 <= i <= MAX_DATABASE_ID for i in args.seed_paper + args.target_paper
    ):
        parser.error("IDs must be positive; use at most 20 seed papers")
    try:
        result = build_reference_set(
            DatabaseReader(load_settings()), args.target_paper, args.seed_paper, args.topic
        )
    except (ConfigError, RagUserError, ValueError) as error:
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
