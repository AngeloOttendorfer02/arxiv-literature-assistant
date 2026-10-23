"""Export private S2 ground truth, unlabeled baseline passages and an independent S3 count."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scilitrag.config import load_settings
from scilitrag.rag.search.database import DatabaseReader


def build_goldens(reader, targets, seed_ids, topic):
    metadata = reader("SELECT dataset_revision FROM schema_metadata WHERE id=1")
    revision = metadata[0]["dataset_revision"] if metadata else None
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
    return {"dataset_revision": revision, "cases": cases, "baseline_annotations": annotations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-paper", type=int, action="append", default=[])
    parser.add_argument("--seed-paper", type=int, action="append", default=[])
    parser.add_argument("--topic", default="retrieval")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.target_paper and not args.seed_paper:
        parser.error("Provide at least one --target-paper or --seed-paper")
    if len(set(args.seed_paper)) > 20 or any(i < 1 for i in args.seed_paper + args.target_paper):
        parser.error("IDs must be positive; use at most 20 seed papers")
    result = build_goldens(
        DatabaseReader(load_settings()), args.target_paper, args.seed_paper, args.topic
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {len(result['cases'])} cases and {len(result['baseline_annotations'])} unlabeled passages"
    )


if __name__ == "__main__":
    main()
