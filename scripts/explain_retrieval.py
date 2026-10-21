"""Capture actual dense and BM25 query plans with the application filters."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from scilitrag.contracts import Plan
from scilitrag.rag.search_backend import BM25_SQL, DENSE_SQL, default_backend


def explain_with(plan: Plan, backend) -> dict:
    params = backend.params(plan, 50)
    params["vector"] = backend.query_vector(params["topic"])
    # Only application-owned templates are executable; input remains bound values.
    return {
        name: backend.reader("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + statement, params)
        for name, statement in [("dense", DENSE_SQL), ("bm25", BM25_SQL)]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--year-to", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    backend = default_backend()
    plans = explain_with(
        Plan(scenario="S1", topic=args.topic, year_from=args.year_from, year_to=args.year_to, k=5),
        backend,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"database": asdict(backend.metadata), "plans": plans}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
