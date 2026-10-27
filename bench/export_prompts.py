"""A-12: S1 benchmark prompts from the real retrieval pipeline (D058).

Usage:
    uv run --locked --group data --group rag --group embed python bench/export_prompts.py

Every LitSearch query (D-03 output in DATA_DIR/filtered/d03) gets the fixed S1 plan of the
planner fallback, so every model tag sees identical prompts and the planner is not part of
the comparison. Retrieval and build_context run as in the app, on the configured database.
Records hold `id` and `messages` for benchmark.py plus the split, the target papers and
the evidence exactly as the prompt shows it, for score_quality.py. They contain LitSearch
text: keep them in eval/private/.
"""

import argparse
import json
import random
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The query split and the arrow reader are shared with the D-20 and D-22 scripts.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compare_embeddings import read_table, split_queries
from select_fixture import data_directory

from scilitrag.config import load_settings
from scilitrag.contracts import Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.evidence.context import build_context
from scilitrag.rag.planner import DEFAULT_RESULT_LIMIT

SEED = 1789
OUTPUT = PROJECT_ROOT / "eval" / "private" / "bench" / "prompts.jsonl"


def litsearch_queries(data_dir: Path) -> list[dict]:
    rows = read_table(data_dir / "filtered" / "d03" / "query.arrow").to_pylist()
    split = split_queries([row["query_set"] for row in rows], seed=SEED)
    return [
        {
            "id": f"q{index:03d}",
            "query": row["query"],
            "query_set": row["query_set"],
            "split": part,
            "relevant": list(row["corpusids"]),
        }
        for index, (row, part) in enumerate(zip(rows, split, strict=True))
    ]


def prompt_record(query: dict, retrieve, settings) -> dict:
    plan = Plan(scenario="S1", query=query["query"], topic=query["query"], k=DEFAULT_RESULT_LIMIT)
    retrieved = retrieve(plan)
    packet = build_context(
        plan,
        retrieved,
        char_budget=settings.context_char_budget,
        item_limit=settings.evidence_item_char_limit,
    )
    # Scoring must see the evidence as the model saw it, after build_context truncation.
    shown = json.loads(packet.messages[-1]["content"])["evidence"]
    evidence = [
        {key: item[key] for key in ("label", "type", "corpus_id", "chunk_id", "text")}
        for item in shown
    ]
    relevant = set(query["relevant"])
    return {
        **query,
        "messages": packet.messages,
        "evidence": evidence,
        "context": {
            "retrieved": sum(item.type != "sql" for item in retrieved),
            "shown": len(shown),
            "omitted": list(packet.omitted_labels),
            "truncated": sum(item["excerpt_truncated"] for item in shown),
            "target_retrieved": any(item.corpus_id in relevant for item in retrieved),
            "target_shown": any(item["corpus_id"] in relevant for item in shown),
        },
    }


def context_stats(records: list[dict]) -> dict:
    """How much retrieved evidence the context budget dropped (D058, gold-paper denominator)."""
    contexts = [record["context"] for record in records]
    if not contexts:
        return {}
    return {
        "retrieved_median": statistics.median(c["retrieved"] for c in contexts),
        "shown_median": statistics.median(c["shown"] for c in contexts),
        "prompts_with_omitted": sum(bool(c["omitted"]) for c in contexts),
        "omitted_share": statistics.fmean(bool(c["omitted"]) for c in contexts),
        "prompts_with_truncated": sum(bool(c["truncated"]) for c in contexts),
        "target_retrieved": sum(c["target_retrieved"] for c in contexts),
        "target_shown": sum(c["target_shown"] for c in contexts),
        "target_dropped_by_budget": sum(
            c["target_retrieved"] and not c["target_shown"] for c in contexts
        ),
    }


def export(queries: list[dict], retrieve, settings) -> tuple[list[dict], dict[str, str]]:
    records, skipped = [], {}
    for query in queries:
        try:
            record = prompt_record(query, retrieve, settings)
        except RagUserError as error:
            skipped[query["id"]] = str(error)
            continue
        if not record["evidence"]:
            skipped[query["id"]] = "no evidence fits the context budget"
            continue
        records.append(record)
        print(f"{query['id']} {len(record['evidence'])} sources", file=sys.stderr)
    return records, skipped


def sample(records: list[dict], size: int, seed: int = SEED) -> list[dict]:
    chosen = set(random.Random(seed).sample([r["id"] for r in records], min(size, len(records))))
    return [record for record in records if record["id"] in chosen]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sample", type=int, default=30, help="extra file for repeats, 0: none")
    parser.add_argument("--limit", type=int, help="first N queries only, for trial runs")
    args = parser.parse_args()

    from scilitrag.rag.retrieval import retrieve_with
    from scilitrag.rag.search.backend import default_backend

    settings = load_settings()
    backend = default_backend(settings)
    queries = litsearch_queries(data_directory())[: args.limit]
    records, skipped = export(queries, lambda plan: retrieve_with(plan, backend), settings)

    write_jsonl(args.output, records)
    if args.sample:
        write_jsonl(
            args.output.with_stem(f"{args.output.stem}_{args.sample}"), sample(records, args.sample)
        )
    meta = {
        **asdict(backend.metadata),
        "seed": SEED,
        "k": DEFAULT_RESULT_LIMIT,
        "context_char_budget": settings.context_char_budget,
        "evidence_item_char_limit": settings.evidence_item_char_limit,
        "queries": len(queries),
        "exported": len(records),
        "skipped": skipped,
        "context": context_stats(records),
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(records)} of {len(queries)} queries exported, {len(skipped)} skipped")
    print(json.dumps(meta["context"], indent=2))


if __name__ == "__main__":
    main()
