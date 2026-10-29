"""D-18: compare database dense, BM25 and hybrid paper retrieval on LitSearch.

Usage: uv run --locked --group data --group rag --group embed python scripts/compare_retrieval.py

Reuses D-22's pinned queries, stratified split (seed 42) and binary paper metrics.
Uses the real search backend, independently of USE_MOCKS, without a planner or LLM.
Only scores and query metadata are exported, never paper IDs or dataset text.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from compare_embeddings import (
    BOOTSTRAP_DRAWS,
    METRICS,
    PROJECT_ROOT,
    REVISION,
    SEED,
    TOP_N,
    mean_metrics,
    paired_bootstrap,
    query_metrics,
    read_table,
    split_queries,
)

from scilitrag.config import ConfigError, load_settings
from scilitrag.contracts import Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.search.backend import default_backend
from scilitrag.rag.search.ranking import reciprocal_rank_fusion

METHODS = ("dense", "bm25", "hybrid")
QUERY_COUNT = 597
RRF_K = 60
COMPARISON_METRICS = ("recall@20", "mrr", "ndcg@10")
OUTPUT = PROJECT_ROOT / "eval" / "results" / "d18_retrieval_comparison.json"
REPORT = PROJECT_ROOT / "docs" / "d18_retrieval_comparison.md"


def load_queries(data_dir: Path) -> tuple[list[dict], dict]:
    folder = data_dir / "filtered" / "d03"
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dataset") != "princeton-nlp/LitSearch"
        or manifest.get("dataset_revision") != REVISION
    ):
        raise ValueError("D-03 must use the pinned LitSearch revision from D-22.")
    query_path = folder / "query.arrow"
    rows = read_table(query_path).to_pylist()
    if len(rows) != QUERY_COUNT:
        raise ValueError(f"Expected all {QUERY_COUNT} LitSearch queries, got {len(rows)}.")
    for index, row in enumerate(rows):
        if (
            not isinstance(row.get("query"), str)
            or not row["query"].strip()
            or not isinstance(row.get("query_set"), str)
            or not row["query_set"].strip()
            or not isinstance(row.get("corpusids"), list)
            or not row["corpusids"]
            or any(type(cid) is not int or cid < 1 for cid in row["corpusids"])
        ):
            raise ValueError(f"Invalid query text, query_set or targets at row {index}.")
    split = split_queries([row["query_set"] for row in rows], seed=SEED)
    queries = [
        {
            "id": f"q{index:03d}",
            "query": row["query"],
            "query_set": row["query_set"],
            "split": part,
            "relevant": sorted(set(row["corpusids"])),
        }
        for index, (row, part) in enumerate(zip(rows, split, strict=True))
    ]
    with query_path.open("rb") as stream:
        query_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return queries, {"manifest": manifest, "query_sha256": query_sha256}


def validate_corpus(backend, queries: list[dict], manifest: dict) -> int:
    """Reject fixtures, missing targets/vectors and corpora from another revision."""
    if backend.metadata.dataset_revision != REVISION:
        raise ValueError("The database dataset revision differs from D-22.")
    expected = manifest.get("retained_corpusids")
    if (
        not isinstance(expected, list)
        or not expected
        or any(type(cid) is not int or cid < 1 for cid in expected)
        or len(expected) != len(set(expected))
    ):
        raise ValueError("D-03 manifest needs unique retained paper IDs.")
    rows = backend.reader("SELECT corpus_id, embedding IS NOT NULL AS embedded FROM papers")
    actual = {row["corpus_id"] for row in rows}
    if actual != set(expected):
        raise ValueError("Database paper IDs differ from D-03; use the full corpus, not a fixture.")
    if any(not row["embedded"] for row in rows):
        raise ValueError("All database papers need embeddings before evaluation.")
    targets = {cid for query in queries for cid in query["relevant"]}
    if not targets <= actual:
        raise ValueError("Some LitSearch target papers are missing from the corpus.")
    return len(actual)


def evaluate(queries: list[dict], backend) -> dict:
    records = []
    for index, query in enumerate(queries):
        plan = Plan(scenario="S1", query=query["query"], topic=query["query"], k=20)
        # At k=20 the application's hybrid also fuses 100 candidates per method.
        # Extending only the fused output to 100 preserves its top-20 ranking.
        candidates = {}
        for method in ("dense", "bm25"):
            try:
                candidates[method] = getattr(backend, method)(plan, limit=TOP_N)
            except RagUserError as error:
                raise RagUserError(
                    f"Evaluation failed at {query['id']} ({method}): {error}"
                ) from None
        hybrid = reciprocal_rank_fusion(
            candidates["dense"], candidates["bm25"], limit=TOP_N, k=RRF_K
        )
        rankings = {
            name: [row["corpus_id"] for row in rows]
            for name, rows in {**candidates, "hybrid": hybrid}.items()
        }
        records.append(
            {
                "id": query["id"],
                "query_set": query["query_set"],
                "split": query["split"],
                "metrics": {
                    name: query_metrics(ranked, query["relevant"])
                    for name, ranked in rankings.items()
                },
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(queries):
            print(f"Evaluated {index + 1}/{len(queries)} queries", file=sys.stderr)
    return summarize(records)


def summarize(records: list[dict]) -> dict:
    """Aggregate the same records for live evaluation and offline validation."""
    groups = {
        "dev": [i for i, row in enumerate(records) if row["split"] == "dev"],
        "test": [i for i, row in enumerate(records) if row["split"] == "test"],
        "all": list(range(len(records))),
    }
    return {
        "queries": len(records),
        "split_counts": {name: len(indices) for name, indices in groups.items()},
        "query_sets": {
            part: dict(Counter(row["query_set"] for row in records if row["split"] == part))
            for part in ("dev", "test")
        },
        "scores": {
            name: {
                group: mean_metrics([row["metrics"][name] for row in records], indices)
                if indices
                else None
                for group, indices in groups.items()
            }
            for name in METHODS
        },
        "per_query": records,
    }


def validate_saved_result(result: dict) -> None:
    """Reject inconsistent evidence before offline statistics can overwrite files."""
    if not isinstance(result, dict):
        raise TypeError("Saved results must be a JSON object.")
    rows, count, smoke = result.get("per_query"), result.get("queries"), result.get("smoke_run")
    if (
        type(smoke) is not bool
        or type(count) is not int
        or not isinstance(rows, list)
        or len(rows) != count
        or not 1 <= count <= QUERY_COUNT
        or (not smoke and count != QUERY_COUNT)
        or (smoke and count == QUERY_COUNT)
    ):
        raise ValueError("Query counts must match the records; a full run needs 597 queries.")
    if result.get("split_seed") != SEED:
        raise ValueError("Saved results must use D-22's split seed 42.")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "query_set", "split", "metrics"}
            or row["id"] != f"q{index:03d}"
            or row["split"] not in ("dev", "test")
            or not isinstance(row["query_set"], str)
            or not row["query_set"].strip()
        ):
            raise ValueError(f"Invalid query ID, order, query_set or split at row {index}.")
        metrics = row["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != set(METHODS):
            raise ValueError(f"Missing search methods at row {index}.")
        for values in metrics.values():
            if (
                not isinstance(values, dict)
                or set(values) != set(METRICS)
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                    for v in values.values()
                )
            ):
                raise ValueError(
                    f"Metrics must be finite numbers between zero and one at row {index}."
                )
    if not smoke and [row["split"] for row in rows] != split_queries(
        [row["query_set"] for row in rows], seed=SEED
    ):
        raise ValueError("Query splits do not match D-22's deterministic assignment.")
    expected = summarize(rows)
    if any(result.get(key) != expected[key] for key in ("split_counts", "query_sets")):
        raise ValueError("Saved split/query_set counts do not match the query records.")
    if not isinstance(result.get("scores"), dict) or set(result["scores"]) != set(METHODS):
        raise ValueError("Saved averages must contain all three search methods.")
    for method in METHODS:
        groups = result["scores"][method]
        if not isinstance(groups, dict) or set(groups) != {"dev", "test", "all"}:
            raise ValueError(f"Saved averages for {method} must contain dev, test and all.")
        for part, scores in expected["scores"][method].items():
            actual = groups[part]
            if scores is None:
                valid = actual is None
            else:
                valid = (
                    isinstance(actual, dict)
                    and set(actual) == set(METRICS)
                    and all(
                        type(actual[m]) in (int, float)
                        and math.isfinite(actual[m])
                        and math.isclose(actual[m], scores[m], rel_tol=0, abs_tol=1e-12)
                        for m in METRICS
                    )
                )
            if not valid:
                raise ValueError(
                    f"Saved averages for {method}/{part} do not match per-query metrics."
                )


def add_bootstrap(result: dict) -> None:
    """Resample matching query pairs within each split, using D-22's procedure."""
    if result["smoke_run"]:
        result["bootstrap"] = None
        return
    intervals = {}
    for part in ("dev", "test"):
        rows = [row for row in result["per_query"] if row["split"] == part]
        if not rows:
            raise ValueError("A full bootstrap comparison needs both dev and test queries.")
        intervals[part] = {
            metric: paired_bootstrap(
                [row["metrics"]["hybrid"][metric] for row in rows],
                [row["metrics"]["dense"][metric] for row in rows],
                draws=BOOTSTRAP_DRAWS,
                seed=SEED,
            )
            for metric in COMPARISON_METRICS
        }
    result["bootstrap"] = {
        "comparison": "hybrid minus dense",
        "method": "paired percentile bootstrap over queries within each split",
        "draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "confidence_level": 0.95,
        "intervals": intervals,
    }


def bootstrap_markdown(result: dict) -> list[str]:
    bootstrap = result.get("bootstrap")
    if bootstrap is None:
        return ["Bootstrap intervals are not reported for this run.", ""]
    intervals = bootstrap["intervals"]
    lines = [
        (
            "Hybrid minus dense: paired percentile bootstrap over queries, separately on "
            f"Dev and Test; {bootstrap['draws']:,} resamples, seed {bootstrap['seed']}. "
            "Intervals are pointwise 95% intervals; all table values are in percentage points."
        ),
        "",
        "| Split | Metric | Difference | 95% interval |",
        "|---|---|---:|---:|",
    ]
    for part, metrics in intervals.items():
        for metric, estimate in metrics.items():
            label = {"recall@20": "Recall@20", "mrr": "MRR@100", "ndcg@10": "NDCG@10"}[metric]
            lines.append(
                f"| {part} | {label} | {100 * estimate['delta']:+.2f} | "
                f"[{100 * estimate['ci_low']:+.2f}, {100 * estimate['ci_high']:+.2f}] |"
            )
    if all(
        intervals[part]["recall@20"]["delta"] > 0
        and all(intervals[part][metric]["delta"] < 0 for metric in ("mrr", "ndcg@10"))
        for part in ("dev", "test")
    ):
        lines += [
            "",
            (
                "Hybrid has higher observed Recall@20, but lower MRR@100 and NDCG@10 on both "
                "splits: broader coverage within 20 results, with worse observed early ranking."
            ),
        ]
    if all(
        estimate["ci_low"] <= 0 <= estimate["ci_high"]
        for metrics in intervals.values()
        for estimate in metrics.values()
    ):
        lines += [
            "",
            (
                "All six intervals include zero. This evaluation does not establish a "
                "statistically clear advantage for either method on these comparisons; "
                "it does not establish equivalence."
            ),
        ]
    lines += [
        "",
        (
            "RRF k=60 and the existing weights remain unchanged. The team should agree on a "
            "primary metric before any tuning, use Dev only for that tuning, then freeze "
            "settings for the final Test evaluation."
        ),
        "",
    ]
    return lines


def report_markdown(result: dict) -> str:
    lines = [
        "# D-18 Database retrieval comparison",
        "",
        f"Status: {'smoke run; not the full benchmark' if result['smoke_run'] else 'full run'}.",
        (
            f"LitSearch revision `{REVISION}`, D-03 corpus: {result['documents']:,} papers, "
            f"{result['queries']} queries. Paper-level binary relevance; title + abstract."
        ),
        f"Embedding model: `{result['database']['embed_model']}`.",
        (
            f"Offline SQL timeout: {result['settings']['statement_timeout_s']:g} seconds "
            "per statement. The API's default 10-second cap is unchanged; this is a "
            "retrieval-quality benchmark, not a response-time or availability test."
        ),
        (
            "Dev/test: D-22's 50/50 split stratified by `query_set`, seed 42. "
            "The split is assigned on all 597 queries before any smoke-run limit."
        ),
        (
            "Dense and BM25 use the application's SQL and read-only database connections. "
            "BM25 title boost is 0.5; abstract boost is the default. Hybrid uses RRF k=60 "
            "on 100 candidates from each method. All returned rankings are capped at 100."
        ),
        (
            "Hybrid top-20 matches the application's candidate budget at k=20; only this "
            "evaluation extends its output to 100 for MRR@100. No planner, LLM or chunk search."
        ),
        "",
    ]
    for group in ("dev", "test", "all"):
        lines += [
            f"## {group.capitalize()} ({result['split_counts'][group]} queries)",
            "",
            "| Method | Recall@5 | Recall@20 | MRR@100 | NDCG@10 |",
            "|---|---:|---:|---:|---:|",
        ]
        for method in METHODS:
            scores = result["scores"][method][group]
            values = " | ".join(f"{scores[m]:.3f}" if scores else "n/a" for m in METRICS)
            lines.append(f"| {method} | {values} |")
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        *bootstrap_markdown(result),
        (
            "Scores are macro-averaged over queries; no failures or empty rankings are dropped. "
            "The JSON key `mrr` follows D-22 and means MRR@100 here. Settings are fixed for this "
            "run, not tuned on test. This split does not undo any earlier use of test queries "
            "to choose settings. PostgreSQL dense retrieval is not D-22's exact cosine search."
        ),
        (
            "NDCG@10 alone does not make these scores directly comparable with MTEB: this run "
            "uses the filtered D-03 corpus and the local database retrieval configuration."
        ),
        "",
    ]
    return "\n".join(lines)


def write_outputs(result: dict, output: Path, report: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report.write_text(report_markdown(result), encoding="utf-8")
    print(f"Results: {output}\nReport: {report}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="JSON scores and query metadata")
    parser.add_argument("--report", type=Path, help="Markdown comparison table")
    parser.add_argument("--limit", type=int, help="First N queries for a smoke run (1..597)")
    parser.add_argument(
        "--score-only",
        type=Path,
        help="Recompute bootstrap intervals and report from saved JSON; no database or model",
    )
    parser.add_argument(
        "--statement-timeout",
        type=float,
        default=60.0,
        help="Offline SQL timeout in seconds per statement (default: 60; API unchanged)",
    )
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= QUERY_COUNT:
        parser.error("--limit must be between 1 and 597.")
    if not math.isfinite(args.statement_timeout) or args.statement_timeout <= 0:
        parser.error("--statement-timeout must be positive and finite.")
    if args.score_only:
        if args.limit is not None:
            parser.error("--limit cannot be combined with --score-only.")
        try:
            result = json.loads(args.score_only.read_text(encoding="utf-8"))
            validate_saved_result(result)
            add_bootstrap(result)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.exit(1, f"Cannot re-score saved results: {error}\n")
        report = args.score_only.with_suffix(".md") if result["smoke_run"] else REPORT
        write_outputs(result, args.output or args.score_only, args.report or report)
        return 0
    try:
        settings = load_settings()
        if not settings.data_dir:
            raise ValueError("Set DATA_DIR in .env or the environment.")
        data_dir = Path(settings.data_dir).expanduser()
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        queries, provenance = load_queries(data_dir)
        backend = default_backend(settings, statement_timeout_s=args.statement_timeout)
        documents = validate_corpus(backend, queries, provenance["manifest"])
        print(
            f"Preflight passed: {documents:,} papers; starting real embedding and search.",
            file=sys.stderr,
        )
        result = evaluate(queries[: args.limit], backend)
        result.update(
            {
                "dataset_revision": REVISION,
                "corpus": "d03",
                "documents": documents,
                "database": asdict(backend.metadata),
                "query_sha256": provenance["query_sha256"],
                "split_seed": SEED,
                "smoke_run": args.limit is not None and args.limit < QUERY_COUNT,
                "settings": {
                    "candidates_per_method": TOP_N,
                    "rrf_k": RRF_K,
                    "title_boost": 0.5,
                    "statement_timeout_s": args.statement_timeout,
                },
                "metric_cutoffs": {"recall@5": 5, "recall@20": 20, "mrr": 100, "ndcg@10": 10},
            }
        )
        add_bootstrap(result)
    except (ConfigError, RagUserError, ValueError, OSError) as error:
        parser.exit(1, f"Retrieval comparison failed: {error}\n")
    smoke = PROJECT_ROOT / "eval" / "private" / "d18_retrieval_smoke"
    output = args.output or (smoke.with_suffix(".json") if result["smoke_run"] else OUTPUT)
    report = args.report or (smoke.with_suffix(".md") if result["smoke_run"] else REPORT)
    write_outputs(result, output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
