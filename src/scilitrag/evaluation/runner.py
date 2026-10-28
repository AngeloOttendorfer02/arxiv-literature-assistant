"""Run fixed retrieval cases through the application pipeline or re-score saved evidence."""

import argparse
import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path

from pydantic import ValidationError

from scilitrag.api.pipeline import run_query
from scilitrag.components import get_components
from scilitrag.config import ConfigError, load_settings
from scilitrag.contracts import Evidence, SqlFacts
from scilitrag.evaluation.datasets import load_reference_set
from scilitrag.evaluation.metrics import score_baselines, score_run, summarize_runs
from scilitrag.evaluation.review import export_reviews, import_reviews
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.scenarios.authors import attach_sql_evidence
from scilitrag.timings import RequestMetrics

CORE = {"plan", "retrieve", "sql_facts", "generate", "render_citations"}
GUARDRAILS = {"check_input", "check_evidence", "check_sentence", "mask_pii"}


def dataset_revision(settings):
    """Read provenance only for real database components; never probe services in mock mode."""
    if settings.use_mocks and not settings.real_components & {"retrieve", "sql_facts"}:
        return None
    from scilitrag.rag.search.database import DatabaseReader

    rows = DatabaseReader(settings)("SELECT dataset_revision FROM schema_metadata WHERE id=1")
    if len(rows) != 1 or not rows[0]["dataset_revision"]:
        raise RagUserError("Publish the frozen dataset revision before evaluating real retrieval.")
    return rows[0]["dataset_revision"]


async def run_case(case, components, settings, *, revision=None, expected_revision=None):
    record = {
        "id": case["id"],
        "model": settings.llm_model,
        "query": case["query"],
        "expected_scenario": case["scenario"],
        "dataset_revision": revision,
        "plan": {},
        "evidence": [],
    }

    def plan(query):
        result = components.plan(query)
        record["plan"] = result.model_dump()
        return result

    def retrieve(request_plan):
        sources = components.retrieve(request_plan)
        record["evidence"] = [item.model_dump() for item in sources]
        return sources

    def screen(evidence):
        result = components.check_evidence(evidence)
        # Score the actual sources admitted to SQL/generation, not dropped inputs.
        record["evidence"] = [item.model_dump() for item in result.evidence]
        record["removed_evidence_labels"] = result.removed
        return result

    selected = replace(components, plan=plan, retrieve=retrieve, check_evidence=screen)
    metrics = RequestMetrics()
    start = time.perf_counter()
    events = [
        {"event": name, "data": event.model_dump()}
        async for name, event in run_query(
            case["query"],
            selected,
            settings.request_timeout_s,
            max_buffer_chars=settings.max_sentence_buffer_chars,
            metrics=metrics,
        )
    ]
    record["elapsed_ms"] = (time.perf_counter() - start) * 1000
    record["component_ms"] = dict(metrics.timings_ms)
    sources = attach_sql_evidence(
        [Evidence.model_validate(item) for item in record["evidence"]],
        [SqlFacts.model_validate(item["data"]) for item in events if item["event"] == "table"],
    )
    record["evidence"] = [item.model_dump() for item in sources]
    record["events"] = events
    record["sentence_supported"] = [None for item in events if item["event"] == "sentence"]
    record["metrics"] = score_run(case, record, expected_revision=expected_revision)
    return record


def rescore_report(bundle: dict, report: dict) -> dict:
    """Reject invalid or mismatched files before calculating success or writing outputs."""
    by_id = {case["id"]: case for case in bundle["cases"]}
    if not isinstance(report, dict):
        raise TypeError("A saved report must be a JSON object")
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("A saved report must contain at least one run")
    candidates = report.get("candidate_models")
    if candidates is not None and (
        not isinstance(candidates, list)
        or not candidates
        or any(m is not None and (not isinstance(m, str) or not m.strip()) for m in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError("candidate_models must contain distinct model names")
    seen = set()
    for record in runs:
        if not isinstance(record, dict) or record.get("id") not in by_id:
            raise ValueError("Saved runs must use case IDs present in the selected reference set")
        if not isinstance(record.get("events"), list):
            raise TypeError("Every saved run must contain an events list")
        key = record.get("model"), record["id"]
        if key in seen:
            raise ValueError("Each candidate/case pair must occur once in a report")
        seen.add(key)
        if record.get("query") is not None and record["query"] != by_id[record["id"]]["query"]:
            raise ValueError("Saved query differs from the selected reference case")
        if report.get("candidate_models") and record.get("model") not in report["candidate_models"]:
            raise ValueError("Saved model is absent from candidate_models")
        record["metrics"] = score_run(
            by_id[record["id"]], record, expected_revision=bundle.get("dataset_revision")
        )
    report["baseline_filter_metrics"] = score_baselines(bundle.get("baseline_annotations", []))
    report["summary"], report["missing_cases"] = summarize_runs(bundle, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="JSON reference set with distinct case IDs")
    parser.add_argument("--models", nargs="+", help="Installed model names; defaults to LLM_MODEL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-mocks", action="store_true", help="Development checks only")
    parser.add_argument("--score-only", type=Path, help="Re-score a saved report without services")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Save a partial diagnostic report after the first failed/misrouted case",
    )
    parser.add_argument(
        "--export-review", type=Path, help="Export source-bound sentence/citation CSV"
    )
    parser.add_argument("--import-review", type=Path, help="Import reviewed CSV with --score-only")
    args = parser.parse_args(argv)
    if args.output.exists() or args.export_review and args.export_review.exists():
        parser.error("Output already exists; choose new paths to preserve runs and reviews")
    if args.import_review and not args.score_only:
        parser.error("--import-review requires --score-only")
    if args.export_review and args.output.resolve() == args.export_review.resolve():
        parser.error("Report and review outputs must use different paths")
    if args.models and (
        len(set(args.models)) != len(args.models) or any(not m.strip() for m in args.models)
    ):
        parser.error("Candidate model names must be distinct and nonblank")
    try:
        bundle = load_reference_set(args.cases)
        if args.score_only:
            report = rescore_report(
                bundle, json.loads(args.score_only.read_text(encoding="utf-8-sig"))
            )
            if args.import_review:
                report = rescore_report(bundle, import_reviews(report, args.import_review))
        else:
            settings = load_settings()
            missing = CORE - settings.real_components if settings.use_mocks else set()
            if missing and not args.allow_mocks:
                parser.error(
                    "Real retrieval evaluation requires components: " + ",".join(sorted(missing))
                )
            mocked_guards = GUARDRAILS - settings.real_components if settings.use_mocks else set()
            revision = dataset_revision(settings)
            expected = bundle.get("dataset_revision")
            if revision is not None and expected is not None and revision != expected:
                parser.error("Reference dataset revision does not match the configured database")
            report = {
                "candidate_models": args.models or [settings.llm_model],
                "development_only": bool(missing or mocked_guards),
                "use_mocks": settings.use_mocks,
                "mocked_guardrails": sorted(mocked_guards),
                "sentence_validator_mocked": "check_sentence" in mocked_guards,
                "real_components": sorted(settings.real_components),
                "dataset_revision": revision,
                "reference_dataset_revision": expected,
                "settings": {
                    "embed_model": settings.embed_model,
                    "request_timeout_s": settings.request_timeout_s,
                    "llm_max_output_tokens": settings.llm_max_output_tokens,
                    "context_char_budget": settings.context_char_budget,
                    "evidence_item_char_limit": settings.evidence_item_char_limit,
                    "max_sentence_buffer_chars": settings.max_sentence_buffer_chars,
                    "nli_model": settings.nli_model,
                    "nli_entailment_threshold": settings.nli_entailment_threshold,
                    "nli_contradiction_threshold": settings.nli_contradiction_threshold,
                    "evidence_guard_threshold": settings.evidence_guard_threshold,
                    "prompt_guard_threshold": settings.prompt_guard_threshold,
                    "topic_gate_margin": settings.topic_gate_margin,
                },
                "runs": [],
            }
            stop = False
            for model in report["candidate_models"]:
                selected_settings = replace(settings, llm_model=model)
                components = get_components(selected_settings, token_delay=0)
                for case in bundle["cases"]:
                    record = asyncio.run(
                        run_case(
                            case,
                            components,
                            selected_settings,
                            revision=revision,
                            expected_revision=expected,
                        )
                    )
                    report["runs"].append(record)
                    passed = record["metrics"]["completed"] and record["metrics"]["scenario_exact"]
                    print(
                        f"{model} / {case['id']} ({case['scenario']}): "
                        f"{'passed' if passed else 'failed'}, {record['elapsed_ms'] / 1000:.2f}s",
                        flush=True,
                    )
                    for event in record["events"]:
                        if event["event"] == "error":
                            print(
                                f"  {event['data']['code']}: {event['data']['message']}", flush=True
                            )
                    if not record["metrics"]["scenario_exact"] and record["plan"]:
                        print(
                            f"  Planned {record['plan']['scenario']}; expected {case['scenario']}",
                            flush=True,
                        )
                    if args.fail_fast and not passed:
                        stop = True
                        break
                if stop:
                    break
            ending_revision = dataset_revision(settings)
            report["ending_dataset_revision"] = ending_revision
            if ending_revision != revision:
                for record in report["runs"]:
                    record["dataset_changed"] = True
            report = rescore_report(bundle, report)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ValidationError,
        ConfigError,
        RagUserError,
    ) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(report['runs'])} runs to {args.output}")
    if args.export_review:
        try:
            count = export_reviews(report, args.export_review)
            print(f"Exported {count} review rows to {args.export_review}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"Report saved; review export failed: {error}")
    for row in report["summary"]:
        print(
            f"{row['model']} / {row['scenario']}: {row['passed']}/{row['expected_cases']} passed; "
            f"{row['semantic_citation_precision_reviewed']}/{row['sentences']} sentences reviewed"
        )
    return (
        0
        if not report["missing_cases"]
        and all(
            run["metrics"]["completed"]
            and run["metrics"]["scenario_exact"]
            and not run.get("dataset_changed", False)
            and run["metrics"]["dataset_revision_exact"] is not False
            for run in report["runs"]
        )
        else 1
    )
