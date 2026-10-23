"""Run the same WP2 cases through the API pipeline for explicitly selected models."""

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from scilitrag.api.pipeline import run_query
from scilitrag.components import get_components
from scilitrag.config import load_settings
from scilitrag.rag.evaluation import score_baselines, score_run

CORE = {"plan", "retrieve", "sql_facts", "generate", "render_citations"}


async def run_case(case, components, settings):
    record = {"id": case["id"], "model": settings.llm_model, "plan": {}, "evidence": []}

    def plan(query):
        result = components.plan(query)
        record["plan"] = result.model_dump()
        return result

    def retrieve(request_plan):
        sources = components.retrieve(request_plan)
        record["evidence"] = [item.model_dump() for item in sources]
        return sources

    selected = replace(components, plan=plan, retrieve=retrieve)
    events = [
        {"event": name, "data": event.model_dump()}
        async for name, event in run_query(
            case["query"],
            selected,
            settings.request_timeout_s,
            max_buffer_chars=settings.max_sentence_buffer_chars,
        )
    ]
    # SQL evidence is appended inside the pipeline; reconstruct its snapshots
    # from the exact tables for offline structural checks.
    from scilitrag.contracts import Evidence, SqlFacts
    from scilitrag.rag.scenarios.authors import attach_sql_evidence

    sources = attach_sql_evidence(
        [Evidence.model_validate(item) for item in record["evidence"]],
        [SqlFacts.model_validate(item["data"]) for item in events if item["event"] == "table"],
    )
    record["evidence"] = [item.model_dump() for item in sources]
    record["events"] = events
    record["sentence_supported"] = [None for item in events if item["event"] == "sentence"]
    record["metrics"] = score_run(case, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="Private JSON file containing a cases array")
    parser.add_argument("--models", nargs="+", help="Explicit LLM_MODEL values; defaults to config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-mocks", action="store_true", help="Development checks only")
    parser.add_argument(
        "--score-only", type=Path, help="Re-score saved runs after human annotation"
    )
    args = parser.parse_args()
    bundle = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = bundle["cases"]
    by_id = {case["id"]: case for case in cases}
    if not cases or len(by_id) != len(cases):
        parser.error("Provide nonempty cases with distinct IDs")
    if args.score_only:
        report = json.loads(args.score_only.read_text(encoding="utf-8"))
        for record in report["runs"]:
            record["metrics"] = score_run(by_id[record["id"]], record)
    else:
        settings = load_settings()
        missing = CORE - settings.real_components if settings.use_mocks else set()
        if missing and not args.allow_mocks:
            parser.error("Real WP2 evaluation requires components: " + ",".join(sorted(missing)))
        report = {
            "development_only": bool(missing),
            "use_mocks": settings.use_mocks,
            "sentence_validator_mocked": settings.use_mocks
            and "check_sentence" not in settings.real_components,
            "real_components": sorted(settings.real_components),
            "runs": [],
        }
        for model in args.models or [settings.llm_model]:
            selected_settings = replace(settings, llm_model=model)
            components = get_components(selected_settings, token_delay=0)
            for case in cases:
                record = asyncio.run(run_case(case, components, selected_settings))
                report["runs"].append(record)
    report["baseline_filter_metrics"] = score_baselines(bundle.get("baseline_annotations", []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(report['runs'])} runs to {args.output}")
    return 0 if all(run["metrics"]["completed"] for run in report["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
