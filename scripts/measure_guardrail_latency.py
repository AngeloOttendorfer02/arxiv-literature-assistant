"""S-14: latency overhead of each guardrail stage, median and p95 (D071).

Replay (default): runs the 100 harmless questions of eval/guardrails through the real input
check, retrieval, evidence guard and sentence checks, with a fixed S1 plan and an answer replayed
from the retrieved sources instead of a language model. The answer arrives without delay, so
the `generate` stage of the log is the time of the sentence checks. Needs the database
(DATABASE_URL) and the models; no LLM.

    uv run --group rag --group guardrails --group embed python scripts/measure_guardrail_latency.py

From a log: reads the requests.jsonl of the real API run against a real LLM (for example the
Modal endpoint of deploy/modal) and reports the same stages. Per sentence times are not in that
log, so the sentence checks stay inside `generate`; the wait for the first sentence is reported.

    uv run python scripts/measure_guardrail_latency.py --from-log logs/requests.jsonl

Results go to eval/results/guardrail_latency_<mode>.json (timings only), aggregates to
docs/guardrail_latency_<mode>.md, with <mode> replay, live or log, so the modes do not overwrite
each other.
"""

import argparse
import asyncio
import json
import os
import platform
import re
import time
from dataclasses import replace
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from scilitrag.evaluation import latency as lat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "eval" / "results"
REPORT_DIR = PROJECT_ROOT / "docs"
REAL = "check_input,mask_pii,check_evidence,retrieve,sql_facts,check_sentence,render_citations"
MAX_SENTENCES = 6
FIRST_SENTENCE = re.compile(r"^(.{20,240}?[.!?])(?:\s|$)")


def questions() -> list[str]:
    from scilitrag.evaluation.guardrail_sets import load_input_cases

    return [c.text for c in load_input_cases() if c.category == "ok"]


def replay_answer(evidence) -> str:
    """One cited sentence per retrieved text source: its first sentence plus the label."""
    sentences = []
    for item in evidence:
        if item.type == "sql":
            continue
        text = item.text.split("Abstract:", 1)[-1].strip()
        match = FIRST_SENTENCE.match(text)
        if match:
            sentences.append(f"{match.group(1).rstrip('.!?')} [{item.label}].")
        if len(sentences) == MAX_SENTENCES:
            break
    return " ".join(sentences)


def timed(name, function):
    from scilitrag.timings import record_timing

    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            record_timing(f"{name}_ms", (time.perf_counter() - started) * 1000)
            if name == "check_sentence":
                record_timing("check_sentence_calls", 1)

    return wrapper


def build_components(settings, *, live: bool = False):
    from scilitrag.components import get_components
    from scilitrag.contracts import Plan

    components = get_components(settings, token_delay=0)
    if live:
        return replace(
            components,
            check_sentence=timed("check_sentence", components.check_sentence),
            mask_pii=timed("mask_pii", components.mask_pii),
            render_citations=timed("render_citations", components.render_citations),
        )

    def plan(masked_query: str) -> Plan:
        return Plan(scenario="S1", query=masked_query, topic=masked_query, k=5)

    async def generate(plan_, evidence):
        for word in replay_answer(evidence).split(" "):
            yield word + " "

    return replace(
        components,
        plan=plan,
        generate=generate,
        check_sentence=timed("check_sentence", components.check_sentence),
        mask_pii=timed("mask_pii", components.mask_pii),
        render_citations=timed("render_citations", components.render_citations),
    )


async def replay(settings, limit: int | None, live: bool = False) -> list[dict]:
    from scilitrag.api.pipeline import run_query
    from scilitrag.api.telemetry import RequestLog

    components = build_components(settings, live=live)
    records = []
    for query in questions()[:limit]:
        log = RequestLog(query_chars=len(query))
        message = None
        async for name, event in run_query(query, components, metrics=log.metrics):
            log.observe(event)
            if name == "error":
                message = event.message[:300]
        record = log.record()
        if message:
            record["error_message"] = message
            print(f"{record['status']}: {message}", flush=True)
        records.append(record)
    return records


def hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    return info


def ms(value) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def row(name: str, entry: dict) -> str:
    return f"| {name} | {entry['n']} | {ms(entry['median'])} | {ms(entry['p95'])} | {ms(entry['max'])} |"


def report(summary: dict, meta: dict) -> str:
    stages = summary["stages"]
    lines = [
        "# Guardrail latency (S-14)",
        "",
        (
            f"Mode: {meta['mode']}. {summary['requests']['total']} requests, "
            f"{summary['requests']['answered']} answered, {summary['requests']['refused']} "
            "refused. The first run of each stage is its cold start and not part of the "
            f"figures; {summary['requests']['in_per_request_figures']} requests enter the "
            f"per-request figures. Hardware: {meta['hardware']['processor'] or meta['hardware']['platform']}, "
            f"{meta['hardware']['cpu_count']} logical CPUs, models on CPU. Generated by "
            "`scripts/measure_guardrail_latency.py`; only timings are stored in "
            f"`eval/results/guardrail_latency_{meta['mode']}.json`."
        ),
        "",
        "Milliseconds per request, median and 95th percentile (nearest rank):",
        "",
        "| Stage | n | median | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[row(name, stages[name]) for name in lat.GUARD_STAGES],
        row("all guardrail stages and sentence checks", summary["guardrail_total_ms"]),
        "",
        "Other stages for scale (not guardrails):",
        "",
        "| Stage | n | median | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[row(name, stages[name]) for name in lat.OTHER_STAGES],
        "",
    ]
    checks = summary["sentence_checks"]
    if checks:
        lines += [
            "Sentence checks (`check_sentence`, `mask_pii`, `render_citations`):",
            "",
            "| Measure | n | median | p95 | max |",
            "| --- | ---: | ---: | ---: | ---: |",
            row("sentence checks per request", checks["sentence_checks_total"]),
            row("check_sentence per sentence", checks["check_sentence_per_call"]),
        ]
        if meta["mode"] == "live" and "generate_without_checks" in checks:
            lines.append(
                row("generate minus sentence checks (the model)", checks["generate_without_checks"])
            )
        lines.append("")
    lines += [
        "Wait for the first sentence caused by the checks (user TTFT minus engine TTFT):",
        "",
        "| Measure | n | median | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: |",
        row("wait for first sentence", summary["wait_for_first_sentence_ms"]),
        "",
    ]
    cold = summary["cold_start_ms"]
    if cold:
        parts = ", ".join(f"{name} {value:.0f}" for name, value in cold.items())
        lines += [f"Cold start, first run of each stage (ms): {parts}.", ""]
    sentences = summary["sentences"]
    lines += [
        f"Sentences checked {sentences['checked']}, removed {sentences['removed']}.",
        "",
        "## Notes",
        "",
    ]
    if meta["mode"] == "replay":
        lines += [
            (
                "- The answer is replayed from the retrieved sources and arrives without delay, "
                "so `generate` is the time of the sentence checks; a language model adds its own "
                "time on top (see the benchmark). The plan is fixed (S1), so `plan` is not the "
                "planner."
            ),
            "- The questions are the harmless questions of `eval/guardrails`, all run as S1.",
        ]
    elif meta["mode"] == "live":
        lines += [
            (
                f"- Live run with the language model `{meta['llm']}` of the configured endpoint "
                "(real planner, retrieval and generation). `generate` includes the model's time; "
                "the sentence checks run between its tokens, so `generate` minus the checks is "
                "about the model's time."
            ),
            "- The questions are the harmless questions of `eval/guardrails`.",
        ]
    else:
        lines += [
            (
                "- The log comes from a run with a language model: `generate` includes its time, "
                "and the wait for the first sentence is the visible cost of the checks."
            ),
        ]
    lines += [
        (
            "- Figures depend on the machine and on the number of retrieved sources and "
            "sentences. The models run on CPU in the API process, also when the language model "
            "runs on Modal."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-log", type=Path, help="Analyse a requests.jsonl of the real API.")
    parser.add_argument("--limit", type=int, help="Only the first N questions.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="All components real, with the language model of LLM_BASE_URL (needs the database).",
    )
    args = parser.parse_args()
    if args.from_log:
        records = [
            json.loads(line)
            for line in args.from_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        mode = f"log {args.from_log.name}"
    else:
        load_dotenv(find_dotenv(usecwd=True))
        from scilitrag.config import load_settings

        if args.live:
            settings = load_settings({**os.environ, "USE_MOCKS": "0"})
        else:
            settings = load_settings({**os.environ, "USE_MOCKS": "1", "REAL_COMPONENTS": REAL})
        records = asyncio.run(replay(settings, args.limit, live=args.live))
        mode = "live" if args.live else "replay"
    summary = lat.summarize(records)
    meta = {
        "mode": mode if mode in ("replay", "live") else "log",
        "source": mode,
        "hardware": hardware(),
        "llm": os.environ.get("LLM_MODEL") if mode == "live" else None,
        "llm_reasoning_effort": os.environ.get("LLM_REASONING_EFFORT") or None,
        "llm_max_output_tokens": os.environ.get("LLM_MAX_OUTPUT_TOKENS"),
    }
    mode_name = meta["mode"]
    result_path = RESULT_DIR / f"guardrail_latency_{mode_name}.json"
    report_path = REPORT_DIR / f"guardrail_latency_{mode_name}.md"
    result_path.write_text(
        json.dumps({"meta": meta, "summary": summary, "records": records}, indent=1) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(report(summary, meta), encoding="utf-8")
    print(
        json.dumps(
            {"requests": summary["requests"], "guardrail_total": summary["guardrail_total_ms"]}
        )
    )


if __name__ == "__main__":
    main()
