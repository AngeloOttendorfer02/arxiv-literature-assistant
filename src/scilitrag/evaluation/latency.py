"""Latency of the guardrail stages from request log records (S-14)."""

import math

INPUT_STAGES = ("prompt_guard", "topic_gate", "pii")
GUARD_STAGES = (*INPUT_STAGES, "evidence_guard")
OTHER_STAGES = ("plan", "retrieve", "sql", "generate")
# Reported by a script that wraps the sentence components; the API does not log them per sentence.
SENTENCE_STEPS = ("check_sentence_ms", "mask_pii_ms", "render_citations_ms")


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; None without values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(fraction * len(ordered)) - 1, 0)]


def distribution(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def sentence_checks(record: dict) -> float | None:
    """Milliseconds the sentence checks took in one request, when the steps were recorded."""
    steps = record.get("component_ms") or {}
    if not any(step in steps for step in SENTENCE_STEPS):
        return None
    return sum(steps.get(step, 0.0) for step in SENTENCE_STEPS)


def summarize(records: list[dict], *, skip_cold: bool = True) -> dict:
    """Median and p95 per stage over the answered requests.

    The first run of each stage loads its model or opens its connections, so it is reported as
    that stage's cold start and left out of the distributions unless skip_cold is false. A
    request that contains a cold start is left out of the per-request figures.
    """
    seen: set[str] = set()
    cold_stages: dict[str, float] = {}
    cold_requests: set[int] = set()
    first_of_stage: dict[str, int] = {}
    for index, record in enumerate(records):
        names = set(record["stage_latency_ms"])
        if sentence_checks(record) is not None:
            names.add("sentence_checks")
        for name in names - seen:
            cold_requests.add(index)
            cold_stages[name] = (
                sentence_checks(record)
                if name == "sentence_checks"
                else record["stage_latency_ms"][name]
            )
            first_of_stage[name] = index
        seen |= names
    if not skip_cold:
        cold_requests = set()
    answered = [(i, r) for i, r in enumerate(records) if r["status"] == "ok"]
    warm = [(i, r) for i, r in answered if i not in cold_requests]

    def stage(name: str) -> list[float]:
        return [
            r["stage_latency_ms"][name]
            for i, r in enumerate(records)
            if name in r["stage_latency_ms"]
            and r["status"] in ("ok", "refused")
            and not (skip_cold and i == first_of_stage[name])
        ]

    stages = {name: distribution(stage(name)) for name in (*GUARD_STAGES, *OTHER_STAGES)}
    gaps = [
        r["ttft_user_ms"] - r["ttft_engine_ms"]
        for _, r in warm
        if r.get("ttft_user_ms") is not None and r.get("ttft_engine_ms") is not None
    ]
    per_request: dict[str, list[float]] = {}
    totals = []
    for _, record in warm:
        sentences = sentence_checks(record)
        guards = sum(record["stage_latency_ms"].get(name, 0.0) for name in GUARD_STAGES)
        totals.append(guards + (sentences or 0.0))
        if sentences is not None:
            calls = (record.get("component_ms") or {}).get("check_sentence_calls") or 0
            per_request.setdefault("sentence_checks_total", []).append(sentences)
            if "generate" in record["stage_latency_ms"]:
                # The checks run between the model's tokens, so what is left is the model's time.
                per_request.setdefault("generate_without_checks", []).append(
                    record["stage_latency_ms"]["generate"] - sentences
                )
            if calls:
                per_request.setdefault("check_sentence_per_call", []).append(
                    record["component_ms"]["check_sentence_ms"] / calls
                )
    checked = sum(r.get("sentences_checked") or 0 for _, r in answered)
    removed = sum(r.get("sentences_removed") or 0 for _, r in answered)
    return {
        "requests": {
            "total": len(records),
            "answered": len(answered),
            "refused": sum(r["status"] == "refused" for r in records),
            "failed": sum(r["status"] not in ("ok", "refused") for r in records),
            "in_per_request_figures": len(warm),
        },
        "cold_start_ms": {k: round(v, 1) for k, v in cold_stages.items()} if skip_cold else {},
        "stages": stages,
        "wait_for_first_sentence_ms": distribution(gaps),
        "sentence_checks": {name: distribution(values) for name, values in per_request.items()},
        "guardrail_total_ms": distribution(totals),
        "sentences": {"checked": checked, "removed": removed},
    }
