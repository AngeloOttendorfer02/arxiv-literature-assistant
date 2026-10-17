import logging
import re
import time
from collections.abc import AsyncIterator

from fastapi.concurrency import run_in_threadpool

from scilitrag.components import Components
from scilitrag.contracts import (
    SENTENCE_REMOVED,
    DoneEvent,
    ErrorEvent,
    Model,
    SentenceEvent,
    StatusEvent,
    TableEvent,
)

logger = logging.getLogger(__name__)

# Naive splitter; WP3 replaces it (S-11).
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Return the complete sentences in buffer and the incomplete rest."""
    parts = SENTENCE_END.split(buffer)
    return [p.strip() for p in parts[:-1] if p.strip()], parts[-1]


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


async def run_query(query: str, c: Components) -> AsyncIterator[tuple[str, Model]]:
    start = time.perf_counter()
    latency: dict[str, float] = {}
    try:
        checked = await run_in_threadpool(c.check_input, query)
        for stage in checked.stages:
            latency[stage.stage] = stage.latency_ms
            yield "status", StatusEvent(**stage.model_dump(exclude={"score"}))
        if not checked.allowed:
            yield "done", DoneEvent(stage_latency_ms=latency, total_ms=_ms(start), refused=True)
            return

        t = time.perf_counter()
        plan = await run_in_threadpool(c.plan, checked.masked_query)
        latency["plan"] = _ms(t)
        yield "status", StatusEvent(stage="plan", result=plan.scenario, latency_ms=latency["plan"])

        t = time.perf_counter()
        evidence = await run_in_threadpool(c.retrieve, plan)
        latency["retrieve"] = _ms(t)
        yield (
            "status",
            StatusEvent(
                stage="retrieve", result=f"{len(evidence)} evidence", latency_ms=latency["retrieve"]
            ),
        )

        t = time.perf_counter()
        facts = await run_in_threadpool(c.sql_facts, plan)
        latency["sql"] = _ms(t)
        yield (
            "status",
            StatusEvent(stage="sql", result=f"{len(facts)} tables", latency_ms=latency["sql"]),
        )
        for table in facts:
            yield "table", TableEvent(**table.model_dump())

        mapping = {e.label: e for e in evidence}
        checked_count = removed = 0
        ttft_engine = ttft_user = None
        buffer = ""
        t = time.perf_counter()

        async def release(sentences: list[str]) -> AsyncIterator[tuple[str, Model]]:
            nonlocal checked_count, removed, ttft_user
            for sentence in sentences:
                check = await run_in_threadpool(c.check_sentence, sentence, mapping)
                checked_count += 1
                if check.status != "ok":
                    removed += 1
                    yield (
                        "status",
                        StatusEvent(
                            stage="generate",
                            result=f"{SENTENCE_REMOVED}{check.status}",
                            latency_ms=check.latency_ms,
                        ),
                    )
                    continue
                masked = await run_in_threadpool(c.mask_pii, sentence)
                rendered = await run_in_threadpool(c.render_citations, masked, mapping)
                if ttft_user is None:
                    ttft_user = _ms(start)
                yield "sentence", SentenceEvent(**rendered.model_dump(), check=check)

        async for token in c.generate(plan, evidence):
            if ttft_engine is None:
                ttft_engine = _ms(start)
            sentences, buffer = split_sentences(buffer + token)
            async for event in release(sentences):
                yield event
        async for event in release([buffer.strip()] if buffer.strip() else []):
            yield event

        latency["generate"] = _ms(t)
        yield (
            "status",
            StatusEvent(
                stage="generate",
                result=f"{checked_count} checked, {removed} removed",
                latency_ms=latency["generate"],
            ),
        )
        yield (
            "done",
            DoneEvent(
                stage_latency_ms=latency,
                ttft_engine_ms=ttft_engine,
                ttft_user_ms=ttft_user,
                total_ms=_ms(start),
                sentences_checked=checked_count,
                sentences_removed=removed,
            ),
        )
    except NotImplementedError as exc:
        logger.warning("%s", exc)
        yield "error", ErrorEvent(code="internal", message=str(exc))
    except Exception:
        logger.exception("query failed")
        yield (
            "error",
            ErrorEvent(code="internal", message="Internal error while processing the query."),
        )
