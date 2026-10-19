import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import aclosing

from fastapi.concurrency import run_in_threadpool

from scilitrag.components import Components
from scilitrag.contracts import (
    SENTENCE_REMOVED,
    DoneEvent,
    ErrorEvent,
    Model,
    SentenceCheck,
    SentenceEvent,
    StatusEvent,
    TableEvent,
)
from scilitrag.rag.citation_labels import citation_status
from scilitrag.rag.errors import LLMClientError
from scilitrag.rag.evidence_index import EvidenceIndex

logger = logging.getLogger(__name__)

# Naive splitter; WP3 replaces it (S-11).
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_END = object()


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Return the complete sentences in buffer and the incomplete rest."""
    parts = SENTENCE_END.split(buffer)
    return [p.strip() for p in parts[:-1] if p.strip()], parts[-1]


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


async def run_query(
    query: str, c: Components, timeout_s: float | None = None
) -> AsyncIterator[tuple[str, Model]]:
    """Events of one query, ending with done or error.

    The pipeline runs as its own task and the caller reads events from a queue with a
    deadline. The time limit therefore never spans a yield, and closing this generator
    (client disconnect) cancels the task, which also closes the LLM stream.
    """
    loop = asyncio.get_running_loop()
    deadline = None if timeout_s is None else loop.time() + timeout_s
    queue: asyncio.Queue = asyncio.Queue()

    async def produce() -> None:
        try:
            async with aclosing(_run(query, c)) as events:
                async for event in events:
                    queue.put_nowait(event)
        finally:
            queue.put_nowait(_END)

    task = asyncio.create_task(produce())
    try:
        while True:
            remaining = None if deadline is None else max(deadline - loop.time(), 0)
            try:
                item = await asyncio.wait_for(queue.get(), remaining)
            except TimeoutError:
                logger.warning("query timed out after %g s", timeout_s)
                yield (
                    "error",
                    ErrorEvent(
                        code="timeout", message=f"The request took longer than {timeout_s:g} s."
                    ),
                )
                return
            if item is _END:
                return
            yield item
    finally:
        task.cancel()


async def _run(query: str, c: Components) -> AsyncIterator[tuple[str, Model]]:
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

        mapping = EvidenceIndex(evidence).mapping()
        checked_count = removed = tokens = 0
        first_token = last_token = None
        ttft_user = None
        buffer = ""
        t = time.perf_counter()

        async def release(sentences: list[str]) -> AsyncIterator[tuple[str, Model]]:
            nonlocal checked_count, removed, ttft_user
            for sentence in sentences:
                structural_status = citation_status(sentence, mapping)
                check = (
                    SentenceCheck(status=structural_status, latency_ms=0)
                    if structural_status != "ok"
                    else await run_in_threadpool(c.check_sentence, sentence, mapping)
                )
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

        async with aclosing(c.generate(plan, evidence)) as stream:
            async for token in stream:
                last_token = time.perf_counter()
                if first_token is None:
                    first_token = last_token
                tokens += 1
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
        rate = None
        if tokens > 1 and last_token > first_token:
            rate = round((tokens - 1) / (last_token - first_token), 2)
        yield (
            "done",
            DoneEvent(
                stage_latency_ms=latency,
                ttft_engine_ms=None if first_token is None else (first_token - start) * 1000,
                ttft_user_ms=ttft_user,
                total_ms=_ms(start),
                sentences_checked=checked_count,
                sentences_removed=removed,
                tokens=tokens,
                tokens_per_s=rate,
            ),
        )
    except LLMClientError as exc:
        yield (
            "error",
            ErrorEvent(code="timeout" if exc.kind == "timeout" else "internal", message=str(exc)),
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
