import time

from scilitrag.config import Settings
from scilitrag.contracts import CheckInput, CheckInputResult, Embed, StageResult
from scilitrag.guardrails.extraction import asks_for_instructions
from scilitrag.guardrails.pii import mask_pii
from scilitrag.guardrails.prompt_guard import shared_guard
from scilitrag.guardrails.prototypes import load_prototypes
from scilitrag.guardrails.topic import TopicGate, decide


def build_component(settings: Settings, embed: Embed | None = None) -> CheckInput:
    guard = shared_guard()
    if embed is None:
        # Mock vectors are hashes, so similarities would be meaningless: the gate always embeds for real.
        from scilitrag.data.embedding import build_component as build_embed

        embed = build_embed(settings)
    gate = TopicGate(embed, load_prototypes(settings.data_dir))

    def check_input(query: str) -> CheckInputResult:
        start = time.perf_counter()
        pii_start = time.perf_counter()
        masked = mask_pii(query)
        pii_ms = (time.perf_counter() - pii_start) * 1000
        stages: list[StageResult] = []

        def finish(category: str) -> CheckInputResult:
            return CheckInputResult(
                allowed=category == "ok",
                category=category,
                masked_query=masked,
                scores={stage.stage: stage.score or 0.0 for stage in stages},
                latency_ms=(time.perf_counter() - start) * 1000,
                stages=stages,
            )

        guard_start = time.perf_counter()
        score = guard.score(query)
        # Prompt Guard misses requests for the system prompt, so a pattern covers them (D066).
        flagged = score >= settings.prompt_guard_threshold or asks_for_instructions(query)
        stages.append(
            StageResult(
                stage="prompt_guard",
                result="injection" if flagged else "ok",
                score=score,
                latency_ms=(time.perf_counter() - guard_start) * 1000,
            )
        )
        if flagged:
            return finish("injection")

        gate_start = time.perf_counter()
        category, gap = decide(gate.scores(masked), settings.topic_gate_margin)
        stages.append(
            StageResult(
                stage="topic_gate",
                result=category,
                score=gap,
                latency_ms=(time.perf_counter() - gate_start) * 1000,
            )
        )
        if category != "ok":
            return finish(category)

        changed = masked != query
        stages.append(
            StageResult(
                stage="pii",
                result="masked" if changed else "ok",
                score=float(changed),
                latency_ms=pii_ms,
            )
        )
        return finish("ok")

    return check_input
