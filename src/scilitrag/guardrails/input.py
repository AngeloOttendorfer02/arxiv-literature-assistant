import time

from scilitrag.config import Settings
from scilitrag.contracts import CheckInput, CheckInputResult, StageResult
from scilitrag.guardrails.pii import mask_pii
from scilitrag.guardrails.prompt_guard import PromptGuard


# Topic gate and write-intent refusal (S-06, S-07) are not part of this stage list yet.
def build_component(settings: Settings) -> CheckInput:
    guard = PromptGuard()

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
        flagged = score >= settings.prompt_guard_threshold
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
