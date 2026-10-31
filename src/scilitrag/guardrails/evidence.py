import time

from scilitrag.config import Settings
from scilitrag.contracts import CheckEvidence, Evidence, EvidenceCheckResult
from scilitrag.guardrails.prompt_guard import RemoteGuard, shared_guard


# Indirect injection: retrieved text is data, but a source that tries to give the model
# instructions is dropped before it reaches the prompt. SQL evidence is built by our code.
def build_component(settings: Settings) -> CheckEvidence:
    guard = shared_guard()
    remote = RemoteGuard(settings.guard_url, settings.llm_api_key) if settings.guard_url else None

    def score_all(texts: list[str]) -> list[float]:
        if remote:
            # All sources in one request; the GPU scores their windows in one batch.
            return remote.scores([guard.windows(text) for text in texts])
        return [guard.score(text) for text in texts]

    def check_evidence(evidence: list[Evidence]) -> EvidenceCheckResult:
        start = time.perf_counter()
        scored = [item for item in evidence if item.type != "sql"]
        texts = [f"{item.title or ''}\n{item.text}".strip() for item in scored]
        scores = dict(zip((item.label for item in scored), score_all(texts), strict=True))
        removed = [
            label for label, score in scores.items() if score >= settings.evidence_guard_threshold
        ]
        return EvidenceCheckResult(
            evidence=[item for item in evidence if item.label not in removed],
            removed=removed,
            scores=scores,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    return check_evidence
