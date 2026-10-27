import time

from scilitrag.config import Settings
from scilitrag.contracts import CheckEvidence, Evidence, EvidenceCheckResult
from scilitrag.guardrails.prompt_guard import shared_guard


# Indirect injection: retrieved text is data, but a source that tries to give the model
# instructions is dropped before it reaches the prompt. SQL evidence is built by our code.
def build_component(settings: Settings) -> CheckEvidence:
    guard = shared_guard()

    def check_evidence(evidence: list[Evidence]) -> EvidenceCheckResult:
        start = time.perf_counter()
        scores = {
            item.label: guard.score(f"{item.title or ''}\n{item.text}".strip())
            for item in evidence
            if item.type != "sql"
        }
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
