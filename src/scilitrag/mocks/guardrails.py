import re
import time

from scilitrag.contracts import (
    CheckInputResult,
    Evidence,
    EvidenceCheckResult,
    EvidenceMapping,
    SentenceCheck,
    StageResult,
)

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
LABEL = re.compile(r"\[(\d+)\]")
INJECTION = re.compile(
    r"ignore (?:all |any )?(?:previous|prior|above) instructions|system prompt"
    r"|disregard (?:all |any )?(?:previous |prior )?instructions|ignoriere .*anweisungen",
    re.IGNORECASE,
)
WRITE_INTENT = re.compile(
    r"\b(?:delete|drop|insert|update|remove)\b|\b(?:lösch|änder)\w*", re.IGNORECASE
)
OFF_TOPIC = re.compile(
    r"\b(?:recipe|football|horoscope|lottery|celebrity|rezept|wetter)\b", re.IGNORECASE
)


def mask_pii(text: str) -> str:
    return EMAIL.sub("<EMAIL>", text)


def check_input(query: str) -> CheckInputResult:
    start = time.perf_counter()
    stages: list[StageResult] = []

    def finish(category: str) -> CheckInputResult:
        return CheckInputResult(
            allowed=category == "ok",
            category=category,
            masked_query=mask_pii(query),
            scores={stage.stage: stage.score or 0.0 for stage in stages},
            latency_ms=(time.perf_counter() - start) * 1000,
            stages=stages,
        )

    injection = bool(INJECTION.search(query))
    stages.append(_stage("prompt_guard", "injection" if injection else "ok", injection))
    if injection:
        return finish("injection")

    if WRITE_INTENT.search(query):
        category = "write_intent"
    elif OFF_TOPIC.search(query):
        category = "off_topic"
    else:
        category = "ok"
    stages.append(_stage("topic_gate", category, category != "ok"))
    if category != "ok":
        return finish(category)

    masked = len(EMAIL.findall(query))
    stages.append(_stage("pii", f"masked {masked}" if masked else "ok", masked > 0))
    return finish("ok")


def check_evidence(evidence: list[Evidence]) -> EvidenceCheckResult:
    flagged = [e.label for e in evidence if e.type != "sql" and INJECTION.search(e.text)]
    return EvidenceCheckResult(
        evidence=[e for e in evidence if e.label not in flagged],
        removed=flagged,
        scores={label: 1.0 for label in flagged},
        latency_ms=0.0,
    )


# Structure stage only (label present and known); no fact or NLI check.
# Open (O12): share of removed sentences that triggers "not enough evidence" must be chosen.
def check_sentence(sentence: str, mapping: EvidenceMapping) -> SentenceCheck:
    labels = {int(n) for n in LABEL.findall(sentence)}
    if not labels:
        status = "no_citation"
    elif labels - mapping.keys():
        status = "unknown_label"
    else:
        status = "ok"
    return SentenceCheck(status=status, latency_ms=0.0)


def _stage(stage: str, result: str, hit: bool) -> StageResult:
    return StageResult(stage=stage, result=result, score=1.0 if hit else 0.0, latency_ms=0.0)
