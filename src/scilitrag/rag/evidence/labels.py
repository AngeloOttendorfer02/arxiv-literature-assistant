"""Parse request-local citation labels and check their structural placement."""

import re

from scilitrag.contracts import EvidenceMapping, SentenceStatus

LABEL = re.compile(r"\[([1-9]\d*)\]")
NUMERIC_BRACKET = re.compile(r"\[\s*\d[^\]]*\]")
TRAILING_LABELS = re.compile(r"(?:\[[1-9]\d*\]\s*)+[.!?]?\s*$")


def citation_labels(sentence: str) -> list[int]:
    """Reject malformed numeric brackets and return labels in first-use order."""
    for match in NUMERIC_BRACKET.finditer(sentence):
        if not LABEL.fullmatch(match.group(0)):
            raise ValueError("Use one positive integer per citation, for example [1][2].")
    return list(dict.fromkeys(int(value) for value in LABEL.findall(sentence)))


def citation_status(sentence: str, mapping: EvidenceMapping) -> SentenceStatus:
    """Check label syntax, membership and placement before semantic validation.

    An ``ok`` result establishes structure only; semantic validation checks
    whether the source supports the claim.
    """
    try:
        labels = citation_labels(sentence)
    except ValueError:
        return "unknown_label"
    if not labels:
        return "no_citation"
    if any(label not in mapping for label in labels):
        return "unknown_label"
    if not TRAILING_LABELS.search(sentence):
        return "unsupported"
    return "ok"
