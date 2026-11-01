"""Parse request-local citation labels and check their structural placement."""

import re

from scilitrag.contracts import EvidenceMapping, SentenceStatus

LABEL = re.compile(r"\[([1-9]\d*)\]")
NUMERIC_BRACKET = re.compile(r"\[\s*\d[^\]]*\]")
TRAILING_LABELS = re.compile(r"(?:\[[1-9]\d*\]\s*)+[.!?]?\s*$")

ABSTRACT_ONLY_SECTION = "Abstract only"


def abstract_notice(labels: list[int]) -> str:
    """Frame sentence written by the code for the sources that have only an abstract."""
    noun = "this result" if len(labels) == 1 else "these results"
    cited = "".join(f"[{label}]" for label in labels)
    return f"Only abstract evidence is present in this context for {noun} {cited}."


def is_abstract_notice(sentence: str, mapping: EvidenceMapping) -> bool:
    """True if the sentence is that frame sentence and every cited source is abstract-only."""
    labels = [int(value) for value in LABEL.findall(sentence)]
    return (
        bool(labels)
        and all(
            label in mapping
            and mapping[label].type == "paper"
            and mapping[label].section == ABSTRACT_ONLY_SECTION
            for label in labels
        )
        and sentence.strip() == abstract_notice(labels)
    )


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
