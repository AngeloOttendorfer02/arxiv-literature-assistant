"""Requests that ask the assistant for its own instructions, which Prompt Guard 2 does not flag."""

import re

ASK = (
    r"(?:print|show|reveal|repeat|display|output|tell|give|list|translate|summari[sz]e|write"
    r"|disclose|leak|expose|dump)"
)
INSTRUCTIONS = (
    r"(?:system|initial|hidden|secret|developer|original|internal)[\s-]*"
    r"(?:prompt|instructions?|message|rules|config(?:uration)?)"
)
PATTERNS = [
    rf"\b{ASK}\b.{{0,60}}\b(?:your|the)\s+{INSTRUCTIONS}",
    rf"\byour\s+{INSTRUCTIONS}",
    r"\brepeat\s+(?:all\s+)?(?:the\s+)?(?:words|text|instructions)\s+(?:above|before)",
    r"\binstructions\s+you\s+(?:were|have been)\s+given",
    r"\b(?:text|words|everything)\b.{0,40}\bbefore\s+my\s+(?:first\s+)?(?:message|question)",
    r"\b(?:api\s*keys?|connection\s+string|credentials|passwords?)\b.{0,30}\byou\s+(?:use|have|know)",
    r"\bsystem[\s-]*prompt\b.{0,40}\b(?:ausgeben|wiederhol|zeig|verrat|gib)",
    r"\b(?:gib|zeig|nenn|wiederhole)\w*\b.{0,60}\b(?:system[\s-]*prompt|anweisungen)",
]
_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in PATTERNS]


def asks_for_instructions(text: str) -> bool:
    """True when a request is aimed at the assistant's own prompt, rules or credentials."""
    return any(pattern.search(text) for pattern in _COMPILED)
