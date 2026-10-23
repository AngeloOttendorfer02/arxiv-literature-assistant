"""D-11 ordinary and grouped e-mail masking at import (O13)."""

import re

# A grouped expression such as {alice,bob}@example.org is one masked span.
EMAIL = re.compile(
    r"(?:"
    r"\{\s*[\w.+-]+(?:\s*[,|]\s*[\w.+-]+)*\s*\}\s*"
    r"|\(\s*[\w.+-]+(?:\s*[,|]\s*[\w.+-]+)*\s*\)\s*"
    r"|[\w.+-]+"
    r")@[\w-]+(?:\.[\w-]+)+"
)
PLACEHOLDER = "<EMAIL>"


def mask_emails(text):
    return EMAIL.sub(PLACEHOLDER, text) if text else text


def count_emails(text):
    """Count matching spans; a grouped expression counts once."""
    return len(EMAIL.findall(text)) if text else 0
