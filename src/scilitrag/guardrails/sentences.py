import re

# A boundary is only reported once the next character is known, so streaming never cuts
# "et al." or "52.3" in half. When unsure the splitter does not split: a merged sentence is
# still checked as a whole, a fragment would lose its citation label.
CANDIDATE = re.compile(
    r"(?P<end>[.!?]+[\"”’')]*(?:\s*\[\d+\])*(?=\s))|(?P<bullet>(?=\n[ \t]*(?:[-*•]|\d{1,2}[.)])[ \t]))"
)
SENTENCE_START = re.compile(r"[A-Z0-9\"“‘'(*•-]")
INITIALS = re.compile(r"(?:[A-Za-z]\.)*[A-Za-z]")
LIST_MARKER = re.compile(r"(?:[-*•]\s*)?(?:\d{1,2}|[A-Za-z])")
# Never end a sentence; the abbreviations are matched case-insensitively without the dot.
ABBREVIATIONS = {"cf", "vs", "fig", "figs", "eq", "eqs", "sec", "tab", "dr", "prof", "mr", "mrs"}
ABBREVIATIONS |= {"ms", "approx", "resp", "ca", "inc", "ltd"}
# Also end a sentence when a capital letter follows.
AMBIGUOUS = re.compile(r"(?:\bet al|\betc)$", re.IGNORECASE)


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Return the complete sentences in buffer and the incomplete rest."""
    sentences: list[str] = []
    start = 0
    for match in CANDIDATE.finditer(buffer):
        end = match.end() if match.group("end") else match.start()
        sentence = buffer[start:end].strip()
        if not sentence:
            continue
        if match.group("end"):
            following = buffer[end:].lstrip()
            if not following or not _ends_sentence(buffer[start : match.start()], match, following):
                continue
        sentences.append(sentence)
        start = end
    return sentences, buffer[start:].lstrip()


def _ends_sentence(before: str, match: re.Match, following: str) -> bool:
    if not SENTENCE_START.match(following):
        return False
    if match.group("end")[0] != ".":
        return True
    before = before.rstrip()
    if AMBIGUOUS.search(before):
        return following[0].isupper()
    word = before.rsplit(None, 1)[-1].lstrip("(\"'“‘") if before else ""
    if word.lower() in ABBREVIATIONS or INITIALS.fullmatch(word):
        return False
    return not LIST_MARKER.fullmatch(before.strip())
