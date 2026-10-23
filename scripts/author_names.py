"""D-08 authors from S2ORC name annotations, normalized as in WP2 entity linking (O2)."""

import html
import json
import re
from collections import Counter, defaultdict

from scilitrag.rag.search.entities import normalize_name

# Affiliations and headers that the parser tags as authors, chosen on the D-03 data.
# "ai" and "alexa" alone also match real people, so they only appear in phrases.
# fmt: off
EXCLUDED_WORDS = frozenset({
    "university", "universit", "univ", "universidad", "universidade", "universitat",
    "universite", "school", "college", "institute", "laboratory", "laboratories", "lab",
    "labs", "research", "academy", "google", "microsoft", "facebook", "meta", "amazon",
    "apple", "ibm", "baidu", "alibaba", "tencent", "huawei", "deepmind", "openai", "nvidia",
    "adobe", "salesforce", "bloomberg", "yahoo", "brain", "bytedance", "inc", "corp",
    "corporation", "ltd", "llc", "gmbh", "workshop", "actes", "conjointe", "language",
})
# fmt: on
EXCLUDED_PHRASES = ("alexa ai", "bruno kessler", "paul g allen", "eth zurich", "eth zürich")
# Chinese institution terms; Chinese text has no spaces, so these match as substrings.
EXCLUDED_SUBSTRINGS = (
    "大學", "大学", "研究院", "研究所", "研究中心", "實驗室", "实验室", "公司", "學院", "学院"
)  # fmt: skip
# Trailing periods are kept for initials such as "Noah A.".
EDGE_NOISE = re.compile(r"^[\W_]+|[^\w.]+$")


def annotation_spans(row, field):
    payload = ((row.get("content") or {}).get("annotations") or {}).get(field)
    entries = json.loads(payload) if isinstance(payload, str) else payload or []
    text = row["content"].get("text") or ""
    spans = []
    for entry in entries:
        start, end = (
            int(value)
            if isinstance(value, str) and value.isascii() and value.isdecimal()
            else value
            for value in (entry.get("start"), entry.get("end"))
        )
        if not (type(start) is int and type(end) is int and 0 <= start < end <= len(text)):
            raise ValueError(f"Paper {row['corpusid']}: invalid {field} text span.")
        spans.append((start, end))
    return sorted(spans)


def join_spans(text, spans, start, end):
    value = " ".join(text[s:e] for s, e in spans if start <= s and e <= end)
    # The source keeps HTML entities such as &apos;; leading and trailing marks are parser noise.
    value = EDGE_NOISE.sub("", html.unescape(value))
    return " ".join(value.split())


def is_excluded(key):
    padded = f" {key} "
    return (
        bool(EXCLUDED_WORDS.intersection(key.split()))
        or any(f" {phrase} " in padded for phrase in EXCLUDED_PHRASES)
        or any(term in key for term in EXCLUDED_SUBSTRINGS)
    )


def extract_authors(rows):
    """Return authors sorted by normalized name, (normalized name, corpus_id) links and counts."""
    counts = Counter()
    variants = defaultdict(Counter)
    links = set()
    for row in rows:
        text = row["content"].get("text") or ""
        firsts = annotation_spans(row, "authorfirstname")
        lasts = annotation_spans(row, "authorlastname")
        for start, end in annotation_spans(row, "author"):
            counts["author_blocks"] += 1
            first = join_spans(text, firsts, start, end)
            last = join_spans(text, lasts, start, end)
            key = normalize_name(f"{first} {last}")
            if not first or not last or not key:
                counts["missing_first_or_last"] += 1
            elif is_excluded(key):
                counts["affiliation_or_header"] += 1
            elif (key, row["corpusid"]) in links:
                counts["duplicate_in_paper"] += 1
            else:
                links.add((key, row["corpusid"]))
                variants[key][(first, last)] += 1
    authors = []
    for key in sorted(variants):
        # Most frequent original spelling for display; ties broken alphabetically.
        (first, last), _ = min(variants[key].items(), key=lambda item: (-item[1], item[0]))
        authors.append((first, last, f"{first} {last}", key))
    counts["authors"] = len(authors)
    counts["paper_authors"] = len(links)
    return authors, sorted(links), dict(counts)
