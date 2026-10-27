"""Validated text spans from corpus_s2orc annotations, shared by D-08 and D-10."""

import json


def annotation_entries(row, field, clip=False):
    """Return validated (start, end, attributes) annotations sorted by position.

    With clip, spans past the end of a truncated text are shortened or skipped.
    """
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
        if clip and type(start) is int and type(end) is int and end > len(text):
            if start >= len(text):
                continue
            end = len(text)
        if not (type(start) is int and type(end) is int and 0 <= start < end <= len(text)):
            raise ValueError(f"Paper {row['corpusid']}: invalid {field} text span.")
        spans.append((start, end, entry.get("attributes") or {}))
    return sorted(spans, key=lambda span: span[:2])


def annotation_spans(row, field):
    return [(start, end) for start, end, _ in annotation_entries(row, field)]
