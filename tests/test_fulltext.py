"""D-10 full-text cleaning checks with synthetic S2ORC rows, never real LitSearch."""

import json

import pytest
from fulltext import clean_sections, fulltext_counts


def paper(parts, corpus_id=1):
    """Build an S2ORC-like row from (annotation field, text, attributes) parts."""
    text, annotations = "", {}
    for field, value, attributes in parts:
        start = len(text)
        text += value + "\n"
        if field:
            entry = {"start": start, "end": start + len(value)}
            if attributes:
                entry["attributes"] = attributes
            annotations.setdefault(field, []).append(entry)
    encoded = {field: json.dumps(entries) for field, entries in annotations.items()}
    return {"corpusid": corpus_id, "content": {"text": text, "annotations": encoded}}


FRONT = [
    ("title", "A Synthetic Paper", None),
    ("author", "Jane Doe jane@example.org", None),
    ("authoraffiliation", "Example University", None),
    ("abstract", "Abstract text.", None),
]


def test_front_matter_and_bibliography_are_removed_and_sections_kept():
    row = paper(
        [
            *FRONT,
            ("sectionheader", "Introduction", {"n": "1"}),
            ("paragraph", "We study things.", None),
            ("sectionheader", "Table 2 :", None),
            ("paragraph", "Table 2: scores of all models.", None),
            ("sectionheader", "Limitations", None),
            ("sectionheader", "Setup", {"n": "2.1"}),
            ("paragraph", "We train for ten epochs.", None),
            ("bibentry", "Doe, J. 2020. Old work.", None),
            ("bibentry", "Roe, R. 2021. Other work.", None),
        ]
    )
    sections, removed = clean_sections(row)
    assert [title for title, _ in sections] == ["1 Introduction", "2.1 Setup"]
    intro = sections[0][1]
    assert "Table 2: scores of all models." in intro and "Limitations" in intro
    assert sections[1][1] == "We train for ten epochs."
    body = " ".join(text for _, text in sections)
    for removed_text in ("Synthetic Paper", "jane@example.org", "Abstract text.", "Doe, J."):
        assert removed_text not in body
    assert removed["front_chars"] > 0 and removed["bibliography_chars"] > 0


def test_paper_without_numbered_headings_is_one_unnamed_section():
    row = paper(
        [
            *FRONT,
            ("sectionheader", "Introduction", None),
            ("paragraph", "Only body text.", None),
        ]
    )
    sections, removed = clean_sections(row)
    assert sections == [(None, "Only body text.")]
    assert removed["bibliography_chars"] == 0


def test_empty_parent_section_is_skipped():
    row = paper(
        [
            ("sectionheader", "Method", {"n": "3"}),
            ("sectionheader", "Model", {"n": "3.1"}),
            ("paragraph", "Details.", None),
        ]
    )
    assert clean_sections(row)[0] == [("3.1 Model", "Details.")]


def test_counts_summarize_all_papers():
    rows = [
        paper([*FRONT, ("paragraph", "Body.", None), ("bibentry", "Ref.", None)], 1),
        paper([("paragraph", "Body only.", None)], 2),
    ]
    counts = fulltext_counts(rows)
    assert counts["papers"] == 2 and counts["sections"] == 2
    assert counts["no_numbered_section"] == 2 and counts["no_bibliography"] == 1
    assert counts["kept_chars"] == len("Body.") + len("Body only.")


def test_bibliography_past_the_end_of_a_truncated_text_is_clipped():
    row = paper([("paragraph", "Body.", None), ("bibentry", "Doe, J. 2020.", None)])
    entries = json.loads(row["content"]["annotations"]["bibentry"])
    entries[0]["end"] += 50
    entries.append({"start": entries[0]["end"], "end": entries[0]["end"] + 20})
    row["content"]["annotations"]["bibentry"] = json.dumps(entries)
    sections, removed = clean_sections(row)
    assert sections == [(None, "Body.")]
    assert removed["bibliography_chars"] == len("Doe, J. 2020.") + 1


def test_invalid_heading_span_is_rejected():
    row = paper([("paragraph", "Body.", None)])
    row["content"]["annotations"]["sectionheader"] = json.dumps([{"start": 0, "end": 999}])
    with pytest.raises(ValueError, match="sectionheader"):
        clean_sections(row)
