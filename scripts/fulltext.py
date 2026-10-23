"""D-10 full-text sections from S2ORC annotations, without front matter and bibliography."""

from collections import Counter

from s2orc_annotations import annotation_entries, annotation_spans


def clean_sections(row):
    """Return (section, text) pairs of the paper body and the removed character counts.

    The body starts at the first paragraph or numbered heading and ends before the first
    bibliography entry. Only headings with a section number start a section; captions and
    unnumbered headings stay in the text of the current section.
    """
    text = row["content"].get("text") or ""
    paragraphs = annotation_spans(row, "paragraph")
    headings = [
        (start, end, attributes["n"])
        for start, end, attributes in annotation_entries(row, "sectionheader")
        if attributes.get("n")
    ]
    # Some texts end inside the bibliography while its annotations still run further.
    bibliography = annotation_entries(row, "bibentry", clip=True)
    starts = [paragraphs[0][0]] if paragraphs else []
    if headings:
        starts.append(headings[0][0])
    body_start = min(starts, default=0)
    body_end = bibliography[0][0] if bibliography else len(text)
    body_end = max(body_end, body_start)
    headings = [heading for heading in headings if body_start <= heading[0] < body_end]

    sections = []
    title, position = None, body_start
    for start, end, number in headings:
        sections.append((title, text[position:start]))
        name = " ".join(text[start:end].split())
        title, position = f"{number} {name}" if name else number, end
    sections.append((title, text[position:body_end]))
    sections = [(title, body.strip()) for title, body in sections if body.strip()]
    removed = {"front_chars": body_start, "bibliography_chars": len(text) - body_end}
    return sections, removed


def fulltext_counts(rows):
    """Clean every row and summarize what was kept and removed."""
    counts = Counter()
    for row in rows:
        sections, removed = clean_sections(row)
        counts["papers"] += 1
        counts["sections"] += len(sections)
        counts["text_chars"] += len(row["content"].get("text") or "")
        counts["kept_chars"] += sum(len(body) for _, body in sections)
        counts.update(removed)
        counts["no_numbered_section"] += all(title is None for title, _ in sections)
        counts["no_bibliography"] += removed["bibliography_chars"] == 0
    return dict(counts)
