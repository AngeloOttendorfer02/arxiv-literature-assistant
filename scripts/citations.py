"""D-09 citation pairs from corpus_clean.citations, where each list holds the papers a paper cites."""

import json
from collections import Counter


def build_citations(graph, ids):
    """Return sorted (citing_id, cited_id) pairs inside ids and counts of dropped entries."""
    counts = Counter()
    pairs = set()
    for citing in sorted(graph):
        for cited in graph[citing]:
            counts["entries"] += 1
            if cited == citing:
                counts["self_citation"] += 1
            elif citing not in ids or cited not in ids:
                counts["outside_corpus"] += 1
            elif (citing, cited) in pairs:
                counts["duplicate"] += 1
            else:
                pairs.add((citing, cited))
    counts["citations"] = len(pairs)
    return sorted(pairs), dict(counts)


def direction_counts(pairs, years):
    """Compare publication years; a citing paper is expected to be the same age or newer."""
    counts = Counter()
    for citing, cited in pairs:
        citing_year, cited_year = years.get(citing), years.get(cited)
        if citing_year is None or cited_year is None:
            counts["year_missing"] += 1
        elif citing_year > cited_year:
            counts["citing_newer"] += 1
        elif citing_year == cited_year:
            counts["same_year"] += 1
        else:
            counts["citing_older"] += 1
    return dict(counts)


def bibliography_ids(row):
    """Corpus IDs that S2ORC matched to entries of this paper's own bibliography."""
    raw = ((row.get("content") or {}).get("annotations") or {}).get("bibentry")
    entries = json.loads(raw) if isinstance(raw, str) else raw or []
    return {
        matched
        for entry in entries
        if type(matched := (entry.get("attributes") or {}).get("matched_paper_id")) is int
    }


def bibliography_counts(pairs, bibliographies):
    """Count pairs whose cited paper is in the citing paper's own bibliography."""
    counts = Counter()
    for citing, cited in pairs:
        found = cited in bibliographies.get(citing, ())
        counts["in_bibliography" if found else "not_in_bibliography"] += 1
    return dict(counts)
