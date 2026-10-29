"""D-23 data statistics for the report, computed from the local D-03 and D-12 files.

Usage: uv run --locked --group data --group rag python scripts/data_statistics.py

Reads only; no database connection. Exports aggregate counts, never IDs, names or text.
"""

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from author_names import extract_authors
from citations import build_citations
from filter_data import batches
from select_fixture import DATASET, PROJECT_ROOT, REVISION, data_directory

OUTPUT = PROJECT_ROOT / "eval" / "results" / "d23_data_statistics.json"


def bibliography_row(row, ids):
    """Entries in one S2ORC bibliography, those S2ORC matched, and those matched into ids."""
    raw = ((row.get("content") or {}).get("annotations") or {}).get("bibentry")
    entries = json.loads(raw) if isinstance(raw, str) else raw or []
    matched = [
        cid
        for entry in entries
        if type(cid := (entry.get("attributes") or {}).get("matched_paper_id")) is int
    ]
    in_corpus = [cid for cid in matched if cid in ids and cid != row["corpusid"]]
    return len(entries), len(matched), len(in_corpus), len(set(in_corpus))


def bibliography_summary(rows):
    sizes = sorted(row[0] for row in rows)
    return {
        "papers": len(rows),
        "entries": sum(sizes),
        "matched_entries": sum(row[1] for row in rows),
        "in_corpus_entries": sum(row[2] for row in rows),
        # A bibliography can list the same paper twice; the database keeps one pair.
        "in_corpus_distinct_pairs": sum(row[3] for row in rows),
        "papers_with_repeated_entries": sum(row[2] > row[3] for row in rows),
        "papers_without_bibliography": sum(entries == 0 for entries in sizes),
        "median_entries_per_paper": sizes[len(sizes) // 2] if sizes else 0,
    }


def link_counts(pairs, ids):
    citing = {pair[0] for pair in pairs}
    cited = {pair[1] for pair in pairs}
    return {
        "papers_citing_no_corpus_paper": len(ids - citing),
        "papers_not_cited": len(ids - cited),
        "papers_without_links": len(ids - citing - cited),
        "mutual_pairs": sum((b, a) in pairs for a, b in pairs) // 2,
    }


def metadata_counts(rows):
    def empty(value):
        return not (value or "").strip()

    return {
        "without_title": sum(empty(row["title"]) for row in rows),
        "without_abstract": sum(empty(row["abstract"]) for row in rows),
        "without_title_and_abstract": sum(
            empty(row["title"]) and empty(row["abstract"]) for row in rows
        ),
        "without_year": sum(row["year"] is None for row in rows),
        "by_year": dict(sorted(Counter(r["year"] for r in rows if r["year"] is not None).items())),
    }


def fold_accents(key):
    return "".join(c for c in unicodedata.normalize("NFKD", key) if not unicodedata.combining(c))


def drop_middle_initials(key):
    tokens = key.split()
    if len(tokens) <= 2:
        return key
    return " ".join([tokens[0], *(t for t in tokens[1:-1] if len(t) > 1), tokens[-1]])


def extra_authors(keys, rule):
    """Authors that would disappear if names equal under rule were merged."""
    groups = defaultdict(set)
    for key in keys:
        groups[rule(key)].add(key)
    return sum(len(group) - 1 for group in groups.values())


def author_statistics(rows, ids):
    authors, links, counts = extract_authors(rows)
    keys = [author[3] for author in authors]
    linked = {cid for _, cid in links}
    return {
        **counts,
        "papers_without_author": len(ids - linked),
        # Candidate merges only; the loaded data keeps these as separate authors (O2).
        "open_variants": {
            "accents_only": extra_authors(keys, fold_accents),
            "middle_initial_only": extra_authors(keys, drop_middle_initials),
            "accents_or_middle_initial": extra_authors(
                keys, lambda key: drop_middle_initials(fold_accents(key))
            ),
        },
    }


def check_ids(found, ids, name):
    """Every manifest paper exactly once; a duplicate cannot hide a missing paper."""
    if len(found) != len(set(found)) or set(found) != ids:
        raise ValueError(f"{name} must contain each D-03 paper exactly once.")


def statistics(data):
    folder = data / "filtered" / "d03"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != DATASET or manifest.get("dataset_revision") != REVISION:
        raise ValueError("D-03 data must match the pinned LitSearch dataset revision.")
    ids = set(manifest["retained_corpusids"])
    graph, metadata = {}, []
    for batch in batches([folder / "corpus_clean.arrow"]):
        for row in batch.select(["corpusid", "title", "abstract", "year", "citations"]).to_pylist():
            graph[row["corpusid"]] = row.pop("citations")
            metadata.append(row)
    check_ids([row["corpusid"] for row in metadata], ids, "corpus_clean")
    pairs, citation_counts = build_citations(graph, ids)
    bibliography_rows, s2orc_ids = [], []

    def s2orc_rows():
        for batch in batches([folder / "corpus_s2orc.arrow"]):
            for row in batch.select(["corpusid", "content"]).to_pylist():
                s2orc_ids.append(row["corpusid"])
                bibliography_rows.append(bibliography_row(row, ids))
                yield row

    authors = author_statistics(s2orc_rows(), ids)
    check_ids(s2orc_ids, ids, "corpus_s2orc")
    bibliography = {**bibliography_summary(bibliography_rows), "citation_pairs": len(pairs)}
    selection = json.loads(
        (data / "fulltext" / "v1" / "selection.json").read_text(encoding="utf-8")
    )
    if not set(selection["corpusids"]) <= ids:
        raise ValueError("The D-12 selection lists papers outside the D-03 corpus.")
    return {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "d03": {"counts": manifest["counts"], "rules": manifest["rules"]},
        "papers": metadata_counts(metadata),
        "citations": {**citation_counts, **link_counts(set(pairs), ids)},
        "bibliography": bibliography,
        "authors": authors,
        "fulltext": {
            "papers": len(selection["corpusids"]),
            "groups": {name: len(group) for name, group in selection["selection_groups"].items()},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        result = statistics(data_directory())
    except (ValueError, KeyError, OSError, ImportError) as error:
        parser.exit(1, f"Data statistics failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
