"""Select the D-12 full-text subset (O3, D034) from local D-03 Arrow files, without downloads.

Usage: uv run --locked --group data python scripts/select_fulltext.py
Writes DATA_DIR/fulltext/v1/selection.json; refuses to overwrite an existing selection.
All papers stay in the database; only the selected papers get full-text chunks (D-13).
"""

import argparse
import json
import random
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from citations import build_citations
from download_data import DATASET
from filter_data import batches
from profile_data import REVISION
from select_fixture import canonical_digest, data_directory, digest

# S1/S3 demo topic: retrieval-augmented generation, matched in title or abstract.
RAG_TOPIC = re.compile(r"(?i:retrieval[- ]augment)|\bRAG\b")
# S2 demo paper: Dense Passage Retrieval, a common baseline in the RAG papers.
S2_PAPERS = (215737187,)


def choose_fulltext(rows, targets, *, count=1000, seed=42, since=2022, s2_papers=S2_PAPERS):
    """Return the selected IDs and the IDs added by each group, in priority order."""
    ids = {row["corpusid"] for row in rows}
    if not targets or targets - ids:
        raise ValueError("Every query target must exist in the filtered corpus.")
    if set(s2_papers) - ids:
        raise ValueError("Every S2 demo paper must exist in the filtered corpus.")
    topic = {
        row["corpusid"]
        for row in rows
        if (row["year"] or 0) >= since
        and RAG_TOPIC.search(f"{row['title'] or ''} {row['abstract'] or ''}")
    }
    pairs, _ = build_citations({row["corpusid"]: row["citations"] or [] for row in rows}, ids)
    citing = {citing_id for citing_id, cited_id in pairs if cited_id in set(s2_papers)}
    selected, groups = set(), {}
    for name, members in (
        ("query_targets", targets),
        ("rag_topic", topic),
        ("s2_demo_papers", set(s2_papers)),
        ("s2_citing", citing),
    ):
        groups[name] = sorted(members - selected)
        selected |= members
    if len(selected) > count:
        raise ValueError(f"The fixed groups need {len(selected)} papers, more than {count}.")
    groups["random"] = sorted(
        random.Random(seed).sample(sorted(ids - selected), count - len(selected))
    )
    selected |= set(groups["random"])
    return sorted(selected), groups


def query_targets(queries, provenance):
    """Return all query targets after checking query and target counts against D-03."""
    counts = provenance.get("counts") or {}
    if len(queries) != counts.get("queries"):
        raise ValueError("Query count differs from the D-03 manifest.")
    if any(
        not isinstance(row.get("corpusids"), list)
        or not row["corpusids"]
        or any(type(cid) is not int for cid in row["corpusids"])
        for row in queries
    ):
        raise ValueError("Every query needs a nonempty list of integer target IDs.")
    targets = {cid for row in queries for cid in row["corpusids"]}
    if len(targets) != counts.get("target_papers"):
        raise ValueError("Target paper count differs from the D-03 manifest.")
    return targets


def write_selection(destination, selection):
    """Publish selection.json only after it is fully written; never overwrite."""
    if destination.exists():
        raise FileExistsError("Full-text selection v1 already exists; it will not be overwritten.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fulltext-", dir=destination.parent) as temporary:
        stage = Path(temporary) / destination.name
        stage.mkdir()
        (stage / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        stage.rename(destination)
    return destination / "selection.json"


def build_selection(data_dir, *, count=1000, seed=42):
    source, destination = data_dir / "filtered" / "d03", data_dir / "fulltext" / "v1"
    if destination.exists():
        raise FileExistsError("Full-text selection v1 already exists; it will not be overwritten.")
    provenance = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if provenance.get("dataset") != DATASET or provenance.get("dataset_revision") != REVISION:
        raise ValueError("D-03 data must match the pinned LitSearch dataset revision.")
    rows = [
        row
        for batch in batches([source / "corpus_clean.arrow"])
        for row in batch.select(["corpusid", "title", "abstract", "year", "citations"]).to_pylist()
    ]
    if {row["corpusid"] for row in rows} != set(provenance["retained_corpusids"]):
        raise ValueError("Filtered corpus IDs differ from the D-03 manifest.")
    queries = [row for batch in batches([source / "query.arrow"]) for row in batch.to_pylist()]
    targets = query_targets(queries, provenance)
    selected, groups = choose_fulltext(rows, targets, count=count, seed=seed)
    selection = {
        "selection_version": 1,
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "filtered_manifest_sha256": canonical_digest(provenance),
        "seed": seed,
        "since": 2022,
        "s2_papers": list(S2_PAPERS),
        "corpusids": selected,
        "selection_groups": groups,
    }
    return write_selection(destination, selection), selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = perf_counter()
    try:
        path, selection = build_selection(data_directory(), count=args.papers, seed=args.seed)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Full-text selection failed: {error}\n")
    counts = {name: len(ids) for name, ids in selection["selection_groups"].items()}
    print(f"Full-text selection: {path}")
    print(f"Papers: {len(selection['corpusids'])}; added per group in order: {counts}")
    print(f"Selection SHA-256: {digest(path)}")
    print(f"Preparation: {perf_counter() - started:.2f}s. No database changes.")


if __name__ == "__main__":
    main()
