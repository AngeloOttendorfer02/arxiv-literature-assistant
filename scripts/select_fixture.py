"""Build a reproducible D-07 fixture from local D-03 Arrow files, without downloads.

Usage: uv run --locked --group data python scripts/select_fixture.py
Writes DATA_DIR/fixtures/v0; refuses to overwrite an existing fixture.
"""

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from dotenv import load_dotenv
from download_data import DATASET, PROJECT_ROOT
from filter_data import batches
from profile_data import REVISION

FILES = ("query.arrow", "corpus_clean.arrow", "corpus_s2orc.arrow")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def data_directory():
    load_dotenv(PROJECT_ROOT / ".env")
    value = os.environ.get("DATA_DIR")
    if not value:
        raise ValueError("Set DATA_DIR in .env first.")
    directory = Path(value).expanduser().resolve()
    if directory.is_relative_to(PROJECT_ROOT) and not directory.is_relative_to(
        PROJECT_ROOT / "data"
    ):
        raise ValueError("Use DATA_DIR outside the repository or in its ignored data/ directory.")
    return directory


def choose_queries(rows, rng, count=20):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if (
            not isinstance(row.get("query_set"), str)
            or not isinstance(row.get("corpusids"), list)
            or not row["corpusids"]
            or any(type(cid) is not int for cid in row["corpusids"])
        ):
            raise ValueError("Queries need a query_set and nonempty target list.")
        groups[row["query_set"]].append(index)
    if not groups:
        raise ValueError("No queries found.")
    selected = []
    for position, name in enumerate(sorted(groups)):
        quota = count // len(groups) + (position < count % len(groups))
        if quota > len(groups[name]):
            raise ValueError("Not enough queries for an evenly stratified selection.")
        selected.extend(rng.sample(groups[name], quota))
    return sorted(selected)


def choose_papers(graph, targets, count, rng):
    available = set(graph)
    if not targets or targets - available:
        raise ValueError("Every selected query target must exist in the filtered corpus.")
    cited = ({cid for target in targets for cid in graph[target]} & available) - targets
    citing = {cid for cid, links in graph.items() if links & targets} - targets
    distractors = available - targets - cited - citing
    pools = [sorted(cited), sorted(citing), sorted(distractors)]
    if count > len(available) or count < len(targets) + 3 or not all(pools):
        raise ValueError("Paper budget must fit all targets and examples from all three pools.")
    for pool in pools:
        rng.shuffle(pool)
    selected = set(targets)
    while len(selected) < count:
        for pool in pools:
            while pool and pool[-1] in selected:
                pool.pop()
            if pool and len(selected) < count:
                selected.add(pool.pop())
    return selected, {
        "targets": sorted(targets),
        "cited_neighbors": sorted(selected & cited),
        "citing_neighbors": sorted(selected & citing),
        "distractors": sorted(selected & distractors),
    }


def export_selection(source, destination, selected, *, queries=False):
    import pyarrow as pa
    from pyarrow import ipc

    count, found, offset = 0, set(), 0
    with pa.memory_map(str(source), "r") as stream:
        schema = ipc.open_stream(stream).schema.remove_metadata()
    with pa.OSFile(str(destination), "wb") as sink, ipc.new_stream(sink, schema) as writer:
        for batch in batches([source]):
            keys = (
                list(range(offset, offset + batch.num_rows))
                if queries
                else batch.column("corpusid").to_pylist()
            )
            indices = [i for i, key in enumerate(keys) if key in selected]
            chosen = [keys[i] for i in indices]
            if found.intersection(chosen) or len(chosen) != len(set(chosen)):
                raise ValueError("Duplicate selected IDs in the source.")
            found.update(chosen)
            if indices:
                writer.write_batch(
                    batch.take(pa.array(indices, type=pa.int64())).replace_schema_metadata(None)
                )
            count += len(indices)
            offset += batch.num_rows
    if found != selected:
        raise ValueError("Selected IDs are missing from an Arrow subset.")
    return count


def build_fixture(data_dir, *, count=300, seed=42):
    if not 200 <= count <= 500:
        raise ValueError("D-07 requires 200 to 500 papers.")
    source, destination = data_dir / "filtered" / "d03", data_dir / "fixtures" / "v0"
    if destination.exists():
        raise FileExistsError("Fixture v0 already exists; it will not be overwritten.")
    provenance = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if provenance.get("dataset") != DATASET or provenance.get("dataset_revision") != REVISION:
        raise ValueError("D-03 data must match the pinned LitSearch dataset revision.")
    queries = [row for batch in batches([source / "query.arrow"]) for row in batch.to_pylist()]
    rng = random.Random(seed)
    query_indices = choose_queries(queries, rng)
    targets = {cid for i in query_indices for cid in queries[i]["corpusids"]}
    graph = {}
    for batch in batches([source / "corpus_clean.arrow"]):
        for row in batch.select(["corpusid", "citations"]).to_pylist():
            cid, links = row["corpusid"], row["citations"]
            if type(cid) is not int or cid in graph or not isinstance(links, list):
                raise ValueError("Invalid or duplicate corpus ID, or missing citation list.")
            if any(type(link) is not int for link in links):
                raise ValueError("Citation IDs must be integers.")
            graph[cid] = set(links)
    if set(graph) != set(provenance["retained_corpusids"]):
        raise ValueError("Filtered corpus IDs differ from the D-03 manifest.")
    selected, groups = choose_papers(graph, targets, count, rng)
    manifest = {
        "fixture_version": 0,
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "filtered_manifest_sha256": canonical_digest(provenance),
        "seed": seed,
        "query_indices": query_indices,
        "corpusids": sorted(selected),
        "selection_groups": groups,
        "citation_count": sum(len(graph[cid] & selected) for cid in selected),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fixture-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "v0"
        stage.mkdir()
        for name in FILES:
            export_selection(
                source / name,
                stage / name,
                set(query_indices) if name == "query.arrow" else selected,
                queries=name == "query.arrow",
            )
        manifest["sha256"] = {name: digest(stage / name) for name in FILES}
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        stage.rename(destination)
    return destination, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = perf_counter()
    try:
        destination, manifest = build_fixture(data_directory(), count=args.papers, seed=args.seed)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Fixture selection failed: {error}\n")
    counts = {name: len(ids) for name, ids in manifest["selection_groups"].items()}
    print(f"Fixture: {destination}")
    print(
        f"Queries: 20; papers: {len(manifest['corpusids'])}; citations: {manifest['citation_count']}"
    )
    print(f"Selection groups (neighbors can overlap): {counts}")
    print(f"Manifest SHA-256: {digest(destination / 'manifest.json')}")
    print(f"Preparation: {perf_counter() - started:.2f}s. No database changes.")


if __name__ == "__main__":
    main()
