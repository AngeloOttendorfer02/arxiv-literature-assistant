"""Apply D-03 quality filters, preserving every query target and the D-01 cache.

Local execution:
    uv run --locked --group data python scripts/filter_data.py

Writes Arrow IPC streams under DATA_DIR/filtered/d03 and two Markdown reports
under docs/. --report-only writes reports without exporting filtered datasets.
An existing filtered/d03 directory is never overwritten.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from download_data import CONFIGS, DATASET, PROJECT_ROOT
from profile_data import REVISION, cached_subset, empty_kind, field_value

RULES = {
    "empty_full_paper": "Empty/blank corpus_clean.full_paper",
    "full_paper_length_1": "Nonblank full_paper of exactly 1 Unicode character",
    "empty_title": "Empty/blank corpus_clean.title",
    "title_over_500": "Title longer than 500 Unicode characters",
    "missing_sectionheader": "No nonblank S2ORC sectionheader annotation",
    "missing_author_names": "Both S2ORC authorfirstname and authorlastname annotations absent",
}


def batches(files):
    import pyarrow as pa
    from pyarrow import ipc

    for path in files:
        with pa.memory_map(str(path), "r") as source, ipc.open_stream(source) as reader:
            yield from reader


def checked_rows(cache):
    info, files = cache
    count = 0
    for batch in batches(files):
        for row in batch.to_pylist():
            count += 1
            yield row
    if count != info["splits"]["full"]["num_examples"]:
        raise ValueError("Cache row count differs from dataset_info.json. Complete D-01 first.")


def normalize_year(value):
    if empty_kind(value):
        return None
    if type(value) is int or (type(value) is float and math.isfinite(value) and value.is_integer()):
        return int(value)
    raise ValueError(f"Invalid year value: {value!r}")


def annotation_present(row, field):
    payload = field_value(row, f"content.annotations.{field}")
    if empty_kind(payload):
        return False
    entries = json.loads(payload) if isinstance(payload, str) else payload
    if entries is None:
        return False
    if not isinstance(entries, list):
        raise TypeError(f"Paper {row['corpusid']}: {field} is not an annotation list.")
    text = field_value(row, "content.text")
    present = False
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"Paper {row['corpusid']}: invalid {field} annotation.")
        bounds = [entry.get("start"), entry.get("end")]
        bounds = [
            int(value)
            if isinstance(value, str) and value.isascii() and value.isdecimal()
            else value
            for value in bounds
        ]
        start, end = bounds
        if not (
            isinstance(text, str)
            and type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(text)
        ):
            raise ValueError(f"Paper {row['corpusid']}: invalid {field} text span.")
        present |= bool(text[start:end].strip())
    return present


def filter_reasons(row):
    reasons = []
    for field in ("title", "full_paper"):
        value = row.get(field)
        if empty_kind(value):
            reasons.append(f"empty_{field}")
        elif not isinstance(value, str):
            raise ValueError(f"Paper {row['corpusid']}: {field} is not text.")
        elif field == "title" and len(value) > 500:
            reasons.append("title_over_500")
        elif field == "full_paper" and len(value) == 1:
            reasons.append("full_paper_length_1")
    return reasons


def unique_id(row, seen):
    cid = row.get("corpusid")
    if type(cid) is not int or cid in seen:
        raise ValueError(f"Missing, invalid or duplicate corpusid: {cid!r}")
    seen.add(cid)
    return cid


def analyze(data_dir):
    caches = {name: cached_subset(data_dir, name) for name in CONFIGS}
    targets = set()
    query_count = 0
    for row in checked_rows(caches["query"]):
        query_count += 1
        ids = row.get("corpusids")
        if not isinstance(ids, list) or not ids or any(type(cid) is not int for cid in ids):
            raise ValueError("Every query must have a nonempty list of integer corpusids.")
        targets.update(ids)
    if not targets:
        raise ValueError("No query targets found; refusing to filter without target protection.")

    print("Checking S2ORC annotations and years...", file=sys.stderr)
    metadata, seen = {}, set()
    for row in checked_rows(caches["corpus_s2orc"]):
        cid = unique_id(row, seen)
        reasons = []
        if not annotation_present(row, "sectionheader"):
            reasons.append("missing_sectionheader")
        first = annotation_present(row, "authorfirstname")
        last = annotation_present(row, "authorlastname")
        if not first and not last:
            reasons.append("missing_author_names")
        metadata[cid] = {"year": normalize_year(row.get("year")), "reasons": reasons}

    print("Checking titles/full texts and protecting query targets...", file=sys.stderr)
    seen, retained, rejected, exceptions = set(), set(), {}, {}
    affected, dropped, protected = Counter(), Counter(), Counter()
    missing_years = Counter()
    for row in checked_rows(caches["corpus_clean"]):
        cid = unique_id(row, seen)
        if cid not in metadata:
            raise ValueError(f"Paper {cid} has no matching S2ORC row.")
        reasons = filter_reasons(row) + metadata[cid]["reasons"]
        target = cid in targets
        discard = bool(reasons) and not target
        affected.update(reasons)
        if discard:
            dropped.update(reasons)
        elif target:
            protected.update(reasons)
        if discard:
            rejected[cid] = reasons
        else:
            retained.add(cid)
        if metadata[cid]["year"] is None:
            missing_years["input"] += 1
            missing_years["rejected" if discard else "retained"] += 1
            if target:
                missing_years["targets"] += 1
        if target and (reasons or metadata[cid]["year"] is None):
            exceptions[cid] = reasons + (["missing_year"] if metadata[cid]["year"] is None else [])
    if seen != metadata.keys():
        raise ValueError("corpus_clean and corpus_s2orc do not have the same corpusids.")
    if missing := targets - retained:
        raise ValueError(f"Query target papers are missing: {sorted(missing)}")
    return {
        "caches": caches,
        "metadata": metadata,
        "retained": retained,
        "rejected": rejected,
        "target_exceptions": exceptions,
        "counts": {
            "input_papers": len(seen),
            "retained_papers": len(retained),
            "rejected_papers": len(rejected),
            "queries": query_count,
            "target_papers": len(targets),
            "retained_target_papers": len(targets & retained),
            "targets_with_rejection_criteria": sum(
                any(r in RULES for r in rs) for rs in exceptions.values()
            ),
        },
        "rules": {
            key: {
                "affected": affected[key],
                "rejected": dropped[key],
                "protected_targets": protected[key],
            }
            for key in RULES
        },
        "missing_years": {
            key: missing_years[key] for key in ("input", "retained", "rejected", "targets")
        },
    }


def export_subset(files, target, retained, metadata, subset):
    import pyarrow as pa
    from pyarrow import ipc

    with pa.memory_map(str(files[0]), "r") as source, ipc.open_stream(source) as reader:
        schema = reader.schema.remove_metadata()
    if subset != "query":
        schema = pa.schema(
            [field for field in schema if field.name != "year"] + [pa.field("year", pa.int64())]
        )
    count, written_ids = 0, set()
    with pa.OSFile(str(target), "wb") as sink, ipc.new_stream(sink, schema) as writer:
        for batch in batches(files):
            if subset != "query":
                ids = batch.column(batch.schema.get_field_index("corpusid")).to_pylist()
                indices = [i for i, cid in enumerate(ids) if cid in retained]
                batch = batch.take(pa.array(indices, type=pa.int64()))
                ids = [ids[i] for i in indices]
                written_ids.update(ids)
                if "year" in batch.schema.names:
                    batch = batch.select([name for name in batch.schema.names if name != "year"])
                batch = batch.append_column(
                    "year", pa.array([metadata[cid]["year"] for cid in ids], type=pa.int64())
                )
            writer.write_batch(batch.replace_schema_metadata(None))
            count += batch.num_rows
    if subset != "query" and (written_ids != retained or count != len(retained)):
        raise ValueError(f"{subset}: exported IDs/row counts do not match the retained papers.")
    return count


def export_filtered(plan, data_dir):
    parent = data_dir / "filtered"
    destination = parent / "d03"
    if destination.exists():
        raise FileExistsError(f"{destination} already exists. Use --report-only to analyze again.")
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".d03-", dir=parent) as temp:
        stage = Path(temp) / "d03"
        stage.mkdir()
        for subset, (_, files) in plan["caches"].items():
            print(f"Exporting {subset}...", file=sys.stderr)
            count = export_subset(
                files, stage / f"{subset}.arrow", plan["retained"], plan["metadata"], subset
            )
            if subset == "query" and count != plan["counts"]["queries"]:
                raise ValueError("Exported query count differs from the input.")
        manifest = {
            "dataset": DATASET,
            "dataset_revision": REVISION,
            "counts": plan["counts"],
            "rules": plan["rules"],
            "missing_years": plan["missing_years"],
            "retained_corpusids": sorted(plan["retained"]),
            "rejected_corpusids_and_reasons": plan["rejected"],
            "protected_target_exceptions": plan["target_exceptions"],
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if stage.resolve().parent.parent != parent.resolve():
            raise ValueError("Staging directory is outside the filtered-data directory.")
        stage.rename(destination)
    return destination


def write_reports(plan, docs_dir, exported):
    counts = plan["counts"]
    mode = (
        "Filtered Arrow files exported."
        if exported
        else "Analysis only; no filtered files exported."
    )
    heading = f"Dataset: `{DATASET}`, revision `{REVISION}`, split `full`.\n\n{mode}\n"
    lines = [
        "# D-03 Quality filters\n",
        heading,
        "\n## Summary\n",
        "| Measure | Papers / queries |",
        "|---|---:|",
    ]
    lines += [f"| {key} | {value:,} |" for key, value in counts.items()]
    lines += [
        "\n## Rejection rules\n",
        "| Criterion | Affected | Rejected | Protected targets |",
        "|---|---:|---:|---:|",
    ]
    for key, label in RULES.items():
        value = plan["rules"][key]
        lines.append(
            f"| {label} | {value['affected']:,} | {value['rejected']:,} | {value['protected_targets']:,} |"
        )
    lines += [
        "\nRule counts overlap. The summary counts each rejected paper once. All query targets are retained, even when a rejection criterion applies.",
        "\nLengths count Unicode characters including surrounding spaces. Blank text is counted as empty, not as a 1-character full text.",
        "\nAuthor absence means that both first-name and last-name annotations lack nonblank text spans. Having only one of these annotations does not reject the paper. No author normalization or person identification is performed.",
        "\n## Missing years\n",
        "Missing years do not reject papers. Output years are nullable integers: missing values become Arrow null (`None` in Python, SQL `NULL` at import).\n",
        "| Scope | Missing years |",
        "|---|---:|",
    ]
    for key, value in plan["missing_years"].items():
        label = "Rejected for other quality criteria" if key == "rejected" else key
        lines.append(f"| {label} | {value:,} |")
    lines += [
        "\n## Output and scope\n",
        "Outputs: `DATA_DIR/filtered/d03/{query,corpus_clean,corpus_s2orc}.arrow` (Arrow IPC streams) and `manifest.json`. Both corpus subsets contain the same retained IDs; queries and their target lists are preserved. Read an output with `pyarrow.ipc.open_stream(path).read_all()`.",
        "\nThe D-01 cache is unchanged. Existing filtered output is never overwritten. Title repair/truncation, email masking, author normalization and citation filtering are not performed. Citation IDs may point outside the retained corpus; D-09 must handle this explicitly.",
        "\nNonblank text can still contain extraction errors; these rules do not certify text quality.",
    ]
    details = [
        "# D-03 Protected target papers\n",
        heading,
        f"\nAll {counts['target_papers']:,} distinct target papers are retained. Listed below are targets with a rejection criterion or missing year. No replacement titles or author names are invented.\n",
        "| corpusid | Issues |",
        "|---|---|",
    ]
    details += [
        f"| {cid} | {', '.join(reasons)} |"
        for cid, reasons in sorted(plan["target_exceptions"].items())
    ]
    if not plan["target_exceptions"]:
        details.append("| None | No target exceptions |")
    lines += ["\n## Protected target papers\n", *details[2:]]
    docs_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in (
        ("d03_quality_filters.md", lines),
        ("d03_target_exceptions.md", details),
    ):
        (docs_dir / filename).write_text("\n".join(content) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Analyze and write Markdown reports without exporting filtered data.",
    )
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    if not (configured := os.getenv("DATA_DIR")):
        parser.error("DATA_DIR is missing. Set it in .env or the environment.")
    data_dir = Path(configured).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    try:
        if not args.report_only and (data_dir / "filtered" / "d03").exists():
            raise FileExistsError("DATA_DIR/filtered/d03 already exists; use --report-only.")
        plan = analyze(data_dir)
        if not args.report_only:
            print(f"Filtered data: {export_filtered(plan, data_dir)}")
        write_reports(plan, PROJECT_ROOT / "docs", exported=not args.report_only)
        print(json.dumps(plan["counts"], indent=2))
        print("Reports: docs/d03_quality_filters.md and docs/d03_target_exceptions.md")
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(1, f"Filtering failed: {exc}\n")


if __name__ == "__main__":
    main()
