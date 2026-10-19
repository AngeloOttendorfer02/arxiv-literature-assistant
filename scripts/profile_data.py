"""Profile the prepared D-01 Arrow cache offline; never download or clean data.

Local: uv run --locked --group data python scripts/profile_data.py
Docker (PowerShell, project root; replace <DATA_DIR> with your Windows data path):
    docker run --rm -it -v "${PWD}:/app" -v "<DATA_DIR>:/data" -v scilitrag-uv-cache:/root/.cache/uv -w /app -e DATA_DIR=/data -e UV_PROJECT_ENVIRONMENT=/opt/venv -e UV_LINK_MODE=copy ghcr.io/astral-sh/uv:python3.13-bookworm-slim uv run --locked --group data python scripts/profile_data.py

Writes DATA_DIR/profiling/d02_profile.json. Use --stdout to print JSON without
writing a report. Text lengths count Unicode characters, including surrounding
spaces, for nonblank strings only. Percentiles use linear interpolation.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

from dotenv import load_dotenv
from download_data import CONFIGS, DATASET, PROJECT_ROOT

REVISION = "9573fb284a1026c998df47024b888a163f0f0e25"
TEXT_FIELDS = ("title", "abstract", "full_paper")
SHORT_FULL_PAPER_LIMITS = (10, 100, 2000)


def empty_kind(value):
    if value is None:
        return "null"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, str):
        if value == "":
            return "empty_string"
        if not value.strip():
            return "whitespace_only"
    if isinstance(value, (list, dict)) and not value:
        return "empty_list" if isinstance(value, list) else "empty_object"
    return None


def length_stats(values):
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    low, high = math.floor(position), math.ceil(position)
    p95 = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": mean(ordered),
        "median": median(ordered),
        "p95": p95,
        "max": ordered[-1],
    }


def field_paths(features, prefix=""):
    for name, feature in features.items():
        path = f"{prefix}.{name}" if prefix else name
        yield path
        if isinstance(feature, dict) and "_type" not in feature:
            yield from field_paths(feature, path)


def field_value(row, path):
    value = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def profile_rows(rows, fields):
    missing = {field: Counter() for field in fields}
    lengths = {field: [] for field in TEXT_FIELDS if field in fields}
    invalid_text = Counter()
    ids = Counter()
    short_papers, short_without_title_abstract = Counter(), Counter()
    missing_ids = invalid_ids = row_count = 0
    for row in rows:
        row_count += 1
        full_paper = row.get("full_paper")
        if isinstance(full_paper, str) and not empty_kind(full_paper):
            no_title_abstract = empty_kind(row.get("title")) and empty_kind(row.get("abstract"))
            for limit in SHORT_FULL_PAPER_LIMITS:
                if len(full_paper) < limit:
                    short_papers[limit] += 1
                    short_without_title_abstract[limit] += bool(no_title_abstract)
        for field in fields:
            value = field_value(row, field)
            if kind := empty_kind(value):
                missing[field][kind] += 1
            if field in lengths and not kind:
                if isinstance(value, str):
                    lengths[field].append(len(value))
                else:
                    invalid_text[field] += 1
        if "corpusid" in fields:
            value = row.get("corpusid")
            if empty_kind(value):
                missing_ids += 1
            elif type(value) is int:
                ids[value] += 1
            else:
                invalid_ids += 1
    result = {
        "rows": row_count,
        "empty_fields": {
            field: {"total": sum(counts.values()), "by_kind": dict(counts)}
            for field, counts in missing.items()
        },
        "text_lengths_characters": {
            field: length_stats(values) for field, values in lengths.items()
        },
        "invalid_text_types": dict(invalid_text),
    }
    if "full_paper" in fields:
        # Nonblank full texts only; completely empty ones are in empty_fields.
        result["short_full_papers"] = {
            f"under_{limit}_characters": {
                "papers": short_papers[limit],
                "without_title_and_abstract": short_without_title_abstract[limit],
            }
            for limit in SHORT_FULL_PAPER_LIMITS
        }
    if "corpusid" in fields:
        result["corpusid"] = {
            "missing_rows": missing_ids,
            "invalid_type_rows": invalid_ids,
            "distinct_ids": len(ids),
            "duplicate_ids": sum(count > 1 for count in ids.values()),
            "duplicate_excess_rows": sum(count - 1 for count in ids.values()),
            "duplicates": {str(key): count for key, count in sorted(ids.items()) if count > 1},
        }
    if "year" in fields:
        result["missing_years"] = result["empty_fields"]["year"]["total"]
    return result, ids


def compare_ids(clean, s2orc):
    common = clean.keys() & s2orc.keys()
    left_only, right_only = sorted(clean.keys() - s2orc.keys()), sorted(s2orc.keys() - clean.keys())
    return {
        "common_distinct_ids": len(common),
        "inner_join_row_pairs": sum(clean[key] * s2orc[key] for key in common),
        "only_in_corpus_clean": {"count": len(left_only), "ids": left_only},
        "only_in_corpus_s2orc": {"count": len(right_only), "ids": right_only},
        "note": "Missing and invalid IDs are excluded; duplicate IDs multiply join row pairs.",
    }


def cached_subset(data_dir, subset):
    root = data_dir / "hf_datasets" / "princeton-nlp___lit_search" / subset
    candidates = list(root.glob(f"*/{REVISION}/dataset_info.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one prepared {subset} cache at revision {REVISION}; found "
            f"{len(candidates)}. Complete D-01 using this DATA_DIR first."
        )
    info_path = candidates[0]
    info = json.loads(info_path.read_text(encoding="utf-8"))
    split = info["splits"]["full"]
    prefix = f"{split.get('dataset_name', 'lit_search')}-full"
    shard_count = len(split.get("shard_lengths", [])) or 1
    names = (
        [f"{prefix}.arrow"]
        if shard_count == 1
        else [f"{prefix}-{i:05d}-of-{shard_count:05d}.arrow" for i in range(shard_count)]
    )
    files = [info_path.parent / name for name in names]
    if any(not path.is_file() for path in files):
        raise ValueError(f"Prepared Arrow shards are missing for {subset}. Complete D-01 first.")
    return info, files


def arrow_rows(files):
    import pyarrow as pa
    from pyarrow import ipc

    for path in files:
        with pa.memory_map(str(path), "r") as source, ipc.open_stream(source) as reader:
            for batch in reader:
                yield from batch.to_pylist()


def build_report(data_dir):
    caches = {subset: cached_subset(data_dir, subset) for subset in CONFIGS}
    report = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "split": "full",
        "length_convention": "Unicode characters in nonblank strings; surrounding spaces retained; "
        "p95 uses linear interpolation at (n-1)*0.95.",
        "empty_convention": "Null, NaN, empty strings, whitespace-only strings and empty lists/objects "
        "are counted separately per field; counts across nested fields overlap.",
        "subsets": {},
    }
    id_counts = {}
    for subset, (info, files) in caches.items():
        print(f"Profiling {subset} ({len(files)} Arrow files)...", file=sys.stderr)
        profile, ids = profile_rows(arrow_rows(files), list(field_paths(info["features"])))
        expected = info["splits"]["full"]["num_examples"]
        if profile["rows"] != expected:
            raise ValueError(
                f"{subset}: read {profile['rows']} rows, but cache metadata lists {expected}."
            )
        report["subsets"][subset] = profile
        id_counts[subset] = ids
    report["corpusid_join"] = compare_ids(id_counts["corpus_clean"], id_counts["corpus_s2orc"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stdout", action="store_true", help="Print JSON without writing a report."
    )
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    if not (configured := os.getenv("DATA_DIR")):
        parser.error("DATA_DIR is missing. Set it in .env or the environment.")
    data_dir = Path(configured).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    try:
        report = build_report(data_dir)
        encoded = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        if args.stdout:
            print(encoded, end="")
            return
        target = data_dir / "profiling" / "d02_profile.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")
        for subset, profile in report["subsets"].items():
            print(f"{subset}: {profile['rows']} rows")
        print(f"Join: {report['corpusid_join']['common_distinct_ids']} common distinct corpus IDs")
        print(f"Report: {target}")
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"Profiling failed: {exc}\n")


if __name__ == "__main__":
    main()
