"""Source-bound sentence and citation worksheets for offline human review."""

import copy
import csv
from pathlib import Path

from scilitrag.contracts import Evidence
from scilitrag.evaluation.datasets import run_fingerprint
from scilitrag.rag.evidence.labels import citation_labels
from scilitrag.rag.evidence.provenance import canonical_json, digest, evidence_fingerprint

FIELDS = [
    "model",
    "case_id",
    "run_sha256",
    "sentence_index",
    "kind",
    "label",
    "sentence",
    "source_hash",
    "source_text",
    "supported",
    "year_exact",
    "count_exact",
    "note",
]
KEY = ["model", "case_id", "run_sha256", "sentence_index", "kind", "label"]


def _flag(value):
    return "" if value is None else str(value).lower()


def _parse(value):
    flags = {"": None, "true": True, "false": False}
    try:
        return flags[value.strip().lower()]
    except KeyError:
        raise ValueError("Review labels must be true, false or blank") from None


def review_rows(report):
    rows = []
    seen = set()
    for run in report["runs"]:
        identity = run.get("model"), run["id"]
        if identity in seen:
            raise ValueError("Each candidate/case pair must occur once in a report")
        seen.add(identity)
        fingerprint = run_fingerprint(run)
        mapping = {e["label"]: Evidence.model_validate(e) for e in run["evidence"]}
        sentences = [e["data"]["text"] for e in run["events"] if e["event"] == "sentence"]
        notes = run.get("review_notes", {})
        for index, sentence in enumerate(sentences):
            labels = citation_labels(sentence)
            sources = [mapping[label] for label in labels]
            base = {field: "" for field in FIELDS}
            base.update(
                model=run.get("model") or "",
                case_id=run["id"],
                run_sha256=fingerprint,
                sentence_index=str(index),
                sentence=sentence,
            )
            whole = {
                **base,
                "kind": "sentence",
                "source_hash": digest([evidence_fingerprint(e) for e in sources]),
                "source_text": canonical_json([e.model_dump() for e in sources]),
            }
            for field, target in (
                ("sentence_supported", "supported"),
                ("sentence_year_exact", "year_exact"),
                ("sentence_count_exact", "count_exact"),
            ):
                whole[target] = _flag((run.get(field) or [None] * len(sentences))[index])
            whole["note"] = notes.get(f"{index}:sentence:", "")
            rows.append(whole)
            judgments = run.get("citation_supported") or [{} for _ in sentences]
            for source in sources:
                label = str(source.label)
                rows.append(
                    {
                        **base,
                        "kind": "citation",
                        "label": label,
                        "source_hash": evidence_fingerprint(source),
                        "source_text": canonical_json(source.model_dump()),
                        "supported": _flag(judgments[index].get(label)),
                        "note": notes.get(f"{index}:citation:{label}", ""),
                    }
                )
    if not rows:
        raise ValueError("No emitted sentences are available for review")
    return rows


def export_reviews(report, path: Path):
    rows = review_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def import_reviews(report, path: Path):
    """Only judgments and notes may change; reject stale, missing or duplicate rows."""
    expected = {tuple(row[key] for key in KEY): row for row in review_rows(report)}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Keep the exported review columns unchanged")
        rows = list(reader)
    seen = set()
    for row in rows:
        key = tuple(row.get(field) for field in KEY)
        if key in seen or key not in expected:
            raise ValueError("Review rows are duplicated or do not match this source report")
        seen.add(key)
        editable = {"supported", "note"} | (
            {"year_exact", "count_exact"} if row["kind"] == "sentence" else set()
        )
        if set(row) != set(FIELDS) or any(
            row[field] != expected[key][field] for field in FIELDS if field not in editable
        ):
            raise ValueError("Review source content changed; edit only judgments and notes")
        for field in editable - {"note"}:
            _parse(row[field])
    if seen != set(expected):
        raise ValueError("Review file must retain every exported row")
    result = copy.deepcopy(report)
    by_run = {(r.get("model") or "", r["id"]): r for r in result["runs"]}
    for run in result["runs"]:
        sentences = [e["data"]["text"] for e in run["events"] if e["event"] == "sentence"]
        for field in ("sentence_supported", "sentence_year_exact", "sentence_count_exact"):
            run[field] = [None] * len(sentences)
        run["citation_supported"] = [{str(n): None for n in citation_labels(s)} for s in sentences]
        run["review_notes"] = {}
    for row in rows:
        run = by_run[row["model"], row["case_id"]]
        index = int(row["sentence_index"])
        if row["kind"] == "sentence":
            run["sentence_supported"][index] = _parse(row["supported"])
            run["sentence_year_exact"][index] = _parse(row["year_exact"])
            run["sentence_count_exact"][index] = _parse(row["count_exact"])
        else:
            run["citation_supported"][index][row["label"]] = _parse(row["supported"])
        run["review_notes"][f"{index}:{row['kind']}:{row['label']}"] = row["note"]
    return result
