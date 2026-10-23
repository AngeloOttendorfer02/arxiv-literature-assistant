"""Retrieval evaluation with explicit scope and separate human semantic judgments."""

from scilitrag.contracts import Evidence
from scilitrag.rag.evidence.index import EvidenceIndex
from scilitrag.rag.evidence.labels import citation_status


def score_run(case: dict, record: dict, *, expected_revision: str | None = None) -> dict:
    """Do not infer semantic support from a syntactically valid citation."""
    evidence = record.get("evidence", [])
    mapping = EvidenceIndex(Evidence.model_validate(item) for item in evidence).mapping()
    sentences = [item["data"] for item in record["events"] if item["event"] == "sentence"]
    checks = [citation_status(item["text"], mapping) == "ok" for item in sentences]
    done = [item["data"] for item in record["events"] if item["event"] == "done"]
    refused = any(item.get("refused", False) for item in done)
    failed = any(item["event"] == "error" for item in record["events"])
    has_table = any(
        item["event"] == "table" and item["data"].get("rows") for item in record["events"]
    )
    actual_revision = record.get("dataset_revision")
    reference_compatible = not record.get("dataset_changed", False) and (
        expected_revision is None or expected_revision == actual_revision
    )
    scores = {
        "completed": bool(done) and not refused and not failed and bool(sentences or has_table),
        "refused": refused,
        "dataset_revision_exact": (
            actual_revision == expected_revision and not record.get("dataset_changed", False)
            if expected_revision is not None and actual_revision is not None
            else None
        ),
        "scenario_exact": record.get("plan", {}).get("scenario") == case.get("scenario"),
        "sentences": len(sentences),
        "structural_citation_validity": sum(checks) / len(checks) if checks else None,
        "semantic_citation_precision": None,
        "semantic_review_coverage": 0.0,
    }
    # Add these manual labels to a saved run after reviewing each sentence and
    # all of its cited passages: true=supported, false=unsupported, null=unreviewed.
    judgments = record.get("sentence_supported", [])
    if judgments and len(judgments) != len(sentences):
        raise ValueError("sentence_supported must have one entry per emitted sentence")
    if any(value is not None and not isinstance(value, bool) for value in judgments):
        raise ValueError("Semantic judgments must be true, false or null")
    reviewed = [(index, value) for index, value in enumerate(judgments) if value is not None]
    if reviewed:
        scores["semantic_citation_precision"] = sum(
            value and checks[index] for index, value in reviewed
        ) / len(reviewed)
        scores["semantic_review_coverage"] = len(reviewed) / len(sentences)
    if case.get("expected_citing_ids") is not None:
        predicted = {
            item["corpus_id"]
            for item in evidence
            if item["type"] == "paper" and item.get("section") != "Citation target"
        }
        expected = set(case["expected_citing_ids"])
        scores["citing_precision"] = (
            len(predicted & expected) / len(predicted) if predicted else None
        )
        scores["citing_recall"] = len(predicted & expected) / len(expected) if expected else None
    if case.get("expected_years") is not None:
        expected = {int(key): value for key, value in case["expected_years"].items()}
        years = [
            item["year"] == expected[item["corpus_id"]]
            for item in evidence
            if item["type"] == "paper"
            and item["corpus_id"] in expected
            and item.get("year") is not None
            and expected[item["corpus_id"]] is not None
        ]
        scores["source_year_exact"] = sum(years) / len(years) if years else None
    if case.get("expected_author_counts") is not None:
        seeds = sorted({item["corpus_id"] for item in evidence if item["type"] == "paper"})
        scores["author_scope_exact"] = seeds == sorted(case["seed_ids"])
        tables = [item["data"] for item in record["events"] if item["event"] == "table"]
        counts = {}
        for table in tables:
            if {"author_id", "citation_count"} <= set(table["columns"]):
                for row in table["rows"]:
                    fields = dict(zip(table["columns"], row, strict=True))
                    counts[str(fields["author_id"])] = fields["citation_count"]
        scores["author_count_exact"] = (
            counts == case["expected_author_counts"]
            if scores["author_scope_exact"] and reference_compatible
            else None
        )
    if not reference_compatible:
        for metric in (
            "citing_precision",
            "citing_recall",
            "source_year_exact",
            "author_count_exact",
        ):
            if metric in scores:
                scores[metric] = None
    return scores


def score_baselines(annotations: list[dict]) -> dict:
    """Score only human-reviewed baseline candidates; never fill missing labels."""
    from scilitrag.rag.scenarios.citation_path import baseline_candidate

    reviewed = [item for item in annotations if item.get("baseline_use") is not None]
    if any(not isinstance(item["baseline_use"], bool) for item in reviewed):
        raise ValueError("baseline_use must be true, false or null")
    tp = fp = fn = tn = 0
    for item in reviewed:
        predicted = baseline_candidate(
            item["text"], item.get("target_title") or "", item.get("section")
        )
        truth = item["baseline_use"]
        tp += predicted and truth
        fp += predicted and not truth
        fn += not predicted and truth
        tn += not predicted and not truth
    return {
        "reviewed": len(reviewed),
        "total": len(annotations),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }
