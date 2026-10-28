"""Retrieval evaluation with explicit scope and separate human semantic judgments."""

import math
from statistics import mean, median

from scilitrag.contracts import Evidence
from scilitrag.rag.evidence.index import EvidenceIndex
from scilitrag.rag.evidence.labels import citation_labels, citation_status


def _judgments(record, field, size):
    values = record.get(field, [])
    if not isinstance(values, list) or values and len(values) != size:
        raise ValueError(f"{field} must have one entry per emitted sentence")
    if any(value is not None and not isinstance(value, bool) for value in values):
        raise ValueError("Semantic judgments must be true, false or null")
    return values


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
        "semantic_reviewed": 0,
    }
    # Add these manual labels to a saved run after reviewing each sentence and
    # all of its cited passages: true=supported, false=unsupported, null=unreviewed.
    judgments = _judgments(record, "sentence_supported", len(sentences))
    reviewed = [(index, value) for index, value in enumerate(judgments) if value is not None]
    if reviewed:
        scores["semantic_citation_precision"] = sum(
            value and checks[index] for index, value in reviewed
        ) / len(reviewed)
        scores["semantic_review_coverage"] = len(reviewed) / len(sentences)
        scores["semantic_reviewed"] = len(reviewed)
    # Per-reference judgments are distinct from whole-sentence support: one good
    # citation must not grant precision credit to an irrelevant co-citation.
    labels = [
        citation_labels(item["text"]) if valid else []
        for item, valid in zip(sentences, checks, strict=True)
    ]
    citation_reviews = record.get("citation_supported", [])
    if (
        not isinstance(citation_reviews, list)
        or citation_reviews
        and len(citation_reviews) != len(sentences)
    ):
        raise ValueError("citation_supported must have one mapping per emitted sentence")
    link_reviews = []
    for expected_labels, reviews in zip(labels, citation_reviews):
        if not isinstance(reviews, dict) or set(reviews) != {str(n) for n in expected_labels}:
            raise ValueError("Citation reviews must match the sentence's source labels")
        if any(v is not None and not isinstance(v, bool) for v in reviews.values()):
            raise ValueError("Citation judgments must be true, false or null")
        link_reviews.extend(v for v in reviews.values() if v is not None)
    scores["citations"] = sum(map(len, labels))
    scores["citation_reviewed"] = len(link_reviews)
    scores["citation_precision"] = mean(link_reviews) if link_reviews else None
    scores["citation_review_coverage"] = (
        len(link_reviews) / scores["citations"] if scores["citations"] else 0.0
    )
    for kind in ("year", "count"):
        values = _judgments(record, f"sentence_{kind}_exact", len(sentences))
        known = [v for v in values if v is not None]
        scores[f"answer_{kind}_exact"] = mean(known) if known else None
        scores[f"answer_{kind}_reviewed"] = len(known)
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
        actual = {
            item["corpus_id"]: item.get("year") for item in evidence if item["type"] == "paper"
        }
        years = [
            actual.get(paper_id) == year for paper_id, year in expected.items() if year is not None
        ]
        scores["source_year_exact"] = sum(years) / len(years) if years else None
        scores["source_year_compared"] = len(years) if reference_compatible else 0
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


def _distribution(values):
    values = sorted(
        v
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
    )
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "p95": values[math.ceil(len(values) * 0.95) - 1] if values else None,
    }


def summarize_runs(bundle, report):
    """Keep model/scenario coverage and unknown reviews visible, including failures."""
    models = report.get("candidate_models") or list(
        dict.fromkeys(r.get("model") for r in report["runs"])
    )
    summary, missing = [], []
    for model in models:
        for scenario in ("S1", "S2", "S3"):
            cases = [c["id"] for c in bundle["cases"] if c["scenario"] == scenario]
            if not cases:
                continue
            runs = [r for r in report["runs"] if r.get("model") == model and r["id"] in cases]
            absent = sorted(set(cases) - {r["id"] for r in runs})
            missing.extend({"model": model, "id": case, "scenario": scenario} for case in absent)
            done = [e["data"] for r in runs for e in r["events"] if e["event"] == "done"]
            row = {
                "model": model,
                "scenario": scenario,
                "expected_cases": len(cases),
                "runs": len(runs),
                "missing_case_ids": absent,
                "completed": sum(r["metrics"]["completed"] for r in runs),
                "scenario_correct": sum(r["metrics"]["scenario_exact"] for r in runs),
                "passed": sum(
                    r["metrics"]["completed"] and r["metrics"]["scenario_exact"] for r in runs
                ),
                "refused": sum(r["metrics"]["refused"] for r in runs),
                "failed": sum(not r["metrics"]["completed"] for r in runs),
                "elapsed_ms": _distribution(r.get("elapsed_ms") for r in runs),
                # Legacy API names include planning/retrieval; these are pipeline
                # observations, not engine-only TTFT or provider token throughput.
                "pipeline_first_delta_ms": _distribution(d.get("ttft_engine_ms") for d in done),
                "first_sentence_ms": _distribution(d.get("ttft_user_ms") for d in done),
                "sentences": sum(r["metrics"]["sentences"] for r in runs),
            }
            for metric, weight in (
                ("semantic_citation_precision", "semantic_reviewed"),
                ("citation_precision", "citation_reviewed"),
                ("answer_year_exact", "answer_year_reviewed"),
                ("answer_count_exact", "answer_count_reviewed"),
                ("source_year_exact", "source_year_compared"),
                ("author_count_exact", None),
            ):
                pairs = [
                    (r["metrics"][metric], r["metrics"].get(weight, 0) if weight else 1)
                    for r in runs
                    if r["metrics"].get(metric) is not None
                ]
                denominator = sum(w for _, w in pairs)
                row[metric] = (
                    sum(value * w for value, w in pairs) / denominator if denominator else None
                )
                row[metric + "_reviewed"] = denominator
            summary.append(row)
    return summary, missing


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
