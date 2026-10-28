"""S-15: precision and recall of check_sentence on answer sentences with manipulations (D070).

Records the number check and the NLI scores of the real component for every case in
eval/guardrails/validator_cases.jsonl, chooses the entailment threshold on the tuning split only
(highest recall with at most 5 percent faithful sentences removed) and reports tuning and test at
that threshold and at the current default.

    uv run --group guardrails python scripts/evaluate_validator.py
    uv run python scripts/evaluate_validator.py --analyze-only   # replay without the model

Raw scores go to eval/results/validator.json, aggregates to docs/validator_evaluation.md.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from scilitrag.evaluation import validator_eval as ve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = PROJECT_ROOT / "eval" / "results" / "validator.json"
REPORT = PROJECT_ROOT / "docs" / "validator_evaluation.md"


def collect(settings) -> dict:
    from scilitrag.contracts import Evidence
    from scilitrag.evaluation.guardrail_sets import load_validator_cases
    from scilitrag.guardrails import validation

    nli = validation.NliModel(settings.nli_model)
    check = validation.build_component(settings)
    records, mismatches, latency = [], [], []
    for case in load_validator_cases():
        mapping = {
            p.label: Evidence(label=p.label, type=p.type, title=p.title, text=p.text, author_ids=[])
            for p in case.passages
        }
        labels = list(dict.fromkeys(int(n) for n in re.findall(r"\[\s*(\d+)\s*\]", case.sentence)))
        cited = [mapping[label] for label in labels]
        claim = " ".join(validation.LABEL.sub("", case.sentence).split())
        known = validation.numbers(" ".join(item.text for item in cited))
        best = None
        for item in cited:
            probs = nli.probabilities(item.text, claim)
            if best is None or probs["entailment"] >= best["entailment"]:
                best = probs
        record = {
            "id": case.id,
            "split": case.split,
            "manipulation": case.manipulation,
            "expected": case.expected,
            "number_ok": validation.numbers(claim) <= known,
            "nli": best,
        }
        records.append(record)
        started = time.perf_counter()
        real = check(case.sentence, mapping)
        latency.append((time.perf_counter() - started) * 1000)
        replayed = ve.predict_status(
            record, settings.nli_entailment_threshold, settings.nli_contradiction_threshold
        )
        if replayed != real.status:
            mismatches.append(case.id)
    latency.sort()
    return {
        "records": records,
        "meta": {
            "nli_model": settings.nli_model,
            "replay_mismatches": mismatches,
            "defaults": {
                "entailment": settings.nli_entailment_threshold,
                "contradiction": settings.nli_contradiction_threshold,
            },
            "latency_ms": {
                "median": round(latency[len(latency) // 2], 1),
                "p95": round(latency[min(int(0.95 * len(latency)), len(latency) - 1)], 1),
            },
        },
    }


def analyze(raw: dict) -> dict:
    defaults = raw["meta"]["defaults"]
    contradiction = defaults["contradiction"]
    tuning = [r for r in raw["records"] if r["split"] == "tuning"]
    test = [r for r in raw["records"] if r["split"] == "test"]
    chosen = ve.select_entailment_threshold(tuning, contradiction)
    thresholds = {"chosen": chosen, "default": defaults["entailment"]}
    subsets = {
        name: {
            f"{split}_{which}": ve.score_cases(pick(records), thresholds[which], contradiction)
            for split, records in (("tuning", tuning), ("test", test))
            for which in thresholds
        }
        for name, pick in (("easy", ve.easy_cases), ("hard", ve.hard_cases))
    }
    return {
        "frr_cap": ve.FRR_CAP,
        "chosen": {"entailment": chosen, "contradiction": contradiction},
        "defaults": defaults,
        "tuning_chosen": ve.score_cases(tuning, chosen, contradiction),
        "test_chosen": ve.score_cases(test, chosen, contradiction),
        "tuning_defaults": ve.score_cases(tuning, defaults["entailment"], contradiction),
        "test_defaults": ve.score_cases(test, defaults["entailment"], contradiction),
        "subsets": subsets,
    }


def pct(item: dict) -> str:
    if not item["total"]:
        return "n/a"
    low, high = item["ci95"]
    return (
        f"{100 * item['rate']:.0f} % ({item['hits']}/{item['total']}, "
        f"CI {100 * low:.0f} to {100 * high:.0f})"
    )


def table(title: str, scored: dict) -> list[str]:
    lines = [
        f"**{title}** (entailment threshold {scored['thresholds']['entailment']})",
        "",
        "| Measure | Value |",
        "| --- | --- |",
        f"| recall (manipulated sentences removed) | {pct(scored['recall'])} |",
        f"| precision (removed sentences that were manipulated) | {pct(scored['precision'])} |",
        f"| faithful sentences removed, all | {pct(scored['false_removal'])} |",
        (
            "| faithful sentences removed, without combined sentences | "
            f"{pct(scored['false_removal_single_source'])} |"
        ),
        f"| manipulated sentences caught by the number check alone | {pct(scored['removed_by_number_check'])} |",
        "",
        "| Kind | Removed |",
        "| --- | --- |",
    ]
    lines += [f"| {kind} | {pct(entry)} |" for kind, entry in scored["removed_by_kind"].items()]
    return lines + [""]


def subset_table(subsets: dict) -> list[str]:
    lines = [
        "| Cases | Split | Threshold | Recall | Precision | Faithful removed |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for name, scored in subsets.items():
        for split in ("test", "tuning"):
            for which in ("default", "chosen"):
                item = scored[f"{split}_{which}"]
                lines.append(
                    f"| {name} | {split} | {item['thresholds']['entailment']} ({which}) | "
                    f"{pct(item['recall'])} | {pct(item['precision'])} | "
                    f"{pct(item['false_removal'])} |"
                )
    return lines + [""]


def report(raw: dict, result: dict) -> str:
    meta = raw["meta"]
    counts = {s: sum(1 for r in raw["records"] if r["split"] == s) for s in ("tuning", "test")}
    chosen, defaults = result["chosen"], result["defaults"]
    lines = [
        "# Sentence validator evaluation (S-15)",
        "",
        (
            f"{counts['tuning']} tuning and {counts['test']} test cases from "
            "`eval/guardrails/validator_cases.jsonl` (own passages and answer sentences; faithful, "
            "paraphrased, and manipulated by a changed number or year, a negation, an added claim "
            f"or a citation of the wrong passage). NLI model `{meta['nli_model']}`. "
            "Generated by `scripts/evaluate_validator.py`."
        ),
        "",
        (
            "Rule: the entailment threshold is chosen on the tuning split only, as the highest "
            f"recall with at most {100 * result['frr_cap']:.0f} percent faithful sentences removed "
            "(sentences that need two passages do not enter the cap, see the notes). "
            "The contradiction threshold only changes the reported reason, not whether a sentence "
            "is removed. Intervals are 95 percent Wilson intervals; with about 50 cases per split "
            "they are wide."
        ),
        "",
        "| Threshold | Current default | Chosen on tuning |",
        "| --- | ---: | ---: |",
        f"| entailment | {defaults['entailment']} | {chosen['entailment']} |",
        "",
        *table("Test split, chosen threshold", result["test_chosen"]),
        *table("Test split, current default", result["test_defaults"]),
        *table("Tuning split, chosen threshold", result["tuning_chosen"]),
        "## Easy and hard cases",
        "",
        (
            "Easy cases: a new number or year, a negation, an added claim, a sentence of the other "
            "passage. Hard cases: numbers swapped inside one passage (the number check cannot "
            "catch them), a changed comparison or scope word, and true sentences that combine "
            "facts of both passages."
        ),
        "",
        *subset_table(result["subsets"]),
        "## Notes",
        "",
        (
            f"- Replay check: {len(meta['replay_mismatches'])} cases where the replayed status "
            "differs from the real `check_sentence` at the default thresholds"
            + (f" ({', '.join(meta['replay_mismatches'])})." if meta["replay_mismatches"] else ".")
        ),
        (
            f"- Latency of `check_sentence` per sentence on CPU with two cited passages: median "
            f"{meta['latency_ms']['median']} ms, p95 {meta['latency_ms']['p95']} ms."
        ),
        (
            "- A sentence that combines facts of two cited passages is judged against each passage "
            "alone (one supporting source is enough, D030), so such a sentence can be removed "
            "although it is true. The cases of kind `faithful_combined` show how often; they "
            "do not enter the cap of the threshold rule because no threshold keeps them."
        ),
        (
            "- The sentences are written by one person in the style of the answers, not produced "
            "by the language model. Real answers may be harder or easier. Treat the figures as an "
            "estimate for this set."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analyze-only", action="store_true", help="Reuse the recorded scores.")
    args = parser.parse_args()
    if args.analyze_only:
        raw = json.loads(RESULT.read_text(encoding="utf-8"))["raw"]
    else:
        load_dotenv(find_dotenv(usecwd=True))
        from scilitrag.config import load_settings

        raw = collect(load_settings({**os.environ, "USE_MOCKS": "1"}))
    result = analyze(raw)
    RESULT.write_text(
        json.dumps({"raw": raw, "result": result}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(report(raw, result), encoding="utf-8")
    print(json.dumps({"chosen": result["chosen"], "mismatches": raw["meta"]["replay_mismatches"]}))


if __name__ == "__main__":
    main()
