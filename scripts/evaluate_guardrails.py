"""S-13: DSR, FRR and confusion matrix of the input guards and the evidence guard (D067).

Records the raw scores of the real components for every case in eval/guardrails, chooses the
thresholds on the tuning split only (highest DSR with at most 5 percent false refusals) and
reports tuning and test at those thresholds and at the current defaults.

    uv run --group guardrails --group embed --group data python scripts/evaluate_guardrails.py
    uv run python scripts/evaluate_guardrails.py --analyze-only   # replay without models

Harmless LitSearch questions (the test half of the D-22 split) join the test split when
DATA_DIR/filtered/d03/query.arrow exists. Raw scores go to eval/results/guardrails.json,
aggregates to docs/guardrail_evaluation.md.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from scilitrag.evaluation import guardrail_eval as ge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = PROJECT_ROOT / "eval" / "results" / "guardrails.json"
REPORT = PROJECT_ROOT / "docs" / "guardrail_evaluation.md"


def collect(settings) -> dict:
    from scilitrag.contracts import Evidence
    from scilitrag.data.embedding import build_component as build_embed
    from scilitrag.evaluation.guardrail_sets import load_evidence_cases, load_input_cases
    from scilitrag.guardrails import evidence as evidence_guard
    from scilitrag.guardrails import input as input_guard
    from scilitrag.guardrails.extraction import asks_for_instructions
    from scilitrag.guardrails.pii import mask_pii
    from scilitrag.guardrails.prompt_guard import shared_guard
    from scilitrag.guardrails.prototypes import litsearch_queries, load_prototypes
    from scilitrag.guardrails.topic import TopicGate

    embed = build_embed(settings)
    guard = shared_guard()
    gate = TopicGate(embed, load_prototypes(settings.data_dir))
    check_input = input_guard.build_component(settings, embed=embed)
    check_evidence = evidence_guard.build_component(settings)

    cases = [
        {
            "id": c.id,
            "text": c.text,
            "category": c.category,
            "subcategory": c.subcategory,
            "split": c.split,
        }
        for c in load_input_cases()
    ]
    litsearch = litsearch_queries(settings.data_dir, "test")
    cases += [
        {
            "id": f"LS-{i:03d}",
            "text": text,
            "category": "ok",
            "subcategory": "litsearch",
            "split": "test",
        }
        for i, text in enumerate(litsearch, 1)
    ]
    inputs, mismatches = [], []
    for case in cases:
        record = {key: case[key] for key in ("id", "category", "subcategory", "split")}
        record["prompt_guard"] = guard.score(case["text"])
        record["pattern"] = asks_for_instructions(case["text"])
        record["gate"] = gate.scores(mask_pii(case["text"]))
        inputs.append(record)
        replayed = ge.predict_input(
            record, settings.prompt_guard_threshold, settings.topic_gate_margin
        )
        if replayed != check_input(case["text"]).category:
            mismatches.append(case["id"])

    evidence = []
    for case in load_evidence_cases():
        item = Evidence(
            label=1, type=case.evidence_type, title=case.title, text=case.text, author_ids=[]
        )
        score = check_evidence([item]).scores.get(1, 0.0)
        evidence.append(
            {
                "id": case.id,
                "kind": case.kind,
                "split": case.split,
                "position": case.position,
                "score": score,
            }
        )
    return {
        "inputs": inputs,
        "evidence": evidence,
        "meta": {
            "replay_mismatches": mismatches,
            "litsearch_harmless": len(litsearch),
            "embed_model": settings.embed_model,
            "defaults": {
                "prompt_guard": settings.prompt_guard_threshold,
                "topic_gate_margin": settings.topic_gate_margin,
                "evidence_guard": settings.evidence_guard_threshold,
            },
        },
    }


def analyze(raw: dict) -> dict:
    defaults = raw["meta"]["defaults"]
    tuning_in = [r for r in raw["inputs"] if r["split"] == "tuning"]
    test_in = [r for r in raw["inputs"] if r["split"] == "test"]
    tuning_ev = [r for r in raw["evidence"] if r["split"] == "tuning"]
    test_ev = [r for r in raw["evidence"] if r["split"] == "test"]
    chosen = ge.select_input_thresholds(tuning_in)
    evidence_threshold = ge.select_evidence_threshold(tuning_ev)
    default_args = (defaults["prompt_guard"], defaults["topic_gate_margin"])
    chosen_args = (chosen["prompt_guard"], chosen["topic_gate_margin"])
    return {
        "frr_cap": ge.FRR_CAP,
        "chosen": {**chosen, "evidence_guard": evidence_threshold},
        "defaults": defaults,
        "inputs": {
            "tuning_chosen": ge.score_inputs(tuning_in, *chosen_args),
            "test_chosen": ge.score_inputs(test_in, *chosen_args),
            "tuning_defaults": ge.score_inputs(tuning_in, *default_args),
            "test_defaults": ge.score_inputs(test_in, *default_args),
        },
        "evidence": {
            "tuning_chosen": ge.score_evidence(tuning_ev, evidence_threshold),
            "test_chosen": ge.score_evidence(test_ev, evidence_threshold),
            "tuning_defaults": ge.score_evidence(tuning_ev, defaults["evidence_guard"]),
            "test_defaults": ge.score_evidence(test_ev, defaults["evidence_guard"]),
        },
    }


def pct(item: dict) -> str:
    if not item["total"]:
        return "n/a"
    low, high = item["ci95"]
    return (
        f"{100 * item['rate']:.0f} % ({item['hits']}/{item['total']}, "
        f"CI {100 * low:.0f} to {100 * high:.0f})"
    )


def input_table(title: str, scored: dict) -> list[str]:
    lines = [
        f"**{title}**",
        "",
        "| Group | DSR / FRR | Refused with the right category |",
        "| --- | --- | --- |",
    ]
    for category in ge.CATEGORIES:
        entry = scored["per_category"][category]
        lines.append(f"| {category} | {pct(entry['dsr'])} | {pct(entry['correct_category'])} |")
    lines.append(f"| all attacks | {pct(scored['dsr'])} | |")
    lines.append(f"| harmless questions (FRR) | {pct(scored['frr'])} | |")
    return lines + [""]


def confusion(scored: dict) -> list[str]:
    lines = [
        "| Expected / predicted | " + " | ".join(ge.PREDICTIONS) + " |",
        "| --- | " + " | ".join("---:" for _ in ge.PREDICTIONS) + " |",
    ]
    for expected, row in scored["confusion"].items():
        lines.append(f"| {expected} | " + " | ".join(str(row[p]) for p in ge.PREDICTIONS) + " |")
    return lines + [""]


def evidence_table(title: str, scored: dict) -> list[str]:
    lines = [
        f"**{title}** (threshold {scored['threshold']})",
        "",
        "| Group | Removed |",
        "| --- | --- |",
        f"| injected passages (DSR) | {pct(scored['dsr'])} |",
    ]
    for position, entry in scored["dsr_by_position"].items():
        lines.append(f"| injected at {position} | {pct(entry)} |")
    lines.append(f"| clean passages (FRR) | {pct(scored['frr'])} |")
    lines.append(f"| papers that discuss attacks | {pct(scored['security_papers_removed'])} |")
    return lines + [""]


def report(raw: dict, result: dict) -> str:
    meta, chosen = raw["meta"], result["chosen"]
    i, e = result["inputs"], result["evidence"]
    counts = {s: sum(1 for r in raw["inputs"] if r["split"] == s) for s in ("tuning", "test")}
    lines = [
        "# Guardrail evaluation (S-13)",
        "",
        "Test sets: `eval/guardrails` (own wording, split into tuning and test, see its README). "
        f"Input cases: {counts['tuning']} tuning, {counts['test']} test"
        + (
            f", of which {meta['litsearch_harmless']} harmless LitSearch questions (test half of "
            "the D-22 split)."
            if meta["litsearch_harmless"]
            else "; no LitSearch questions (DATA_DIR not available), harmless cases are our own."
        )
        + f" Embedding model `{meta['embed_model']}`. Generated by `scripts/evaluate_guardrails.py`.",
        "",
        (
            "Rule: thresholds are chosen on the tuning split only, as the highest DSR with at "
            f"most {100 * result['frr_cap']:.0f} percent false refusals (for the evidence guard: "
            "clean passages removed). The test split is read with the chosen values. Intervals "
            "are 95 percent Wilson intervals; with about 25 cases per group they are wide."
        ),
        "",
        "| Threshold | Current default | Chosen on tuning |",
        "| --- | ---: | ---: |",
        f"| Prompt Guard (query) | {result['defaults']['prompt_guard']} | {chosen['prompt_guard']} |",
        f"| Topic gate margin | {result['defaults']['topic_gate_margin']} | {chosen['topic_gate_margin']} |",
        f"| Evidence guard | {result['defaults']['evidence_guard']} | {chosen['evidence_guard']} |",
        "",
        "## Input guards (check_input)",
        "",
        *input_table("Test split, chosen thresholds", i["test_chosen"]),
        *input_table("Test split, current defaults", i["test_defaults"]),
        *input_table("Tuning split, chosen thresholds", i["tuning_chosen"]),
        "Confusion matrix, test split, chosen thresholds:",
        "",
        *confusion(i["test_chosen"]),
        (
            "Correct outcome by subcategory, test split, chosen thresholds "
            "(refused for attacks, allowed for harmless):"
        ),
        "",
        "| Subcategory | Correct |",
        "| --- | --- |",
        *[f"| {k} | {pct(v)} |" for k, v in i["test_chosen"]["correct_by_subcategory"].items()],
        "",
        "## Evidence guard (check_evidence)",
        "",
        *evidence_table("Test split, chosen threshold", e["test_chosen"]),
        *evidence_table("Test split, current default", e["test_defaults"]),
        *evidence_table("Tuning split, chosen threshold", e["tuning_chosen"]),
        "## Notes",
        "",
        f"- Replay check: {len(meta['replay_mismatches'])} cases where the replayed category "
        "differs from the real `check_input` at the default thresholds"
        + (f" ({', '.join(meta['replay_mismatches'])})." if meta["replay_mismatches"] else "."),
        (
            "- A threshold cannot fix attacks whose Prompt Guard score is near zero. The "
            "evidence guard scores papers that merely discuss attacks like real attacks, so they "
            "are reported separately and do not enter the cap."
        ),
        (
            "- The evidence figures use our own clean passages. A sample of real chunks "
            "(docs/evidence_real_sample.md, D069) scores higher than they do, so a low "
            "threshold chosen here would remove many real sources."
        ),
        (
            "- The cases are our own wording, written by one person, and the patterns of D066 "
            "were written after the tuning cases were known. Treat the figures as an estimate "
            "for this set, not as general detection rates."
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
    print(
        json.dumps(
            {"chosen": result["chosen"], "replay_mismatches": raw["meta"]["replay_mismatches"]}
        )
    )


if __name__ == "__main__":
    main()
