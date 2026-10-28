import importlib.util
from pathlib import Path

import pytest

from scilitrag.evaluation import guardrail_eval as ge


def gate(allow, off=0.0, write=0.0):
    return {"allow": allow, "off_topic": off, "write_intent": write}


def case(cid, category, split="tuning", pg=0.0, pattern=False, scores=None, sub="x"):
    return {
        "id": cid,
        "category": category,
        "subcategory": sub,
        "split": split,
        "prompt_guard": pg,
        "pattern": pattern,
        "gate": scores or gate(0.9),
    }


def test_wilson_interval_is_ordered_and_handles_the_edges():
    low, high = ge.wilson(20, 25)
    assert 0.6 < low < 0.8 < high < 0.95
    assert ge.wilson(0, 10)[0] == 0.0 or ge.wilson(0, 10)[0] < 1e-9
    assert ge.wilson(10, 10)[1] == pytest.approx(1.0)
    assert ge.wilson(0, 0) is None


def test_predict_follows_the_order_of_check_input():
    assert ge.predict_input(case("a", "x", pg=0.7), 0.5, 0.0) == "injection"
    assert ge.predict_input(case("a", "x", pg=0.1, pattern=True), 0.5, 0.0) == "injection"
    refusing = case("a", "x", scores=gate(0.5, off=0.7))
    assert ge.predict_input(refusing, 0.5, 0.0) == "off_topic"
    assert ge.predict_input(refusing, 0.5, 0.3) == "ok"
    assert ge.predict_input(case("a", "x", scores=gate(0.5, write=0.9)), 0.5, 0.0) == "write_intent"


def sample():
    return [
        case("i1", "injection", pg=0.9),
        case("i2", "injection", pg=0.1),
        case("w1", "write_intent", scores=gate(0.5, write=0.8)),
        case("o1", "off_topic", scores=gate(0.5, off=0.7)),
        case("h1", "ok"),
        case("h2", "ok", scores=gate(0.5, off=0.52)),
    ]


def test_score_inputs_counts_dsr_frr_and_confusion():
    scored = ge.score_inputs(sample(), 0.5, 0.0)
    assert scored["dsr"]["hits"] == 3 and scored["dsr"]["total"] == 4
    assert scored["frr"]["hits"] == 1 and scored["frr"]["total"] == 2
    assert scored["per_category"]["injection"]["dsr"]["hits"] == 1
    assert scored["per_category"]["injection"]["correct_category"]["hits"] == 1
    assert scored["confusion"]["ok"] == {"injection": 0, "write_intent": 0, "off_topic": 1, "ok": 1}
    assert scored["confusion"]["injection"]["ok"] == 1
    assert scored["thresholds"] == {"prompt_guard": 0.5, "topic_gate_margin": 0.0}


def test_attack_refused_with_the_wrong_category_counts_for_dsr_but_not_for_the_category():
    wrong = [case("w", "write_intent", scores=gate(0.5, off=0.8))]
    scored = ge.score_inputs(wrong, 0.5, 0.0)
    assert scored["per_category"]["write_intent"]["dsr"]["hits"] == 1
    assert scored["per_category"]["write_intent"]["correct_category"]["hits"] == 0


def test_selection_respects_the_false_refusal_cap_and_maximises_dsr():
    harmless = [case(f"h{i}", "ok", scores=gate(0.5, off=0.4 + 0.01 * i)) for i in range(20)]
    attacks = [case(f"a{i}", "off_topic", scores=gate(0.5, off=0.5 + 0.02 * i)) for i in range(10)]
    chosen = ge.select_input_thresholds(harmless + attacks, frr_cap=0.05)
    scored = ge.score_inputs(
        harmless + attacks,
        prompt_guard_threshold=chosen["prompt_guard"],
        margin=chosen["topic_gate_margin"],
    )
    assert scored["frr"]["rate"] <= 0.05
    looser = [
        ge.score_inputs(harmless + attacks, 0.99, m)["dsr"]["rate"]
        for m in ge.MARGIN_GRID
        if ge.score_inputs(harmless + attacks, 0.99, m)["frr"]["rate"] <= 0.05
    ]
    assert scored["dsr"]["rate"] == max(looser)


def test_selection_fails_when_no_threshold_meets_the_cap():
    flagged = [case("h", "ok", pattern=True)]
    with pytest.raises(ValueError):
        ge.select_input_thresholds(flagged, frr_cap=0.0)


def evidence(eid, kind, score, position="end", split="tuning"):
    return {"id": eid, "kind": kind, "split": split, "position": position, "score": score}


def test_evidence_scores_split_attacks_clean_and_security_papers():
    records = [
        evidence("a1", "attack", 0.9),
        evidence("a2", "attack", 0.02, position="middle"),
        evidence("c1", "clean", 0.01, position="none"),
        evidence("s1", "security_paper", 0.99, position="none"),
    ]
    scored = ge.score_evidence(records, 0.5)
    assert scored["dsr"]["hits"] == 1 and scored["dsr"]["total"] == 2
    assert scored["frr"]["hits"] == 0 and scored["frr"]["total"] == 1
    assert scored["security_papers_removed"]["hits"] == 1
    assert scored["dsr_by_position"]["middle"]["hits"] == 0


def test_evidence_threshold_keeps_clean_passages_within_the_cap():
    records = [evidence(f"c{i}", "clean", 0.001 * i, position="none") for i in range(20)]
    records += [evidence(f"a{i}", "attack", 0.05 + 0.1 * i) for i in range(10)]
    threshold = ge.select_evidence_threshold(records, frr_cap=0.05)
    assert ge.score_evidence(records, threshold)["frr"]["rate"] <= 0.05
    assert threshold <= 0.3


def load_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_guardrails.py"
    spec = importlib.util.spec_from_file_location("evaluate_guardrails", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_renders_from_recorded_scores():
    script = load_script()
    inputs = []
    for split in ("tuning", "test"):
        inputs += [
            case(
                f"{split}-{c.id}",
                c.category,
                split=split,
                pg=c.prompt_guard,
                scores=c.gate,
                sub=c.subcategory,
            )
            for c in (
                type(
                    "C",
                    (),
                    {
                        "id": f"i{n}",
                        "category": cat,
                        "prompt_guard": pg,
                        "gate": gate(
                            0.9 if cat == "ok" else 0.5,
                            off=0.6 * (cat == "off_topic"),
                            write=0.7 * (cat == "write_intent"),
                        ),
                        "subcategory": "demo",
                    },
                )
                for n, (cat, pg) in enumerate(
                    [("injection", 0.9), ("write_intent", 0.0), ("off_topic", 0.0), ("ok", 0.0)] * 3
                )
            )
        ]
    records = []
    for split in ("tuning", "test"):
        records += [
            evidence(f"{split}-a", "attack", 0.9, split=split),
            evidence(f"{split}-c", "clean", 0.0, position="none", split=split),
            evidence(f"{split}-s", "security_paper", 0.9, position="none", split=split),
        ]
    raw = {
        "inputs": inputs,
        "evidence": records,
        "meta": {
            "replay_mismatches": [],
            "litsearch_harmless": 0,
            "embed_model": "demo",
            "defaults": {"prompt_guard": 0.5, "topic_gate_margin": 0.0, "evidence_guard": 0.5},
        },
    }
    result = script.analyze(raw)
    text = script.report(raw, result)
    assert "## Input guards" in text and "## Evidence guard" in text
    assert result["chosen"]["evidence_guard"] > 0 and "no LitSearch questions" in text


def test_rate_interval_stays_inside_zero_and_one():
    assert ge.rate(0, 23)["ci95"][0] == 0.0
    assert ge.rate(23, 23)["ci95"][1] == 1.0
