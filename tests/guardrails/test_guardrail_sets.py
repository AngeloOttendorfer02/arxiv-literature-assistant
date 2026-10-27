from collections import Counter

import pytest

from scilitrag.evaluation.guardrail_sets import load_evidence_cases, load_input_cases
from scilitrag.guardrails import prototypes

ATTACK_CATEGORIES = {"injection", "write_intent", "off_topic"}


@pytest.fixture(scope="module")
def inputs():
    return load_input_cases()


@pytest.fixture(scope="module")
def evidence():
    return load_evidence_cases()


def test_ids_are_unique_across_all_files(inputs, evidence):
    ids = [case.id for case in inputs] + [case.id for case in evidence]
    assert len(ids) == len(set(ids))


def test_input_attacks_cover_every_category_in_both_splits(inputs):
    counts = Counter((c.category, c.split) for c in inputs if c.category != "ok")
    for category in ATTACK_CATEGORIES:
        for split in ("tuning", "test"):
            assert counts[category, split] >= 10, (category, split)


def test_injection_has_override_exfiltration_and_jailbreak_in_both_splits(inputs):
    found = {(c.subcategory, c.split) for c in inputs if c.category == "injection"}
    for sub in ("direct_override", "exfiltration", "jailbreak"):
        assert {(sub, "tuning"), (sub, "test")} <= found


def test_harmless_questions_are_enough_and_split(inputs):
    harmless = [c for c in inputs if c.category == "ok"]
    assert len(harmless) >= 100
    assert {c.split for c in harmless} == {"tuning", "test"}
    assert {"german", "security_topic", "word_trap"} <= {c.subcategory for c in harmless}


def test_total_attacks_include_indirect_injection_and_reach_the_minimum(inputs, evidence):
    direct = sum(1 for c in inputs if c.category != "ok")
    indirect = sum(1 for c in evidence if c.kind == "attack")
    assert direct + indirect >= 100
    assert indirect >= 20


def test_no_text_appears_twice(inputs, evidence):
    texts = [c.text.casefold() for c in inputs] + [c.text.casefold() for c in evidence]
    assert len(texts) == len(set(texts))


def test_inputs_do_not_repeat_topic_gate_prototypes(inputs):
    known = {
        text.casefold() for items in prototypes.load_prototypes(None).values() for text in items
    }
    assert not [c.id for c in inputs if c.text.casefold() in known]


def test_evidence_pairs_share_the_split_and_the_attack_extends_the_clean_text(evidence):
    by_id = {c.id: c for c in evidence}
    attacks = [c for c in evidence if c.kind == "attack"]
    assert attacks
    for attack in attacks:
        clean = by_id[attack.pair]
        assert clean.kind == "clean" and clean.pair == attack.id
        assert clean.split == attack.split and clean.title == attack.title
        assert all(part in attack.text for part in clean.text.split(". "))
        assert len(attack.text) > len(clean.text)
        assert attack.expected == "removed" and clean.expected == "kept"


def test_evidence_covers_positions_and_security_papers_in_both_splits(evidence):
    assert {"start", "middle", "end"} <= {c.position for c in evidence if c.kind == "attack"}
    for kind in ("attack", "security_paper"):
        assert {c.split for c in evidence if c.kind == kind} == {"tuning", "test"}
    assert {c.evidence_type for c in evidence} >= {"chunk", "paper"}
