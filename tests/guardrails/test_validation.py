import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from scilitrag.config import ConfigError, load_settings
from scilitrag.contracts import Evidence
from scilitrag.guardrails import validation

CHUNK = (
    "Our retriever reaches an average nDCG@10 of 52.3 on BEIR, 4.1 points above BM25. "
    "Training takes 18 hours on eight A100 GPUs."
)


def settings(**env):
    return load_settings({"USE_MOCKS": "1", **env})


def chunk(label=1, text=CHUNK):
    return Evidence(label=label, type="chunk", text=text)


def sql(label=2):
    table = {"rows": [["Hinton", 42]], "parameters": {"year_from": 2018}, "sql": "SELECT 7"}
    return Evidence(label=label, type="sql", query_id="q1", text=json.dumps(table))


@pytest.fixture
def probs(monkeypatch):
    """Set the NLI output; calls records the premises that were scored."""
    state = {"values": {"contradiction": 0.0, "entailment": 1.0, "neutral": 0.0}, "calls": []}

    def fake(self, premise, hypothesis):
        state["calls"].append((premise, hypothesis))
        values = state["values"]
        return values(premise) if callable(values) else values

    monkeypatch.setattr(validation.NliModel, "probabilities", fake)
    return state


def check(sentence, *evidence, **env):
    mapping = {item.label: item for item in evidence}
    return validation.build_component(settings(**env))(sentence, mapping)


def test_supported_sentence_is_ok(probs):
    result = check("The model reaches an nDCG@10 of 52.3 on BEIR [1].", chunk())
    assert result.status == "ok"
    assert result.nli_scores["entailment"] == 1.0
    assert probs["calls"][0][1] == "The model reaches an nDCG@10 of 52.3 on BEIR ."


def test_changed_number_is_a_fact_mismatch_without_nli(probs):
    result = check("The model reaches an nDCG@10 of 61.0 on BEIR [1].", chunk())
    assert result.status == "fact_mismatch"
    assert probs["calls"] == []


def test_number_formats_match():
    assert validation.numbers("1,200 papers and 52,3 points") == {"1200", "52.3"}


def test_contradiction_is_a_fact_mismatch(probs):
    probs["values"] = {"contradiction": 0.9, "entailment": 0.05, "neutral": 0.05}
    assert check("It does not outperform BM25 [1].", chunk()).status == "fact_mismatch"


def test_neutral_is_unsupported(probs):
    probs["values"] = {"contradiction": 0.1, "entailment": 0.2, "neutral": 0.7}
    assert check("The authors work in Austria [1].", chunk()).status == "unsupported"


def test_one_supporting_source_is_enough(probs):
    probs["values"] = lambda premise: (
        {"contradiction": 0.0, "entailment": 1.0, "neutral": 0.0}
        if premise == "good"
        else {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0}
    )
    assert check("Claim [1][2].", chunk(1, "bad"), chunk(2, "good")).status == "ok"
    assert check("Claim [1][2].", chunk(1, "bad"), chunk(2, "bad")).status == "unsupported"


def test_sql_evidence_uses_only_the_number_check(probs):
    assert check("Hinton has 42 cited papers [2].", sql()).status == "ok"
    assert check("Hinton has 43 cited papers [2].", sql()).status == "fact_mismatch"
    assert probs["calls"] == []


def test_numbers_in_sql_text_are_not_evidence(probs):
    assert check("There are 7 authors [2].", sql()).status == "fact_mismatch"


def test_unknown_label(probs):
    assert check("Claim [5].", chunk()).status == "unknown_label"


def test_thresholds_come_from_settings(probs):
    probs["values"] = {"contradiction": 0.0, "entailment": 0.6, "neutral": 0.4}
    assert check("Claim [1].", chunk(), NLI_ENTAILMENT_THRESHOLD="0.5").status == "ok"
    assert check("Claim [1].", chunk(), NLI_ENTAILMENT_THRESHOLD="0.7").status == "unsupported"


def test_invalid_threshold_is_rejected():
    with pytest.raises(ConfigError):
        settings(NLI_CONTRADICTION_THRESHOLD="2")


@pytest.fixture(scope="module")
def model():
    nli = validation.NliModel(settings().nli_model)
    try:
        nli.probabilities("a", "b")
    except OSError:
        pytest.skip("NLI model not available offline")
    return nli


@pytest.mark.parametrize(
    ("hypothesis", "expected"),
    [
        ("The model reaches an average nDCG@10 of 52.3 on BEIR.", "entailment"),
        ("The model does not outperform BM25 on BEIR.", "contradiction"),
        ("The authors are affiliated with a university in Austria.", "neutral"),
    ],
)
def test_model_label_order(model, hypothesis, expected):
    probs = model.probabilities(CHUNK, hypothesis)
    assert max(probs, key=probs.get) == expected
