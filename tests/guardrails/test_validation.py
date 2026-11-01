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


def test_scoring_stops_at_the_first_supporting_source(probs):
    probs["values"] = lambda premise: (
        {"contradiction": 0.0, "entailment": 0.9, "neutral": 0.1}
        if premise == "good"
        else {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0}
    )
    assert check("Claim [1][2].", chunk(1, "good"), chunk(2, "bad")).status == "ok"
    assert [premise for premise, _ in probs["calls"]] == ["good"]


def fake_nli_server(seen, model="dleemiller/ModernCE-large-nli", drop_first=False):
    httpx = pytest.importorskip("httpx")

    def handler(request):
        if drop_first and not seen:
            seen.append(None)
            raise httpx.ConnectError("dropped")
        body = json.loads(request.content)
        seen.append((request.headers.get("authorization"), body))
        probs = [
            [0.0, 0.9, 0.1] if premise == "good" else [0.1, 0.1, 0.8]
            for premise, _ in body["pairs"]
        ]
        return httpx.Response(200, json={"model": model, "probabilities": probs})

    return httpx.MockTransport(handler)


REMOTE_NLI = validation.RemoteNli


def remote_check(monkeypatch, transport, sentence, *evidence):
    monkeypatch.setattr(validation, "RemoteNli", lambda *args: REMOTE_NLI(*args, transport))
    component = validation.build_component(settings(NLI_URL="http://gpu/nli", LLM_API_KEY="t"))
    return component(sentence, {item.label: item for item in evidence})


def test_remote_nli_sends_all_sources_of_a_sentence_in_one_request(monkeypatch):
    seen = []
    result = remote_check(
        monkeypatch, fake_nli_server(seen), "Claim [1][2].", chunk(1, "bad"), chunk(2, "good")
    )
    assert result.status == "ok" and result.nli_scores["entailment"] == 0.9
    ((auth, body),) = seen
    assert auth == "Bearer t"
    assert body["model"] == "dleemiller/ModernCE-large-nli"
    assert [premise for premise, _ in body["pairs"]] == ["bad", "good"]
    bad = remote_check(monkeypatch, fake_nli_server([]), "Claim [1].", chunk(1, "bad"))
    assert bad.status == "unsupported"


def test_remote_nli_retries_a_dropped_connection_once(monkeypatch):
    seen = []
    transport = fake_nli_server(seen, drop_first=True)
    assert remote_check(monkeypatch, transport, "Claim [1].", chunk(1, "good")).status == "ok"
    assert len(seen) == 2


def test_remote_nli_rejects_another_model(monkeypatch):
    transport = fake_nli_server([], model="dleemiller/EttinX-nli-s")
    with pytest.raises(RuntimeError, match="EttinX"):
        remote_check(monkeypatch, transport, "Claim [1].", chunk(1, "good"))


def test_sql_evidence_uses_only_the_number_check(probs):
    assert check("Hinton has 42 cited papers [2].", sql()).status == "ok"
    assert check("Hinton has 43 cited papers [2].", sql()).status == "fact_mismatch"
    assert probs["calls"] == []


def test_numbers_in_sql_text_are_not_evidence(probs):
    assert check("There are 7 authors [2].", sql()).status == "fact_mismatch"


def test_unknown_label(probs):
    assert check("Claim [5].", chunk()).status == "unknown_label"


def test_uncited_sentence_is_no_citation(probs):
    assert check("Transformers were invented by aliens.", chunk()).status == "no_citation"
    assert probs["calls"] == []


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


def abstract_only(label=1):
    return Evidence(
        label=label,
        type="paper",
        corpus_id=7,
        title="A paper",
        section="Abstract only",
        text="We study retrieval.",
    )


NOTICE = "Only abstract evidence is present in this context for this result [1]."


def test_code_written_abstract_notice_passes_without_nli(probs):
    probs["values"] = {"contradiction": 0.9, "entailment": 0.0, "neutral": 0.1}
    assert check(NOTICE, abstract_only()).status == "ok"
    assert probs["calls"] == []


def test_abstract_notice_is_only_exempt_for_an_abstract_only_source(probs):
    probs["values"] = {"contradiction": 0.9, "entailment": 0.0, "neutral": 0.1}
    full_text = Evidence(
        label=1, type="paper", corpus_id=7, section="Abstract", text="We study retrieval."
    )
    assert check(NOTICE, full_text).status == "fact_mismatch"
    assert check(NOTICE, chunk()).status == "fact_mismatch"


def test_changed_notice_text_is_still_checked(probs):
    probs["values"] = {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0}
    sentence = "Only abstract evidence is present and it proves the model is perfect [1]."
    assert check(sentence, abstract_only()).status == "unsupported"
    assert len(probs["calls"]) == 1


def test_combined_notice_passes_only_if_every_source_is_abstract_only(probs):
    probs["values"] = {"contradiction": 0.9, "entailment": 0.0, "neutral": 0.1}
    combined = "Only abstract evidence is present in this context for these results [1][2]."
    assert check(combined, abstract_only(1), abstract_only(2)).status == "ok"
    assert probs["calls"] == []
    assert check(combined, abstract_only(1), chunk(2)).status == "fact_mismatch"


def test_generator_and_check_share_the_notice_text():
    import asyncio

    from scilitrag.contracts import Plan
    from scilitrag.rag.generation import generate_with

    class Silent:
        settings = None

        async def stream(self, messages):
            return
            yield

    async def collect():
        return [
            t
            async for t in generate_with(
                Plan(scenario="S1", topic="x", k=3), [abstract_only()], Silent()
            )
        ]

    first = asyncio.run(collect())[0]
    assert check(first, abstract_only()).status == "ok"
