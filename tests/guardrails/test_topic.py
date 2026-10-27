import asyncio
import importlib.util
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from scilitrag.guardrails import prototypes
from scilitrag.guardrails.topic import TopicGate, decide

AXES = {"allow": 0, "off_topic": 1, "write_intent": 2}
CLASS_OF = {
    text: label for label, texts in prototypes.load_prototypes(None).items() for text in texts
}
KEYWORDS = {"delete": "write_intent", "insert": "write_intent", "recipe": "off_topic"}


def fake_embed(texts):
    """One axis per class; prototypes by their class, other texts by keyword, default allow."""
    vectors = []
    for text in texts:
        label = CLASS_OF.get(text) or next(
            (v for k, v in KEYWORDS.items() if k in text.lower()), "allow"
        )
        vector = [0.0, 0.0, 0.0]
        vector[AXES[label]] = 1.0
        vectors.append(vector)
    return vectors


def test_decide_refuses_only_when_a_refusing_class_wins_by_more_than_the_margin():
    scores = {"allow": 0.5, "off_topic": 0.6, "write_intent": 0.2}
    assert decide(scores, 0.0) == ("off_topic", pytest.approx(0.1))
    assert decide(scores, 0.2)[0] == "ok"
    assert decide({"allow": 0.5, "off_topic": 0.1, "write_intent": 0.4}, 0.0)[0] == "ok"


def test_write_intent_wins_over_off_topic_when_it_is_closer():
    scores = {"allow": 0.3, "off_topic": 0.4, "write_intent": 0.7}
    assert decide(scores, 0.0)[0] == "write_intent"


def test_gate_takes_the_best_prototype_per_class_and_embeds_them_once():
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return [[1.0, 0.0] if "ml" in t else [0.0, 1.0] for t in texts]

    gate = TopicGate(
        embed, {"allow": ["ml a", "ml b"], "off_topic": ["pie"], "write_intent": ["rm"]}
    )
    first = gate.scores("ml question")
    gate.scores("pie question")
    assert first == {"allow": pytest.approx(1.0), "off_topic": 0.0, "write_intent": 0.0}
    assert calls == [4, 1, 1]


def test_scores_work_with_unnormalized_vectors():
    gate = TopicGate(
        lambda texts: [[3.0, 4.0] for _ in texts], {"allow": ["a"], "off_topic": ["b"]}
    )
    assert gate.scores("q")["allow"] == pytest.approx(1.0)


def test_builtin_prototypes_cover_all_classes_without_data():
    found = prototypes.load_prototypes(None)
    assert set(found) == {"allow", "off_topic", "write_intent"}
    assert all(len(texts) >= 10 for texts in found.values())
    assert found == prototypes.load_prototypes("missing-folder-for-test")


def write_queries(root: Path, queries, query_sets):
    pa = pytest.importorskip("pyarrow")
    ipc = pytest.importorskip("pyarrow.ipc")
    folder = root / "filtered" / "d03"
    folder.mkdir(parents=True)
    table = pa.table({"query": queries, "query_set": query_sets})
    with (
        pa.OSFile(str(folder / "query.arrow"), "wb") as sink,
        ipc.new_stream(sink, table.schema) as w,
    ):
        w.write_table(table)


def test_dev_queries_are_added_to_allow(tmp_path):
    queries = [f"query {i}" for i in range(20)]
    write_queries(tmp_path, queries, ["a", "b"] * 10)
    found = prototypes.load_prototypes(str(tmp_path))
    added = [t for t in found["allow"] if t.startswith("query ")]
    assert len(added) == 10
    assert found["off_topic"] == prototypes.OFF_TOPIC


def test_dev_split_matches_the_d22_script():
    pytest.importorskip("numpy")
    path = Path(__file__).resolve().parents[2] / "scripts" / "compare_embeddings.py"
    spec = importlib.util.spec_from_file_location("compare_embeddings", path)
    script = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(script)
    except ImportError:
        pytest.skip("script dependencies missing")
    import numpy as np

    sets = ["a"] * 7 + ["b"] * 9 + ["c"] * 4
    expected = [i for i, s in enumerate(script.split_queries(sets)) if s == "dev"]
    assert prototypes.dev_indices(sets, np) == expected


# Pipeline level: S4 and A3 are refused by the real input check and nothing downstream runs.
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("presidio_analyzer")

from scilitrag.api.pipeline import run_query
from scilitrag.components import mock_components
from scilitrag.config import ConfigError, load_settings
from scilitrag.guardrails import input as real_input
from scilitrag.guardrails.prompt_guard import PromptGuard


def settings(**env):
    return load_settings({"USE_MOCKS": "1", **env})


@pytest.fixture(autouse=True)
def quiet_guard(monkeypatch):
    monkeypatch.setattr(PromptGuard, "score", lambda self, text: 0.0)


def spied(components):
    calls = Counter()

    def spy(name, fn):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)

        return wrapper

    names = ("plan", "retrieve", "sql_facts", "check_evidence")
    wrapped = {name: spy(name, getattr(components, name)) for name in names}

    def generate(plan, evidence):
        calls["generate"] += 1
        return components.generate(plan, evidence)

    return replace(components, generate=generate, **wrapped), calls


def run(components, query):
    async def collect():
        return [event async for event in run_query(query, components)]

    return asyncio.run(collect())


@pytest.mark.parametrize(
    ("query", "category"),
    [
        ("Delete all papers by Yann LeCun from the database.", "write_intent"),
        ("What is a good recipe for apple pie?", "off_topic"),
    ],
)
def test_refused_before_any_database_or_model_call(query, category):
    check = real_input.build_component(settings(), embed=fake_embed)
    components, calls = spied(replace(mock_components(0), check_input=check))
    events = run(components, query)
    assert dict(calls) == {}
    names = [name for name, _ in events]
    assert names[-1] == "done" and events[-1][1].refused
    assert {"sentence", "table"}.isdisjoint(names)
    gate = [e for n, e in events if n == "status" and e.stage == "topic_gate"]
    assert [s.result for s in gate] == [category]


def test_allowed_query_reaches_the_pipeline():
    check = real_input.build_component(settings(), embed=fake_embed)
    components, calls = spied(replace(mock_components(0), check_input=check))
    events = run(components, "Which papers use dense retrieval?")
    assert not events[-1][1].refused
    assert calls["retrieve"] == 1 and calls["generate"] == 1


def test_gate_runs_after_prompt_guard_and_before_pii():
    check = real_input.build_component(settings(), embed=fake_embed)
    result = check("Which papers use dense retrieval? Mail me at jane@uni.edu")
    assert [s.stage for s in result.stages] == ["prompt_guard", "topic_gate", "pii"]
    assert result.allowed and "topic_gate" in result.scores


def test_margin_comes_from_settings():
    query = "What is a good recipe for apple pie?"
    assert not real_input.build_component(settings(), embed=fake_embed)(query).allowed
    strict = settings(TOPIC_GATE_MARGIN="1")
    assert real_input.build_component(strict, embed=fake_embed)(query).allowed


@pytest.mark.parametrize("value", ["abc", "1.5", "-2"])
def test_invalid_margin_is_rejected(value):
    with pytest.raises(ConfigError):
        settings(TOPIC_GATE_MARGIN=value)
