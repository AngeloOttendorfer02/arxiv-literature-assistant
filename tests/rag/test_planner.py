import json

import pytest

from scilitrag.components import get_components
from scilitrag.config import load_settings
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.llm.client import ChatClient, LLMClientError
from scilitrag.rag.planner import build_plan


def payload(**updates):
    value = {
        "scenario": "S2",
        "topic": "retrieval baseline",
        "year_from": None,
        "year_to": None,
        "paper_ref": "BERT",
        "author_ref": None,
        "require_baseline": True,
        "k": 3,
    }
    return json.dumps({**value, **updates})


class Client:
    def __init__(self, value):
        self.value = value

    def complete(self, *args, **kwargs):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def test_validated_plan_preserves_full_question():
    query = "Which papers cite BERT as an experimental baseline?"
    result = build_plan(query, Client(payload()))
    assert result.scenario == "S2" and result.require_baseline
    assert result.query == query and result.paper_ref == "BERT"


@pytest.mark.parametrize("reference", ["DPR", "BERT"])
@pytest.mark.parametrize(
    "template",
    [
        "Which papers use {reference} as a baseline?",
        "Which papers cite {reference} and use it as a baseline?",
        'Which papers compare against "{reference}" as a baseline?',
        "List papers that use {reference} as an experimental baseline.",
    ],
)
def test_explicit_baseline_request_is_resolved_before_the_llm(reference, template):
    query = template.format(reference=reference)
    result = build_plan(query, Client(AssertionError("LLM must not be called")))
    assert result.scenario == "S2" and result.require_baseline
    assert result.paper_ref == reference and result.query == query
    assert result.author_ref is None and result.k == 5


@pytest.mark.parametrize("reference", ["DPR", "BERT"])
def test_deterministic_baseline_route_preserves_explicit_filters(reference):
    query = f"Find top 10 papers since 2022 that use {reference} as a baseline until 2024."
    result = build_plan(query, Client(AssertionError("LLM must not be called")))
    assert result.scenario == "S2" and result.paper_ref == reference
    assert result.year_from == 2022 and result.year_to == 2024 and result.k == 10


@pytest.mark.parametrize(
    "query",
    [
        "What is DPR?",
        "Which papers discuss BERT baselines?",
        "Which papers do not use DPR as a baseline?",
        "Which papers use DPR-PAQ as a baseline?",
        "Which papers use DPRv2 as a baseline?",
        "Which papers use ADPR as a baseline?",
        "Which papers use BERTv2 as a baseline?",
        "Which papers use RoBERTa as a baseline?",
        "Which papers use DPR-based models as a baseline?",
        "Which papers use DPR as a baseline and BERT for reranking?",
        "Which papers compare against BM25 as a baseline?",
        "Which authors use DPR as a baseline?",
    ],
)
def test_other_questions_keep_model_planning(query):
    result = build_plan(
        query,
        Client(payload(scenario="S1", paper_ref=None, require_baseline=False)),
    )
    assert result.scenario == "S1" and result.paper_ref is None and result.k == 3


@pytest.mark.parametrize("reference", ["DPR", "BERT"])
def test_component_factory_routes_reviewed_baselines_without_llm_or_db(monkeypatch, reference):
    def forbidden(*args, **kwargs):
        raise AssertionError("Neither LLM nor DB must be needed for this question")

    monkeypatch.setattr(ChatClient, "complete", forbidden)
    monkeypatch.setattr("scilitrag.rag.search.database.DatabaseReader.__call__", forbidden)
    settings = load_settings({"USE_MOCKS": "1", "REAL_COMPONENTS": "plan"})
    result = get_components(settings).plan(
        f"Which papers cite {reference} and use it as a baseline?"
    )
    assert result.scenario == "S2" and result.paper_ref == reference and result.require_baseline


@pytest.mark.parametrize("scenario", ["S1", "S3"])
@pytest.mark.parametrize("topic", ["retrieval baseline", None])
@pytest.mark.parametrize(
    "reference", ["DPR", "Karpukhin et al. (2020)", "DPR (Karpukhin et al., 2020)"]
)
def test_target_baseline_request_overrides_incorrect_model_scenario(scenario, topic, reference):
    # This phrasing uses model planning, so test the post-model correction too.
    query = f"Identify top 10 papers since 2022 that use {reference} as a baseline."
    result = build_plan(
        query,
        Client(payload(scenario=scenario, topic=topic, paper_ref=reference, year_from=2020)),
    )
    assert result.scenario == "S2" and result.require_baseline
    assert result.paper_ref == reference and result.query == query
    assert result.year_from == 2022 and result.year_to is None and result.k == 10


@pytest.mark.parametrize(
    ("scenario", "paper_ref", "require_baseline"),
    [("S1", "DPR", False), ("S3", "DPR", False), ("S1", None, True)],
)
def test_other_requests_keep_model_scenario(scenario, paper_ref, require_baseline):
    result = build_plan(
        "retrieval question",
        Client(payload(scenario=scenario, paper_ref=paper_ref, require_baseline=require_baseline)),
    )
    assert result.scenario == scenario


@pytest.mark.parametrize(
    "reference", ["DPR", "Karpukhin et al. (2020)", "DPR (Karpukhin et al., 2020)"]
)
def test_target_citation_is_preserved_without_inventing_result_year_filters(reference):
    query = f"Which papers use {reference} as an experimental baseline?"
    result = build_plan(query, Client(payload(paper_ref=reference, year_from=2020, year_to=2020)))
    assert result.scenario == "S2" and result.require_baseline
    assert result.paper_ref == reference
    assert result.author_ref is None
    assert result.year_from is None and result.year_to is None


def test_citation_year_and_explicit_result_filter_are_independent():
    query = "Find top 10 papers since 2022 that use Karpukhin et al. (2020) as a baseline."
    result = build_plan(query, Client(payload(paper_ref="Karpukhin et al. (2020)", year_from=2020)))
    assert result.paper_ref == "Karpukhin et al. (2020)"
    assert result.year_from == 2022 and result.year_to is None and result.k == 10


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        payload(k="3"),
        payload(paper_ref=None),
        LLMClientError("offline", kind="unavailable"),
    ],
)
def test_invalid_or_unavailable_planning_recovers_to_s1(raw):
    result = build_plan("retrieval question", Client(raw))
    assert result.scenario == "S1" and result.paper_ref is None
    assert result.query == "retrieval question"


def test_invalid_input_is_rejected_before_inference():
    with pytest.raises(RagUserError):
        build_plan(" ", Client(AssertionError("called")))


def test_component_factory_uses_the_explicit_selected_model(monkeypatch):
    seen = []

    def complete(self, *args, **kwargs):
        seen.append(self.settings.llm_model)
        return payload(scenario="S1", paper_ref=None, require_baseline=False)

    monkeypatch.setattr(ChatClient, "complete", complete)
    settings = load_settings(
        {
            "USE_MOCKS": "1",
            "REAL_COMPONENTS": "plan",
            "LLM_MODEL": "selected-model",
            "LLM_BASE_URL": "http://model/v1",
        }
    )
    assert get_components(settings).plan("retrieval").scenario == "S1"
    assert seen == ["selected-model"]


@pytest.mark.parametrize("null_value", ["null", " NULL ", None])
def test_s3_stringified_null_references_are_not_author_filters(null_value):
    result = build_plan(
        "Rank up to 3 most cited authors for dense passage retrieval.",
        Client(
            payload(
                scenario="S3",
                topic="dense passage retrieval",
                paper_ref=null_value,
                author_ref=null_value,
                require_baseline=False,
            )
        ),
    )
    assert result.scenario == "S3"
    assert result.author_ref is None and result.paper_ref is None


def test_s3_explicit_author_filter_is_preserved():
    result = build_plan(
        "Rank authors named Yann LeCun for dense passage retrieval.",
        Client(
            payload(
                scenario="S3",
                topic="dense passage retrieval",
                paper_ref=None,
                author_ref="Yann LeCun",
                require_baseline=False,
            )
        ),
    )
    assert result.author_ref == "Yann LeCun"


def test_s2_stringified_null_paper_is_not_a_valid_target():
    result = build_plan(
        "Find papers that cite a paper.",
        Client(payload(paper_ref="null", author_ref=None, require_baseline=False)),
    )
    assert result.scenario == "S1" and result.paper_ref is None


def test_s2_does_not_use_unmentioned_target_acronym_as_author_filter():
    query = "Which papers cite Dense Passage Retrieval for Open-Domain Question Answering?"
    result = build_plan(
        query,
        Client(
            payload(
                paper_ref="DPR (Karpukhin et al., 2020)",
                author_ref="DPR",
                require_baseline=False,
            )
        ),
    )
    assert result.scenario == "S2"
    assert result.paper_ref == "DPR (Karpukhin et al., 2020)"
    assert result.author_ref is None


def test_s2_preserves_explicit_author_filter():
    query = "Which papers by Yann LeCun cite DPR?"
    result = build_plan(
        query,
        Client(
            payload(
                paper_ref="DPR (Karpukhin et al., 2020)",
                author_ref="Yann LeCun",
                require_baseline=False,
            )
        ),
    )
    assert result.author_ref == "Yann LeCun"


def test_s2_does_not_silently_drop_unrelated_unresolved_author():
    query = "Which papers cite Dense Passage Retrieval?"
    result = build_plan(
        query,
        Client(
            payload(
                paper_ref="DPR (Karpukhin et al., 2020)",
                author_ref="Jane Doe",
                require_baseline=False,
            )
        ),
    )
    assert result.author_ref == "Jane Doe"


def test_s1_topic_is_not_treated_as_a_specific_paper_reference():
    query = "Find up to 3 papers about dense passage retrieval published from 2020 to 2023."
    result = build_plan(
        query,
        Client(
            payload(
                scenario="S1",
                topic="dense passage retrieval",
                paper_ref="dense passage retrieval",
                author_ref=None,
                require_baseline=False,
            )
        ),
    )
    assert result.scenario == "S1" and result.paper_ref is None
    assert result.topic == "dense passage retrieval"
    assert result.year_from == 2020 and result.year_to == 2023


def test_s1_preserves_explicit_paper_reference():
    query = "Find the paper Dense Passage Retrieval for Open-Domain Question Answering."
    title = "Dense Passage Retrieval for Open-Domain Question Answering"
    result = build_plan(
        query,
        Client(
            payload(
                scenario="S1", topic=title, paper_ref=title, author_ref=None, require_baseline=False
            )
        ),
    )
    assert result.paper_ref == title
