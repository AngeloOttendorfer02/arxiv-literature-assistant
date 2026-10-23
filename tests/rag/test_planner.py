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
