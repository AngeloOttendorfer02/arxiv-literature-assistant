import pytest

from scilitrag.rag.errors import RagUserError
from scilitrag.rag.llm.client import LLMClientError
from scilitrag.rag.planner import build_plan
from scilitrag.rag.search.filters import parse_filters


@pytest.mark.parametrize(
    "query,expected",
    [
        ("since 2020", (2020, None, None)),
        ("after 2020 before 2024", (2021, 2023, None)),
        ("between 2019 and 2022", (2019, 2022, None)),
        ("2018–2021", (2018, 2021, None)),
        ("in 2022", (2022, 2022, None)),
        ("top 7 papers seit 2020", (2020, None, 7)),
        ("nach 2019 bis 2023", (2020, 2023, None)),
        ("retrieval 512 tokens", (None, None, None)),
    ],
)
def test_explicit_constraints(query, expected):
    value = parse_filters(query)
    assert (value.year_from, value.year_to, value.k) == expected


@pytest.mark.parametrize("query", ["after 2024 before 2020", "top 30 papers", "top 3 and top 4"])
def test_conflicting_requests_require_clarification(query):
    with pytest.raises(RagUserError):
        parse_filters(query)


def test_fallback_preserves_years_and_explicit_limit():
    class Offline:
        def complete(self, *args, **kwargs):
            raise LLMClientError("offline")

    result = build_plan("top 4 retrieval papers since 2020", Offline())
    assert (result.scenario, result.year_from, result.k) == ("S1", 2020, 4)
