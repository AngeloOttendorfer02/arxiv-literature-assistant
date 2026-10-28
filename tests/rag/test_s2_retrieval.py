import pytest

from scilitrag.contracts import Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.retrieval import retrieve_with
from scilitrag.rag.scenarios.citation_path import CITING_SQL, baseline_candidate
from scilitrag.rag.search.backend import SearchBackend

PLAN = Plan(scenario="S2", paper_ref="1", topic="retrieval", k=3)


class Linker:
    def paper_record(self, reference):
        assert reference == "1"
        return {"corpus_id": 1, "title": "BERT", "year": 2019}

    def author(self, reference):
        return 10


class Backend:
    linker = Linker()

    def reader(self, sql, params):
        assert sql == CITING_SQL
        assert params["target_id"] == 1
        return [
            {"corpus_id": 2, "title": "Retrieval A", "has_indexed_chunks": True},
            {"corpus_id": 3, "title": "Survey", "has_indexed_chunks": True},
        ]

    def chunks(self, plan, ids, per_paper=4):
        assert "BERT" in plan.topic
        assert ids == [2, 3]
        return [
            {
                "corpus_id": 2,
                "chunk_id": "2-0",
                "section": "Experiments",
                "text": "We use BERT as a baseline for retrieval.",
            },
            {
                "corpus_id": 3,
                "chunk_id": "3-0",
                "section": "Related work",
                "text": "BERT is discussed in related work.",
            },
        ]


def test_s2_routes_to_citation_edges_not_topic_search():
    evidence = retrieve_with(PLAN, Backend())
    assert evidence[0].section == "Citation target"
    assert {e.corpus_id for e in evidence if e.section == "Citing paper"} == {2, 3}
    assert [e.label for e in evidence] == list(range(1, len(evidence) + 1))


def test_s2_baseline_requires_target_specific_experimental_passage():
    evidence = retrieve_with(PLAN.model_copy(update={"require_baseline": True}), Backend())
    assert {e.corpus_id for e in evidence} == {1, 2}
    assert any(e.chunk_id == "2-0" for e in evidence)


@pytest.mark.parametrize(
    "text,section",
    [
        ("We cite BERT.", "Experiments"),
        ("We do not use BERT as a baseline.", "Experiments"),
        ("We use BERT as a baseline.", "Related work"),
        ("We use RoBERTa as a baseline.", "Experiments"),
        ("BERT is mentioned. We use RoBERTa as a baseline.", "Experiments"),
        ("Earlier studies compare their methods against BERT.", "Introduction"),
        ("BERT is a baseline in the literature.", "Experiments"),
        ("We could compare our method against BERT.", "Experiments"),
        ("Will we compare against BERT?", "Experiments"),
        ("We compare against BERT without running experiments.", "Experiments"),
    ],
)
def test_mentions_negations_and_other_models_are_not_baseline_candidates(text, section):
    assert not baseline_candidate(text, "BERT", section)


@pytest.mark.parametrize(
    "text",
    [
        "We use BERT as a baseline for retrieval.",
        "We compare our method against BERT on the retrieval task.",
        "BERT serves as our baseline.",
        "Our experiments used BERT as the baseline.",
    ],
)
def test_explicit_target_use_is_a_candidate(text):
    assert baseline_candidate(text, "BERT", "Experiments")


def test_baseline_without_indexed_support_is_controlled():
    class NoPassages(Backend):
        def chunks(self, *args, **kwargs):
            return []

    with pytest.raises(RagUserError, match="no explicit baseline-use candidate"):
        retrieve_with(PLAN.model_copy(update={"require_baseline": True}), NoPassages())


def test_missing_target_requests_clarification():
    with pytest.raises(RagUserError, match="Which paper"):
        retrieve_with(PLAN.model_copy(update={"paper_ref": None}), Backend())


def test_cross_paper_passage_is_rejected():
    class WrongScope(Backend):
        def chunks(self, *args, **kwargs):
            return [{"corpus_id": 99, "chunk_id": "wrong", "text": "unrelated"}]

    with pytest.raises(RagUserError, match="citing papers"):
        retrieve_with(PLAN, WrongScope())


def test_baseline_search_keeps_citation_and_filter_scope():
    class Scoped(Backend):
        def reader(self, sql, params):
            assert params == {
                "target_id": 1,
                "low": 2020,
                "high": 2023,
                "author_id": 10,
                "limit": 100,
            }
            return super().reader(sql, params)

    plan = PLAN.model_copy(
        update={"require_baseline": True, "year_from": 2020, "year_to": 2023, "author_ref": "Alice"}
    )
    evidence = retrieve_with(plan, Scoped())
    assert {e.corpus_id for e in evidence if e.section != "Citation target"} == {2}


def test_s2_real_citation_direction_years_and_author_filter(fixture_reader):
    backend = SearchBackend(fixture_reader, lambda _: [[1.0] + [0.0] * 767])
    evidence = retrieve_with(PLAN.model_copy(update={"year_from": 2022}), backend)
    assert {e.corpus_id for e in evidence if e.type == "paper"} == {1, 3, 5}
    with pytest.raises(RagUserError, match="No indexed papers cite"):
        retrieve_with(PLAN.model_copy(update={"year_to": 2020}), backend)
