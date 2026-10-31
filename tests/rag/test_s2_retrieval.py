import pytest

from scilitrag.contracts import Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.retrieval import retrieve_with
from scilitrag.rag.scenarios.citation_path import CITING_SQL, baseline_candidate
from scilitrag.rag.search.backend import SearchBackend
from scilitrag.rag.search.entities import AUTHOR_YEAR_SQL

PLAN = Plan(scenario="S2", paper_ref="1", topic="retrieval", k=3)
DPR_ID = 215737187
DPR_TITLE = "Dense Passage Retrieval for Open-Domain Question Answering"


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


@pytest.mark.parametrize(
    "text",
    [
        "We use BERT as a baseline.",
        "We use BERT-base as our baseline.",
        "We use BERT-large as a baseline.",
    ],
)
@pytest.mark.parametrize("target_id", [None, 52967399])
def test_title_prefix_preserves_main_matching_for_hyphenated_bert_variants(text, target_id):
    title = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    assert baseline_candidate(text, title, "Experiments", target_id=target_id)


@pytest.mark.parametrize(
    "text",
    [
        "We use DPR as a baseline.",
        "We use DPR and BERT as the baseline.",
        "We use DPR for retrieval as its pretrained baseline.",
        "Our experiments used DPR as the pretrained baseline.",
        "Our comparisons with DPR use the same training data.",
        "Our comparison against DPR uses the same training data.",
        "We compare our method against DPR.",
        "DPR (Karpukhin et al., 2020) is our baseline.",
        "We use DPR (Karpukhin et al. (2020)) as a baseline.",
        "DPR [12] serves as our pretrained baseline.",
        "We use Karpukhin et al., 2020 as a baseline.",
        "We use Karpukhin et al. (2020) as a baseline.",
        "We use Karpukhin et al. 2020 as a baseline.",
        "We use Karpukhin  et al.,\n2020 as a baseline.",
        "We use dpr as a baseline.",
    ],
)
def test_reviewed_aliases_and_explicit_baseline_phrasings(text):
    assert baseline_candidate(text, DPR_TITLE, "Experiments", target_id=DPR_ID)


@pytest.mark.parametrize(
    "text",
    [
        "DPR was used as a baseline in our experiments.",
        "In our evaluation, DPR was used as the baseline.",
        "DPR was used as a pretrained baseline for our evaluation.",
        "During our study, DPR has been evaluated as a baseline.",
        "DPR is adopted as a baseline in our study.",
        "DPR was used as our baseline.",
        "Our method was compared with DPR.",
        "Our approach has been compared against DPR.",
        "DPR was compared with our model.",
        "In our evaluation, Karpukhin et al. (2020) was used as a baseline.",
        "DPR [12] was used as a baseline in our experiments.",
    ],
)
def test_passive_use_requires_directly_linked_own_study_context(text):
    assert baseline_candidate(text, DPR_TITLE, "Experiments", target_id=DPR_ID)


@pytest.mark.parametrize(
    "text,section",
    [
        ("DPR was used as a baseline.", "Experiments"),
        ("DPR was used as a baseline in earlier studies.", "Experiments"),
        ("In prior work, DPR was used as a baseline.", "Experiments"),
        ("The method was compared with DPR.", "Experiments"),
        ("Their method was compared with DPR.", "Experiments"),
        ("DPR was compared with their model.", "Experiments"),
        ("Our method was compared with DPR in earlier work.", "Experiments"),
        ("In prior work, our method was compared with DPR.", "Experiments"),
        ("DPR was used as our baseline in previous studies.", "Experiments"),
        ("Our method discusses studies where DPR was used as a baseline.", "Experiments"),
        (
            "In our evaluation, BM25 was used as a baseline, while DPR was used as a baseline in earlier work.",
            "Experiments",
        ),
        ("DPR was used as a baseline. In our experiments, we evaluated BM25.", "Experiments"),
        ("In our evaluation, DPR may be used as a baseline.", "Experiments"),
        ("DPR could be used as a baseline in our experiments.", "Experiments"),
        ("DPR was not used as a baseline in our experiments.", "Experiments"),
        ("Our method was never compared with DPR.", "Experiments"),
        ("Our method was compared with DPR without running experiments.", "Experiments"),
        ("DPR was used as a baseline in our experiments?", "Experiments"),
        ("DPR was used as a baseline in our experiments.", "Related work"),
        ("DPR-PAQ was used as a baseline in our experiments.", "Experiments"),
        ("ADPR was used as a baseline in our experiments.", "Experiments"),
    ],
)
def test_passive_mentions_do_not_borrow_context_from_other_clauses_or_studies(text, section):
    assert not baseline_candidate(text, DPR_TITLE, section, target_id=DPR_ID)


@pytest.mark.parametrize(
    "text,section",
    [
        ("DPR substantially outperforms BM25.", "Experiments"),
        ("In our experiments, DPR substantially outperforms BM25.", "Experiments"),
        ("We find that DPR significantly outperforms BM25.", "Experiments"),
        ("Our model outperforms DPR.", "Experiments"),
        ("DPR outperforms our method.", "Experiments"),
        ("DPR is a baseline in the literature.", "Experiments"),
        ("DPR (Karpukhin et al., 2020) is a baseline in earlier work.", "Experiments"),
        ("Earlier comparisons with DPR are promising.", "Experiments"),
        ("We use DPR as a baseline.", "Related work"),
        ("We use DPR as a baseline.", "References"),
        ("We do not use DPR as a baseline.", "Experiments"),
        ("We might use DPR as a baseline.", "Experiments"),
        ("Will we compare against DPR?", "Experiments"),
        ("Our comparisons with DPR would be useful.", "Experiments"),
        ("We use DPRv2 as a baseline.", "Experiments"),
        ("We use ADPR as a baseline.", "Experiments"),
        ("We use DPR-PAQ as a baseline.", "Experiments"),
        ("We use DPR-based rerankers as our baseline.", "Experiments"),
        ("We use pre-DPR as a baseline.", "Experiments"),
        ("DPR is mentioned. We use BM25 as a baseline.", "Experiments"),
        ("We use Karpukhin et al., 2021 as a baseline.", "Experiments"),
        ("We use Another et al., 2020 as a baseline.", "Experiments"),
        ("We use S2TARGETMENTION as a baseline; DPR is mentioned.", "Experiments"),
    ],
)
def test_aliases_do_not_admit_performance_claims_mentions_or_hypothetical_use(text, section):
    assert not baseline_candidate(text, DPR_TITLE, section, target_id=DPR_ID)


@pytest.mark.parametrize("target_id", [None, 999])
def test_short_alias_requires_the_reviewed_target_identity(target_id):
    assert not baseline_candidate(
        "We use DPR as a baseline.", DPR_TITLE, "Experiments", target_id=target_id
    )


def test_unregistered_titles_still_match_literally():
    assert baseline_candidate("We use Atlas as a baseline.", "Atlas: A retrieval model")
    assert baseline_candidate("We use A (B) as a baseline.", "A (B)")
    assert not baseline_candidate("We use A B as a baseline.", "A (B)")
    assert not baseline_candidate("We use anything as a baseline.", "")


@pytest.mark.parametrize("require_baseline", [True, False])
def test_full_title_dpr_search_uses_aliases_only_for_baselines_and_keeps_limits(require_baseline):
    class DprLinker:
        def paper_record(self, reference):
            assert reference == DPR_TITLE
            return {"corpus_id": DPR_ID, "title": DPR_TITLE, "year": 2020}

    class DprBackend:
        linker = DprLinker()

        def reader(self, sql, params):
            assert sql == CITING_SQL
            assert params["target_id"] == DPR_ID
            assert params["limit"] == (100 if require_baseline else PLAN.k)
            return [{"corpus_id": 2, "title": "Comparison", "has_indexed_chunks": True}]

        def chunks(self, plan, ids, per_paper):
            assert ids == [2] and per_paper == 4
            assert DPR_TITLE in plan.topic
            assert ("DPR" in plan.topic) == require_baseline
            assert ("Karpukhin et al., 2020" in plan.topic) == require_baseline
            return [
                {
                    "corpus_id": 2,
                    "chunk_id": "2-0",
                    "section": "Experiments",
                    "text": "DPR (Karpukhin et al., 2020) serves as our pretrained baseline.",
                }
            ]

    evidence = retrieve_with(
        PLAN.model_copy(update={"paper_ref": DPR_TITLE, "require_baseline": require_baseline}),
        DprBackend(),
    )
    assert {item.corpus_id for item in evidence} == {DPR_ID, 2}
    assert any(item.chunk_id == "2-0" for item in evidence)


@pytest.mark.parametrize(
    "reference",
    [DPR_TITLE, "DPR", "Karpukhin et al. (2020)", "DPR (Karpukhin et al., 2020)"],
)
def test_natural_dpr_references_flow_through_real_linker_to_baseline_filter(reference):
    target = {"corpus_id": DPR_ID, "title": DPR_TITLE, "year": 2020, "abstract": "Retrieval"}

    def reader(sql, params):
        if sql == CITING_SQL:
            assert params["target_id"] == DPR_ID and params["author_id"] is None
            return [{"corpus_id": 2, "title": "Comparison", "has_indexed_chunks": True}]
        if sql == AUTHOR_YEAR_SQL or "ANY(%(ids)s)" in sql or "year,abstract" in sql:
            return [target.copy()]
        if "lower(title)=" in sql and params["ref"] == DPR_TITLE:
            return [target.copy()]
        return []

    class AliasBackend(SearchBackend):
        def chunks(self, plan, ids, per_paper):
            assert ids == [2] and per_paper == 4
            assert "DPR" in plan.topic
            return [
                {
                    "corpus_id": 2,
                    "chunk_id": "2-0",
                    "section": "Experiments",
                    "text": "DPR was used as a baseline in our experiments.",
                }
            ]

    backend = AliasBackend(reader, lambda _: pytest.fail("Fake passages must not load a model"))
    evidence = retrieve_with(
        PLAN.model_copy(update={"paper_ref": reference, "require_baseline": True}), backend
    )
    assert {item.corpus_id for item in evidence} == {DPR_ID, 2}
    assert any(item.chunk_id == "2-0" for item in evidence)


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


def test_composite_baseline_identifies_the_target_retriever():
    text = (
        "Our baseline is a supervised QA model-DPR-BERT (Si et al., 2022) "
        "with a dense passage retriever (DPR; to feed passages to the reader)."
    )
    assert baseline_candidate(text, DPR_TITLE, "Evaluation setup", target_id=DPR_ID)


def test_named_target_based_comparison_in_baselines_section():
    text = "DPR Wiki directly uses the DPR trained on Wikipedia for retrieval."
    assert baseline_candidate(text, DPR_TITLE, "RACO vs. Baseline Methods", target_id=DPR_ID)
    assert not baseline_candidate(text, DPR_TITLE, "Related work", target_id=DPR_ID)
    assert not baseline_candidate(text, DPR_TITLE, "Introduction", target_id=DPR_ID)


def test_composite_baseline_does_not_match_an_unrelated_retriever():
    text = "Our baseline is BERT with a dense passage retriever (ANCE; not DPR)."
    assert not baseline_candidate(text, DPR_TITLE, "Evaluation setup", target_id=DPR_ID)
    assert not baseline_candidate(
        "Our baseline is not a model with a dense passage retriever (DPR; details).",
        DPR_TITLE, "Evaluation setup", target_id=DPR_ID,
    )


def test_without_finetuning_is_not_negated_baseline_use():
    text = "DPR Wiki directly uses the DPR trained on Wikipedia without any fine-tuning."
    assert baseline_candidate(text, DPR_TITLE, "Baseline Methods", target_id=DPR_ID)
    assert not baseline_candidate(
        "Our baseline works without using DPR and without fine-tuning.",
        DPR_TITLE, "Baseline Methods", target_id=DPR_ID,
    )
