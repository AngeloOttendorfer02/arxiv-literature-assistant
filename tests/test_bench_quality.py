import pytest

pytest.importorskip("numpy")

from score_quality import opens_with_not_found, per_query, score_answer, summarize

EVIDENCE = [
    {"label": 1, "type": "paper", "corpus_id": 10, "chunk_id": None, "text": "alpha"},
    {"label": 2, "type": "chunk", "corpus_id": 10, "chunk_id": "10-0", "text": "beta"},
    {"label": 3, "type": "chunk", "corpus_id": 20, "chunk_id": "20-0", "text": "gamma"},
]


def nli(supported):
    """Premise entails the claim when it contains every word listed for that claim."""

    def entails(premise, claim):
        return all(word in premise for word in supported.get(claim, ["never"]))

    return entails


def test_single_supported_citation_counts_for_recall_and_precision():
    score = score_answer("A uses alpha [1].", EVIDENCE, [10], nli({"A uses alpha .": ["alpha"]}))
    assert score["citation_recall"] == 1.0
    assert score["citation_precision"] == 1.0
    assert score["gold_cited"] is True


def test_unsupported_and_uncited_sentences_lower_recall():
    answer = "A uses alpha [1]. B is new. C is gamma [3]."
    score = score_answer(answer, EVIDENCE, [10], nli({"A uses alpha .": ["alpha"]}))
    assert score["citation_recall"] == pytest.approx(1 / 3)
    assert score["citation_precision"] == pytest.approx(1 / 2)
    assert score["uncited_sentences"] == 1


def test_redundant_citation_is_imprecise_but_needed_ones_are_not():
    redundant = score_answer("X [1][3].", EVIDENCE, [], nli({"X .": ["alpha"]}))
    assert redundant["citation_recall"] == 1.0
    assert redundant["sentences"][0]["precise"] == [True, False]
    # Neither source alone entails, both together do: both are needed.
    joint = score_answer("Y [1][2].", EVIDENCE, [], nli({"Y .": ["alpha", "beta"]}))
    assert joint["sentences"][0]["precise"] == [True, True]


def test_unknown_or_malformed_labels_never_support():
    score = score_answer("Z [1][9]. W [1, 2].", EVIDENCE, [], nli({"Z .": ["alpha"]}))
    assert score["citation_recall"] == 0.0
    assert score["sentences"][0]["precise"] == [False, False]
    assert score["sentences"][1]["labels"] == []


def test_gold_paper_needs_a_target_in_the_evidence():
    assert score_answer("Q [3].", EVIDENCE, [10], nli({}))["gold_cited"] is False
    assert score_answer("Q [3].", EVIDENCE, [99], nli({}))["gold_cited"] is None


def test_label_only_tail_is_not_a_sentence():
    score = score_answer("A uses alpha [1]. [2]", EVIDENCE, [], nli({"A uses alpha .": ["alpha"]}))
    assert len(score["sentences"]) == 1


def test_repeats_are_averaged_and_compared_with_the_baseline():
    def run(model, id_, split, recall):
        return {
            "model": model,
            "id": id_,
            "split": split,
            "not_found": False,
            "citation_recall": recall,
            "citation_precision": None,
            "gold_cited": True,
        }

    scores = [
        run("bf16", "q1", "dev", 1.0),
        run("bf16", "q2", "test", 0.5),
        run("q4", "q1", "dev", 0.5),
        run("q4", "q1", "dev", 1.0),
        run("q4", "q2", "test", 0.5),
    ]
    queries = per_query(scores)
    assert queries["q4", "q1"]["citation_recall"] == 0.75
    summary = summarize(queries, ["bf16", "q4"], "bf16")
    assert summary["q4"]["all"]["citation_recall"] == pytest.approx(0.625)
    assert summary["q4"]["all"]["citation_recall_vs_baseline"]["delta"] == pytest.approx(-0.125)
    assert summary["q4"]["dev"]["citation_recall_n"] == 1
    assert summary["q4"]["all"]["citation_precision"] is None
    assert "citation_recall_vs_baseline" not in summary["bf16"]["all"]


@pytest.mark.parametrize(
    "first",
    [
        "The provided evidence does not contain a paper that uses an NLI model.",
        "None of the supplied papers addresses multilingual parsing.",
        "No paper in the evidence describes such a method.",
        "I could not find a matching paper in the sources.",
        "There is no evidence about this in the provided abstracts.",
        "Such a dataset is not present in the supplied passages.",
    ],
)
def test_not_found_openings_are_recognized(first):
    assert opens_with_not_found([first])


@pytest.mark.parametrize(
    "first",
    [
        "XtremeDistil compresses multilingual models with staged distillation.",
        "The paper does not use labeled data and still beats the baseline.",
        "No improvement over BM25 was found for short queries.",
    ],
)
def test_ordinary_openings_are_not_flagged(first):
    assert not opens_with_not_found([first])


def test_not_found_answers_leave_recall_and_precision_open_but_keep_gold():
    answer = "The evidence does not contain a matching paper [1]. Paper B studies gamma [3]."
    score = score_answer(answer, EVIDENCE, [10], nli({"Paper B studies gamma .": ["gamma"]}))
    assert score["not_found"] is True
    assert score["citation_recall"] is None and score["citation_precision"] is None
    assert score["gold_cited"] is True
    later = score_answer(
        "A uses alpha [1]. The sources do not mention beta [2].", EVIDENCE, [], nli({})
    )
    assert later["not_found"] is False and later["citation_recall"] == 0.0
