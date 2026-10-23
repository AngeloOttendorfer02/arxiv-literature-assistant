import pytest

from scilitrag.guardrails.sentences import split_sentences
from scilitrag.rag.sentence_stream import SentenceBuffer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("First claim [1]. Second claim [2].", ["First claim [1].", "Second claim [2]."]),
        ("Is it so? Yes it is! Done.", ["Is it so?", "Yes it is!", "Done."]),
        ("It reaches 52.3 on BEIR [1]. Next one.", ["It reaches 52.3 on BEIR [1].", "Next one."]),
        ("Smith et al. showed it [1]. Then more.", ["Smith et al. showed it [1].", "Then more."]),
        (
            "Proposed by Smith et al. [1]. Then more.",
            ["Proposed by Smith et al. [1].", "Then more."],
        ),
        ("Methods, e.g. BM25, work [1]. Next one.", ["Methods, e.g. BM25, work [1].", "Next one."]),
        ("See Fig. 3 and Eq. 5 [1]. Next one.", ["See Fig. 3 and Eq. 5 [1].", "Next one."]),
        ("Work by J. Devlin [1]. Next one.", ["Work by J. Devlin [1].", "Next one."]),
        ("Used in the U.S. market [1]. Next one.", ["Used in the U.S. market [1].", "Next one."]),
        ("A claim. [1] Another claim. [2]", ["A claim. [1]", "Another claim. [2]"]),
        ("It was said. [1][2]. Next one.", ["It was said. [1][2].", "Next one."]),
        ('Said "so." Then more.', ['Said "so."', "Then more."]),
        ("Uses BM25, dense, etc. Then more.", ["Uses BM25, dense, etc.", "Then more."]),
        (
            "Uses BM25, etc. and dense [1]. Next one.",
            ["Uses BM25, etc. and dense [1].", "Next one."],
        ),
        ("1. First item [1]. 2. Second item [2].", ["1. First item [1].", "2. Second item [2]."]),
        (
            "Results:\n- First item [1]\n- Second item [2]\n- Third item [3]",
            ["Results:", "- First item [1]", "- Second item [2]"],
        ),
        (
            "Results:\n1) First item [1].\n2) Second item [2].",
            ["Results:", "1) First item [1].", "2) Second item [2]."],
        ),
    ],
)
def test_split(text, expected):
    # The probe word makes the last boundary decidable; it stays in the rest.
    sentences, _ = split_sentences(text + " X")
    assert sentences[: len(expected)] == expected


def test_unknown_next_character_waits():
    for text in ("A claim [1]. ", "A claim [1]."):
        sentences, rest = split_sentences(text)
        assert sentences == [] and rest.strip() == "A claim [1]."


def test_lowercase_continuation_does_not_split():
    text = "Compare to vs. baselines and more"
    assert split_sentences(text) == ([], text)


def test_stream_token_by_token_matches_whole_text():
    text = "Smith et al. report 52.3 on BEIR [1]. A second claim, e.g. BM25 [2]. Last one [1]."
    buffer = SentenceBuffer(split_sentences)
    released = []
    for char in text:
        released += buffer.feed(char)
    released += buffer.finish()
    assert released == [
        "Smith et al. report 52.3 on BEIR [1].",
        "A second claim, e.g. BM25 [2].",
        "Last one [1].",
    ]
