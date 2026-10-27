"""D-19 passage sampling and question filters, without LitSearch data or a running model."""

import json
import random

import pytest
from build_chunk_eval_set import (
    ask,
    jaccard,
    longest_copied_run,
    pick_passage,
    rejection,
    sentence_spans,
    usable_sentence,
)

PASSAGE = (
    "We fine-tune the retriever on Natural Questions for three epochs. "
    "The hybrid model improves top-20 accuracy from 78.4 to 81.2 points."
)


def test_sentence_spans_point_into_the_section_text():
    text = "Intro words here et al. continue. Second sentence follows. Third"
    spans = sentence_spans(text)
    assert [text[a:b] for a, b in spans] == [
        "Intro words here et al. continue.",
        "Second sentence follows.",
        "Third",
    ]


def test_captions_noise_and_short_sentences_are_not_usable():
    assert usable_sentence("The model is trained with a contrastive loss on hard negatives.")
    assert not usable_sentence("Figure 3: Accuracy of the model on all eight benchmark sets.")
    assert not usable_sentence("Table 2 lists the results on the development set of SQuAD.")
    assert not usable_sentence("Too short to ask about.")
    assert not usable_sentence("0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 a b")
    assert not usable_sentence("Contact <EMAIL> for the code and the data of this project.")


def test_pick_passage_returns_a_span_within_word_limits_and_skips_acknowledgements():
    body = " ".join(f"Sentence {i} describes the method with several more words." for i in range(6))
    sections = [("7 Acknowledgments", body), ("2 Method", body)]
    index, start, end = pick_passage(sections, random.Random(1789))
    assert index == 1
    assert 15 <= len(body[start:end].split()) <= 60
    assert pick_passage(sections, random.Random(1789)) == (index, start, end)
    assert pick_passage([("1 A", "Too short.")], random.Random(1789)) is None


def test_copy_and_overlap_measures():
    assert longest_copied_run("which retriever on natural questions works", PASSAGE) == 4
    assert longest_copied_run("unrelated words only", PASSAGE) == 0
    assert jaccard(PASSAGE, PASSAGE) == 1.0
    assert jaccard("nothing shared", PASSAGE) == 0.0


@pytest.mark.parametrize(
    ("question", "usable", "reason"),
    [
        ("How much does hybrid retrieval raise top-20 accuracy on open-domain QA?", True, None),
        ("Which accuracy gain is reported?", False, "not_usable"),
        ("Accuracy gain?", True, "too_short"),
        ("What does this paper report about hybrid retrieval accuracy?", True, "self_reference"),
        ("Does the hybrid model improve top-20 accuracy from 78.4 to 81.2?", True, "copied"),
        ("Which model improves top-20 accuracy points on fine-tune retriever?", True, None),
    ],
)
def test_rejection_reasons(question, usable, reason):
    assert rejection(question, usable, PASSAGE) == reason


def test_ask_sends_schema_and_seed_and_parses_the_answer():
    httpx = pytest.importorskip("httpx")
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        content = json.dumps({"question": " Which gain? ", "usable": True})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        answer = ask(client, "http://model/v1/", "m", "Title", PASSAGE)
    assert answer == ("Which gain?", True)
    assert seen["seed"] == 1789 and seen["temperature"] == 0 and seen["model"] == "m"
    assert seen["response_format"]["json_schema"]["strict"] is True
    assert PASSAGE in seen["messages"][0]["content"]
