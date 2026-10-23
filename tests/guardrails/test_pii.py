import pytest

pytest.importorskip("presidio_analyzer")

from scilitrag.guardrails.pii import mask_pii


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Write to jane.doe@uni.edu now", "Write to <EMAIL> now"),
        ("Call +43 1 58801 12345 please", "Call <PHONE> please"),
        ("Card 4111 1111 1111 1111", "Card <CARD>"),
        ("IBAN DE89370400440532013000", "IBAN <IBAN>"),
        ("Server at 192.168.0.1", "Server at <IP>"),
    ],
)
def test_masks_pii(text, expected):
    assert mask_pii(text) == expected


def test_masks_every_occurrence():
    assert mask_pii("a@b.org and c@d.org") == "<EMAIL> and <EMAIL>"


@pytest.mark.parametrize(
    "text",
    [
        "Papers by Geoffrey Hinton on dropout since 2014",
        "BERT 110M reaches 0.95 F1 on SQuAD 2.0, see arXiv 1706.03762",
        "Which papers cite Attention Is All You Need between 2018 and 2020?",
        "",
    ],
)
def test_keeps_text_without_pii(text):
    assert mask_pii(text) == text
