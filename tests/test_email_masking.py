"""D-11 e-mail masking checks with synthetic text, never real LitSearch."""

import json

import pytest
from email_masking import count_emails, mask_emails
from fulltext import clean_sections, fulltext_counts


def test_addresses_are_masked_and_counted():
    text = "Write to x_y@uni.edu or z.w+tag@lab.ac.uk, not to @handle or user@localhost."
    assert count_emails(text) == 2
    assert mask_emails(text) == "Write to <EMAIL> or <EMAIL>, not to @handle or user@localhost."
    assert count_emails(mask_emails(text)) == 0
    assert mask_emails(None) is None and count_emails(None) == 0


@pytest.mark.parametrize(
    ("text", "expected", "count"),
    [
        ("{alice,bob}@example.org", "<EMAIL>", 1),
        ("Contact {alice, bob, carol}@sub.example.org.", "Contact <EMAIL>.", 1),
        ("{ alice , bob+tag } @example.org", "<EMAIL>", 1),
        ("{alice,\nbob}@example.org", "<EMAIL>", 1),
        ("{alice|bob}@uni.edu", "<EMAIL>", 1),
        ("(alice | bob)@uni.edu", "<EMAIL>", 1),
        ("(alice,bob)@uni.edu", "<EMAIL>", 1),
        ("(alice, bob)@uni.edu", "<EMAIL>", 1),
        ("( alice,\nbob+tag ) @sub.uni.edu.", "<EMAIL>.", 1),
        ("{alice}@uni.edu", "<EMAIL>", 1),
        ("{ alice } @uni.edu", "<EMAIL>", 1),
        ("(alice)@uni.edu", "<EMAIL>", 1),
        ("(alice@uni.edu)", "(<EMAIL>)", 1),
        ("{alice}@uni.edu and (bob,carol)@lab.org", "<EMAIL> and <EMAIL>", 2),
        ("{x_y,z.w+tag}@lab.ac.uk; jane@uni.edu", "<EMAIL>; <EMAIL>", 2),
        ("Jane Doe: {alice,bob}@example.org", "Jane Doe: <EMAIL>", 1),
        ("{alice,bob}@localhost", "{alice,bob}@localhost", 0),
        ("(alice,bob)@localhost", "(alice,bob)@localhost", 0),
        ("(alice}@uni.edu", "(alice}@uni.edu", 0),
        ("{alice)@uni.edu", "{alice)@uni.edu", 0),
        ("Names {alice,bob} and @handle", "Names {alice,bob} and @handle", 0),
    ],
)
def test_grouped_addresses_are_masked_and_counted(text, expected, count):
    assert count_emails(text) == count
    assert mask_emails(text) == expected
    assert count_emails(expected) == 0
    assert mask_emails(expected) == expected


@pytest.mark.parametrize(
    ("heading_address", "body_address"),
    [
        ("jane@uni.edu", "jane.doe@uni.edu"),
        ("{alice,bob}@uni.edu", "{alice, bob}@uni.edu"),
        ("(alice,bob)@uni.edu", "{alice}@uni.edu"),
        ("{alice}@uni.edu", "(alice, bob)@uni.edu"),
        ("{alice|bob}@uni.edu", "(alice | bob)@uni.edu"),
    ],
)
def test_full_text_sections_are_masked_before_indexing(heading_address, body_address):
    heading = f"1 Data {heading_address}"
    text = f"{heading}\nContact: Jane <{body_address}>.\n"
    heading_end = len(heading)
    row = {
        "corpusid": 1,
        "content": {
            "text": text,
            "annotations": {
                "sectionheader": json.dumps(
                    [{"start": 2, "end": heading_end, "attributes": {"n": "1"}}]
                ),
                "paragraph": json.dumps([{"start": heading_end + 1, "end": len(text) - 1}]),
            },
        },
    }
    sections, removed = clean_sections(row)
    assert sections == [("1 Data <EMAIL>", "Contact: Jane <<EMAIL>>.")]
    assert removed["emails_masked"] == 2
    counts = fulltext_counts([row])
    assert counts["emails_masked"] == 2 and counts["emails_left"] == 0
