import pytest

pytest.importorskip("streamlit")

from scilitrag.ui.answer_review import summarize


def test_review_counts_exclude_unreviewed_judgments():
    assert summarize([True, False, None]) == {
        "correct": 1,
        "reviewed": 2,
        "total": 3,
        "rate": 0.5,
    }
    assert summarize([None])["rate"] is None
