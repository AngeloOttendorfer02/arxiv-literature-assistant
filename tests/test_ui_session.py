from scilitrag.contracts import (
    Citation,
    DoneEvent,
    SentenceCheck,
    SentenceEvent,
    StatusEvent,
    TableEvent,
)
from scilitrag.ui.session import Answer, export_conversation


def test_export_includes_answers_sources_and_tables():
    sentence = SentenceEvent(
        text="The paper reports a result [1].",
        citations=[Citation(label=1, title="Example", year=2024, corpus_id=7)],
        check=SentenceCheck(status="ok", latency_ms=1),
    )
    table = TableEvent(
        query_id="q1",
        sql="SELECT 1",
        scope="Top authors",
        columns=["name", "count"],
        rows=[["A | B", 0], ["Two\nlines", None]],
    )
    answer = Answer(sentences=[sentence], tables=[table], done=DoneEvent(total_ms=10))

    result = export_conversation([("user", "Find papers"), ("assistant", answer)])

    assert "## Question\n\nFind papers" in result
    assert "The paper reports a result [1]." in result
    assert "- [1] Example (2024; corpus ID: 7)" in result
    assert "**Scope:** Top authors" in result
    assert "| A \\| B | 0 |" in result
    assert "| Two lines |  |" in result


def test_export_preserves_refusals_and_incomplete_errors():
    refused = Answer(
        statuses=[StatusEvent(stage="topic_gate", result="write_intent", latency_ms=1)],
        done=DoneEvent(total_ms=1, refused=True),
    )
    partial = Answer(error="The response was interrupted.")

    result = export_conversation([("assistant", refused), ("assistant", partial)])

    assert "only search and read papers" in result
    assert "**Response status:** The response was interrupted." in result
