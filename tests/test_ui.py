import pytest

pytest.importorskip("streamlit")  # needs --group ui

from scilitrag.contracts import (
    SENTENCE_REMOVED,
    Citation,
    DoneEvent,
    ErrorEvent,
    SentenceCheck,
    SentenceEvent,
    StatusEvent,
    TableEvent,
)
from scilitrag.ui.app import Answer, stage_badge

OK = SentenceCheck(status="ok", latency_ms=0)


def status(stage, result):
    return StatusEvent(stage=stage, result=result, latency_ms=1)


def sentence(*labels):
    citations = [Citation(label=n, title=f"Paper {n}") for n in labels]
    return SentenceEvent(text="x", citations=citations, check=OK)


def test_add_routes_events():
    answer = Answer()
    for event in [
        status("plan", "S1"),
        status("generate", f"{SENTENCE_REMOVED}no_citation"),
        sentence(1),
        DoneEvent(total_ms=1),
        ErrorEvent(code="internal", message="boom"),
    ]:
        answer.add(event)
    assert [s.stage for s in answer.statuses] == ["plan"]
    assert len(answer.removed) == 1 and len(answer.sentences) == 1
    assert answer.done is not None
    assert answer.error == "The API reported an error: boom"


@pytest.mark.parametrize(
    ("stage", "result", "expected"),
    [
        ("topic_gate", "write_intent", "only search and read papers"),
        ("prompt_guard", "injection", "change the assistant's instructions"),
        ("topic_gate", "off_topic", "outside the scope"),
        ("topic_gate", "something_new", "refused (something_new)"),
    ],
)
def test_refusal_message_by_reason(stage, result, expected):
    answer = Answer(statuses=[status(stage, result)], done=DoneEvent(total_ms=1, refused=True))
    assert expected in answer.refusal()


def test_no_refusal_without_refused_done():
    assert Answer(done=DoneEvent(total_ms=1)).refusal() is None


@pytest.mark.parametrize(("index", "scenario"), [(0, "S1"), (1, "S2"), (2, "S3")])
def test_example_questions_match_mock_scenarios(index, scenario):
    from scilitrag.mocks.rag import plan
    from scilitrag.ui.app import EXAMPLES

    assert plan(EXAMPLES[index][1]).scenario == scenario


def test_citations_are_unique_and_sorted():
    answer = Answer(sentences=[sentence(3, 1), sentence(1, 2)])
    assert [c.label for c in answer.citations()] == [1, 2, 3]


def test_stage_badge_colors():
    blocked = Answer(statuses=[status("topic_gate", "write_intent")], done=DoneEvent(total_ms=1))
    assert stage_badge(blocked, "topic_gate")[1] == "red"
    assert stage_badge(blocked, "plan") == ("skipped", "gray")

    masked = Answer(statuses=[status("pii", "masked 1")])
    assert stage_badge(masked, "pii")[1] == "green"
    assert stage_badge(masked, "plan") == ("waiting", "gray")

    removed = Answer(
        statuses=[status("generate", "4 checked, 1 removed")],
        removed=[status("generate", f"{SENTENCE_REMOVED}unsupported")],
    )
    assert stage_badge(removed, "generate")[1] == "orange"


def test_example_button_sends_its_question(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from scilitrag.ui.app import EXAMPLES

    # Nothing listens on port 9, so the request fails fast and the UI shows an error.
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file("../src/scilitrag/ui/app.py", default_timeout=30).run()
    label, question = EXAMPLES[1]
    at.button(key=f"example-{label}").click().run()
    history = at.session_state["history"]
    assert history[0] == ("user", question)
    assert "not reachable" in history[1][1].error


def test_table_is_rendered_before_stream_ends(monkeypatch):
    from contextlib import nullcontext

    from scilitrag.ui import app

    order = []

    class Placeholder:
        def container(self):
            return nullcontext()

        def markdown(self, text):
            pass

    class Response:
        def __init__(self):
            self.headers = {}

        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(())

    table = TableEvent(columns=["name"], rows=[["Ada"]], query_id="q1", sql="SELECT 1")

    def events(_lines):
        yield "table", table
        order.append("stream finished")

    monkeypatch.setattr(app.st, "empty", Placeholder)
    monkeypatch.setattr(app.st, "dataframe", lambda *a, **k: order.append("table rendered"))
    monkeypatch.setattr(app.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(app, "render_guardrails", lambda answer: None)
    monkeypatch.setattr(app.httpx2, "stream", lambda *a, **k: nullcontext(Response()))
    monkeypatch.setattr(app, "iter_events", events)
    answer = app.stream_answer("question", Placeholder())
    assert order == ["table rendered", "stream finished"]
    assert answer.tables == [table]
