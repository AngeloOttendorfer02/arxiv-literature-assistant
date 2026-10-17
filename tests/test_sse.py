from scilitrag.contracts import DoneEvent, StatusEvent
from scilitrag.ui.sse import iter_events


def test_iter_events_parses_events_and_skips_comments():
    lines = [
        ": ping",
        "event: status",
        'data: {"stage":"plan","result":"S1","latency_ms":1.5}',
        "",
        "event: done",
        'data: {"total_ms":10.0}',
    ]
    events = list(iter_events(lines))
    assert [name for name, _ in events] == ["status", "done"]
    assert events[0][1] == StatusEvent(stage="plan", result="S1", latency_ms=1.5)
    assert isinstance(events[1][1], DoneEvent)


def test_iter_events_yields_while_streaming():
    def lines():
        yield "event: status"
        yield 'data: {"stage":"plan","result":"S1","latency_ms":0}'
        yield ""
        raise AssertionError("read past the first event")

    assert next(iter_events(lines()))[0] == "status"
