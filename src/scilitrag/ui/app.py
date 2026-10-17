"""Streamlit chat. Talks to the API over HTTP only; imports nothing but the contracts.

Run: uv run --group ui streamlit run src/scilitrag/ui/app.py
"""

import os
from dataclasses import dataclass, field

import httpx2
import streamlit as st
from dotenv import find_dotenv, load_dotenv

from scilitrag.contracts import (
    Citation,
    DoneEvent,
    ErrorEvent,
    SentenceEvent,
    StatusEvent,
    TableEvent,
)
from scilitrag.ui.sse import iter_events

load_dotenv(find_dotenv(usecwd=True))
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = httpx2.Timeout(10.0, read=120.0)

GUARD_STAGES = ("prompt_guard", "topic_gate", "pii")
STAGE_LABELS = {
    "prompt_guard": "Injection check",
    "topic_gate": "Topic check",
    "pii": "PII masking",
    "plan": "Query plan",
    "retrieve": "Retrieval",
    "sql": "SQL facts",
    "generate": "Answer check",
}
REFUSALS = {
    "write_intent": "This assistant can only search and read papers. Changing or deleting data "
    "is not possible.",
    "injection": "The request was blocked because it tries to change the assistant's instructions.",
    "off_topic": "The request is outside the scope of ML and NLP literature.",
}


@dataclass
class Answer:
    statuses: list[StatusEvent] = field(default_factory=list)
    sentences: list[SentenceEvent] = field(default_factory=list)
    tables: list[TableEvent] = field(default_factory=list)
    done: DoneEvent | None = None
    error: str | None = None

    def refusal(self) -> str | None:
        if not (self.done and self.done.refused):
            return None
        blocked = [s for s in self.statuses if s.stage in GUARD_STAGES and s.result != "ok"]
        reason = blocked[-1].result if blocked else ""
        return REFUSALS.get(reason, f"The request was refused ({reason or 'no reason given'}).")


def status_line(status: StatusEvent) -> str:
    return f"{STAGE_LABELS.get(status.stage, status.stage)}: {status.result} ({status.latency_ms:.0f} ms)"


def answer_text(answer: Answer) -> str:
    return " ".join(sentence.text for sentence in answer.sentences)


def citations(answer: Answer) -> list[Citation]:
    seen: dict[int, Citation] = {}
    for sentence in answer.sentences:
        for citation in sentence.citations:
            seen.setdefault(citation.label, citation)
    return [seen[label] for label in sorted(seen)]


def render_table(table: TableEvent) -> None:
    st.dataframe([dict(zip(table.columns, row)) for row in table.rows], hide_index=True)
    st.caption(f"SQL query `{table.query_id}`")


def render_details(answer: Answer) -> None:
    """Everything below the answer text: tables, sources, metrics."""
    for table in answer.tables:
        render_table(table)
    sources = citations(answer)
    if sources:
        st.markdown(
            "\n".join(
                f"- [{c.label}] {c.title or 'Untitled'}"
                + (f" ({c.year})" if c.year else "")
                + (f", corpus id {c.corpus_id}" if c.corpus_id else "")
                for c in sources
            )
        )
    if answer.done and not answer.done.refused:
        d = answer.done
        parts = [f"{d.sentences_checked} sentences checked, {d.sentences_removed} removed"]
        if d.ttft_user_ms is not None:
            parts.append(f"first sentence after {d.ttft_user_ms:.0f} ms")
        parts.append(f"total {d.total_ms:.0f} ms")
        st.caption(" · ".join(parts))


def render_answer(answer: Answer) -> None:
    with st.expander("Checks", expanded=False):
        for status in answer.statuses:
            st.write(status_line(status))
    if answer.error:
        st.error(answer.error)
    elif refusal := answer.refusal():
        st.warning(refusal)
    else:
        st.markdown(answer_text(answer) or "_No answer._")
    render_details(answer)


def stream_answer(query: str) -> Answer:
    """Send the query and show events live while they arrive."""
    answer = Answer()
    checks = st.status("Checking the request", expanded=False)
    text = st.empty()
    try:
        url = f"{API_URL}/query"
        with httpx2.stream("POST", url, json={"query": query}, timeout=TIMEOUT) as response:
            response.raise_for_status()
            for _, event in iter_events(response.iter_lines()):
                if isinstance(event, StatusEvent):
                    answer.statuses.append(event)
                    checks.write(status_line(event))
                elif isinstance(event, SentenceEvent):
                    answer.sentences.append(event)
                    text.markdown(answer_text(answer))
                elif isinstance(event, TableEvent):
                    answer.tables.append(event)
                elif isinstance(event, DoneEvent):
                    answer.done = event
                elif isinstance(event, ErrorEvent):
                    answer.error = f"The API reported an error: {event.message}"
    except httpx2.HTTPStatusError as exc:
        answer.error = f"The API answered with HTTP {exc.response.status_code}."
    except httpx2.HTTPError:
        answer.error = f"The API at {API_URL} is not reachable."

    failed = answer.error is not None or answer.refusal() is not None
    checks.update(
        label="Request refused" if answer.refusal() else "Checks done",
        state="error" if failed else "complete",
    )
    return answer


def main() -> None:
    st.set_page_config(page_title="Literature Assistant", layout="wide")
    st.title("Literature Assistant")
    st.caption("Ask about ML and NLP papers. Every statement is backed by a cited source.")

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, content in st.session_state.history:
        with st.chat_message(role):
            if role == "user":
                st.markdown(content)
            else:
                render_answer(content)

    query = st.chat_input("Ask about papers, citations or authors")
    if not query:
        return

    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        answer = stream_answer(query)
    st.session_state.history.append(("assistant", answer))
    st.rerun()


main()
