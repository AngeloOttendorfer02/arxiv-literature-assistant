"""Streamlit chat. Talks to the API over HTTP only; imports nothing but the contracts.

Run: uv run --group ui streamlit run src/scilitrag/ui/app.py
"""

import os
from dataclasses import dataclass, field

import httpx2
import streamlit as st
from dotenv import find_dotenv, load_dotenv

from scilitrag.contracts import (
    SENTENCE_REMOVED,
    Citation,
    DoneEvent,
    ErrorEvent,
    SentenceEvent,
    StatusEvent,
    TableEvent,
)
from scilitrag.ui.baseline_review import render_baseline_review
from scilitrag.ui.citation_details import citation_details
from scilitrag.ui.sse import iter_events

load_dotenv(find_dotenv(usecwd=True))
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
# SSE requests may legitimately take longer than the API's default 120-second
# deadline. Keep the connection open while the server sends its keep-alives.
TIMEOUT = httpx2.Timeout(10.0, read=None)

GUARD_STAGES = ("prompt_guard", "topic_gate", "pii")
STAGE_LABELS = {
    "prompt_guard": "Injection check",
    "topic_gate": "Topic check",
    "pii": "PII masking",
    "plan": "Query plan",
    "retrieve": "Retrieval",
    "evidence_guard": "Source check",
    "sql": "SQL facts",
    "generate": "Answer check",
}
# One question per scenario (A-15); S4 is refused on purpose. Not copied from the topic-gate
# prototypes, so the refusal is not a lookup.
EXAMPLES = (
    (
        "S1 Topic search",
        "Which papers propose dense passage retrieval for open-domain question answering?",
    ),
    (
        "S2 Citation path",
        (
            "Which papers cite Dense Passage Retrieval for Open-Domain Question Answering "
            "and use it as a baseline?"
        ),
    ),
    ("S3 Author ranking", "Which authors are cited most in papers on dense passage retrieval?"),
    ("S4 Write request", "Please remove all papers by Geoffrey Hinton from the database."),
)
REFUSALS = {
    "write_intent": "This assistant can only search and read papers. Changing or deleting data "
    "is not possible.",
    "injection": "The request was blocked because it tries to change the assistant's instructions.",
    "off_topic": "The request is outside the scope of ML and NLP literature.",
}


@dataclass
class Answer:
    statuses: list[StatusEvent] = field(default_factory=list)
    removed: list[StatusEvent] = field(default_factory=list)
    sentences: list[SentenceEvent] = field(default_factory=list)
    tables: list[TableEvent] = field(default_factory=list)
    done: DoneEvent | None = None
    error: str | None = None
    request_id: str | None = None

    def add(self, event: object) -> None:
        match event:
            case StatusEvent(result=result) if result.startswith(SENTENCE_REMOVED):
                self.removed.append(event)
            case StatusEvent():
                self.statuses.append(event)
            case SentenceEvent():
                self.sentences.append(event)
            case TableEvent():
                self.tables.append(event)
            case DoneEvent():
                self.done = event
            case ErrorEvent(message=message):
                self.error = f"The API reported an error: {message}"

    def refusal(self) -> str | None:
        if not (self.done and self.done.refused):
            return None
        blocked = [s for s in self.statuses if s.stage in GUARD_STAGES and s.result != "ok"]
        reason = blocked[-1].result if blocked else ""
        return REFUSALS.get(reason, f"The request was refused ({reason or 'no reason given'}).")

    def citations(self) -> list[Citation]:
        seen: dict[int, Citation] = {}
        for sentence in self.sentences:
            for citation in sentence.citations:
                seen.setdefault(citation.label, citation)
        return [seen[label] for label in sorted(seen)]


def stage_badge(answer: Answer, stage: str) -> tuple[str, str]:
    """Text and color of one stage in the guardrail panel."""
    status = next((s for s in reversed(answer.statuses) if s.stage == stage), None)
    if status is None:
        finished = answer.done is not None or answer.error is not None
        return ("skipped", "gray") if finished else ("waiting", "gray")
    text = f"{status.result} · {status.latency_ms:.0f} ms"
    if stage in GUARD_STAGES and status.result != "ok" and not status.result.startswith("masked"):
        return text, "red"
    if stage == "evidence_guard" and status.result != "ok":
        return text, "orange"
    if stage == "generate" and answer.removed:
        return text, "orange"
    return text, "green"


def render_guardrails(answer: Answer | None) -> None:
    st.subheader("Guardrails")
    if answer is None:
        st.caption("Status of the last request appears here.")
        return
    for stage, label in STAGE_LABELS.items():
        row = st.container(horizontal=True, vertical_alignment="center")
        row.write(label)
        text, color = stage_badge(answer, stage)
        row.badge(text, color=color)
    checked = len(answer.sentences) + len(answer.removed)
    removed = len(answer.removed)
    st.caption(f"Sentences: {checked} checked, {removed} removed")
    for event in answer.removed:
        st.caption(f"Removed sentence: {event.result.removeprefix(SENTENCE_REMOVED)}")


def render_drawer(history: list) -> None:
    st.subheader("Source")
    selected = st.session_state.get("selected")
    if not selected:
        st.caption("Click a citation below an answer to inspect its source.")
        return
    index, label = selected
    answer = history[index][1]
    citation = next((c for c in answer.citations() if c.label == label), None)
    if citation is None:
        st.caption("Citation not found.")
        return
    st.markdown(f"**[{citation.label}] {citation.title or 'Untitled'}**")
    facts = [
        f"Year: {citation.year}" if citation.year else None,
        f"Corpus id: {citation.corpus_id}" if citation.corpus_id else None,
        f"Chunk id: {citation.chunk_id}" if citation.chunk_id else None,
        f"Section: {citation.section}" if citation.section else None,
    ]
    st.caption(" · ".join(f for f in facts if f))
    details = citation_details(citation, answer.tables)
    if citation.author_ids:
        st.caption("Author IDs: " + ", ".join(str(value) for value in citation.author_ids))
    if citation.source_hash:
        st.caption(f"Source fingerprint: {citation.source_hash[:16]}")
    if details["kind"] != "sql" and citation.text:
        st.text(citation.text)
    if citation.query_id:
        table = next((t for t in answer.tables if t.query_id == citation.query_id), None)
        st.caption(f"SQL query `{citation.query_id}`")
        if table:
            if table.scope:
                st.write(table.scope)
            if table.dataset_revision:
                st.caption(f"Dataset revision: {table.dataset_revision}")
            st.code(table.sql, language="sql")
            if table.parameters:
                st.json(table.parameters)


def select_citation(index: int, label: int) -> None:
    st.session_state.selected = (index, label)


def ask_example(question: str) -> None:
    st.session_state.example = question


def render_examples() -> None:
    row = st.container(horizontal=True)
    for label, question in EXAMPLES:
        row.button(
            label, key=f"example-{label}", help=question, on_click=ask_example, args=(question,)
        )


def answer_text(answer: Answer) -> str:
    return " ".join(sentence.text for sentence in answer.sentences)


def render_answer(answer: Answer, index: int) -> None:
    if answer.error:
        st.error(answer.error)
    if refusal := answer.refusal():
        st.warning(refusal)
        return
    st.markdown(answer_text(answer) or "_No answer._")
    for table in answer.tables:
        if table.scope:
            st.caption(table.scope)
        st.dataframe([dict(zip(table.columns, row)) for row in table.rows], hide_index=True)
        st.caption(f"SQL query `{table.query_id}`")
    citations = answer.citations()
    if citations:
        row = st.container(horizontal=True)
        for c in citations:
            title = c.title or "Untitled"
            row.button(
                f"[{c.label}] {title[:40]}",
                key=f"cite-{index}-{c.label}",
                help=title,
                on_click=select_citation,
                args=(index, c.label),
            )
    if answer.done:
        d = answer.done
        parts = [f"{d.sentences_checked} sentences checked, {d.sentences_removed} removed"]
        if d.ttft_user_ms is not None:
            parts.append(f"first sentence after {d.ttft_user_ms:.0f} ms")
        parts.append(f"total {d.total_ms:.0f} ms")
        if d.tokens_per_s is not None:
            parts.append(f"{d.tokens_per_s:.0f} tokens/s")
        if answer.request_id:
            parts.append(f"request {answer.request_id[:8]}")
        st.caption(" · ".join(parts))


def stream_answer(query: str, panel) -> Answer:
    """Send the query, show the answer and the guardrail panel live while events arrive."""
    answer = Answer()
    text = st.empty()
    tables = st.empty()
    try:
        url = f"{API_URL}/query"
        with httpx2.stream("POST", url, json={"query": query}, timeout=TIMEOUT) as response:
            response.raise_for_status()
            answer.request_id = response.headers.get("x-request-id")
            for _, event in iter_events(response.iter_lines()):
                answer.add(event)
                if isinstance(event, SentenceEvent):
                    text.markdown(answer_text(answer))
                if isinstance(event, TableEvent):
                    with tables.container():
                        for table in answer.tables:
                            if table.scope:
                                st.caption(table.scope)
                            st.dataframe(
                                [dict(zip(table.columns, row)) for row in table.rows],
                                hide_index=True,
                            )
                            st.caption(f"SQL query `{table.query_id}`")
                with panel.container():
                    render_guardrails(answer)
    except httpx2.HTTPStatusError as exc:
        answer.error = f"The API answered with HTTP {exc.response.status_code}."
    except httpx2.HTTPError:
        answer.error = f"The API at {API_URL} is not reachable."
    except (KeyError, ValueError):
        # Unknown event name (KeyError) or payload (pydantic ValidationError and JSON errors
        # are ValueErrors): API and UI were built from different contracts.
        answer.error = (
            "The API sent an event this UI does not understand (contract version mismatch?)."
        )
    return answer


def main() -> None:
    st.set_page_config(page_title="Literature Assistant", layout="wide")
    st.title("Literature Assistant")
    st.caption("Ask about ML and NLP papers and inspect the sources cited in each answer.")
    render_examples()
    render_baseline_review()

    history = st.session_state.setdefault("history", [])
    answers = [content for role, content in history if role == "assistant"]

    with st.sidebar:
        panel = st.empty()
        with panel.container():
            render_guardrails(answers[-1] if answers else None)
        st.divider()
        render_drawer(history)

    for index, (role, content) in enumerate(history):
        with st.chat_message(role):
            if role == "user":
                st.markdown(content)
            else:
                render_answer(content, index)

    typed = st.chat_input("Ask about papers, citations or authors")
    query = typed or st.session_state.pop("example", None)
    if not query:
        return

    history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        answer = stream_answer(query, panel)
    history.append(("assistant", answer))
    st.rerun()


# Streamlit runs this file as __main__; tests can import it without starting the app.
if __name__ == "__main__":
    main()
