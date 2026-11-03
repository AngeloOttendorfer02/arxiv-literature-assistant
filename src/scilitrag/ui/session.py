"""Chat state and a Markdown export, independent of the Streamlit rendering layer."""

from dataclasses import dataclass, field

from scilitrag.contracts import (
    SENTENCE_REMOVED,
    Citation,
    DoneEvent,
    ErrorEvent,
    SentenceEvent,
    StatusEvent,
    TableEvent,
)

GUARD_STAGES = ("prompt_guard", "topic_gate", "pii")
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


def answer_text(answer: Answer) -> str:
    return " ".join(sentence.text for sentence in answer.sentences)


def _table_cell(value: object) -> str:
    """Keep arbitrary SQL results inside one Markdown table cell."""
    if value is None:
        return ""
    text = str(value).replace("\\", "\\\\").replace("|", "\\|")
    return text.replace("\n", " ").replace("\r", " ")


def export_conversation(history: list[tuple[str, str | Answer]]) -> str:
    """Export the conversation, cited source metadata, and displayed tables as Markdown."""
    sections = ["# SciLitRAG conversation"]
    for role, content in history:
        if role == "user" and isinstance(content, str):
            sections.append(f"## Question\n\n{content}")
            continue
        if role != "assistant" or not isinstance(content, Answer):
            continue
        response = content.refusal() or answer_text(content) or "_No answer._"
        sections.append(f"## Answer\n\n{response}")
        if content.error:
            sections.append(f"**Response status:** {content.error}")
        for table in content.tables:
            if table.scope:
                sections.append(f"**Scope:** {table.scope}")
            if table.columns:
                rows = ["| " + " | ".join(_table_cell(c) for c in table.columns) + " |"]
                rows.append("| " + " | ".join("---" for _ in table.columns) + " |")
                for row in table.rows:
                    values = list(row[: len(table.columns)])
                    values.extend([None] * (len(table.columns) - len(values)))
                    rows.append("| " + " | ".join(_table_cell(v) for v in values) + " |")
                sections.append("\n".join(rows))
        if citations := content.citations():
            sources = ["### Sources"]
            for citation in citations:
                details = []
                if citation.year is not None:
                    details.append(str(citation.year))
                if citation.corpus_id is not None:
                    details.append(f"corpus ID: {citation.corpus_id}")
                if citation.chunk_id:
                    details.append(f"chunk: {citation.chunk_id}")
                if citation.query_id:
                    details.append(f"query: {citation.query_id}")
                suffix = f" ({'; '.join(details)})" if details else ""
                sources.append(f"- [{citation.label}] {citation.title or 'Untitled'}{suffix}")
            sections.append("\n".join(sources))
    return "\n\n".join(sections) + "\n"
