"""Shared data types and interfaces. Changes only by PR with a ping to everyone."""

from collections.abc import AsyncIterator
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "0.3"

InputCategory = Literal["ok", "off_topic", "injection", "write_intent"]
Scenario = Literal["S1", "S2", "S3"]
EvidenceType = Literal["chunk", "paper", "sql"]
SentenceStatus = Literal["ok", "no_citation", "unknown_label", "fact_mismatch", "unsupported"]
Stage = Literal["prompt_guard", "topic_gate", "pii", "plan", "retrieve", "sql", "generate"]
ErrorCode = Literal["internal", "timeout", "bad_request"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Data


class StageResult(Model):
    stage: Stage
    result: str
    score: float | None = None
    latency_ms: float


class CheckInputResult(Model):
    allowed: bool
    category: InputCategory
    masked_query: str
    scores: dict[str, float] = Field(default_factory=dict)
    latency_ms: float
    stages: list[StageResult] = Field(default_factory=list)


class Plan(Model):
    scenario: Scenario
    topic: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    paper_ref: str | None = None
    author_ref: str | None = None
    k: int


class Evidence(Model):
    label: int
    type: EvidenceType
    chunk_id: str | None = None
    corpus_id: int | None = None
    title: str | None = None
    year: int | None = None
    section: str | None = None
    query_id: str | None = None
    text: str
    score: float | None = None


EvidenceMapping = dict[int, Evidence]


class SqlFacts(Model):
    query_id: str
    sql: str
    columns: list[str]
    rows: list[list[str | int | float | None]]


class SentenceCheck(Model):
    status: SentenceStatus
    nli_scores: dict[str, float] = Field(default_factory=dict)
    latency_ms: float


class Citation(Model):
    label: int
    corpus_id: int | None = None
    chunk_id: str | None = None
    title: str | None = None
    year: int | None = None
    query_id: str | None = None
    section: str | None = None
    text: str | None = None


class RenderedSentence(Model):
    text: str
    citations: list[Citation] = Field(default_factory=list)


# Interfaces


class CheckInput(Protocol):
    def __call__(self, query: str) -> CheckInputResult: ...


class MaskPii(Protocol):
    def __call__(self, text: str) -> str: ...


class Embed(Protocol):
    def __call__(self, texts: list[str]) -> list[list[float]]: ...


class PlanQuery(Protocol):
    def __call__(self, masked_query: str) -> Plan: ...


class Retrieve(Protocol):
    def __call__(self, plan: Plan) -> list[Evidence]: ...


class GetSqlFacts(Protocol):
    def __call__(self, plan: Plan) -> list[SqlFacts]: ...


class Generate(Protocol):
    def __call__(self, plan: Plan, evidence: list[Evidence]) -> AsyncIterator[str]: ...


class CheckSentence(Protocol):
    def __call__(self, sentence: str, mapping: EvidenceMapping) -> SentenceCheck: ...


class RenderCitations(Protocol):
    def __call__(self, sentence: str, mapping: EvidenceMapping) -> RenderedSentence: ...


# API: POST /query request and SSE events


class QueryRequest(Model):
    query: str = Field(min_length=1)


class StatusEvent(Model):
    stage: Stage
    result: str
    latency_ms: float


# Result of a "generate" status event sent for each sentence that failed check_sentence,
# followed by the SentenceStatus, e.g. "removed: unsupported".
SENTENCE_REMOVED = "removed: "


class SentenceEvent(RenderedSentence):
    check: SentenceCheck


class TableEvent(SqlFacts):
    pass


class DoneEvent(Model):
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    ttft_engine_ms: float | None = None
    ttft_user_ms: float | None = None
    total_ms: float
    sentences_checked: int = 0
    sentences_removed: int = 0
    refused: bool = False
    tokens: int = 0
    tokens_per_s: float | None = None


class ErrorEvent(Model):
    code: ErrorCode
    message: str


EVENT_TYPES: dict[str, type[Model]] = {
    "status": StatusEvent,
    "sentence": SentenceEvent,
    "table": TableEvent,
    "done": DoneEvent,
    "error": ErrorEvent,
}
