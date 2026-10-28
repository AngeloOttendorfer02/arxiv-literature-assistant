"""Validated input for reference export and reproducible retrieval evaluation."""

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from scilitrag.contracts import Scenario
from scilitrag.rag.evidence.provenance import digest

MAX_DATABASE_ID = 9_223_372_036_854_775_807


class ReferenceCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=6000)
    scenario: Scenario
    expected_citing_ids: list[int] | None = None
    expected_years: dict[str, int | None] | None = None
    seed_ids: list[int] | None = None
    expected_author_counts: dict[str, int] | None = None
    source_run_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if not self.id.strip() or not self.query.strip():
            raise ValueError("Case IDs and queries must not be blank")
        for ids in (self.expected_citing_ids, self.seed_ids):
            if ids is not None and (
                len(ids) != len(set(ids))
                or any(
                    isinstance(value, bool) or not 1 <= value <= MAX_DATABASE_ID for value in ids
                )
            ):
                raise ValueError("Reference IDs must be distinct positive database IDs")
        if self.seed_ids is not None and len(self.seed_ids) > 20:
            raise ValueError("At most 20 seed papers are supported")
        if self.expected_years is not None and any(
            not key.isdecimal() or not 1 <= int(key) <= MAX_DATABASE_ID
            for key in self.expected_years
        ):
            raise ValueError("Year references require positive paper IDs")
        if self.expected_author_counts is not None:
            if self.scenario != "S3" or not self.seed_ids:
                raise ValueError("Author-count references require S3 and nonempty seed_ids")
            if any(
                not key.isdecimal() or not 1 <= int(key) <= MAX_DATABASE_ID or count < 0
                for key, count in self.expected_author_counts.items()
            ):
                raise ValueError("Author IDs must be positive and counts nonnegative")
        if self.expected_citing_ids is not None and self.scenario != "S2":
            raise ValueError("Citing-paper references require S2")
        return self


class BaselineAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_id: str
    corpus_id: int = Field(ge=1, le=MAX_DATABASE_ID)
    section: str | None = None
    text: str
    target_title: str | None = None
    target_id: int = Field(ge=1, le=MAX_DATABASE_ID)
    baseline_use: StrictBool | None = None
    reviewer_note: str = ""
    source_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.source_hash and self.source_hash != digest(
            self.model_dump(exclude={"source_hash", "baseline_use", "reviewer_note"})
        ):
            raise ValueError("Baseline source content has changed")
        return self


class ReferenceSet(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_revision: str | None = None
    cases: list[ReferenceCase] = Field(min_length=1)
    baseline_annotations: list[BaselineAnnotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Case IDs must be unique")
        if self.dataset_revision is not None and not self.dataset_revision.strip():
            raise ValueError("dataset_revision must be nonempty when supplied")
        keys = [(row.target_id, row.chunk_id) for row in self.baseline_annotations]
        if len(keys) != len(set(keys)):
            raise ValueError("Baseline target/chunk pairs must be unique")
        return self


def load_reference_set(path: Path) -> dict:
    """Validate input before contacting a model or database; preserve absent expectations."""
    value = ReferenceSet.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return value.model_dump(exclude_unset=True)


def run_fingerprint(record: dict) -> str:
    """Bind reviews/references to immutable run content, excluding scores and labels."""
    return digest(
        {
            key: record.get(key)
            for key in (
                "id",
                "model",
                "query",
                "dataset_revision",
                "dataset_changed",
                "plan",
                "evidence",
                "events",
            )
        }
    )
