"""Test sets for the guardrails (S-12): input prompts and indirect-injection evidence."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scilitrag.contracts import EvidenceType, InputCategory

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "eval" / "guardrails"
Split = Literal["tuning", "test"]


class InputCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    category: InputCategory
    subcategory: str
    language: Literal["en", "de"]
    split: Split
    source: Literal["hand", "ai"]
    reviewed: bool


class EvidenceCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    kind: Literal["attack", "clean", "security_paper"]
    evidence_type: EvidenceType
    title: str
    text: str = Field(min_length=1)
    expected: Literal["removed", "kept"]
    position: Literal["start", "middle", "end", "none"]
    pair: str | None
    split: Split
    source: Literal["hand", "ai"]
    reviewed: bool


def _read(path: Path, model):
    with path.open(encoding="utf-8") as handle:
        return [model.model_validate(json.loads(line)) for line in handle if line.strip()]


def load_input_cases(directory: Path = DEFAULT_DIR) -> list[InputCase]:
    """Attacks and harmless questions together; the harmless ones have category ok."""
    return _read(directory / "input_attacks.jsonl", InputCase) + _read(
        directory / "input_harmless.jsonl", InputCase
    )


def load_evidence_cases(directory: Path = DEFAULT_DIR) -> list[EvidenceCase]:
    return _read(directory / "evidence_cases.jsonl", EvidenceCase)
