"""Stable source identities and consistency checks for serialized evidence ledgers."""

import hashlib
import json
from typing import Any

from scilitrag.contracts import CONTRACT_VERSION, Evidence
from scilitrag.rag.evidence.index import EvidenceIndex


def canonical_json(value: object) -> str:
    """Serialize JSON deterministically for hashing, rejecting non-finite numbers."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value: object) -> str:
    """Hash canonical UTF-8 JSON with SHA-256."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def evidence_fingerprint(item: Evidence) -> str:
    """Identify a source snapshot independently of its label and retrieval score."""
    # Labels/scores describe a retrieval result; content and metadata identify
    # the source snapshot. Sort author IDs so their input order is irrelevant.
    payload = item.model_dump(exclude={"label", "score"})
    payload["author_ids"] = sorted(set(payload["author_ids"]))
    return digest(payload)


def build_ledger(evidence: list[Evidence]) -> dict[str, Any]:
    """Capture validated evidence, source hashes and the label-to-source mapping."""
    index = EvidenceIndex(evidence)
    entries = [
        {
            "label": item.label,
            "source_hash": evidence_fingerprint(item),
            "evidence": item.model_dump(),
        }
        for item in index.values()
    ]
    payload = {"contract_version": CONTRACT_VERSION, "entries": entries}
    return {**payload, "ledger_hash": digest(payload)}


def verify_ledger(ledger: dict[str, Any]) -> EvidenceIndex:
    """Verify ledger consistency and rebuild a defensive request evidence index."""
    payload = {key: ledger[key] for key in ("contract_version", "entries")}
    if ledger["contract_version"] != CONTRACT_VERSION or digest(payload) != ledger["ledger_hash"]:
        raise ValueError("The evidence ledger content or contract version has changed.")
    values = []
    for entry in ledger["entries"]:
        item = Evidence.model_validate(entry["evidence"])
        if entry["label"] != item.label or evidence_fingerprint(item) != entry["source_hash"]:
            raise ValueError("An evidence ledger source has changed.")
        values.append(item)
    return EvidenceIndex(values)
