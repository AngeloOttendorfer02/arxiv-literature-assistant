import hashlib
import json

from scilitrag.contracts import CONTRACT_VERSION, Evidence
from scilitrag.rag.evidence_index import EvidenceIndex


def canonical_json(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def evidence_fingerprint(item: Evidence) -> str:
    # Label and search score can change between runs; database source content cannot.
    payload = item.model_dump(exclude={"label", "score"})
    payload["author_ids"] = sorted(set(payload["author_ids"]))
    return digest(payload)


def build_ledger(evidence: list[Evidence]) -> dict:
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


def verify_ledger(ledger: dict) -> EvidenceIndex:
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
