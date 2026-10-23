from copy import deepcopy

import pytest

from scilitrag.contracts import Evidence
from scilitrag.rag.evidence.citations import render_citations
from scilitrag.rag.evidence.provenance import build_ledger, evidence_fingerprint, verify_ledger


def source():
    return Evidence(label=1, type="chunk", corpus_id=7, chunk_id="7-0", text="Exact passage.")


def test_ledger_round_trip_and_rendered_fingerprint():
    item = source()
    ledger = build_ledger([item])
    index = verify_ledger(ledger)
    citation = render_citations("Claim [1].", index.mapping()).citations[0]
    assert citation.source_hash == ledger["entries"][0]["source_hash"]
    assert index.resolve(1).text == "Exact passage."


def test_source_hash_is_stable_across_labels_and_scores():
    item = source()
    changed = item.model_copy(update={"label": 9, "score": 0.8})
    assert evidence_fingerprint(item) == evidence_fingerprint(changed)
    assert evidence_fingerprint(item) != evidence_fingerprint(
        item.model_copy(update={"text": "Other."})
    )


def test_ledger_detects_content_and_mapping_changes():
    ledger = build_ledger([source()])
    for field, value in [("text", "Changed."), ("label", 8)]:
        changed = deepcopy(ledger)
        changed["entries"][0]["evidence"][field] = value
        with pytest.raises(ValueError):
            verify_ledger(changed)
