import pytest

from scilitrag.contracts import Evidence, Plan, SqlFacts
from scilitrag.rag.evidence_index import EvidenceIndex, allocate_evidence


def test_snapshot_is_isolated_from_callers_and_other_requests():
    source = Evidence(label=7, type="paper", corpus_id=1, text="original", author_ids=[3, 2, 3])
    index = EvidenceIndex([source])
    source.text = "changed"
    result = index.resolve(7)
    assert result.text == "original" and result.author_ids == [2, 3]
    result.author_ids.append(9)
    assert index.resolve(7).author_ids == [2, 3]
    with pytest.raises(ValueError, match="Unknown"):
        EvidenceIndex([]).resolve(7)


@pytest.mark.parametrize(
    "items",
    [
        [Evidence(label=0, type="paper", corpus_id=1, text="x")],
        [Evidence(label=1, type="paper", text="x")],
        [Evidence(label=1, type="chunk", corpus_id=1, text="x")],
        [Evidence(label=1, type="sql", text="x")],
    ],
)
def test_invalid_source_identities_are_rejected(items):
    with pytest.raises(ValueError):
        EvidenceIndex(items)


def test_label_allocation_is_predictable_without_mutating_input():
    source = Evidence(label=0, type="paper", corpus_id=17, text="abstract")
    assert allocate_evidence([source])[0].label == 1 and source.label == 0
    with pytest.raises(ValueError):
        EvidenceIndex([source.model_copy(update={"label": 1})] * 2)


def test_new_metadata_has_compatible_defaults():
    assert Plan(scenario="S1", k=1).query is None
    table = SqlFacts(query_id="q", sql="SELECT", columns=[], rows=[])
    assert table.parameters == {} and table.dataset_revision is None
