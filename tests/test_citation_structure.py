import pytest

from scilitrag.contracts import Evidence
from scilitrag.rag.evidence.labels import citation_labels, citation_status

MAPPING = {1: Evidence(label=1, type="paper", corpus_id=1, text="source")}


@pytest.mark.parametrize(
    "sentence,status",
    [
        ("A supported result [1].", "ok"),
        ("Uncited result.", "no_citation"),
        ("Unknown [2].", "unknown_label"),
        ("Sources [1,2].", "unknown_label"),
        ("Label [1] precedes an unsupported ending.", "unsupported"),
    ],
)
def test_sentence_structure(sentence, status):
    assert citation_status(sentence, MAPPING) == status


@pytest.mark.parametrize("sentence", ["Claim [0].", "Claim [01].", "Claim [1-3]."])
def test_malformed_numeric_syntax_is_rejected(sentence):
    with pytest.raises(ValueError):
        citation_labels(sentence)
