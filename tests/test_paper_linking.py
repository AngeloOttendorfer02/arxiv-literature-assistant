import pytest

from scilitrag.rag.entities import EntityLinker
from scilitrag.rag.errors import RagUserError


def test_numeric_ids_are_bound_and_cached():
    calls = []

    def reader(sql, params):
        calls.append((sql, params))
        return [{"corpus_id": 7, "title": "A"}]

    linker = EntityLinker(reader)
    assert linker.paper("7") == linker.paper("7") == 7
    assert len(calls) == 1 and calls[0][1] == {"id": 7}


def test_explicit_title_prefix_resolves_bert():
    def reader(sql, params):
        return (
            [{"corpus_id": 1, "title": "BERT: Pre-training of Deep Bidirectional Transformers"}]
            if "split_part" in sql
            else []
        )

    assert EntityLinker(reader).paper("BERT") == 1


@pytest.mark.parametrize("scores", [[0.4], [0.9, 0.85]])
def test_uncertain_fuzzy_matches_require_a_choice(scores):
    def reader(sql, params):
        return (
            [
                {"corpus_id": i + 1, "title": f"Paper {i}", "similarity": score}
                for i, score in enumerate(scores)
            ]
            if "similarity(title" in sql
            else []
        )

    with pytest.raises(RagUserError, match="Which paper"):
        EntityLinker(reader).paper("query title")


def test_fuzzy_winner_requires_clear_margin():
    def reader(sql, params):
        return (
            [
                {"corpus_id": 1, "title": "A", "similarity": 0.95},
                {"corpus_id": 2, "title": "B", "similarity": 0.70},
            ]
            if "similarity(title" in sql
            else []
        )

    assert EntityLinker(reader).paper("specific title") == 1


def test_missing_paper_has_a_clear_message():
    with pytest.raises(RagUserError, match="not found"):
        EntityLinker(lambda *_: []).paper("unknown")
