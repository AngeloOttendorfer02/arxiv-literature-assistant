import pytest

from scilitrag.rag.errors import RagUserError
from scilitrag.rag.search.entities import AUTHOR_YEAR_SQL, EntityLinker
from scilitrag.rag.search.paper_aliases import (
    VERIFIED_ALIASES,
    VerifiedAliases,
    author_year_reference,
    reviewed_reference_ids,
)


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


@pytest.mark.parametrize(
    "reference",
    [
        "DPR",
        "dpr",
        "“DPR”",
        "Karpukhin et al., 2020",
        "Karpukhin et al. (2020)",
        "Karpukhin et al. 2020",
        "DPR (Karpukhin et al., 2020)",
        "DPR(Karpukhin et al.,2020)",
    ],
)
def test_reviewed_user_references_resolve_and_cache_without_an_id(reference):
    calls = []

    def reader(sql, params):
        calls.append((sql, params))
        if "ANY(%(ids)s)" in sql:
            assert params == {"ids": [215737187]}
            return [{"corpus_id": 215737187, "title": "Dense Passage Retrieval"}]
        if sql == AUTHOR_YEAR_SQL:
            assert params == {"last_name": "Karpukhin", "year": 2020}
            return [{"corpus_id": 215737187, "title": "Dense Passage Retrieval"}]
        return []

    linker = EntityLinker(reader)
    assert linker.paper(reference) == 215737187
    initial_calls = len(calls)
    assert linker.paper(reference.lower()) == 215737187
    assert len(calls) == initial_calls


@pytest.mark.parametrize(
    "reference,surname",
    [
        ("Smith et al. (2021)", "Smith"),
        ("Smith, 2021", "Smith"),
        ("Smith et al. 2021", "Smith"),
        ("van der Waals et al., 2021", "van der Waals"),
        ("O'Neil (2021)", "O'Neil"),
        ("Müller-Schmidt et al. (2021)", "Müller-Schmidt"),
    ],
)
def test_unregistered_author_year_lookup_uses_bound_metadata(reference, surname):
    def reader(sql, params):
        if sql == AUTHOR_YEAR_SQL:
            assert params == {"last_name": surname, "year": 2021}
            assert surname not in sql
            return [{"corpus_id": 42, "title": "A study"}]
        return []

    assert EntityLinker(reader).paper(reference) == 42


@pytest.mark.parametrize("reference", ["et al. (2020)", "Smith et al.", "DPR", "Smith (2020a)"])
def test_incomplete_citation_metadata_is_not_guessed(reference):
    assert author_year_reference(reference) is None


@pytest.mark.parametrize(
    "reference", ["BERT 2019", "Findings of WMT 2020", "Attention is all you need, 2017"]
)
def test_titles_ending_in_years_keep_the_ordinary_title_lookup_path(reference):
    assert author_year_reference(reference) is None

    def reader(sql, params):
        if sql == AUTHOR_YEAR_SQL:
            pytest.fail("A title ending in a year must not trigger surname lookup")
        if "similarity(title" in sql:
            assert params == {"ref": reference}
            return [{"corpus_id": 42, "title": reference, "similarity": 0.95}]
        return []

    assert EntityLinker(reader).paper(reference) == 42


def test_bare_surname_year_is_not_automatically_a_citation():
    assert author_year_reference("Smith 2021") is None


def test_exact_titles_take_priority_over_generic_author_year_interpretation():
    def reader(sql, params):
        if "lower(title)=" in sql:
            return [{"corpus_id": 42, "title": "Smith (2021)"}]
        pytest.fail("An exact title must not trigger generic citation lookup")

    assert EntityLinker(reader).paper("Smith (2021)") == 42


@pytest.mark.parametrize("reference", ["DPR-PAQ", "ADPR", "DPRv2", "Karpukhin-et al., 2020"])
def test_registry_lookup_preserves_name_and_hyphen_boundaries(reference):
    assert reviewed_reference_ids(reference) == []


@pytest.mark.parametrize("reference", ["DPR", "Karpukhin et al. (2020)", "Smith (2021)"])
def test_missing_identity_or_citation_never_substitutes_a_fuzzy_other_paper(reference):
    def reader(sql, params):
        if "similarity(title" in sql:
            pytest.fail("An explicit identity must not fall back to a different fuzzy paper")
        return []

    with pytest.raises(RagUserError, match="Paper not found"):
        EntityLinker(reader).paper(reference)


@pytest.mark.parametrize("reference", ["DPR", "Karpukhin et al. (2020)", "Smith (2021)"])
def test_conflicting_alias_or_author_year_matches_require_a_title(reference):
    def reader(sql, params):
        if "lower(title)=" in sql:
            return [{"corpus_id": 42, "title": "Another paper"}] if reference == "DPR" else []
        if "ANY(%(ids)s)" in sql:
            return [{"corpus_id": 215737187, "title": "Dense Passage Retrieval"}]
        if sql == AUTHOR_YEAR_SQL:
            return [
                {"corpus_id": 42, "title": "Another paper"},
                {"corpus_id": 43, "title": "A second paper"},
            ]
        return []

    with pytest.raises(RagUserError, match="Which paper.*full title"):
        EntityLinker(reader).paper(reference)


def test_a_reviewed_short_name_can_have_multiple_paper_identities(monkeypatch):
    monkeypatch.setitem(VERIFIED_ALIASES, 999, VerifiedAliases(("DPR",), "Other et al.", 2021))

    def reader(sql, params):
        if "ANY(%(ids)s)" in sql:
            assert params["ids"] == [999, 215737187]
            return [
                {"corpus_id": 999, "title": "Another DPR"},
                {"corpus_id": 215737187, "title": "Dense Passage Retrieval"},
            ]
        return []

    with pytest.raises(RagUserError, match="Which paper"):
        EntityLinker(reader).paper("DPR")


def test_reviewed_dpr_long_name_with_acronym_resolves_exactly():
    assert reviewed_reference_ids("Dense Passage Retrieval (DPR)") == [215737187]
    assert reviewed_reference_ids("dense passage retrieval(dpr)") == [215737187]
    assert reviewed_reference_ids("Dense Passage Retrieval (DPR-PAQ)") == []
    assert reviewed_reference_ids("Other Retrieval (DPR)") == []
