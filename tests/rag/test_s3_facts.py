import json

import pytest

from scilitrag.config import load_settings
from scilitrag.contracts import Evidence, Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.evidence.citations import render_citations
from scilitrag.rag.scenarios.authors import AUTHOR_SQL, attach_sql_evidence, author_facts
from scilitrag.rag.sql_facts import SqlFactsComponent

PLAN = Plan(scenario="S3", topic="retrieval", k=3)
SEEDS = [
    Evidence(label=1, type="paper", corpus_id=2, text="Retrieval A"),
    Evidence(label=2, type="chunk", corpus_id=2, chunk_id="2-0", text="passage"),
    Evidence(label=3, type="paper", corpus_id=5, text="Retrieval C"),
]
ROWS = [
    {
        "author_id": 20,
        "author_name": "Bob",
        "citation_count": 4,
        "example_papers": "1: BERT; 4: Vision model",
    },
    {"author_id": 10, "author_name": "Alice", "citation_count": 2, "example_papers": "1: BERT"},
    {
        "author_id": 30,
        "author_name": "Carol",
        "citation_count": 2,
        "example_papers": "4: Vision model",
    },
]


def reader(sql, params=None):
    assert sql == AUTHOR_SQL
    assert json.loads(params["seed_ids"]) == [2, 5]
    assert params["limit"] == 3
    return ROWS


def facts():
    return author_facts(PLAN, SEEDS, reader, dataset_revision="synthetic-v1")


def test_exact_seed_scope_and_reproducible_query_id():
    table = facts()[0]
    assert table.rows[0][:3] == [20, "Bob", 4]
    assert table.dataset_revision == "synthetic-v1"
    assert "not global citation counts" in table.scope
    assert "stored author IDs" in table.scope
    assert "merge namesakes" in table.scope
    assert table.query_id == facts()[0].query_id
    changed = author_facts(PLAN, SEEDS, reader, dataset_revision="next")[0]
    assert changed.query_id != table.query_id
    assert "DISTINCT c.citing_id,c.cited_id" in table.sql
    assert 'r.author_name COLLATE "C",r.author_id' in table.sql


def test_no_seed_papers_or_edges_produces_clear_message():
    with pytest.raises(RagUserError, match="No topic papers"):
        author_facts(PLAN, [], reader)
    with pytest.raises(RagUserError, match="No author citation credits"):
        author_facts(PLAN, SEEDS, lambda *_: [])


def test_sql_sources_keep_existing_labels_and_do_not_duplicate():
    evidence = attach_sql_evidence(SEEDS, facts())
    assert [e.label for e in evidence] == [1, 2, 3, 4]
    assert evidence[3].query_id == facts()[0].query_id
    assert attach_sql_evidence(evidence, facts()) == evidence
    assert len(SEEDS) == 3
    rendered = render_citations(
        "The ranking uses indexed citation edges [4].", {e.label: e for e in evidence}
    )
    assert rendered.citations[0].query_id == facts()[0].query_id
    assert rendered.citations[0].source_hash


def test_non_s3_sql_component_never_opens_database():
    component = SqlFactsComponent(load_settings({"USE_MOCKS": "1"}))
    assert component(Plan(scenario="S1", k=1)) == []
    assert component.with_evidence(Plan(scenario="S2", k=1), []) == []


def test_real_sql_matches_independent_pair_credits_and_tie_order(fixture_reader):
    table = author_facts(PLAN, SEEDS, fixture_reader, dataset_revision="synthetic-v1")[0]
    assert [row[:3] for row in table.rows] == [[20, "Bob", 4], [10, "Alice", 2], [30, "Carol", 2]]
    # A chunk for seed 2 must not count that paper a second time.
    assert json.loads(table.parameters["seed_ids"]) == [2, 5]
    assert "1: BERT" in table.rows[0][3] and "4: Vision model" in table.rows[0][3]


def test_real_s3_author_filter_restricts_credited_authors_not_seeds(fixture_reader):
    table = author_facts(PLAN.model_copy(update={"author_ref": "Alice"}), SEEDS, fixture_reader)[0]
    assert table.rows == [[10, "Alice", 2, "1: BERT"]]
