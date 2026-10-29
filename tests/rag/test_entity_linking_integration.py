"""Entity-linking SQL and API clarification checks in an isolated test schema."""

import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.rows import dict_row

from scilitrag.api.app import create_app
from scilitrag.components import mock_components
from scilitrag.config import load_settings
from scilitrag.contracts import Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.retrieval import retrieve_with
from scilitrag.rag.search.backend import SearchBackend
from scilitrag.rag.search.entities import EntityLinker, normalize_name
from scilitrag.ui.sse import iter_events

PAPERS = [
    (101, "A Deterministic Study of Neural Document Retrieval"),
    (102, "Atlas: Dense Retrieval"),
    (103, "Atlas: Sparse Retrieval"),
    (104, "Orion: Scientific Language Models"),
    (105, "Another study by Smith"),
    (215737187, "Dense Passage Retrieval for Open-Domain Question Answering"),
]
AUTHORS = [
    (201, "Alice Smith"),
    (202, "Alice Smith"),
    (203, "Bob Jones"),
    (204, "Vladimir Karpukhin"),
]


@pytest.fixture(scope="module")
def entity_reader():
    dsn = os.environ.get("RETRIEVAL_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set RETRIEVAL_TEST_DATABASE_URL to a disposable ParadeDB database.")
    schema = "entity_test_" + uuid4().hex
    root = Path(__file__).resolve().parents[2]
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as connection:
        for extension in ("vector", "pg_trgm"):
            connection.execute(
                sql.SQL("CREATE EXTENSION IF NOT EXISTS {}").format(sql.Identifier(extension))
            )
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            connection.execute(
                sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema))
            )
            schema_sql = (root / "docker/db/schema.sql").read_text(encoding="utf-8")
            connection.execute(
                "\n".join(line for line in schema_sql.splitlines() if not line.startswith("\\"))
            )
            with connection.cursor() as cursor:
                cursor.executemany("INSERT INTO papers (corpus_id,title) VALUES (%s,%s)", PAPERS)
                cursor.executemany(
                    "INSERT INTO authors (author_id,name,normalized_name) "
                    "OVERRIDING SYSTEM VALUE VALUES (%s,%s,%s)",
                    [(author_id, name, normalize_name(name)) for author_id, name in AUTHORS],
                )
                cursor.executemany(
                    "INSERT INTO paper_authors (corpus_id,author_id) VALUES (%s,%s)",
                    [(101, 201), (102, 202), (104, 203), (105, 201), (215737187, 204)],
                )
                cursor.executemany(
                    "UPDATE papers SET year=%s WHERE corpus_id=%s",
                    [(2020, 101), (2021, 102), (2020, 104), (2021, 105), (2020, 215737187)],
                )
                cursor.executemany(
                    "UPDATE authors SET last_name=%s WHERE author_id=%s",
                    [("Smith", 201), ("Smith", 202), ("Jones", 203), ("Karpukhin", 204)],
                )

            def reader(statement, params=None):
                with connection.transaction():
                    connection.execute("SET TRANSACTION READ ONLY")
                    return connection.execute(statement, params or {}).fetchall()

            yield reader
        finally:
            connection.execute("SET search_path TO public")
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.parametrize(
    "reference,expected",
    [
        ("101", 101),
        ("a deterministic study of neural document retrieval", 101),
        ("Orion", 104),
    ],
)
def test_real_paper_ids_exact_titles_and_prefixes(entity_reader, reference, expected):
    assert EntityLinker(entity_reader).paper(reference) == expected


def test_real_trigram_query_resolves_a_clear_typo(entity_reader):
    statements = []

    def observed_reader(statement, params):
        statements.append(statement)
        return entity_reader(statement, params)

    reference = "A Deterministic Study of Neural Document Retreival"
    assert EntityLinker(observed_reader).paper(reference) == 101
    assert any("similarity(title" in statement and " %% " in statement for statement in statements)


@pytest.mark.parametrize(
    "reference,expected",
    [
        ("DPR", 215737187),
        ("Karpukhin et al. (2020)", 215737187),
        ("DPR (Karpukhin et al., 2020)", 215737187),
        ("Jones et al., 2020", 104),
        ("Smith (2020)", 101),
    ],
)
def test_real_alias_and_author_year_resolution(entity_reader, reference, expected):
    assert EntityLinker(entity_reader).paper(reference) == expected


def test_real_same_surname_and_year_requires_disambiguation(entity_reader):
    with pytest.raises(RagUserError, match="Which paper") as error:
        EntityLinker(entity_reader).paper("Smith et al. (2021)")
    assert "102: Atlas: Dense Retrieval" in str(error.value)
    assert "105: Another study by Smith" in str(error.value)


def test_real_missing_author_year_does_not_substitute_a_fuzzy_title(entity_reader):
    with pytest.raises(RagUserError, match="Paper not found"):
        EntityLinker(entity_reader).paper("Jones et al. (2021)")


def test_real_author_lookup_normalizes_case_spacing_and_punctuation(entity_reader):
    assert EntityLinker(entity_reader).author("  BOB--Jones ") == 203


def test_real_paper_ambiguity_lists_both_candidate_ids(entity_reader):
    with pytest.raises(RagUserError, match="Which paper") as error:
        EntityLinker(entity_reader).paper("Atlas")
    assert "102: Atlas: Dense Retrieval" in str(error.value)
    assert "103: Atlas: Sparse Retrieval" in str(error.value)


def test_real_author_ambiguity_keeps_distinct_namesakes(entity_reader):
    with pytest.raises(RagUserError, match="Which author") as error:
        EntityLinker(entity_reader).author("ALICE-SMITH")
    assert "201: Alice Smith" in str(error.value)
    assert "202: Alice Smith" in str(error.value)


def test_real_author_context_and_explicit_id_resolve_ambiguity(entity_reader):
    linker = EntityLinker(entity_reader)
    assert linker.author("Alice Smith", paper_ref="101") == 201
    assert linker.author("Alice Smith", paper_ref="102") == 202
    assert linker.author("202") == 202


@pytest.mark.parametrize("kind,reference", [("paper", "999999"), ("author", "999999")])
def test_real_unknown_entities_are_controlled(entity_reader, kind, reference):
    with pytest.raises(RagUserError, match="not found"):
        getattr(EntityLinker(entity_reader), kind)(reference)


@pytest.mark.parametrize("kind", ["paper", "author"])
def test_real_entity_values_cannot_inject_sql(entity_reader, kind):
    reference = "zzzxqv'; DROP TABLE papers; --"
    with pytest.raises(RagUserError):
        getattr(EntityLinker(entity_reader), kind)(reference)
    assert entity_reader("SELECT count(*) AS n FROM papers")[0]["n"] == len(PAPERS)
    assert entity_reader("SELECT count(*) AS n FROM authors")[0]["n"] == len(AUTHORS)


@pytest.mark.parametrize(
    "references,expected,choices",
    [
        ({"paper_ref": "Atlas"}, "Which paper", ["102", "103"]),
        ({"author_ref": "Alice Smith"}, "Which author", ["201", "202"]),
        ({"paper_ref": "999999"}, "Paper not found", []),
        ({"author_ref": "999999"}, "Author not found", []),
    ],
)
def test_api_delivers_real_linker_clarifications_before_generation(
    entity_reader, references, expected, choices
):
    def unexpected_embedding(_texts):
        pytest.fail("Entity failure must stop before embedding or generation.")

    async def unexpected_generation(_plan, _evidence):
        pytest.fail("Entity failure must stop before generation.")
        yield "unreachable"

    backend = SearchBackend(entity_reader, unexpected_embedding)
    plan = Plan(scenario="S1", topic="retrieval", k=5, **references)
    components = replace(
        mock_components(0),
        plan=lambda _query: plan,
        retrieve=lambda request_plan: retrieve_with(request_plan, backend),
        generate=unexpected_generation,
    )
    settings = load_settings({"USE_MOCKS": "1", "LOG_DIR": ""})
    with TestClient(create_app(components, settings)) as client:
        response = client.post("/query", json={"query": "Find retrieval papers"})
    assert response.status_code == 200
    events = list(iter_events(response.iter_lines()))
    assert not any(name in {"sentence", "table"} for name, _event in events)
    assert events[-1][0] == "error"
    error = events[-1][1]
    assert error.code == "bad_request"
    assert expected in error.message
    assert all(choice in error.message for choice in choices)
