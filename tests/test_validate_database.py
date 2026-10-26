"""D-17 validation checks against a fake read-only database, never LitSearch."""

import pytest
from validate_database import (
    ENTITIES,
    FOREIGN_KEYS,
    INFO_CHECKS,
    NULL_CHECKS,
    TEXT_FIELDS,
    validate,
)

COUNTS = {"papers": 56115, "authors": 90000, "paper_authors": 150000, "citations": 300, "chunks": 0}
DECLARED = [(child, column, parent, key, True) for child, column, parent, key in FOREIGN_KEYS]


def fake_db(counts=COUNTS, declared=DECLARED, orphans=None, violations=None, texts=None):
    """Answer each validation query from small in-memory tables."""
    orphans, violations, texts = orphans or {}, violations or {}, texts or {}
    sql_names = {sql: name for name, sql in NULL_CHECKS + INFO_CHECKS}

    def query(sql):
        if sql in sql_names:
            return [(violations.get(sql_names[sql], 0),)]
        if "pg_constraint" in sql:
            return declared
        if "LEFT JOIN" in sql:
            child, column = sql.split("FROM ")[1].split()[0], sql.split("= c.")[1].split()[0]
            return [(orphans.get(f"{child}.{column}", 0),)]
        return [(counts[sql.removeprefix("SELECT count(*) FROM ")],)]

    def stream(sql):
        name = next(name for name, text_sql in TEXT_FIELDS if text_sql == sql)
        return iter(texts.get(name, [("Plain title", None)]))

    return query, stream


def statuses(results):
    return {name: ok for name, ok, _ in results}


def test_clean_database_passes_and_reports_exceptions_as_info():
    results = validate(*fake_db(violations={"papers without title and abstract": 5}))
    assert [name for name, ok, _ in results if ok is False] == []
    info = {name: detail for name, ok, detail in results if ok is None}
    assert info == {"papers without title and abstract": "5", "papers without year": "0"}
    assert len(results) == 1 + len(FOREIGN_KEYS) + len(NULL_CHECKS) + 2 + len(TEXT_FIELDS)


def test_row_rule_needs_three_entities_with_1000_rows():
    small = {table: 999 for table in ENTITIES} | {"papers": 1000, "authors": 1000}
    results = validate(*fake_db(counts=small))
    assert statuses(results)["rows"] is False
    assert "2 entities with at least 1,000" in results[0][2]


@pytest.mark.parametrize(
    "key", ["paper_authors.author_id", "citations.cited_id", "chunks.corpus_id"]
)
def test_orphans_fail_their_foreign_key(key):
    results = statuses(validate(*fake_db(orphans={key: 3})))
    assert results[f"fk {key}"] is False
    assert sum(ok is False for ok in results.values()) == 1


def test_missing_or_unvalidated_constraint_fails():
    declared = [row for row in DECLARED if row[1] != "citing_id"]
    declared = [(c, col, p, key, col != "cited_id") for c, col, p, key, _ in declared]
    results = statuses(validate(*fake_db(declared=declared)))
    assert results["fk citations.citing_id"] is False
    assert results["fk citations.cited_id"] is False
    assert results["fk chunks.corpus_id"] is True


@pytest.mark.parametrize("name", [name for name, _ in NULL_CHECKS])
def test_null_violations_fail(name):
    results = statuses(validate(*fake_db(violations={name: 1})))
    assert results[name] is False


def test_emails_are_counted_in_every_text_field():
    texts = {
        "papers.title and abstract": [("Contact a@b.org", "{x,y}@uni.edu and c@d.com")],
        "chunks.section and text": [(None, "Masked <EMAIL> only")],
        "authors.name": [("Ann Lee", "Ann", "Lee")],
    }
    results = validate(*fake_db(texts=texts))
    emails = {name: (ok, detail) for name, ok, detail in results if name.startswith("e-mails")}
    assert emails["e-mails in papers.title and abstract"] == (False, "3")
    assert emails["e-mails in chunks.section and text"] == (True, "0")
    assert emails["e-mails in authors.name"] == (True, "0")
