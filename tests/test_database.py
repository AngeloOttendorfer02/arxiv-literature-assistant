import psycopg
import pytest

from scilitrag.config import load_settings
from scilitrag.rag.database import DatabaseReader
from scilitrag.rag.database_contract import REQUIRED_COLUMNS, validate_database
from scilitrag.rag.errors import RagUserError


def metadata_reader(*, model="model", revision="r1", dimension=768, missing=None):
    def reader(statement, params=None):
        if "information_schema" in statement:
            return [
                {"table_name": table, "column_name": column}
                for table, columns in REQUIRED_COLUMNS.items()
                for column in columns
                if (table, column) != missing
            ]
        if "pg_extension" in statement:
            return [{"extname": name} for name in ["vector", "pg_search", "pg_trgm"]]
        if "pg_attribute" in statement:
            return [
                {"table_name": table, "vector_type": f"vector({dimension})"}
                for table in ["papers", "chunks"]
            ]
        assert "FROM schema_metadata WHERE id=1" in statement
        return [{"schema_version": 1, "dataset_revision": revision, "embed_model": model}]

    return reader


def test_schema_readiness_allows_unpopulated_search_metadata():
    reader = metadata_reader(model=None, revision=None)
    metadata = validate_database(reader, "model", require_search=False)
    assert metadata.embed_dim == 768 and metadata.dataset_revision is None
    with pytest.raises(RagUserError, match="Search is not ready"):
        validate_database(reader, "model")


def test_search_requires_matching_model_and_declared_dimensions():
    assert validate_database(metadata_reader(), "model").dataset_revision == "r1"
    with pytest.raises(RagUserError, match="does not match"):
        validate_database(metadata_reader(), "other")
    with pytest.raises(RagUserError, match=r"vector\(768\)"):
        validate_database(metadata_reader(dimension=3), "model")


def test_missing_schema_field_is_reported_before_metadata_query():
    with pytest.raises(RagUserError, match="papers.corpus_id"):
        validate_database(metadata_reader(missing=("papers", "corpus_id")), "model")


def test_reader_binds_values_and_sets_readonly_before_first_statement():
    calls = []

    class Connection:
        read_only = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, statement, params):
            assert self.read_only
            calls.append((statement, params))
            return self

        def fetchall(self):
            return [{"value": 1}]

    settings = load_settings({"USE_MOCKS": "1", "DATABASE_URL": "postgresql://local/db"})
    reader = DatabaseReader(settings, connect=lambda *args, **kwargs: Connection())
    assert reader("SELECT %(value)s", {"value": 1}) == [{"value": 1}]
    assert calls[0][1] == ("10000",) and calls[1][1] == {"value": 1}


@pytest.mark.parametrize("error", [psycopg.OperationalError, psycopg.errors.QueryCanceled])
def test_database_errors_are_controlled_and_do_not_expose_credentials(error):
    def connect(*args, **kwargs):
        raise error("postgresql://private:secret@database")

    settings = load_settings({"USE_MOCKS": "1", "DATABASE_URL": "postgresql://local/db"})
    with pytest.raises(RagUserError) as captured:
        DatabaseReader(settings, connect=connect)("SELECT 1")
    assert "secret" not in str(captured.value)
