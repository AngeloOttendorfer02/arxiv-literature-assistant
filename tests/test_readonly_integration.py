import psycopg
import pytest


def test_real_fixture_reads_and_blocks_writes(fixture_reader):
    assert fixture_reader("SELECT COUNT(*) AS n FROM papers")[0]["n"] == 5
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        fixture_reader("CREATE TABLE should_not_exist(id integer)")
