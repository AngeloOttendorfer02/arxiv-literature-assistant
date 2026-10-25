"""D-16 full-corpus loader checks using synthetic D-03 and D-12 files, never real LitSearch."""

import json
import re
from unittest.mock import MagicMock

import load_full
import pyarrow as pa
import pytest
from pyarrow import ipc
from select_fixture import DATASET, REVISION, canonical_digest

PAPERS = range(1, 31)
SELECTED = [1, 5, 9]


def write_arrow(path, rows):
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(path), "wb") as sink, ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=7)


def words(text):
    return [match.span() for match in re.finditer(r"\S+", text)]


def write_manifest(data, ids):
    manifest = {"dataset": DATASET, "dataset_revision": REVISION, "retained_corpusids": ids}
    (data / "filtered/d03/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def write_selection(data, manifest, ids=SELECTED):
    selection = {"filtered_manifest_sha256": canonical_digest(manifest), "corpusids": ids}
    (data / "fulltext/v1/selection.json").write_text(json.dumps(selection), encoding="utf-8")


@pytest.fixture
def data(tmp_path, monkeypatch):
    d03 = tmp_path / "filtered" / "d03"
    d03.mkdir(parents=True)
    (tmp_path / "fulltext" / "v1").mkdir(parents=True)
    clean = [
        {
            "corpusid": cid,
            "title": f"Synthetic paper {cid}",
            "abstract": None if cid == 2 else "Synthetic abstract",
            "year": 2023,
            # The next paper, a self-citation and, for paper 30, a paper outside the corpus.
            "citations": [cid + 1, cid],
        }
        for cid in PAPERS
    ]
    write_arrow(d03 / "corpus_clean.arrow", clean)
    write_arrow(
        d03 / "corpus_s2orc.arrow",
        [{"corpusid": cid, "content": {"text": f"Body of paper {cid}."}} for cid in PAPERS],
    )
    write_arrow(d03 / "query.arrow", [{"query_set": "a", "query": "q", "corpusids": [1, 2]}])
    write_selection(tmp_path, write_manifest(tmp_path, list(PAPERS)))
    monkeypatch.setattr(load_full, "data_directory", lambda: tmp_path)
    monkeypatch.setattr(load_full, "token_offsets", lambda model: words)
    return tmp_path


def test_check_only_loads_every_paper_but_chunks_only_the_selection(data, monkeypatch, capsys):
    monkeypatch.setattr("psycopg.connect", lambda *a, **kw: pytest.fail("Unexpected DB connection"))
    monkeypatch.setattr("sys.argv", ["load_full.py", "--check-only"])
    load_full.main()
    output = capsys.readouterr().out
    assert "Valid corpus: 30 papers, 3 with full-text chunks." in output
    assert "Chunks: 3 from 3 papers" in output
    assert "Citations: 29 from 60 list entries" in output
    assert "No database connection" in output


def test_import_copies_all_papers_and_selected_chunks(data, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(load_full, "import_records", lambda *args: calls.append(args))
    monkeypatch.setattr("psycopg.connect", lambda *a, **kw: MagicMock())
    monkeypatch.setenv("INGEST_DATABASE_URL", "postgresql://ingest@localhost/scilitrag")
    monkeypatch.setattr("sys.argv", ["load_full.py"])
    load_full.main()
    _, papers, citations, _, _, chunks = calls[0]
    assert [paper[0] for paper in papers] == list(PAPERS)
    assert len(citations) == 29
    assert sorted({chunk[1] for chunk in chunks}) == SELECTED
    assert "Loaded 30 papers, 29 citations" in capsys.readouterr().out


def test_selection_from_other_d03_data_is_refused(data):
    write_selection(data, {"dataset": DATASET, "other": True})
    with pytest.raises(ValueError, match="different D-03 data"):
        load_full.full_records(data / "filtered/d03", data / "fulltext/v1/selection.json")


def test_corpus_must_match_the_d03_manifest(data):
    write_selection(data, write_manifest(data, list(PAPERS)[:-1]))
    with pytest.raises(ValueError, match="differ from the D-03 manifest"):
        load_full.full_records(data / "filtered/d03", data / "fulltext/v1/selection.json")


def test_selection_outside_the_corpus_is_refused(data):
    write_selection(data, write_manifest(data, list(PAPERS)), ids=[1, 999])
    with pytest.raises(ValueError, match="distinct D-03 papers"):
        load_full.full_records(data / "filtered/d03", data / "fulltext/v1/selection.json")
