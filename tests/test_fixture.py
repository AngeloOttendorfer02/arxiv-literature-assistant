"""D-07 selection and import checks using synthetic records, never real LitSearch."""

import json
import random
import re
from collections import Counter
from unittest.mock import MagicMock

import load_fixture
import pyarrow as pa
import pytest
import select_fixture
from load_fixture import fixture_records, import_records
from pyarrow import ipc
from select_fixture import DATASET, REVISION, build_fixture, choose_papers, choose_queries, digest


def write_arrow(path, rows):
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(path), "wb") as sink, ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=37)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "filtered" / "d03"
    root.mkdir(parents=True)
    clean = [
        {
            "corpusid": cid,
            "title": None if cid == 1 else f"Synthetic paper {cid}",
            "abstract": "Synthetic abstract. * a.b@example.org"
            if cid == 3
            else "Synthetic abstract",
            "full_paper": "Synthetic body",
            "year": None if cid == 2 else 2023,
            "citations": [cid + 40, cid + 40, 999999]
            if cid <= 20
            else ([cid - 80] if 81 <= cid <= 100 else []),
        }
        for cid in range(1, 261)
    ]
    queries = [
        {"query_set": f"group_{i % 4}", "query": f"Synthetic query {i}", "corpusids": [i + 1]}
        for i in range(20)
    ]
    write_arrow(root / "corpus_clean.arrow", clean)
    write_arrow(
        root / "corpus_s2orc.arrow",
        [{"corpusid": r["corpusid"], "content": {"text": r["full_paper"]}} for r in clean],
    )
    write_arrow(root / "query.arrow", queries)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "dataset_revision": REVISION,
                "retained_corpusids": list(range(1, 261)),
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_query_selection_is_balanced_and_reproducible():
    rows = [{"query_set": f"group_{i % 4}", "corpusids": [i]} for i in range(80)]
    indices = choose_queries(rows, random.Random(42))
    assert indices == choose_queries(rows, random.Random(42))
    assert indices != choose_queries(rows, random.Random(43))
    assert Counter(rows[i]["query_set"] for i in indices) == dict.fromkeys(
        [f"group_{i}" for i in range(4)], 5
    )


def test_selection_keeps_targets_and_both_citation_directions():
    graph = {i: set() for i in range(20)}
    graph[0] = {1, 99999}
    graph[2] = {0}
    selected, groups = choose_papers(graph, {0}, 10, random.Random(42))
    assert len(selected) == 10
    assert {0, 1, 2} <= selected
    assert groups["cited_neighbors"] == [1]
    assert groups["citing_neighbors"] == [2]
    assert groups["distractors"]
    assert 99999 not in selected


@pytest.mark.parametrize("targets,count", [({999}, 10), ({0}, 25), ({0}, 2)])
def test_invalid_targets_or_budget_are_rejected(targets, count):
    graph = {i: {(i + 1) % 20} for i in range(20)}
    with pytest.raises(ValueError):
        choose_papers(graph, targets, count, random.Random(42))


def test_export_preserves_sources_targets_nulls_and_alignment(source):
    before = {path.name: path.read_bytes() for path in (source / "filtered/d03").iterdir()}
    folder, manifest = build_fixture(source, count=200)
    papers, citations, _ = fixture_records(folder)
    assert len(papers) == 200
    assert {cid for cid, *_ in papers} >= set(range(1, 21))
    assert {row[0]: row[1] for row in papers}[1] is None
    assert {row[0]: row[3] for row in papers}[2] is None
    assert (1, 41) in citations and (81, 1) in citations
    assert len(citations) == len(set(citations)) == manifest["citation_count"]
    assert all(
        citing in manifest["corpusids"] and cited in manifest["corpusids"]
        for citing, cited in citations
    )
    assert {path.name: path.read_bytes() for path in (source / "filtered/d03").iterdir()} == before
    with pytest.raises(FileExistsError):
        build_fixture(source, count=200)


def test_emails_in_titles_and_abstracts_are_masked(source):
    folder, _ = build_fixture(source, count=200)
    papers = {cid: (title, abstract) for cid, title, abstract, _ in fixture_records(folder)[0]}
    assert papers[3][1] == "Synthetic abstract. * <EMAIL>"
    assert not any("@" in (text or "") for pair in papers.values() for text in pair)
    assert load_fixture.mask_emails(None) is None
    assert load_fixture.mask_emails("Write to x_y@uni.edu or z@lab.ac.uk.") == (
        "Write to <EMAIL> or <EMAIL>."
    )


def test_second_build_has_identical_selection_and_file_hashes(source):
    folder, manifest = build_fixture(source, count=200)
    # A separate local output directory simulates a colleague with the same source.
    original = folder.rename(folder.with_name("first-build"))
    second, again = build_fixture(source, count=200)
    assert manifest == again
    assert (original / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_manifest_hash_ignores_source_json_format_and_output_uses_lf(source):
    provenance = source / "filtered/d03/manifest.json"
    content = json.loads(provenance.read_text(encoding="utf-8"))
    provenance.write_bytes((json.dumps(content, indent=2) + "\n").encode("utf-8"))
    first, manifest = build_fixture(source, count=200)
    first_bytes = (first / "manifest.json").read_bytes()
    assert b"\r" not in first_bytes
    assert first_bytes.endswith(b"\n")
    first.rename(first.with_name("lf-build"))

    reordered = dict(reversed(list(content.items())))
    windows_json = (json.dumps(reordered, indent=4) + "\n").replace("\n", "\r\n")
    provenance.write_bytes(windows_json.encode("utf-8"))
    second, again = build_fixture(source, count=200)
    assert manifest == again
    assert first_bytes == (second / "manifest.json").read_bytes()
    assert manifest["filtered_manifest_sha256"] == select_fixture.canonical_digest(content)


def test_wrong_revision_is_rejected_before_export(source):
    path = source / "filtered/d03/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["dataset_revision"] = "wrong"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        build_fixture(source, count=200)
    assert not (source / "fixtures/v0").exists()


def test_changed_fixture_file_is_rejected(source):
    folder, _ = build_fixture(source, count=200)
    with (folder / "query.arrow").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        fixture_records(folder)


def test_loader_rechecks_target_coverage_even_with_updated_checksum(source):
    folder, manifest = build_fixture(source, count=200)
    rows = [{"query_set": "test", "corpusids": [99999]}] * 20
    write_arrow(folder / "query.arrow", rows)
    manifest["sha256"]["query.arrow"] = digest(folder / "query.arrow")
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="target"):
        fixture_records(folder)


def test_nonempty_database_is_refused_before_copy():
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = (True,)
    with pytest.raises(ValueError, match="not empty"):
        import_records(connection, [], [])
    connection.cursor.assert_not_called()


def test_import_uses_copy_values_and_parameterized_metadata():
    connection = MagicMock()
    connection.execute.return_value.fetchone.side_effect = [(False,), (1, None)]
    paper = (1, "'); DROP TABLE papers; --", None, None)
    import_records(connection, [paper], [(1, 1)])
    cursor = connection.cursor.return_value.__enter__.return_value
    assert [call.args[0] for call in cursor.copy.call_args_list] == [
        load_fixture.PAPER_COPY,
        load_fixture.CITATION_COPY,
    ]
    assert cursor.copy.return_value.__enter__.return_value.write_row.call_args_list[0].args == (
        paper,
    )
    assert connection.execute.call_args.args[1] == (REVISION,)


def words(text):
    return [match.span() for match in re.finditer(r"\S+", text)]


def test_check_only_never_connects_to_database(source, monkeypatch, capsys):
    folder, _ = build_fixture(source, count=200)
    monkeypatch.setattr(load_fixture, "data_directory", lambda: source)
    monkeypatch.setattr(load_fixture, "token_offsets", lambda model: words)
    monkeypatch.setattr("psycopg.connect", lambda *a, **kw: pytest.fail("Unexpected DB connection"))
    monkeypatch.setattr("sys.argv", ["load_fixture.py", "--check-only"])
    load_fixture.main()
    output = capsys.readouterr().out
    assert "No database connection" in output
    assert "Chunks: " in output
    assert f"Manifest SHA-256: {digest(folder / 'manifest.json')}" in output


def test_cli_selection_uses_environment_directory(source, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(source))
    monkeypatch.setattr("sys.argv", ["select_fixture.py", "--papers", "200"])
    select_fixture.main()
    manifest = source / "fixtures/v0/manifest.json"
    assert manifest.is_file()
    assert f"Manifest SHA-256: {digest(manifest)}" in capsys.readouterr().out


def test_chunks_are_copied_after_papers():
    connection = MagicMock()
    connection.execute.return_value.fetchone.side_effect = [(False,), (1, None)]
    chunk = ("1-0", 1, "1 Intro", 0, "Body text.")
    import_records(connection, [(1, "T", None, None)], [], chunks=[chunk])
    cursor = connection.cursor.return_value.__enter__.return_value
    statements = [call.args[0] for call in cursor.copy.call_args_list]
    assert statements[0] == load_fixture.PAPER_COPY and statements[-1] == load_fixture.CHUNK_COPY
    rows = cursor.copy.return_value.__enter__.return_value.write_row.call_args_list
    assert rows[-1].args == (chunk,)
