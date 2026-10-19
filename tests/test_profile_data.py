import json
from collections import Counter

import profile_data
import pytest
from profile_data import (
    REVISION,
    arrow_rows,
    build_report,
    cached_subset,
    compare_ids,
    empty_kind,
    length_stats,
    profile_rows,
)


@pytest.mark.parametrize(
    "value,kind",
    [
        (None, "null"),
        (float("nan"), "nan"),
        ("", "empty_string"),
        (" \t", "whitespace_only"),
        ([], "empty_list"),
        ({}, "empty_object"),
        (0, None),
        (False, None),
        ("paper", None),
    ],
)
def test_empty_values_are_distinguished(value, kind):
    assert empty_kind(value) == kind


def test_lengths_exclude_missing_and_blank_but_keep_surrounding_spaces():
    rows = [{"title": value} for value in (None, "", "  ", "a", " abc ", "ab", 17)]
    report, _ = profile_rows(rows, ["title"])
    assert report["rows"] == 7
    assert report["empty_fields"]["title"]["total"] == 3
    assert report["text_lengths_characters"]["title"] == {
        "count": 3,
        "min": 1,
        "mean": 8 / 3,
        "median": 2,
        "p95": pytest.approx(4.7),
        "max": 5,
    }
    assert report["invalid_text_types"] == {"title": 1}
    assert length_stats([])["median"] is None
    assert length_stats([4])["p95"] == 4


def test_short_full_papers_count_nonblank_texts_and_missing_title_and_abstract():
    rows = [
        {"full_paper": "1", "title": "", "abstract": ""},
        {"full_paper": "x" * 50, "title": "T", "abstract": ""},
        {"full_paper": "x" * 500, "title": " ", "abstract": None},
        {"full_paper": "x" * 3000, "title": "", "abstract": ""},
        {"full_paper": "", "title": "", "abstract": ""},
        {"full_paper": "   ", "title": "", "abstract": ""},
    ]
    report, _ = profile_rows(rows, ["title", "abstract", "full_paper"])
    assert report["short_full_papers"] == {
        "under_10_characters": {"papers": 1, "without_title_and_abstract": 1},
        "under_100_characters": {"papers": 2, "without_title_and_abstract": 1},
        "under_2000_characters": {"papers": 3, "without_title_and_abstract": 2},
    }
    assert "short_full_papers" not in profile_rows(rows, ["title"])[0]


def test_ids_and_join_do_not_hide_duplicates_or_match_missing_values():
    fields = ["corpusid"]
    clean, left = profile_rows([{"corpusid": i} for i in (1, 1, 2, None, "1")], fields)
    _, right = profile_rows([{"corpusid": i} for i in (1, 3, 3, None)], fields)
    assert clean["corpusid"]["missing_rows"] == 1
    assert clean["corpusid"]["invalid_type_rows"] == 1
    assert clean["corpusid"]["duplicates"] == {"1": 2}
    assert clean["corpusid"]["duplicate_excess_rows"] == 1
    joined = compare_ids(left, right)
    assert joined["common_distinct_ids"] == 1
    assert joined["inner_join_row_pairs"] == 2
    assert joined["only_in_corpus_clean"] == {"count": 1, "ids": [2]}
    assert joined["only_in_corpus_s2orc"] == {"count": 1, "ids": [3]}
    assert compare_ids(Counter(), Counter())["inner_join_row_pairs"] == 0


def test_missing_years_include_nan_but_do_not_apply_filtering_rules():
    rows = [{"year": year} for year in (None, float("nan"), 2020.0, 0.0)]
    report, _ = profile_rows(rows, ["year"])
    assert report["missing_years"] == 2


def test_missing_cache_fails_without_downloading(tmp_path):
    other_revision = (
        tmp_path
        / "hf_datasets"
        / "princeton-nlp___lit_search"
        / "corpus_clean"
        / "0.0.0"
        / "another_revision"
    )
    other_revision.mkdir(parents=True)
    (other_revision / "dataset_info.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Complete D-01"):
        cached_subset(tmp_path, "corpus_clean")


def test_prepared_arrow_cache_is_read_offline_and_revision_is_enforced(tmp_path, monkeypatch):
    import socket

    pa = pytest.importorskip("pyarrow")
    ipc = pytest.importorskip("pyarrow.ipc")
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Network access"))
    features = {"corpusid": {"_type": "Value", "dtype": "int64"}}
    for subset in ("query", "corpus_clean", "corpus_s2orc"):
        root = tmp_path / "hf_datasets" / "princeton-nlp___lit_search" / subset / "0.0.0" / REVISION
        root.mkdir(parents=True)
        table = pa.table({"corpusid": [42]})
        with (
            pa.OSFile(str(root / "lit_search-full.arrow"), "wb") as sink,
            ipc.new_stream(sink, table.schema) as writer,
        ):
            writer.write_table(table)
        (root / "dataset_info.json").write_text(
            json.dumps(
                {
                    "features": features,
                    "splits": {"full": {"num_examples": 1, "dataset_name": "lit_search"}},
                }
            ),
            encoding="utf-8",
        )
    report = build_report(tmp_path)
    assert report["subsets"]["corpus_s2orc"]["rows"] == 1
    assert report["corpusid_join"]["common_distinct_ids"] == 1
    json.dumps(report, allow_nan=False)
    root = tmp_path / "hf_datasets" / "princeton-nlp___lit_search" / "query" / "0.0.0" / REVISION
    assert list(arrow_rows([root / "lit_search-full.arrow"])) == [{"corpusid": 42}]
    info_path = root / "dataset_info.json"
    metadata = json.loads(info_path.read_text(encoding="utf-8"))
    metadata["splits"]["full"]["num_examples"] = 2
    info_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="read 1 rows, but cache metadata lists 2"):
        build_report(tmp_path)
    (root / "lit_search-full.arrow").unlink()
    with pytest.raises(ValueError, match="shards are missing"):
        cached_subset(tmp_path, "query")


@pytest.mark.parametrize("container_override", [False, True])
@pytest.mark.parametrize("stdout", [False, True])
def test_cli_uses_local_env_or_container_override(
    tmp_path, monkeypatch, capsys, container_override, stdout
):
    monkeypatch.setattr(profile_data, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("DATA_DIR=local_data\n", encoding="utf-8")
    expected = tmp_path / ("mounted_data" if container_override else "local_data")
    if container_override:
        monkeypatch.setenv("DATA_DIR", str(expected))
    else:
        monkeypatch.delenv("DATA_DIR", raising=False)
    report = {"subsets": {"query": {"rows": 1}}, "corpusid_join": {"common_distinct_ids": 1}}
    visited = []

    def fake_report(data_dir):
        visited.append(data_dir)
        return report

    monkeypatch.setattr(profile_data, "build_report", fake_report)
    monkeypatch.setattr("sys.argv", ["profile_data.py"] + (["--stdout"] if stdout else []))
    profile_data.main()
    assert visited == [expected]
    target = expected / "profiling" / "d02_profile.json"
    if stdout:
        assert json.loads(capsys.readouterr().out) == report
        assert not target.exists()
    else:
        assert json.loads(target.read_text(encoding="utf-8")) == report
