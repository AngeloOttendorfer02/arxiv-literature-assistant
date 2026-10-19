import copy
import json

import filter_data
import pytest
from filter_data import (
    RULES,
    analyze,
    annotation_present,
    export_filtered,
    filter_reasons,
    normalize_year,
    write_reports,
)
from profile_data import REVISION


def s2_row(cid, first=True, last=True, section=True, year=2020.0):
    span = json.dumps([{"start": "0", "end": "4"}])
    return {
        "corpusid": cid,
        "year": year,
        "content": {
            "text": "Anna Smith Introduction",
            "annotations": {
                "authorfirstname": span if first else None,
                "authorlastname": span if last else None,
                "sectionheader": span if section else None,
            },
        },
    }


def paper_row(cid, title="A paper", full_paper="Some full text"):
    return {
        "corpusid": cid,
        "title": title,
        "abstract": "Abstract",
        "full_paper": full_paper,
        "citations": [999],
    }


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (float("nan"), None),
        ("", None),
        ("  ", None),
        (2020.0, 2020),
        (0.0, 0),
        (2021, 2021),
    ],
)
def test_missing_years_become_null_without_discarding_valid_years(value, expected):
    assert normalize_year(value) == expected


@pytest.mark.parametrize("value", [True, 2020.5, float("inf"), "unknown"])
def test_invalid_years_abort_instead_of_silently_guessing(value):
    with pytest.raises(ValueError, match="Invalid year"):
        normalize_year(value)


def test_title_and_full_text_boundaries_and_overlapping_reasons():
    assert filter_reasons(paper_row(1, "x" * 500, "ab")) == []
    assert filter_reasons(paper_row(1, "x" * 501, "a")) == ["title_over_500", "full_paper_length_1"]
    assert filter_reasons(paper_row(1, "  ", "\n")) == ["empty_title", "empty_full_paper"]
    assert filter_reasons(paper_row(1, None, None)) == ["empty_title", "empty_full_paper"]


def test_annotation_presence_parses_json_and_numeric_offsets():
    row = s2_row(1)
    assert annotation_present(row, "authorfirstname")
    row["content"]["annotations"]["authorfirstname"] = "[]"
    assert not annotation_present(row, "authorfirstname")
    row["content"]["annotations"]["authorfirstname"] = '[{"start":4,"end":5}]'
    assert not annotation_present(row, "authorfirstname")
    row["content"]["annotations"]["authorfirstname"] = '[{"start":0,"end":999}]'
    with pytest.raises(ValueError, match="text span"):
        annotation_present(row, "authorfirstname")
    row["content"]["annotations"]["authorfirstname"] = "invalid JSON"
    with pytest.raises(ValueError):
        annotation_present(row, "authorfirstname")


def sample_rows():
    return {
        "query": [{"corpusids": [1, 2]}, {"corpusids": [1]}],
        "corpus_clean": [
            paper_row(1, "", ""),
            paper_row(2),
            paper_row(3, "x" * 501, "1"),
            paper_row(4),
            paper_row(5),
        ],
        "corpus_s2orc": [
            s2_row(1, False, False, False, float("nan")),
            s2_row(2, False, True),
            s2_row(3),
            s2_row(4, False, False),
            s2_row(5, True, False, True, None),
        ],
    }


def analyze_rows(monkeypatch, rows):
    monkeypatch.setattr(filter_data, "cached_subset", lambda root, name: name)
    monkeypatch.setattr(filter_data, "checked_rows", lambda name: iter(rows[name]))
    return analyze(None)


def test_protected_targets_survive_all_rules_and_rule_counts_do_not_sum_to_total(monkeypatch):
    rows = sample_rows()
    original = copy.deepcopy(rows)
    plan = analyze_rows(monkeypatch, rows)
    assert plan["retained"] == {1, 2, 5}
    assert set(plan["rejected"]) == {3, 4}
    assert plan["counts"]["retained_target_papers"] == 2
    assert plan["counts"]["targets_with_rejection_criteria"] == 1
    assert set(plan["target_exceptions"][1]) == {
        "empty_title",
        "empty_full_paper",
        "missing_sectionheader",
        "missing_author_names",
        "missing_year",
    }
    assert plan["rules"]["missing_author_names"] == {
        "affected": 2,
        "rejected": 1,
        "protected_targets": 1,
    }
    assert plan["missing_years"] == {"input": 2, "retained": 2, "rejected": 0, "targets": 1}
    assert sum(count["rejected"] for count in plan["rules"].values()) == 3
    assert plan["counts"]["rejected_papers"] == 2
    assert rows["corpus_clean"] == original["corpus_clean"]
    assert rows["query"] == original["query"]


@pytest.mark.parametrize(
    "defect", ["missing_target", "missing_s2orc", "duplicate", "invalid_query"]
)
def test_missing_targets_and_invalid_joins_abort_before_export(monkeypatch, defect):
    rows = sample_rows()
    if defect == "missing_target":
        rows["query"].append({"corpusids": [999]})
    elif defect == "missing_s2orc":
        rows["corpus_s2orc"].pop()
    elif defect == "duplicate":
        rows["corpus_clean"].append(paper_row(5))
    else:
        rows["query"] = [{"corpusids": []}]
    with pytest.raises(ValueError):
        analyze_rows(monkeypatch, rows)


def test_reports_include_all_counts_and_separate_protected_papers(tmp_path, monkeypatch):
    plan = analyze_rows(monkeypatch, sample_rows())
    write_reports(plan, tmp_path / "docs", exported=False)
    report = (tmp_path / "docs" / "d03_quality_filters.md").read_text(encoding="utf-8")
    details = (tmp_path / "docs" / "d03_target_exceptions.md").read_text(encoding="utf-8")
    assert "Analysis only" in report
    for label in RULES.values():
        assert label in report
    assert "| 1 |" in details and "missing_year" in details
    assert "| 3 |" not in details
    target_section = report.split("\n## Protected target papers\n", 1)[1]
    assert "\n| 1 |" in target_section and "missing_year" in target_section
    assert "\n| 3 |" not in target_section
    assert not (tmp_path / "filtered").exists()


def test_targets_with_oversized_titles_and_one_character_texts_survive(monkeypatch):
    rows = sample_rows()
    rows["corpus_clean"][1] = paper_row(2, "x" * 501, "1")
    plan = analyze_rows(monkeypatch, rows)
    assert 2 in plan["retained"]
    assert set(plan["target_exceptions"][2]) == {"title_over_500", "full_paper_length_1"}
    assert plan["rules"]["title_over_500"]["protected_targets"] == 1


def test_failed_export_does_not_publish_partial_dataset(tmp_path, monkeypatch):
    plan = {"caches": {"query": ({}, [])}, "retained": set(), "metadata": {}}

    def fail_export(*args):
        raise OSError("Simulated disk error")

    monkeypatch.setattr(filter_data, "export_subset", fail_export)
    with pytest.raises(OSError, match="Simulated disk error"):
        export_filtered(plan, tmp_path)
    assert not (tmp_path / "filtered" / "d03").exists()
    assert list((tmp_path / "filtered").iterdir()) == []


def test_cli_report_only_honors_environment_override_and_does_not_export(tmp_path, monkeypatch):
    plan = analyze_rows(monkeypatch, sample_rows())
    monkeypatch.setattr(filter_data, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("DATA_DIR=local_data\n", encoding="utf-8")
    mounted = tmp_path / "mounted_data"
    monkeypatch.setenv("DATA_DIR", str(mounted))
    visited = []

    def inspect(data_dir):
        visited.append(data_dir)
        return plan

    monkeypatch.setattr(filter_data, "analyze", inspect)
    monkeypatch.setattr(filter_data, "export_filtered", lambda *a: pytest.fail("Unexpected export"))
    monkeypatch.setattr("sys.argv", ["filter_data.py", "--report-only"])
    filter_data.main()
    assert visited == [mounted]
    assert (tmp_path / "docs" / "d03_quality_filters.md").is_file()
    assert not mounted.exists()


def test_arrow_export_keeps_corpora_aligned_query_targets_unchanged_and_years_null(tmp_path):
    import pyarrow as pa
    from pyarrow import ipc

    rows = sample_rows()
    source_bytes = {}
    for name, records in rows.items():
        root = tmp_path / "hf_datasets" / "princeton-nlp___lit_search" / name / "0.0.0" / REVISION
        root.mkdir(parents=True)
        table = pa.Table.from_pylist(records)
        path = root / "lit_search-full.arrow"
        with pa.OSFile(str(path), "wb") as sink, ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        source_bytes[path] = path.read_bytes()
        (root / "dataset_info.json").write_text(
            json.dumps({"splits": {"full": {"num_examples": len(records)}}}), encoding="utf-8"
        )
    plan = analyze(tmp_path)
    destination = export_filtered(plan, tmp_path)
    for subset in ("corpus_clean", "corpus_s2orc"):
        with pa.memory_map(str(destination / f"{subset}.arrow"), "r") as source:
            table = ipc.open_stream(source).read_all()
        assert table.column("corpusid").to_pylist() == [1, 2, 5]
        assert table.column("year").to_pylist() == [None, 2020, None]
        assert table.schema.field("year").type == pa.int64()
    with pa.memory_map(str(destination / "query.arrow"), "r") as source:
        assert ipc.open_stream(source).read_all().to_pylist() == rows["query"]
    with pa.memory_map(str(destination / "corpus_clean.arrow"), "r") as source:
        assert ipc.open_stream(source).read_all().column("citations").to_pylist() == [[999]] * 3
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["retained_corpusids"] == [1, 2, 5]
    assert all(path.read_bytes() == before for path, before in source_bytes.items())
    with pytest.raises(FileExistsError):
        export_filtered(plan, tmp_path)
