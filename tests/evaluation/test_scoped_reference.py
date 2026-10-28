import copy

import pytest
from pydantic import ValidationError

from scilitrag.contracts import Evidence, Plan, SqlFacts
from scilitrag.evaluation.datasets import ReferenceSet, run_fingerprint
from scilitrag.evaluation.reference import build_reference_set, main
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.scenarios.authors import author_facts


def source_run():
    plan = Plan(
        scenario="S3", topic="retrieval", query="Rank up to 2 cited authors for retrieval.", k=2
    )
    return {
        "id": "actual-s3",
        "model": "candidate",
        "dataset_revision": "v1",
        "plan": plan.model_dump(),
        "evidence": [Evidence(type="paper", label=1, corpus_id=2, text="seed").model_dump()],
        "events": [
            {
                "event": "table",
                "data": SqlFacts(
                    query_id="q",
                    sql="SELECT production",
                    dataset_revision="v1",
                    parameters={"seed_ids": "[2]", "limit": 2, "author_id": None},
                    columns=["author_id", "citation_count"],
                    rows=[[10, 999]],
                ).model_dump(),
            }
        ],
    }


def reader(sql, params=None):
    if sql.startswith("SELECT corpus_id"):
        return [{"corpus_id": i} for i in params["ids"]]
    if sql.startswith("SELECT dataset_revision"):
        return [{"dataset_revision": "v1"}]
    if sql.startswith("SELECT p.corpus_id"):
        return [{"corpus_id": 2, "year": 2021}]
    if sql.startswith("SELECT c.chunk_id"):
        return [
            {
                "chunk_id": f"{i}-0",
                "corpus_id": i,
                "section": "Experiments",
                "text": "We use BERT as a baseline." if i % 2 else "BERT was mentioned.",
                "target_title": "BERT",
            }
            for i in range(2, 42)
        ]
    assert "AUTHOR_SQL" not in sql and params["seeds"] == [2]
    # Recompute distinct pair credits rather than trusting the source's wrong 999.
    return [
        {"citing_id": 2, "cited_id": p, "author_id": a, "name": n}
        for p, a, n in [(1, 10, "Alice"), (1, 10, "Alice"), (4, 20, "Bob"), (5, 20, "Bob")]
    ]


def test_reference_uses_recorded_query_seeds_limit_and_independent_credits():
    source = source_run()
    bundle = build_reference_set(reader, [], [], "ignored", source_run=source)
    case = bundle["cases"][0]
    assert case["id"] == "actual-s3" and case["query"] == source["plan"]["query"]
    assert case["seed_ids"] == [2]
    assert case["expected_author_counts"] == {"20": 2, "10": 1}
    assert case["source_run_sha256"] == run_fingerprint(source)


@pytest.mark.parametrize("change", ["revision", "seeds", "limit", "scenario", "author", "changed"])
def test_incompatible_source_scope_is_rejected(change):
    source = source_run()
    if change == "revision":
        source["dataset_revision"] = "other"
    elif change == "seeds":
        source["events"][0]["data"]["parameters"]["seed_ids"] = "[3]"
    elif change == "limit":
        source["events"][0]["data"]["parameters"]["limit"] = 5
    elif change == "scenario":
        source["plan"]["scenario"] = "S1"
    elif change == "author":
        source["events"][0]["data"]["parameters"]["author_id"] = 10
    else:
        source["dataset_changed"] = True
    with pytest.raises((ValueError, RagUserError)):
        build_reference_set(reader, [], [], "retrieval", source_run=source)


def test_sample_is_twenty_total_across_targets_with_empty_labels_and_bound_sources():
    bundle = build_reference_set(reader, [1, 3], [], "retrieval")
    rows = bundle["baseline_annotations"]
    assert len(rows) == 20 and {row["target_id"] for row in rows} == {1, 3}
    assert all(row["baseline_use"] is None for row in rows)
    assert len({(r["target_id"], r["corpus_id"]) for r in rows}) == 20
    changed = copy.deepcopy(bundle)
    changed["baseline_annotations"][0]["text"] += " changed"
    with pytest.raises(ValidationError, match="source content"):
        ReferenceSet.model_validate(changed)
    bundle["baseline_annotations"][0]["baseline_use"] = False
    ReferenceSet.model_validate(bundle)


def test_cli_preserves_existing_reference_file(tmp_path):
    path = tmp_path / "reviewed.json"
    path.write_text("keep reviewed labels")
    with pytest.raises(SystemExit):
        main(["--target-paper", "1", "--output", str(path)])
    assert path.read_text() == "keep reviewed labels"


def test_independent_reference_matches_production_count_scope(fixture_reader):
    plan = Plan(
        scenario="S3", topic="retrieval", query="Rank up to 2 cited authors for retrieval.", k=2
    )
    evidence = [
        Evidence(type="paper", label=i, corpus_id=p, text="seed") for i, p in enumerate([2, 5], 1)
    ]
    table = author_facts(plan, evidence, fixture_reader)[0]
    record = {
        "id": "s3",
        "model": "candidate",
        "dataset_revision": "synthetic-v1",
        "plan": plan.model_dump(),
        "evidence": [e.model_dump() for e in evidence],
        "events": [{"event": "table", "data": table.model_dump()}],
    }
    bundle = build_reference_set(fixture_reader, [], [], "retrieval", source_run=record)
    expected = {str(row[0]): row[2] for row in table.rows}
    assert bundle["cases"][0]["expected_author_counts"] == expected
