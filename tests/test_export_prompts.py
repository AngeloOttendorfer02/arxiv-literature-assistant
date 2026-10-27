import json

import pytest

pytest.importorskip("numpy")

from benchmark import load_prompts
from export_prompts import context_stats, export, sample, write_jsonl

from scilitrag.config import load_settings
from scilitrag.contracts import Evidence
from scilitrag.rag.errors import RagUserError

SETTINGS = load_settings({"USE_MOCKS": "1"})


def query(id_, text):
    return {"id": id_, "query": text, "query_set": "s", "split": "dev", "relevant": [10]}


def fake_retrieve(plan):
    if plan.query == "nothing":
        raise RagUserError("No matching papers were found in the indexed corpus.")
    assert plan.scenario == "S1" and plan.topic == plan.query and plan.k == 5
    return [
        Evidence(label=1, type="paper", corpus_id=10, title="T", text="abstract " * 50),
        Evidence(label=2, type="chunk", corpus_id=10, chunk_id="10-0", title="T", text="x" * 5000),
    ]


def test_export_keeps_the_evidence_as_the_prompt_shows_it(tmp_path):
    records, skipped = export(
        [query("q000", "dense retrieval"), query("q001", "nothing")], fake_retrieve, SETTINGS
    )
    assert skipped == {"q001": "No matching papers were found in the indexed corpus."}
    [record] = records
    assert [item["label"] for item in record["evidence"]] == [1, 2]
    assert len(record["evidence"][1]["text"]) == SETTINGS.evidence_item_char_limit
    assert record["relevant"] == [10] and record["split"] == "dev"
    assert record["context"] == {
        "retrieved": 2,
        "shown": 2,
        "omitted": [],
        "truncated": 1,
        "target_retrieved": True,
        "target_shown": True,
    }
    path = tmp_path / "prompts.jsonl"
    write_jsonl(path, records)
    [loaded] = load_prompts(path)
    assert json.loads(loaded["messages"][-1]["content"])["question"] == "dense retrieval"


def test_sample_is_reproducible_and_keeps_order():
    records = [{"id": f"q{i:03d}"} for i in range(100)]
    first = sample(records, 30)
    assert first == sample(records, 30)
    assert len(first) == 30 and first == sorted(first, key=lambda r: r["id"])
    assert sample(records[:5], 30) == records[:5]


def test_context_stats_count_targets_dropped_by_the_budget():
    def context(retrieved, shown, omitted, target_retrieved, target_shown):
        return {
            "context": {
                "retrieved": retrieved,
                "shown": shown,
                "omitted": omitted,
                "truncated": 0,
                "target_retrieved": target_retrieved,
                "target_shown": target_shown,
            }
        }

    stats = context_stats(
        [context(15, 15, [], True, True), context(15, 12, [13, 14, 15], True, False)]
    )
    assert stats["retrieved_median"] == 15 and stats["shown_median"] == 13.5
    assert stats["prompts_with_omitted"] == 1 and stats["omitted_share"] == 0.5
    assert stats["target_shown"] == 1 and stats["target_dropped_by_budget"] == 1
    assert context_stats([]) == {}
