import pytest

from scilitrag.evaluation import latency as lat


def record(status="ok", pg=30.0, gen=100.0, ttft=(50.0, 70.0), steps=None, checked=3):
    return {
        "status": status,
        "total_ms": 500.0,
        "ttft_engine_ms": ttft[0],
        "ttft_user_ms": ttft[1],
        "stage_latency_ms": {
            "prompt_guard": pg,
            "topic_gate": 5.0,
            "pii": 1.0,
            "evidence_guard": 200.0,
            "generate": gen,
        },
        "component_ms": steps or {},
        "sentences_checked": checked,
        "sentences_removed": 1,
    }


def test_percentile_uses_nearest_rank():
    values = [float(n) for n in range(1, 21)]
    assert lat.percentile(values, 0.5) == 10.0
    assert lat.percentile(values, 0.95) == 19.0
    assert lat.percentile([7.0], 0.95) == 7.0
    assert lat.percentile([], 0.5) is None


def test_the_first_run_of_each_stage_is_its_cold_start_and_not_in_the_distributions():
    records = [record(pg=900.0), record(pg=30.0), record(pg=40.0), record("refused")]
    summary = lat.summarize(records)
    assert summary["cold_start_ms"]["prompt_guard"] == 900.0
    assert summary["stages"]["prompt_guard"]["n"] == 3
    assert summary["stages"]["prompt_guard"]["median"] == 30.0
    assert summary["requests"]["total"] == 4 and summary["requests"]["refused"] == 1
    assert summary["requests"]["in_per_request_figures"] == 2


def test_a_refused_first_request_still_marks_the_cold_start_of_its_stages():
    refused = {
        "status": "refused",
        "total_ms": 9.0,
        "stage_latency_ms": {"prompt_guard": 5000.0, "topic_gate": 900.0},
        "component_ms": {},
    }
    summary = lat.summarize([refused, record(pg=30.0), record(pg=40.0)])
    assert summary["cold_start_ms"]["prompt_guard"] == 5000.0
    assert summary["cold_start_ms"]["evidence_guard"] == 200.0
    assert summary["stages"]["prompt_guard"]["median"] == 30.0


def test_skip_cold_can_be_turned_off():
    summary = lat.summarize([record(pg=900.0), record(pg=30.0)], skip_cold=False)
    assert summary["cold_start_ms"] == {}
    assert summary["stages"]["prompt_guard"]["n"] == 2


def test_wait_for_the_first_sentence_is_the_user_minus_the_engine_ttft():
    summary = lat.summarize(
        [record(), record(), record(ttft=(50.0, 90.0)), record(ttft=(10.0, 70.0))]
    )
    assert summary["wait_for_first_sentence_ms"]["median"] == 40.0


def test_sentence_check_time_and_guardrail_total_come_from_recorded_steps():
    steps = {
        "check_sentence_ms": 300.0,
        "mask_pii_ms": 3.0,
        "render_citations_ms": 2.0,
        "check_sentence_calls": 3,
    }
    summary = lat.summarize([record(steps=steps)] * 3)
    assert summary["sentence_checks"]["sentence_checks_total"]["median"] == 305.0
    assert summary["sentence_checks"]["check_sentence_per_call"]["median"] == 100.0
    assert summary["sentence_checks"]["generate_without_checks"]["median"] == -205.0
    assert summary["guardrail_total_ms"]["median"] == pytest.approx(30 + 5 + 1 + 200 + 305)


def test_without_recorded_steps_the_total_covers_the_stages_only():
    summary = lat.summarize([record(), record(), record()])
    assert summary["sentence_checks"] == {}
    assert summary["guardrail_total_ms"]["median"] == pytest.approx(236.0)
    assert summary["sentences"] == {"checked": 9, "removed": 3}
