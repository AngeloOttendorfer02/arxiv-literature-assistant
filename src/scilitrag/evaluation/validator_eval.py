"""Offline scoring and threshold choice for the sentence validator (S-15).

Per case the number check and the best NLI scores are recorded once, so the entailment threshold
can be replayed from them and chosen on the tuning split only.
"""

from scilitrag.evaluation.guardrail_eval import FRR_CAP, rate

ENTAILMENT_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
FAITHFUL = (
    "faithful_verbatim",
    "faithful_paraphrase",
    "faithful_two_sources",
    "faithful_combined",
)
MANIPULATED = (
    "number",
    "year",
    "negation",
    "addition",
    "wrong_paper",
    "number_swap",
    "comparison",
    "quantifier",
)
# Number swaps inside one passage, changed comparisons and scopes, and true sentences that
# combine two passages: the number check cannot help and NLI has to decide.
HARD = ("number_swap", "comparison", "quantifier", "faithful_combined")
# True sentences that need both cited passages: check_sentence judges each passage alone (D030),
# so no threshold keeps them. They are reported on their own and do not enter the cap.
COMBINED = "faithful_combined"


def predict_status(record: dict, entailment: float, contradiction: float) -> str:
    """The status check_sentence returns for recorded scores at the given thresholds."""
    if not record["number_ok"]:
        return "fact_mismatch"
    nli = record["nli"]
    if nli is None:
        return "ok"
    if nli["entailment"] >= entailment:
        return "ok"
    if nli["contradiction"] >= contradiction:
        return "fact_mismatch"
    return "unsupported"


def score_cases(records: list[dict], entailment: float, contradiction: float) -> dict:
    """Recall and precision of removing manipulated sentences, and false removals of faithful ones."""
    removed = {r["id"]: predict_status(r, entailment, contradiction) != "ok" for r in records}
    manipulated = [r for r in records if r["expected"] == "removed"]
    faithful = [r for r in records if r["expected"] == "ok"]
    single = [r for r in faithful if r["manipulation"] != COMBINED]
    hits = sum(removed[r["id"]] for r in manipulated)
    all_removed = sum(removed.values())
    by_kind = {}
    for kind in (*MANIPULATED, *FAITHFUL):
        items = [r for r in records if r["manipulation"] == kind]
        if not items:
            continue
        by_kind[kind] = rate(sum(removed[r["id"]] for r in items), len(items))
    by_number_stage = sum(1 for r in manipulated if removed[r["id"]] and not r["number_ok"])
    return {
        "thresholds": {"entailment": entailment, "contradiction": contradiction},
        "recall": rate(hits, len(manipulated)),
        "precision": rate(hits, all_removed),
        "false_removal": rate(sum(removed[r["id"]] for r in faithful), len(faithful)),
        "false_removal_single_source": rate(sum(removed[r["id"]] for r in single), len(single)),
        "removed_by_kind": by_kind,
        "removed_by_number_check": rate(by_number_stage, len(manipulated)),
    }


def select_entailment_threshold(
    tuning: list[dict], contradiction: float, frr_cap: float = FRR_CAP, default: float = 0.5
) -> float:
    """Highest recall on tuning with at most the cap of faithful sentences removed.

    Sentences that need two passages (`faithful_combined`) do not enter the cap. Ties go to the lower false removal rate, then the threshold closest to the default.
    """
    best = None
    for threshold in ENTAILMENT_GRID:
        scored = score_cases(tuning, threshold, contradiction)
        false_rate = scored["false_removal_single_source"]["rate"]
        if false_rate is None or false_rate > frr_cap:
            continue
        key = (scored["recall"]["rate"], -false_rate, -abs(threshold - default))
        if best is None or key > best[0]:
            best = (key, threshold)
    if best is None:
        raise ValueError("No threshold keeps faithful sentences within the cap.")
    return best[1]


def hard_cases(records: list[dict]) -> list[dict]:
    return [r for r in records if r["manipulation"] in HARD]


def easy_cases(records: list[dict]) -> list[dict]:
    return [r for r in records if r["manipulation"] not in HARD]
