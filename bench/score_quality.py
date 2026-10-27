"""A-12: answer quality per model tag from benchmark.py records (D058).

Usage:
    uv run --locked --group data --group guardrails python bench/score_quality.py \
        eval/private/bench/prompts.jsonl eval/private/bench/engine.jsonl \
        --baseline qwen3.8:27b-bf16 --output eval/private/bench/quality.json

(A) Citation recall and precision as in ALCE (Gao et al., EMNLP 2023), with the app's NLI
model and entailment threshold: a sentence is supported when its cited evidence, joined,
entails it; a citation is precise when its sentence is supported and, with several
citations, the citation either entails the sentence alone or the others no longer do
without it. Labels that are not in the evidence count as imprecise citations. Answers whose
first sentence says the evidence holds nothing matching (NOT_FOUND) are counted as a rate
and left out of recall and precision.
(B) Gold-paper citation: the answer cites a LitSearch target paper, among queries whose
evidence contains one, "not found" answers included. Each quantization is compared with
the baseline tag by a paired bootstrap over queries. The output holds scores and labels
only, no dataset text; --review prints first sentences for a manual check.
"""

import argparse
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compare_embeddings import BOOTSTRAP_DRAWS, paired_bootstrap

from scilitrag.guardrails.sentences import split_sentences
from scilitrag.rag.evidence.labels import citation_labels

SEED = 1789
METRICS = ("not_found", "citation_recall", "citation_precision", "gold_cited")
SPLITS = ("dev", "test", "all")
LABEL = re.compile(r"\[\s*\d+\s*\]")
SOURCES = r"(?:evidence|sources?|papers?|passages?|context|abstracts?|excerpts?)"
# NLI cannot entail a statement about what the evidence lacks; answers that open with one
# are counted separately (D065). Check the flagged answers by hand.
NOT_FOUND = re.compile(
    rf"\b(?:{SOURCES}\s+(?:\w+\s+){{0,3}}(?:does|do)\s+not\s+(?:contain|include|mention|describe"
    rf"|discuss|address|provide|cover|report|identify|present)"
    rf"|none\s+of\s+the\s+(?:\w+\s+){{0,2}}{SOURCES}"
    rf"|no\s+(?:\w+\s+){{0,2}}{SOURCES}\s+(?:\w+\s+){{0,3}}(?:matches|match|addresses|address"
    rf"|describes|describe|mentions|mention|discusses|discuss)"
    rf"|(?:could|can)\s*not\s+(?:find|identify|locate)|unable\s+to\s+(?:find|identify|locate)"
    rf"|there\s+(?:is|are)\s+no\s+(?:\w+\s+){{0,2}}{SOURCES}"
    rf"|not\s+(?:found|present|available)\s+in\s+the\s+(?:\w+\s+){{0,2}}{SOURCES})",
    re.IGNORECASE,
)


def sentences(answer: str) -> list[str]:
    done, rest = split_sentences(answer)
    return [s for s in [*done, rest.strip()] if s]


def opens_with_not_found(claims: list[str]) -> bool:
    return bool(claims) and NOT_FOUND.search(claims[0]) is not None


def score_answer(answer: str, evidence: list[dict], relevant: list[int], entails) -> dict:
    """entails(premise, hypothesis) -> bool; sentences without a claim are skipped."""
    by_label = {item["label"]: item for item in evidence}
    rows, supported, precise, citations, claims = [], 0, 0, 0, []
    cited_papers = set()
    for sentence in sentences(answer):
        claim = " ".join(LABEL.sub(" ", sentence).split())
        if not any(ch.isalnum() for ch in claim):
            continue
        try:
            labels = citation_labels(sentence)
        except ValueError:
            labels = []
        known = [label for label in labels if label in by_label]
        cited_papers |= {by_label[label]["corpus_id"] for label in known}

        def joint(subset, claim=claim):
            return bool(subset) and entails("\n".join(by_label[x]["text"] for x in subset), claim)

        ok = bool(labels) and len(known) == len(labels) and joint(known)
        good = []
        for label in labels:
            if not ok:
                good.append(False)
            elif len(labels) == 1 or entails(by_label[label]["text"], claim):
                good.append(True)
            else:
                # ALCE: irrelevant when the remaining citations still entail without it.
                good.append(not joint([x for x in known if x != label]))
        supported += ok
        precise += sum(good)
        citations += len(labels)
        claims.append(claim)
        rows.append({"labels": labels, "supported": ok, "precise": good})
    in_evidence = {item["corpus_id"] for item in evidence} & set(relevant)
    not_found = opens_with_not_found(claims)
    return {
        "sentences": rows,
        "not_found": not_found,
        "citation_recall": None if not_found else supported / len(rows) if rows else 0.0,
        "citation_precision": None if not_found or not citations else precise / citations,
        "gold_cited": bool(cited_papers & set(relevant)) if in_evidence else None,
        "uncited_sentences": sum(not row["labels"] for row in rows),
    }


def load_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def per_query(scores: list[dict]) -> dict[tuple[str, str], dict]:
    """Average repeats of one prompt, so every query counts once per model."""
    grouped = defaultdict(list)
    for score in scores:
        grouped[score["model"], score["id"]].append(score)
    result = {}
    for key, runs in grouped.items():
        result[key] = {"split": runs[0]["split"]}
        for metric in METRICS:
            values = [float(r[metric]) for r in runs if r[metric] is not None]
            result[key][metric] = statistics.fmean(values) if values else None
    return result


def summarize(queries: dict, models: list[str], baseline: str | None) -> dict:
    summary = {}
    for model in models:
        own = {i: q for (m, i), q in queries.items() if m == model}
        summary[model] = {}
        for split in SPLITS:
            ids = [i for i, q in own.items() if split == "all" or q["split"] == split]
            row = {}
            for metric in METRICS:
                values = [own[i][metric] for i in ids if own[i][metric] is not None]
                row[metric] = statistics.fmean(values) if values else None
                row[f"{metric}_n"] = len(values)
                if baseline and model != baseline:
                    pairs = [
                        (own[i][metric], queries[baseline, i][metric])
                        for i in ids
                        if own[i][metric] is not None
                        and queries.get((baseline, i), {}).get(metric) is not None
                    ]
                    if pairs:
                        a, b = zip(*pairs, strict=True)
                        row[f"{metric}_vs_baseline"] = paired_bootstrap(a, b, BOOTSTRAP_DRAWS, SEED)
            summary[model][split] = row
    return summary


def print_summary(summary: dict, baseline: str | None) -> None:
    print("model\tsplit\t" + "\t".join(METRICS))
    for model, splits in summary.items():
        for split, row in splits.items():
            cells = []
            for metric in METRICS:
                value = row[metric]
                cell = "" if value is None else f"{value:.3f}"
                diff = row.get(f"{metric}_vs_baseline")
                if diff:
                    cell += (
                        f" ({diff['delta']:+.3f} [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}])"
                    )
                cells.append(cell)
            print(f"{model}\t{split}\t" + "\t".join(cells))
    if baseline:
        print(f"Differences against {baseline}, paired bootstrap, 95% interval.")


def review(requests: list[dict], scores: list[dict], size: int, seed: int = SEED) -> None:
    """Sample for checking NOT_FOUND by hand; console only, since answers quote papers."""
    answers = {(r["model"], r["id"], r.get("repeat", 0)): r["answer"] for r in requests}
    rng = random.Random(seed)
    for flagged in (True, False):
        group = [s for s in scores if s["not_found"] is flagged]
        print(f"\n== {'flagged' if flagged else 'not flagged'}: {len(group)}")
        for score in rng.sample(group, min(size, len(group))):
            first = sentences(answers[score["model"], score["id"], score["repeat"]])[:1]
            print(f"{score['model']} {score['id']}: {' '.join(first)[:300]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompts", type=Path, help="export_prompts.py output")
    parser.add_argument("records", type=Path, help="benchmark.py output")
    parser.add_argument("--baseline", help="model tag the others are compared with")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--review", type=int, default=0, help="print N flagged and N other first sentences"
    )
    args = parser.parse_args()

    from scilitrag.config import load_settings
    from scilitrag.guardrails.validation import NliModel

    prompts = {p["id"]: p for p in load_jsonl(args.prompts)}
    requests = [r for r in load_jsonl(args.records) if r.get("kind") == "request"]
    unknown = sorted({r["id"] for r in requests} - prompts.keys())
    if unknown:
        parser.error(f"records for ids missing in {args.prompts}: {', '.join(unknown[:5])}")
    models = list(dict.fromkeys(r["model"] for r in requests))
    if args.baseline and args.baseline not in models:
        parser.error(f"--baseline {args.baseline} has no records")

    settings = load_settings()
    nli = NliModel(settings.nli_model)
    cache = {}

    def entails(premise, hypothesis):
        if (premise, hypothesis) not in cache:
            probs = nli.probabilities(premise, hypothesis)
            cache[premise, hypothesis] = probs["entailment"] >= settings.nli_entailment_threshold
        return cache[premise, hypothesis]

    scores, failed = [], defaultdict(int)
    for record in requests:
        if "error" in record:
            failed[record["model"]] += 1
            continue
        prompt = prompts[record["id"]]
        score = score_answer(record["answer"], prompt["evidence"], prompt["relevant"], entails)
        scores.append(
            {
                "model": record["model"],
                "id": record["id"],
                "repeat": record.get("repeat", 0),
                "split": prompt["split"],
                "done_reason": record.get("done_reason"),
                **score,
            }
        )
        recall = score["citation_recall"]
        shown = "not found" if recall is None else f"recall {recall:.2f}"
        print(f"{record['model']} {record['id']} {shown}", file=sys.stderr)

    if args.review:
        review(requests, scores, args.review)
    summary = summarize(per_query(scores), models, args.baseline)
    result = {
        "nli_model": settings.nli_model,
        "nli_entailment_threshold": settings.nli_entailment_threshold,
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "baseline": args.baseline,
        "failed_requests": dict(failed),
        "truncated_answers": {
            model: sum(s["done_reason"] == "length" for s in scores if s["model"] == model)
            for model in models
        },
        "summary": summary,
        "answers": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_summary(summary, args.baseline)


if __name__ == "__main__":
    main()
