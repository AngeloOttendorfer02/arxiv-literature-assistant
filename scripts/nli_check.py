"""S-09 NLI check (D057): synthetic sentence pairs from LitSearch abstracts, three NLI models.

Runs outside the project lock because the project pins CPU-only torch:
    uv run --no-project --python 3.13 --index https://download.pytorch.org/whl/cu130 \
        --with torch==2.14.1+cu130 --with transformers==5.20.2 --with numpy==2.4.6 \
        --with pyarrow==26.0.0 --with python-dotenv==1.2.4 python scripts/nli_check.py

Per abstract one hypothesis of each kind: a sentence of the abstract (entail, must pass), the
abstract's sentences negated or with a changed number, and a sentence of the most similar
other abstract by granite embedding (wrong_paper); these three must fail. Premise: the abstract alone
(short) or shuffled with two random abstracts (long, about chunk length). Needs the D-22
granite cache in DATA_DIR/embeddings. Scores on the GPU in fp16 (--device cpu: fp32, unbatched,
as a check of that shortcut); latency always on the CPU in fp32 with 4 threads.
Full results (scores only) go to eval/results/s09_nli_check.json, aggregates to
docs/s09_nli_check.md.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
from compare_embeddings import read_table
from select_fixture import data_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = PROJECT_ROOT / "eval" / "results" / "s09_nli_check.json"
REPORT = PROJECT_ROOT / "docs" / "s09_nli_check.md"
# The D057 numbers were produced with this seed; another seed draws other pairs.
SEED = 42
PAPERS = 150
MODELS = {
    "EttinX-nli-s": ("dleemiller/EttinX-nli-s", "db0dfbf65c68dcd4747f384257a44f18776be709"),
    "ModernCE-base-nli": (
        "dleemiller/ModernCE-base-nli",
        "c7bdbbe134ed451db33a946bab9221712e589c35",
    ),
    "ModernCE-large-nli": (
        "dleemiller/ModernCE-large-nli",
        "0c0575b16b1e4614c4fa8fbf5dbc1a98476f2aaf",
    ),
}
EMBEDDINGS = "granite-embedding-english-r2@47ea694-d03.npz"
KINDS = ("entail", "negation", "number", "wrong_paper")
LENGTHS = ("short", "long")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
LATENCY_PAIRS = 15
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
AUX = re.compile(r"\b(is|are|was|were|can|does|do|will|has|have|could|should)\b")
NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
# The entailment index is read from these pairs, since config.json label orders disagree (D029).
SANITY = [
    (
        "The model was trained on 10,000 English documents.",
        "The model was trained on English documents.",
    ),
    (
        "The model was trained on 10,000 English documents.",
        "The model was not trained on English documents.",
    ),
]


def sentences(text):
    return [s.strip() for s in SPLIT.split(text or "") if 8 <= len(s.split()) <= 45]


def negate(sentence):
    return AUX.sub(lambda m: m.group(1) + " not", sentence, count=1)


def change_number(sentence):
    def bump(match):
        value = match.group(0)
        return str(round(float(value) * 2 + 1, 2)) if "." in value else str(int(value) * 2 + 1)

    return NUM.sub(bump, sentence, count=1)


def build_pairs(abstracts, nearest, papers, seed=SEED):
    """(paper, kind, premise length, premise, hypothesis) for randomly drawn abstracts."""
    rng = random.Random(seed)
    candidates = [i for i, a in enumerate(abstracts) if len(sentences(a)) >= 4]
    pairs = []
    for i in rng.sample(candidates, papers):
        abstract, own = abstracts[i], sentences(abstracts[i])
        others = [abstracts[j] for j in rng.sample(candidates, 2)]
        long_premise = "\n\n".join(rng.sample([abstract, *others], 3))
        hypotheses = {"entail": rng.choice(own)}
        if with_aux := [s for s in own if AUX.search(s)]:
            hypotheses["negation"] = negate(rng.choice(with_aux))
        if with_number := [s for s in own if NUM.search(s)]:
            hypotheses["number"] = change_number(rng.choice(with_number))
        if neighbor := sentences(abstracts[nearest(i)]):
            hypotheses["wrong_paper"] = rng.choice(neighbor)
        for kind, hypothesis in hypotheses.items():
            pairs.append((i, kind, "short", abstract, hypothesis))
            pairs.append((i, kind, "long", long_premise, hypothesis))
    return pairs


def pass_rates(scores, pairs):
    """Share of pairs per kind whose entailment probability reaches each threshold."""
    table = {}
    for length in LENGTHS:
        for threshold in THRESHOLDS:
            row = {}
            for kind in KINDS:
                chosen = [s for s, p in zip(scores, pairs, strict=True) if p[1:3] == (kind, length)]
                row[kind] = sum(s >= threshold for s in chosen) / len(chosen)
            table[f"{length}@{threshold}"] = row
    return table


def evaluate(name, pairs, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_id, revision = MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).eval()
    model = model.cuda().half() if device == "cuda" else model
    batch_size = 32 if device == "cuda" else 1

    def probabilities(batch):
        encoded = tokenizer(
            [p for p, _ in batch],
            [h for _, h in batch],
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(device)
        with torch.no_grad():
            return model(**encoded).logits.float().softmax(-1).cpu().numpy()

    sanity = probabilities(SANITY)
    entail = int(sanity[0].argmax())
    scores = []
    for start in range(0, len(pairs), batch_size):
        batch = [(p[3], p[4]) for p in pairs[start : start + batch_size]]
        scores.extend(probabilities(batch)[:, entail].tolist())
    lengths = [len(tokenizer(p[3], p[4])["input_ids"]) for p in pairs]

    torch.set_num_threads(4)
    cpu = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).eval()
    latency = {}
    for length in LENGTHS:
        times = []
        for pair in [p for p in pairs if p[2] == length][:LATENCY_PAIRS]:
            encoded = tokenizer(pair[3], pair[4], return_tensors="pt", truncation=True)
            start = time.perf_counter()
            with torch.no_grad():
                cpu(**encoded)
            times.append((time.perf_counter() - start) * 1000)
        latency[length] = float(np.median(times))
    return {
        "revision": revision,
        "pass_rate": pass_rates(scores, pairs),
        "counts": {k: sum(p[1:3] == (k, "short") for p in pairs) for k in KINDS},
        "cpu_ms": latency,
        "median_tokens": {
            length: float(
                np.median([n for n, p in zip(lengths, pairs, strict=True) if p[2] == length])
            )
            for length in LENGTHS
        },
        "entail_index": entail,
        "sanity_argmax": [int(row.argmax()) for row in sanity],
    }


def write_report(result, path):
    lines = [
        "# S-09 NLI check",
        "",
        (
            f"{result['pairs']:,} synthetic pairs from {result['papers']} random LitSearch abstracts "
            f"(D-03 corpus, seed {result['seed']}). Per abstract: one of its sentences (entail, "
            "must pass), a sentence of the abstract negated (negation) or with a changed number "
            "(number), and a sentence of the most similar other abstract by granite embedding "
            "(wrong_paper); the last three must fail. Premise: the abstract (short) or the "
            "abstract shuffled with two random ones (long). A pair passes when the entailment "
            "probability reaches the threshold. Scores on "
            f"{result['device']}, latency on the CPU in fp32 with 4 threads (median of "
            f"{LATENCY_PAIRS} pairs). Generated by `scripts/nli_check.py`; basis of D057."
        ),
        "",
        "## Pass rate at threshold 0.5",
        "",
        "| Model | Premise | Median tokens | " + " | ".join(KINDS) + " |",
        "|---|---|---:|" + "---:|" * len(KINDS),
    ]
    for name, model in result["models"].items():
        for length in LENGTHS:
            row = model["pass_rate"][f"{length}@0.5"]
            values = " | ".join(f"{row[k]:.2f}" for k in KINDS)
            lines.append(f"| {name} | {length} | {model['median_tokens'][length]:.0f} | {values} |")
    lines += [
        "",
        "## Entailed sentences passing, by threshold",
        "",
        "| Model | Premise | " + " | ".join(str(t) for t in THRESHOLDS) + " |",
        "|---|---|" + "---:|" * len(THRESHOLDS),
    ]
    for name, model in result["models"].items():
        for length in LENGTHS:
            values = " | ".join(
                f"{model['pass_rate'][f'{length}@{t}']['entail']:.2f}" for t in THRESHOLDS
            )
            lines.append(f"| {name} | {length} | {values} |")
    lines += [
        "",
        "## Highest pass rate of a wrong sentence (negation, number, wrong_paper), by threshold",
        "",
        "| Model | Premise | " + " | ".join(str(t) for t in THRESHOLDS) + " |",
        "|---|---|" + "---:|" * len(THRESHOLDS),
    ]
    for name, model in result["models"].items():
        for length in LENGTHS:
            values = " | ".join(
                f"{max(model['pass_rate'][f'{length}@{t}'][k] for k in KINDS[1:]):.2f}"
                for t in THRESHOLDS
            )
            lines.append(f"| {name} | {length} | {values} |")
    lines += [
        "",
        "## Size and CPU latency per pair",
        "",
        "| Model | Revision | Short premise, ms | Long premise, ms |",
        "|---|---|---:|---:|",
    ]
    for name, model in result["models"].items():
        lines.append(
            f"| {name} | `{model['revision'][:7]}` | {model['cpu_ms']['short']:.0f} "
            f"| {model['cpu_ms']['long']:.0f} |"
        )
    counts = next(iter(result["models"].values()))["counts"]
    lines += [
        "",
        "Pairs per kind and premise: " + ", ".join(f"{k} {counts[k]}" for k in KINDS) + ".",
        f"GPU: {result['gpu']}. CPU: {result['cpu']}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--papers", type=int, default=PAPERS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    import platform

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.exit(1, "CUDA is not available. Run with the CUDA torch build (see docstring).\n")
    data_dir = data_directory()
    corpus = read_table(data_dir / "filtered" / "d03" / "corpus_clean.arrow")
    ids = corpus.column("corpusid").to_pylist()
    abstracts = corpus.column("abstract").to_pylist()
    cache = np.load(data_dir / "embeddings" / EMBEDDINGS)
    if cache["ids"].tolist() != ids:
        parser.exit(
            1, "The granite cache does not match the D-03 corpus; run compare_embeddings.\n"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = torch.from_numpy(cache["emb"]).to(device)

    def nearest(i):
        similarity = emb @ emb[i]
        similarity[i] = -1
        return int(torch.argmax(similarity))

    pairs = build_pairs(abstracts, nearest, args.papers)
    print(f"{len(pairs)} pairs from {args.papers} abstracts")
    result = {
        "seed": SEED,
        "papers": args.papers,
        "pairs": len(pairs),
        "device": "GPU in fp16, batches of 32"
        if args.device == "cuda"
        else "CPU in fp32, unbatched",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
        "cpu": platform.processor() or platform.machine(),
        "versions": {"torch": torch.__version__},
        "models": {},
    }
    for name in args.models:
        result["models"][name] = evaluate(name, pairs, args.device)
        model = result["models"][name]
        print(
            name,
            "entail@0.5 short/long:",
            model["pass_rate"]["short@0.5"]["entail"],
            model["pass_rate"]["long@0.5"]["entail"],
            "cpu ms:",
            model["cpu_ms"],
        )
        if args.device == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    if not args.no_report:
        write_report(result, REPORT)


if __name__ == "__main__":
    main()
