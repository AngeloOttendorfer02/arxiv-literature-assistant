"""D-22: compare embedding models on LitSearch paper retrieval (title + abstract).

Runs outside the project lock because the project pins CPU-only torch:
    uv run --no-project --python 3.13 --index https://download.pytorch.org/whl/cu130 \
        --with torch==2.14.1 --with sentence-transformers==6.2.0 --with numpy==2.5.4 \
        --with pyarrow==26.0.0 --with python-dotenv==1.2.4 \
        python scripts/compare_embeddings.py

Reads the D-03 output from DATA_DIR/filtered/d03 (--raw: the D-01 cache instead).
Embeddings are cached under DATA_DIR/embeddings. Full results go to
eval/private/embedding_comparison.json, the aggregate report to
docs/d22_embedding_comparison.md. No dataset text is written to the repository.
"""

import argparse
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVISION = "9573fb284a1026c998df47024b888a163f0f0e25"
MAX_SEQ_LENGTH = 512
TOP_N = 100
SEED = 42
BOOTSTRAP_DRAWS = 10_000

# key: (HF model id, pinned revision, query prefix, document prefix); prefixes from the model cards.
MODELS = {
    "bge-base-en-v1.5": (
        "BAAI/bge-base-en-v1.5",
        "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    # BAAI: the v1.5 query instruction has little effect; test both settings on our task.
    "bge-base-en-v1.5-noinstr": (
        "BAAI/bge-base-en-v1.5",
        "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "",
        "",
    ),
    "granite-embedding-english-r2": (
        "ibm-granite/granite-embedding-english-r2",
        "47ea694b257b703fee9253d75c2b1f2985180498",
        "",
        "",
    ),
    "cadet-embed-base-v1": (
        "manveertamber/cadet-embed-base-v1",
        "0f8d867073d8fc0c239768a1965938889ed4acb2",
        "query: ",
        "passage: ",
    ),
}
BASELINE = "bge-base-en-v1.5"
METRICS = ("recall@5", "recall@20", "mrr", "ndcg@10")


def query_metrics(ranked, relevant):
    """Paper-level metrics for one query with binary relevance."""
    relevant = set(relevant)
    hits = [doc in relevant for doc in ranked]
    first = next((i for i, hit in enumerate(hits) if hit), None)
    dcg = sum(1 / math.log2(i + 2) for i, hit in enumerate(hits[:10]) if hit)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), 10)))
    return {
        "recall@5": sum(hits[:5]) / len(relevant),
        "recall@20": sum(hits[:20]) / len(relevant),
        "mrr": 0.0 if first is None else 1 / (first + 1),
        "ndcg@10": dcg / ideal,
    }


def split_queries(query_sets, seed=SEED):
    """Deterministic 50/50 dev/test split, stratified by query_set."""
    rng = np.random.default_rng(seed)
    split = [""] * len(query_sets)
    for name in sorted(set(query_sets)):
        members = [i for i, value in enumerate(query_sets) if value == name]
        rng.shuffle(members)
        half = (len(members) + 1) // 2
        for position, index in enumerate(members):
            split[index] = "dev" if position < half else "test"
    return split


def paired_bootstrap(a, b, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """Mean difference a - b with a 95% percentile interval over resampled queries."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    samples = np.random.default_rng(seed).integers(0, len(diff), size=(draws, len(diff)))
    means = diff[samples].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"delta": float(diff.mean()), "ci_low": float(low), "ci_high": float(high)}


def mean_metrics(rows, indices):
    return {m: float(np.mean([rows[i][m] for i in indices])) for m in METRICS}


def read_table(path):
    import pyarrow as pa
    from pyarrow import ipc

    with pa.memory_map(str(path), "r") as source, ipc.open_stream(source) as reader:
        return reader.read_all()


def load_data(data_dir, raw):
    if raw:
        base = data_dir / "hf_datasets" / "princeton-nlp___lit_search"
        import pyarrow as pa

        def table(config):
            files = sorted((base / config / "0.0.0" / REVISION).glob("*.arrow"))
            return pa.concat_tables(read_table(f) for f in files)

        return table("query"), table("corpus_clean"), "raw"
    folder = data_dir / "filtered" / "d03"
    return read_table(folder / "query.arrow"), read_table(folder / "corpus_clean.arrow"), "d03"


def document_text(title, abstract):
    title, abstract = (title or "").strip(), (abstract or "").strip()
    return f"{title}\n{abstract}" if title and abstract else title or abstract


def select_docs(ids, targets, limit):
    """All targets plus random distractors, for smoke runs."""
    if not limit:
        return list(range(len(ids)))
    keep = [i for i, doc in enumerate(ids) if doc in targets]
    others = [i for i, doc in enumerate(ids) if doc not in targets]
    rng = np.random.default_rng(SEED)
    extra = rng.choice(others, size=max(0, min(limit - len(keep), len(others))), replace=False)
    return sorted(keep + [int(i) for i in extra])


def load_model(name, device):
    import torch
    from sentence_transformers import SentenceTransformer

    model_id, revision, _, _ = MODELS[name]
    # fp16 on GPU, fp32 on CPU (granite would otherwise load in bf16, slow on most CPUs).
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = SentenceTransformer(
        model_id, revision=revision, device=device, model_kwargs={"dtype": dtype}
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def encode(model, texts, batch_size):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)


def search(query_emb, doc_emb, doc_ids, device):
    import torch

    q = torch.from_numpy(query_emb).to(device)
    d = torch.from_numpy(doc_emb).to(device)
    top = torch.topk(q @ d.T, k=min(TOP_N, d.shape[0]), dim=1).indices.cpu().numpy()
    return [[doc_ids[j] for j in row] for row in top]


def cpu_query_latency(name, queries, runs=50):
    """Median single-query encode time on CPU, as at runtime and for topic gating."""
    import torch

    torch.set_num_threads(os.cpu_count() or 1)
    model = load_model(name, "cpu")
    prefix = MODELS[name][2]
    model.encode(prefix + queries[0], normalize_embeddings=True)
    times = []
    for text in queries[:runs]:
        start = time.perf_counter()
        model.encode(prefix + text, normalize_embeddings=True)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def write_report(result, path):
    lines = [
        "# D-22 Embedding comparison",
        "",
        (
            f"Dataset: `princeton-nlp/LitSearch`, revision `{REVISION}`, corpus `{result['corpus']}` "
            f"({result['documents']:,} papers, title + abstract), {result['queries']} queries. "
            f"Exact cosine search, max {MAX_SEQ_LENGTH} tokens, fp16 on {result['gpu']}."
        ),
        "",
        (
            "Dev/test: 50/50, stratified by `query_set`, seed 42. Binary paper-level relevance "
            "against `corpusids`. Generated by `scripts/compare_embeddings.py`."
        ),
        "",
    ]
    for split in ("dev", "test", "all"):
        lines += [f"## {split.capitalize()}", "", "| Model | " + " | ".join(METRICS) + " |"]
        lines.append("|---|" + "---:|" * len(METRICS))
        for name, model in result["models"].items():
            values = " | ".join(f"{model['scores'][split][m]:.3f}" for m in METRICS)
            lines.append(f"| {name} | {values} |")
        lines.append("")
    lines += [
        f"## Difference to {BASELINE} (paired bootstrap, {BOOTSTRAP_DRAWS:,} draws, 95% CI)",
        "",
        "| Model | Split | Metric | Delta | CI |",
        "|---|---|---|---:|---|",
    ]
    for name, model in result["models"].items():
        for split, rows in model.get("vs_baseline", {}).items():
            for metric, d in rows.items():
                lines.append(
                    f"| {name} | {split} | {metric} | {d['delta']:+.3f} | "
                    f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] |"
                )
    lines += ["", "## NDCG@10 per query_set (all queries)", ""]
    sets = sorted(next(iter(result["models"].values()))["per_query_set"])
    lines += ["| Model | " + " | ".join(sets) + " |", "|---|" + "---:|" * len(sets)]
    for name, model in result["models"].items():
        values = " | ".join(f"{model['per_query_set'][s]['ndcg@10']:.3f}" for s in sets)
        lines.append(f"| {name} | {values} |")
    lines += [
        "",
        "## Speed and truncation",
        "",
        (
            "| Model | GPU encode, docs/s | CPU query latency, ms (median) "
            f"| Median doc tokens | Docs > {MAX_SEQ_LENGTH} tokens |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for name, model in result["models"].items():
        lines.append(
            f"| {name} | {model['docs_per_s']:.0f} | {model['cpu_query_ms']:.1f} "
            f"| {model['median_doc_tokens']:.0f} | {model['truncated_docs']:.1%} |"
        )
    lines += ["", f"CPU: {result['cpu']}.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--raw", action="store_true", help="Use the D-01 cache, not D-03.")
    parser.add_argument("--limit-docs", type=int, default=0, help="Smoke run: targets + random.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-report", action="store_true", help="Skip the docs/ report.")
    args = parser.parse_args()
    if BASELINE not in args.models:
        parser.error(f"{BASELINE} is the baseline and must be included.")
    load_dotenv(PROJECT_ROOT / ".env")
    if not (configured := os.getenv("DATA_DIR")):
        parser.error("DATA_DIR is missing. Set it in .env or the environment.")
    data_dir = Path(configured).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    import torch

    if not torch.cuda.is_available():
        parser.exit(1, "CUDA is not available. Run with the CUDA torch build (see docstring).\n")

    queries, corpus, variant = load_data(data_dir, args.raw)
    query_texts = queries.column("query").to_pylist()
    relevant = queries.column("corpusids").to_pylist()
    query_sets = queries.column("query_set").to_pylist()
    split = split_queries(query_sets)
    all_ids = corpus.column("corpusid").to_pylist()
    targets = {doc for docs in relevant for doc in docs}
    picked = select_docs(all_ids, targets, args.limit_docs)
    titles, abstracts = corpus.column("title").to_pylist(), corpus.column("abstract").to_pylist()
    doc_ids = [all_ids[i] for i in picked]
    doc_texts = [document_text(titles[i], abstracts[i]) for i in picked]
    corpus_name = variant if not args.limit_docs else f"{variant}-limit{args.limit_docs}"
    print(f"{len(doc_ids):,} documents ({corpus_name}), {len(query_texts)} queries")

    cache_dir = data_dir / "embeddings"
    cache_dir.mkdir(exist_ok=True)
    groups = {
        "dev": [i for i, s in enumerate(split) if s == "dev"],
        "test": [i for i, s in enumerate(split) if s == "test"],
        "all": list(range(len(split))),
    }
    result = {
        "dataset_revision": REVISION,
        "corpus": corpus_name,
        "documents": len(doc_ids),
        "queries": len(query_texts),
        "split": split,
        "gpu": torch.cuda.get_device_name(0),
        "cpu": platform.processor() or platform.machine(),
        "versions": {"torch": torch.__version__},
        "models": {},
    }
    import sentence_transformers

    result["versions"]["sentence_transformers"] = sentence_transformers.__version__

    for name in args.models:
        model_id, revision, query_prefix, doc_prefix = MODELS[name]
        print(f"== {name} ({model_id}@{revision[:7]})")
        # Keyed by model, not by name: variants that differ only in the query prefix share it.
        cache = cache_dir / f"{model_id.split('/')[1]}@{revision[:7]}-{corpus_name}.npz"
        if cache.exists() and np.load(cache)["ids"].tolist() == doc_ids:
            data = np.load(cache)
            doc_emb, docs_per_s = data["emb"], float(data["docs_per_s"])
            model = load_model(name, "cuda")
        else:
            model = load_model(name, "cuda")
            start = time.perf_counter()
            doc_emb = encode(model, [doc_prefix + t for t in doc_texts], args.batch_size)
            docs_per_s = len(doc_texts) / (time.perf_counter() - start)
            np.savez(cache, emb=doc_emb, ids=np.array(doc_ids), docs_per_s=docs_per_s)
        query_emb = encode(model, [query_prefix + q for q in query_texts], args.batch_size)
        # Silent truncation: share of documents (with prefix) above the token limit.
        lengths = [
            len(ids) for ids in model.tokenizer([doc_prefix + t for t in doc_texts])["input_ids"]
        ]
        truncated = sum(n > MAX_SEQ_LENGTH for n in lengths) / len(lengths)
        del model
        torch.cuda.empty_cache()
        ranked = search(query_emb, doc_emb, doc_ids, "cuda")
        rows = [query_metrics(r, rel) for r, rel in zip(ranked, relevant)]
        per_set = {
            s: mean_metrics(rows, [i for i, v in enumerate(query_sets) if v == s])
            for s in sorted(set(query_sets))
        }
        result["models"][name] = {
            "model_id": model_id,
            "revision": revision,
            "scores": {g: mean_metrics(rows, idx) for g, idx in groups.items()},
            "per_query_set": per_set,
            "per_query": rows,
            "docs_per_s": docs_per_s,
            "truncated_docs": truncated,
            "median_doc_tokens": statistics.median(lengths),
            "cpu_query_ms": cpu_query_latency(name, query_texts),
        }
        print(json.dumps(result["models"][name]["scores"], indent=2))

    base = result["models"][BASELINE]["per_query"]
    for name, model in result["models"].items():
        if name == BASELINE:
            continue
        model["vs_baseline"] = {
            g: {
                m: paired_bootstrap(
                    [model["per_query"][i][m] for i in idx], [base[i][m] for i in idx]
                )
                for m in ("ndcg@10", "recall@20")
            }
            for g, idx in groups.items()
            if g != "all"
        }

    output = PROJECT_ROOT / "eval" / "private" / "embedding_comparison.json"
    if args.limit_docs:
        output = output.with_name(f"embedding_comparison_{corpus_name}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results: {output}")
    if not args.no_report and not args.limit_docs:
        report = PROJECT_ROOT / "docs" / "d22_embedding_comparison.md"
        write_report(result, report)
        print(f"Report: {report}")


if __name__ == "__main__":
    main()
