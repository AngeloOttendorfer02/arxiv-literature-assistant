"""D-20 chunking experiment (O5): the 9 size/overlap settings of D-13, hybrid retrieval, hit rates.

Runs outside the project lock because the project pins CPU-only torch:
    uv run --no-project --python 3.13 --index https://download.pytorch.org/whl/cu130 \
        --with torch==2.14.1 --with sentence-transformers==6.2.0 --with transformers==5.20.2 \
        --with numpy==2.4.6 --with pyarrow==26.0.0 --with python-dotenv==1.2.4 \
        --with pydantic==2.14.1 python scripts/chunk_experiment.py
with PYTHONPATH=src, so the package resolves without installing the project.

Pool: the D-12 papers, chunked like the database build. Retrieval: granite dense (as D-14)
plus BM25, fused with RRF k=60 (as D-15 and the S1 search). Benchmarks:
  A  LitSearch queries, paper level through chunks (targets inside the pool).
  B  D-19 questions: Hit@k of the source span globally, within the source paper as S1 picks
     passages, and at an equal token budget.
Full results (scores only) go to eval/results/d20_chunk_experiment.json, aggregates to
docs/d20_chunking_experiment.md. No dataset text is written to the repository.
"""

import argparse
import bisect
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from build_chunk_eval_set import OUTPUT as EVAL_SET
from build_chunk_eval_set import selected_papers, words
from chunker import OVERLAPS, SIZES, chunk_sections, token_offsets
from compare_embeddings import METRICS, paired_bootstrap, query_metrics, split_queries
from filter_data import batches
from select_fixture import data_directory

from scilitrag.config import DEFAULT_EMBED_MODEL
from scilitrag.data.embedding import MODEL_REVISIONS
from scilitrag.rag.search.passages import select_passages
from scilitrag.rag.search.ranking import reciprocal_rank_fusion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = PROJECT_ROOT / "eval" / "results" / "d20_chunk_experiment.json"
REPORT = PROJECT_ROOT / "docs" / "d20_chunking_experiment.md"
SEED = 1789
BOOTSTRAP_DRAWS = 10_000
DEFAULT = (512, 10)
CONFIGS = [(size, overlap) for size in SIZES for overlap in OVERLAPS]
POOL_TOP = 200
# As SearchBackend.chunks: candidates per list, fused list length, passages per paper, query.
CANDIDATES, FUSED, PER_PAPER = 10, 20, 2
S1_SUFFIX = " methods metrics experimental results"
HIT_KS = (1, 3, 5, 10)
BUDGET_TOKENS = 1024
MIN_COVERAGE = 0.5
# Prompt limits from config.py: one evidence item and the whole context, in characters.
ITEM_CHARS, CONTEXT_CHARS = 2400, 16000
PRIMARY = ("litsearch_dev_ndcg@10", "s1_hit@2")


def located_chunks(corpus_id, sections, offsets, size, overlap):
    """D-13 chunks of one paper with their section index and character span in the section."""
    chunks = []
    for index, (title, text) in enumerate(sections):
        spans = offsets(text)
        ends = [end for _, end in spans]
        position = 0
        for *_, body in chunk_sections(
            corpus_id, [(title, text)], offsets, size=size, overlap=overlap
        ):
            start = text.find(body, position)
            if start < 0:
                raise ValueError(f"Chunk text not found in section {index} of {corpus_id}.")
            end = start + len(body)
            first = bisect.bisect_right(ends, start)
            chunks.append(
                {
                    "corpus_id": corpus_id,
                    "section_index": index,
                    "section": title,
                    "start": start,
                    "end": end,
                    "text": body,
                    "tokens": bisect.bisect_left(ends, end) + 1 - first,
                }
            )
            position = start + 1
    return chunks


def span_coverage(chunk, item):
    """Share of the source span's characters inside the chunk."""
    if (chunk["corpus_id"], chunk["section_index"]) != (item["corpus_id"], item["section_index"]):
        return 0.0
    inside = min(chunk["end"], item["end"]) - max(chunk["start"], item["start"])
    return max(0, inside) / (item["end"] - item["start"])


class Bm25:
    """Okapi BM25 with Lucene/Tantivy defaults; any query term may match (like `|||`)."""

    def __init__(self, texts, k1=1.2, b=0.75):
        self.size = len(texts)
        postings = {}
        lengths = np.zeros(self.size, dtype=np.float32)
        for doc, text in enumerate(texts):
            tokens = words(text)
            lengths[doc] = len(tokens)
            counts = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                postings.setdefault(token, ([], []))
                postings[token][0].append(doc)
                postings[token][1].append(count)
        norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1))
        self.postings = {}
        for token, (docs, counts) in postings.items():
            docs, tf = np.array(docs), np.array(counts, dtype=np.float32)
            idf = math.log(1 + (self.size - len(docs) + 0.5) / (len(docs) + 0.5))
            self.postings[token] = (docs, idf * tf * (k1 + 1) / (tf + norm[docs]))

    def scores(self, query):
        result = np.zeros(self.size, dtype=np.float32)
        for token in set(words(query)):
            if token in self.postings:
                docs, weights = self.postings[token]
                result[docs] += weights
        return result


def top(scores, n, among=None, positive=False):
    """Indices of the n best scores (ties by index), optionally within `among`."""
    index = np.arange(len(scores)) if among is None else np.asarray(among)
    values = scores[index]
    if positive:
        index, values = index[values > 0], values[values > 0]
    order = np.lexsort((index, -values))[:n]
    return index[order].tolist()


def fused(dense, sparse, limit, key="i", rows=None):
    rows = rows or (lambda i: {key: i})
    return reciprocal_rank_fusion(
        [rows(i) for i in dense], [rows(i) for i in sparse], limit=limit, k=60, key_field=key
    )


def papers_in_order(chunk_ranking, chunks):
    seen = []
    for row in chunk_ranking:
        paper = chunks[row["i"]]["corpus_id"]
        if paper not in seen:
            seen.append(paper)
    return seen


def evaluate(chunks, size, chunk_emb, query_emb, queries, items, question_emb, s1_emb):
    """Per-query scores of benchmarks A and B for one chunk configuration."""
    import torch

    bm25 = Bm25([c["text"] for c in chunks])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix = torch.from_numpy(chunk_emb).to(device)

    def dense_scores(vectors):
        return (torch.from_numpy(vectors).to(device) @ matrix.T).cpu().numpy()

    litsearch = []
    for start in range(0, len(queries), 64):
        block = dense_scores(query_emb[start : start + 64])
        for row, query in zip(block, queries[start : start + 64], strict=True):
            ranking = fused(
                top(row, POOL_TOP),
                top(bm25.scores(query["query"]), POOL_TOP, positive=True),
                POOL_TOP,
            )
            litsearch.append(query_metrics(papers_in_order(ranking, chunks), query["relevant"]))

    budget_k = round(BUDGET_TOKENS / size)
    by_paper = {}
    for i, chunk in enumerate(chunks):
        by_paper.setdefault(chunk["corpus_id"], []).append(i)
    synthetic = []

    def passage_row(i):
        return {"i": i, "section": chunks[i]["section"], "text": chunks[i]["text"]}

    global_dense, s1_dense = dense_scores(question_emb), dense_scores(s1_emb)
    for item, g_row, s_row in zip(items, global_dense, s1_dense, strict=True):
        ranking = fused(
            top(g_row, POOL_TOP),
            top(bm25.scores(item["question"]), POOL_TOP, positive=True),
            max(HIT_KS + (budget_k,)),
        )
        cover = [span_coverage(chunks[row["i"]], item) >= MIN_COVERAGE for row in ranking]
        scores = {f"hit@{k}": float(any(cover[:k])) for k in HIT_KS}
        scores["hit@budget"] = float(any(cover[:budget_k]))
        # S1: the source paper is among the top papers; pick its passages like SearchBackend.chunks.
        own = by_paper[item["corpus_id"]]
        query = item["question"] + S1_SUFFIX
        passages = select_passages(
            fused(
                top(s_row, CANDIDATES, among=own),
                top(bm25.scores(query), CANDIDATES, among=own, positive=True),
                FUSED,
                rows=passage_row,
            ),
            limit=PER_PAPER,
        )
        scores["s1_hit@2"] = float(
            any(span_coverage(chunks[p["i"]], item) >= MIN_COVERAGE for p in passages)
        )
        scores["source_chunks"] = len(own)
        synthetic.append(scores)
    return litsearch, synthetic, budget_k


def nli_latency(chunk_sets, items, runs):
    """Median CPU time of one NLI pair with a chunk as premise, per chunk size (ModernCE, D057)."""
    from scilitrag.config import load_settings
    from scilitrag.guardrails.validation import NliModel

    model = NliModel(load_settings().nli_model)
    hypotheses = [item["passage"] for item in items[:runs]]
    model.probabilities("warm up", "warm up")
    result = {}
    for size, chunks in chunk_sets.items():
        full = [c["text"] for c in chunks if c["tokens"] > 0.9 * size][:runs]
        times = []
        for premise, hypothesis in zip(full, hypotheses, strict=False):
            start = time.perf_counter()
            model.probabilities(premise, hypothesis)
            times.append((time.perf_counter() - start) * 1000)
        result[size] = statistics.median(times)
    return result


def litsearch_queries(data_dir, pool):
    rows = [
        r for b in batches([data_dir / "filtered" / "d03" / "query.arrow"]) for r in b.to_pylist()
    ]
    split = split_queries([row["query_set"] for row in rows], seed=SEED)
    queries = []
    for row, part in zip(rows, split, strict=True):
        relevant = [cid for cid in row["corpusids"] if cid in pool]
        if relevant:
            queries.append(
                {
                    "query": row["query"],
                    "relevant": relevant,
                    "split": part,
                    "set": row["query_set"],
                }
            )
    return queries, len(rows)


def embed_chunks(encoder, chunks, cache, batch_size):
    digest = hashlib.sha256("\x00".join(c["text"] for c in chunks).encode()).hexdigest()
    if cache.exists():
        data = np.load(cache)
        if str(data["digest"]) == digest:
            return data["emb"], float(data["seconds"])
    start = time.perf_counter()
    emb = encode(encoder, [c["text"] for c in chunks], batch_size)
    seconds = time.perf_counter() - start
    np.savez(cache, emb=emb, digest=digest, seconds=seconds)
    return emb, seconds


def encode(encoder, texts, batch_size):
    return encoder.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)


def mean(rows, key, indices=None):
    indices = range(len(rows)) if indices is None else indices
    return float(np.mean([rows[i][key] for i in indices]))


def decide(configs):
    """Pre-registered rule: keep settings not significantly worse than the best on both
    primary metrics; among them fewer truncated chunks, then the smaller size (NLI latency)."""
    tied = []
    for name, config in configs.items():
        if all(config["vs_best"][m]["ci_high"] >= 0 for m in PRIMARY):
            tied.append(name)
    return min(
        tied,
        key=lambda n: (
            round(configs[n]["truncated_share"], 2),
            configs[n]["size"],
            -sum(configs[n]["primary"][m] for m in PRIMARY),
        ),
    ), tied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-papers", type=int, default=0, help="Smoke run on fewer papers.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nli-runs", type=int, default=30, help="0 skips the NLI latency.")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--report-only", action="store_true", help=f"Rewrite the report from {RESULT.name}."
    )
    args = parser.parse_args()
    if args.report_only:
        write_report(json.loads(RESULT.read_text(encoding="utf-8")), REPORT)
        return
    import torch

    if not torch.cuda.is_available():
        parser.exit(1, "CUDA is not available. Run with the CUDA torch build (see docstring).\n")
    data_dir = data_directory()
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    items = [item for item in eval_set["items"] if item["rejected"] is None]
    if any(item["review"] is False for item in items):
        items = [item for item in items if item["review"] is not False]
    selection, _, sections = selected_papers(data_dir)
    papers = sorted(sections)
    if args.limit_papers:
        keep = {item["corpus_id"] for item in items}
        papers = sorted(keep | set(papers[: args.limit_papers]))
    pool = set(papers)
    items = [item for item in items if item["corpus_id"] in pool]
    queries, total_queries = litsearch_queries(data_dir, pool)
    print(
        f"{len(papers)} papers, {len(queries)}/{total_queries} LitSearch queries, {len(items)} questions"
    )

    from scilitrag.data.embedding import load_model

    model = DEFAULT_EMBED_MODEL
    encoder = load_model(model)
    tokenizer_offsets = token_offsets(model)
    known = {text: tokenizer_offsets(text) for cid in papers for _, text in sections[cid]}

    def offsets(text):
        return known[text] if text in known else tokenizer_offsets(text)

    query_emb = encode(encoder, [q["query"] for q in queries], args.batch_size)
    question_emb = encode(encoder, [i["question"] for i in items], args.batch_size)
    s1_emb = encode(encoder, [i["question"] + S1_SUFFIX for i in items], args.batch_size)
    groups = {
        "dev": [i for i, q in enumerate(queries) if q["split"] == "dev"],
        "test": [i for i, q in enumerate(queries) if q["split"] == "test"],
    }
    cache_dir = data_dir / "embeddings"
    cache_dir.mkdir(exist_ok=True)
    corpus_name = f"limit{args.limit_papers}" if args.limit_papers else "d12"
    configs, per_query, chunk_sets = {}, {}, {}
    for size, overlap in CONFIGS:
        name = f"{size}/{overlap}"
        chunks = [
            chunk
            for cid in papers
            for chunk in located_chunks(cid, sections[cid], offsets, size, overlap)
        ]
        cache = (
            cache_dir / f"chunks-{size}-{overlap}@{MODEL_REVISIONS[model][:7]}-{corpus_name}.npz"
        )
        emb, seconds = embed_chunks(encoder, chunks, cache, args.batch_size)
        litsearch, synthetic, budget_k = evaluate(
            chunks, size, emb, query_emb, queries, items, question_emb, s1_emb
        )
        chars = [len(c["text"]) for c in chunks]
        evidence = statistics.mean(min(c, ITEM_CHARS) for c in chars)
        config = {
            "size": size,
            "overlap": overlap,
            "chunks": len(chunks),
            "mean_tokens": statistics.mean(c["tokens"] for c in chunks),
            "mean_chars": statistics.mean(chars),
            "truncated_share": sum(c > ITEM_CHARS for c in chars) / len(chars),
            # S1 with k=3 papers and 2 passages each.
            "six_passages_share_of_context": 6 * evidence / CONTEXT_CHARS,
            "encode_seconds": seconds,
            "budget_k": budget_k,
            "litsearch": {
                split: {m: mean(litsearch, m, idx) for m in METRICS}
                for split, idx in groups.items()
            },
            "synthetic": {
                key: mean(synthetic, key)
                for key in [f"hit@{k}" for k in HIT_KS] + ["hit@budget", "s1_hit@2"]
            },
        }
        config["primary"] = {
            "litsearch_dev_ndcg@10": config["litsearch"]["dev"]["ndcg@10"],
            "s1_hit@2": config["synthetic"]["s1_hit@2"],
        }
        configs[name] = config
        per_query[name] = {
            "litsearch_dev_ndcg@10": [litsearch[i]["ndcg@10"] for i in groups["dev"]],
            "litsearch_test_ndcg@10": [litsearch[i]["ndcg@10"] for i in groups["test"]],
            "s1_hit@2": [s["s1_hit@2"] for s in synthetic],
            "hit@5": [s["hit@5"] for s in synthetic],
            "hit@budget": [s["hit@budget"] for s in synthetic],
        }
        if overlap == 10:
            chunk_sets[size] = chunks
        print(
            f"{name}: {len(chunks):,} chunks, encode {seconds:.0f}s, "
            f"nDCG@10 dev {config['primary']['litsearch_dev_ndcg@10']:.3f}, "
            f"Hit@5 {config['synthetic']['hit@5']:.2f}, S1 Hit@2 {config['synthetic']['s1_hit@2']:.2f}"
        )

    for metric in PRIMARY:
        best = max(configs, key=lambda n: configs[n]["primary"][metric])
        for name, config in configs.items():
            config.setdefault("vs_best", {})[metric] = paired_bootstrap(
                per_query[name][metric], per_query[best][metric], BOOTSTRAP_DRAWS, SEED
            )
    for name, config in configs.items():
        config["vs_default"] = {
            metric: paired_bootstrap(
                per_query[name][metric],
                per_query[f"{DEFAULT[0]}/{DEFAULT[1]}"][metric],
                BOOTSTRAP_DRAWS,
                SEED,
            )
            for metric in per_query[name]
        }
    chosen, tied = decide(configs)
    nli = nli_latency(chunk_sets, items, args.nli_runs) if args.nli_runs else {}
    for config in configs.values():
        config["nli_ms"] = nli.get(config["size"])
    result = {
        "task": "D-20",
        "corpus": corpus_name,
        "papers": len(papers),
        "selection_seed": selection["seed"],
        "seed": SEED,
        "embed_model": f"{model}@{MODEL_REVISIONS[model]}",
        "litsearch_queries": len(queries),
        "litsearch_queries_total": total_queries,
        "split_sizes": {k: len(v) for k, v in groups.items()},
        "questions": len(items),
        "question_model": eval_set["model"],
        "reviewed": sum(item["review"] is not None for item in items),
        "gpu": torch.cuda.get_device_name(0),
        "cpu": platform.processor() or platform.machine(),
        "configs": configs,
        "per_query": per_query,
        "tied": tied,
        "chosen": chosen,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    out = (
        RESULT
        if not args.limit_papers
        else PROJECT_ROOT / "eval" / "private" / f"chunk_experiment_{corpus_name}.json"
    )
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Chosen: {chosen} (equivalent: {', '.join(tied)}). Results: {out}")
    if not args.no_report and not args.limit_papers:
        write_report(result, REPORT)


def write_report(result, path):
    configs = result["configs"]
    lines = [
        "# D-20 Chunking experiment",
        "",
        (
            f"Pool: the {result['papers']:,} D-12 full-text papers, cleaned and chunked like the "
            f"database build (D-10, D-11, D-13). Embeddings: `{result['embed_model']}`, fp32, "
            f"exact cosine search on {result['gpu']}. Retrieval: dense and BM25 "
            "(k1 = 1.2, b = 0.75, Python implementation, not pg_search), fused with RRF k = 60. "
            f"Seed {result['seed']}. Generated by `scripts/chunk_experiment.py`."
        ),
        "",
        "## Benchmarks",
        "",
        (
            f"- **A, LitSearch (S1 topic search):** {result['litsearch_queries']} of "
            f"{result['litsearch_queries_total']} queries have at least one target paper in the "
            "pool; targets outside the pool are ignored. Chunks of the whole pool are ranked and "
            "reduced to papers by their first chunk. Dev/test 50/50, stratified by `query_set` "
            f"({result['split_sizes']['dev']}/{result['split_sizes']['test']}); a different split "
            "than D-22 (seed 42)."
        ),
        (
            f"- **B, synthetic passage questions (D-19, O16):** {result['questions']} questions "
            f"generated by `{result['question_model']}` from 1 to 2 sentence passages, independent "
            "of chunk boundaries. A hit is a returned chunk that holds at least "
            f"{MIN_COVERAGE:.0%} of the source span. Hit@k ranks all chunks of the pool. "
            "S1 Hit@2 assumes the source paper is among the top papers and picks its 2 passages "
            "exactly like `SearchBackend.chunks` (question plus the fixed S1 terms, 10 dense and "
            "10 BM25 candidates, RRF, `select_passages`). Hit@budget returns about "
            f"{BUDGET_TOKENS} tokens (k = 4, 3, 2 for 256, 384, 512)."
        ),
        "",
        "## Decision rule (fixed before the run)",
        "",
        (
            "Primary metrics: LitSearch dev NDCG@10 and S1 Hit@2. A setting whose paired "
            f"bootstrap interval ({BOOTSTRAP_DRAWS:,} draws, 95 %) against the best setting "
            "includes 0 on both metrics counts as equivalent. Among equivalent settings the one "
            f"with fewer chunks above the {ITEM_CHARS:,}-character evidence limit wins, then the "
            "smaller size (lower NLI latency per sentence). Test values are reported only."
        ),
        "",
        f"**Result: {result['chosen']}** (equivalent: {', '.join(result['tied'])}).",
        "",
        "## Difference to the best setting on the primary metrics (decides equivalence)",
        "",
        "| Size/overlap | LitSearch dev NDCG@10 | S1 Hit@2 | Equivalent |",
        "|---|---|---|---|",
    ]
    for name, config in configs.items():
        cells = [
            f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
            for d in (config["vs_best"][m] for m in PRIMARY)
        ]
        tied = "yes" if name in result["tied"] else "no"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {tied} |")
    lines += [
        "",
        "## A: LitSearch, paper level through chunks",
        "",
        "| Size/overlap | Split | " + " | ".join(METRICS) + " |",
        "|---|---|" + "---:|" * len(METRICS),
    ]
    for name, config in configs.items():
        for split in ("dev", "test"):
            values = " | ".join(f"{config['litsearch'][split][m]:.3f}" for m in METRICS)
            lines.append(f"| {name} | {split} | {values} |")
    keys = [f"hit@{k}" for k in HIT_KS] + ["hit@budget", "s1_hit@2"]
    lines += [
        "",
        "## B: synthetic passage questions",
        "",
        "| Size/overlap | "
        + " | ".join(k.replace("hit", "Hit").replace("s1_", "S1 ") for k in keys)
        + " |",
        "|---|" + "---:|" * len(keys),
    ]
    for name, config in configs.items():
        lines.append(
            f"| {name} | " + " | ".join(f"{config['synthetic'][k]:.2f}" for k in keys) + " |"
        )
    lines += [
        "",
        f"## Difference to the default {DEFAULT[0]}/{DEFAULT[1]} (paired bootstrap, 95 % CI)",
        "",
        (
            "| Size/overlap | LitSearch dev NDCG@10 | LitSearch test NDCG@10 | S1 Hit@2 | Hit@5 "
            "| Hit@budget |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for name, config in configs.items():
        cells = []
        for metric in (
            "litsearch_dev_ndcg@10",
            "litsearch_test_ndcg@10",
            "s1_hit@2",
            "hit@5",
            "hit@budget",
        ):
            d = config["vs_default"][metric]
            cells.append(f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Cost",
        "",
        (
            "| Size/overlap | Chunks | Mean tokens | Mean chars | Chunks > 2,400 chars "
            "| 6 passages / 16,000 chars | GPU encode, s | NLI per pair, ms (CPU, median) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, config in configs.items():
        nli = f"{config['nli_ms']:.0f}" if config.get("nli_ms") else "-"
        lines.append(
            f"| {name} | {config['chunks']:,} | {config['mean_tokens']:.0f} "
            f"| {config['mean_chars']:.0f} | {config['truncated_share']:.1%} "
            f"| {config['six_passages_share_of_context']:.0%} | {config['encode_seconds']:.0f} | {nli} |"
        )
    lines += [
        "",
        (
            "NLI: `NLI_MODEL` (ModernCE-large-nli, D057) with a full-size chunk as premise and a "
            f"D-19 passage as hypothesis, {result['cpu']}."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
