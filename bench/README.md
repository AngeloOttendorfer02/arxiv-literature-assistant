# bench

Benchmark of the inference engine across precisions (fp16, q8_0, q4_K_M) and the load test.

## Engine benchmark

`benchmark.py` sends each prompt to Ollama's native `/api/chat`, directly and without the app (E11). Per model tag it records:

- VRAM and CPU offload from `/api/ps`, after warm-up
- TTFT, measured two ways:
  - engine: Ollama's load plus prompt processing time, without network
  - client: time to the first streamed token, network included
- tokens/s, from `eval_count` and `eval_duration`
- the answer text, for quality scoring later

Settings that are the same for every tag:
- temperature 0, seed 0, `num_ctx` 8192, at most 1024 output tokens (the app's `LLM_MAX_OUTPUT_TOKENS`), thinking off
- each request gets a unique prefix, so Ollama's prompt cache cannot shorten TTFT
- warm-up prompts (2 by default) load the model and are not counted

Set `LLM_BASE_URL` (with or without `/v1`) and, for the Modal endpoint, `LLM_API_KEY` in `.env`. The endpoint and its deployment are described in [deploy/modal](../deploy/modal/README.md). Then run:

```
uv run --group rag python bench/benchmark.py bench/prompts.example.jsonl --models qwen3.8:27b-q4_K_M,qwen3.8:27b-q8_0,qwen3.8:27b-bf16 --output eval/private/bench/engine.jsonl
uv run --group rag python bench/benchmark.py bench/prompts.example.jsonl --output eval/private/bench/engine.jsonl --summary-only
```

Each request is appended to `--output` as one JSON line. A run that stops halfway keeps what it measured, and the summary table (p50 and p95) is computed from the whole file.

## Prompts

The input is JSON lines with `id` and `messages`, in the format the app sends to the LLM.

- `prompts.example.jsonl` holds 30 synthetic questions with made-up evidence, built with the app's `build_context`. It is for trial runs only.
- For the reported numbers, `export_prompts.py` builds one prompt per LitSearch query from the real retrieval on the configured database. Every query gets the fixed S1 plan of the planner fallback (k = 5), so all tags see the same prompts and the planner is not part of the comparison. It also writes a sample of 30 for the repeats (seed 1789) and a `.meta.json` with dataset revision, embedding model and skipped queries. The prompts contain LitSearch text, so keep them and all outputs in `eval/private/`.

```
uv run --locked --group data --group rag --group embed python bench/export_prompts.py
uv run --group rag python bench/benchmark.py eval/private/bench/prompts.jsonl --models qwen3.8:27b-q4_K_M,qwen3.8:27b-q8_0,qwen3.8:27b-bf16 --repeats 1 --output eval/private/bench/engine.jsonl
uv run --group rag python bench/benchmark.py eval/private/bench/prompts_30.jsonl --models qwen3.8:27b-q4_K_M,qwen3.8:27b-q8_0,qwen3.8:27b-bf16 --output eval/private/bench/engine_30.jsonl
```

## Quality

`score_quality.py` scores the stored answers offline with the app's NLI model (D058):

- citation recall and precision as in ALCE (Gao et al., EMNLP 2023): a sentence is supported when its cited evidence, joined, entails it at `NLI_ENTAILMENT_THRESHOLD`; a citation is precise when its sentence is supported and it is not redundant. Labels outside the evidence count as wrong.
- "not found" rate: answers whose first sentence says the evidence holds nothing matching. NLI cannot entail such a statement, so these answers are left out of recall and precision. Check the pattern with `--review 20`, which prints first sentences of flagged and unflagged answers.
- gold-paper citation: the answer cites a LitSearch target paper, counted only for queries whose evidence holds one, "not found" answers included.

Repeats of a prompt are averaged first. Each tag is compared with `--baseline` by a paired bootstrap over queries (10,000 draws, seed 1789), per split (dev, test, all). The output has scores and labels, no dataset text. NLI runs on the CPU and caches repeated pairs.

```
uv run --locked --group data --group guardrails python bench/score_quality.py eval/private/bench/prompts.jsonl eval/private/bench/engine.jsonl --baseline qwen3.8:27b-bf16 --output eval/private/bench/quality.json
```
