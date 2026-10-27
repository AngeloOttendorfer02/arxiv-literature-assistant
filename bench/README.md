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
- temperature 0, seed 0, `num_ctx` 8192, at most 512 output tokens, thinking off
- each request gets a unique prefix, so Ollama's prompt cache cannot shorten TTFT
- warm-up prompts (2 by default) load the model and are not counted

Set `LLM_BASE_URL` (with or without `/v1`) and, for the Modal endpoint, `LLM_API_KEY` in `.env`. Then run:

```
uv run --group rag python bench/benchmark.py bench/prompts.example.jsonl --models qwen3.8:27b-q4_K_M,qwen3.8:27b-q8_0,qwen3.8:27b-bf16 --output eval/private/bench/engine.jsonl
uv run --group rag python bench/benchmark.py bench/prompts.example.jsonl --output eval/private/bench/engine.jsonl --summary-only
```

Each request is appended to `--output` as one JSON line. A run that stops halfway keeps what it measured, and the summary table (p50 and p95) is computed from the whole file.

## Prompts

The input is JSON lines with `id` and `messages`, in the format the app sends to the LLM.

- `prompts.example.jsonl` holds 30 synthetic questions with made-up evidence, built with the app's `build_context`. It is for trial runs only.
- For the reported numbers, use prompts from real pipeline runs. They contain LitSearch text, so keep them and their outputs in `eval/private/`.
