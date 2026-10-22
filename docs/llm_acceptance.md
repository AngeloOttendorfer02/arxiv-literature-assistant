# LLM client acceptance

Run the following command from the project root with Ollama running
and the configured model installed:

```powershell
$AcceptanceRecord = "data/acceptance/llm-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
uv run --locked --group rag python scripts/check_llm.py --output $AcceptanceRecord
```

The command uses `LLM_BASE_URL`, `LLM_MODEL`, `REQUEST_TIMEOUT_S`
and the existing request policy from configuration. It always uses
the real `ChatClient`; database availability and mock component
selection do not affect these direct calls.

## Verification procedure

A successful run completes one non-streaming request and consumes
one streaming request through the configured provider.

It then checks timeout translation for both modes using a deliberately
stalled loopback HTTP server, a 0.25-second request budget and no retries.
This controlled probe is identified separately in the report; it does
not establish that the model provider itself stalled. Existing client
unit tests cover request URLs, model/config forwarding, SSE parsing
and retry rules.

The command exits nonzero if a check fails and writes a success report
only after all checks pass. Use a fresh output filename for each run;
an existing record is not removed on failure.

Reports contain timings and counts without prompts, answers, provider
response bodies, database credentials or configured endpoint URLs.
Generated JSON files remain under the ignored `data/` directory.
The observed acceptance output is embedded below for repository review.

For CPU acceptance, confirm device placement while the model is loaded:

```powershell
ollama --version
ollama ps
```

The script does not infer CPU placement. First-load latency can include
model loading; a separate warm run is needed to distinguish that cost.
A short response is not a throughput benchmark. Stream text deltas are
not tokenizer tokens, so this tool does not measure tokens per second
or verify the approximately 34 tokens/s reference performance.

## Local validation record

- Date: 2026-10-22
- Model: `qwen3.5:2b-mtp-q4_K_M`
- Model ID reported by Ollama: `f64538b732a5`
- Ollama version: `0.40.6`
- Device placement: `100% CPU`
- Context window reported by Ollama: `4096`
- Runner: `llamacpp`
- Request timeout: 120 seconds
- Maximum output tokens: 1024
- Real completion: passed
- Real streaming: passed
- Controlled completion timeout: passed
- Controlled streaming timeout: passed
- Offline tests: **218 passed, 3 skipped, 1 xfailed in 10.75 seconds**

### Captured acceptance output

```json
{
  "status": "passed",
  "model": "qwen3.5:2b-mtp-q4_K_M",
  "request_timeout_s": 120.0,
  "max_output_tokens": 1024,
  "completion": {
    "status": "passed",
    "elapsed_s": 64.0368,
    "characters": 104
  },
  "streaming": {
    "status": "passed",
    "elapsed_s": 48.7812,
    "first_text_delta_s": 47.8876,
    "text_deltas": 16,
    "characters": 104
  },
  "timeout_probe": {
    "endpoint": "controlled local stalled HTTP server",
    "checks": {
      "completion": {
        "status": "passed",
        "error_kind": "timeout",
        "elapsed_s": 0.8033
      },
      "stream": {
        "status": "passed",
        "error_kind": "timeout",
        "elapsed_s": 0.7046
      }
    }
  },
  "throughput": "Not measured: text deltas are not provider tokens.",
  "hardware": "Record Ollama device placement separately; not inferred by this script."
}
```

The hardware field is preserved from the original output. CPU placement
was confirmed separately through `ollama ps`.

Observed timeout-probe durations include client overhead and therefore
exceed the configured 0.25-second HTTP timeout.

## Acceptance result

R-02 is complete: the configured CPU model successfully handled real
completion and streaming requests, and both controlled timeout checks
passed. Existing automated tests verify endpoint/model forwarding,
response parsing and retry behavior.

These results establish LLM client functionality. Token throughput,
answer quality, corpus retrieval and guardrail quality were not measured
by this acceptance run.