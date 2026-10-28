# Retrieval evaluation examples

`cases.example.json` contains synthetic S1/S2/S3 questions for the mock pipeline.
It contains no LitSearch records or reviewed quality labels.

`cases.live.example.json` supplies public S1/S2/S3 questions, including a baseline-use
request, for initial real-service diagnostics. Verify the named target is indexed and
replace cases without usable full-text coverage before freezing the final quality set.

```powershell
$env:USE_MOCKS = "1"
$env:REAL_COMPONENTS = ""
uv run --locked --group rag python scripts/evaluate_retrieval.py eval/retrieval/cases.example.json --allow-mocks --output eval/private/development.json
```

See [the evaluation guide](../../docs/evaluation.md) for frozen reference export,
model comparisons, metric definitions and human review. Real sources and outputs belong
under the ignored `eval/private/` directory.
