# Evaluation

Evaluation inputs are organized by capability. Executable implementations live in the
`scilitrag.evaluation` package; repository scripts expose command-line entry points.

| Location | Contents |
| --- | --- |
| `retrieval/` | Synthetic public cases and retrieval-evaluation usage |
| `guardrails/` | Attack and harmless test sets for the guardrails, split into tuning and test |
| `private/` | Ignored real references, source passages, annotations and recorded runs |

See [retrieval evaluation](../docs/evaluation.md) for reference export, explicit model
comparisons, human review and offline scoring. Other capability-specific evaluations can
be added alongside retrieval using feature-based names.
