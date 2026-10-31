# WP2 Streamlit review

Use the prepared corpus and the team's configured inference endpoint with real
components (`USE_MOCKS=0`). The chat examples exercise S1, S2 and S3, and the
guardrail panel shows completed stages. Open citations and compare each claim
with its source; accepted labels alone do not establish factual support. S3
tables appear as soon as the API emits them, even when later generation fails.

To review baseline passages, expand **Review baseline passages** and upload an
existing private reference JSON produced by `build_reference_set.py`. Review
each passage in context: an incidental mention of the target does not prove
that it was used as an experimental baseline. Leave uncertain cases uncertain.
Download the reviewed JSON and keep it privately in `eval/private/`. This page
does not write to the database and does not produce or infer labels for you.
The sample is bounded and stratified; it is not a corpus-wide estimate.

For S3, compare the displayed table with independently calculated counts for
the exact seed IDs, filters and dataset revision. A reference derived from a
failed run does not make that answer successful. Compare candidate models with
WP4 on the same cases and record the model, endpoint, settings, coverage and
results; a few UI examples do not establish model superiority. Citation
precision and year/count exact match require recorded human judgments and
denominators. Use **Review answer quality** below a completed chat answer to
record and download these judgments. Unreviewed items are excluded; rates apply
only to that answer. Current UI interactions do not create the structured
evaluation run reports described in `evaluation.md`, so keep them separate.

Do not commit reference JSON, reviewed excerpts, API keys or database credentials.
