"""Load the full D-03 corpus into an empty D-04 database (D-16).

All papers, D-09 citations and D-08 authors; D-13 chunks only for the D-12 full-text
selection (D048). Uses the same checks and COPY import as load_fixture.py.

Usage: uv run --locked --group data --group rag python scripts/load_full.py
Use --check-only to validate the local D-03 and D-12 files without connecting to PostgreSQL.
Embeddings follow with scripts/embed_database.py.
"""

import argparse
import json
import os
from collections import Counter
from time import perf_counter

from author_names import extract_authors
from chunker import TOKENIZER_REVISIONS, chunk_sections, token_offsets
from citations import bibliography_counts, bibliography_ids, build_citations, direction_counts
from email_masking import count_emails, mask_emails
from filter_data import batches
from fulltext import clean_sections, fulltext_counts
from load_fixture import database_error_detail, import_records
from select_fixture import DATASET, REVISION, canonical_digest, data_directory

from scilitrag.config import DEFAULT_EMBED_MODEL


def full_records(folder, selection_path):
    """Papers and citations of every D-03 paper, plus the D-12 IDs that get chunks."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != DATASET or manifest.get("dataset_revision") != REVISION:
        raise ValueError("D-03 data must match the pinned LitSearch dataset revision.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("filtered_manifest_sha256") != canonical_digest(manifest):
        raise ValueError("The D-12 selection was made from different D-03 data.")
    papers, graph = [], {}
    for batch in batches([folder / "corpus_clean.arrow"]):
        for row in batch.select(["corpusid", "title", "abstract", "year", "citations"]).to_pylist():
            cid = row["corpusid"]
            if type(cid) is not int or cid in graph:
                raise ValueError("D-03 data has invalid or duplicate corpus IDs.")
            if any(
                row[name] is not None and not isinstance(row[name], str)
                for name in ("title", "abstract")
            ):
                raise ValueError("Title and abstract must be text or NULL.")
            if row["year"] is not None and type(row["year"]) is not int:
                raise ValueError("Years must be integers or NULL.")
            if not isinstance(row["citations"], list) or any(
                type(c) is not int for c in row["citations"]
            ):
                raise ValueError("Citation lists must contain integer corpus IDs.")
            papers.append(
                (cid, mask_emails(row["title"]), mask_emails(row["abstract"]), row["year"])
            )
            graph[cid] = row["citations"]
    ids = set(graph)
    if ids != set(manifest["retained_corpusids"]):
        raise ValueError("Filtered corpus IDs differ from the D-03 manifest.")
    selected = set(selection["corpusids"])
    if len(selected) != len(selection["corpusids"]) or not selected <= ids:
        raise ValueError("The D-12 selection must list distinct D-03 papers.")
    queries = [row for batch in batches([folder / "query.arrow"]) for row in batch.to_pylist()]
    if any(not row["corpusids"] or set(row["corpusids"]) - ids for row in queries):
        raise ValueError("Every query target must be in the D-03 corpus.")
    citations, citation_counts = build_citations(graph, ids)
    return sorted(papers), citations, citation_counts, selected


def s2orc_pass(folder, ids, selected, offsets, size, overlap):
    """Read the large S2ORC file once: authors, bibliographies and chunks of selected papers."""
    bibliographies, chunks, full = {}, [], Counter()

    def rows():
        for batch in batches([folder / "corpus_s2orc.arrow"]):
            for row in batch.select(["corpusid", "content"]).to_pylist():
                cid = row["corpusid"]
                bibliographies[cid] = bibliography_ids(row)
                if cid in selected:
                    full.update(fulltext_counts([row]))
                    sections = clean_sections(row)[0]
                    chunks.extend(
                        chunk_sections(cid, sections, offsets, size=size, overlap=overlap)
                    )
                yield row

    authors, links, author_counts = extract_authors(rows())
    if bibliographies.keys() != ids:
        raise ValueError("D-03 corpora are not aligned.")
    return authors, links, author_counts, bibliographies, chunks, full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=10, help="Chunk overlap in percent.")
    args = parser.parse_args()
    started = perf_counter()
    try:
        data = data_directory()
        folder = data / "filtered" / "d03"
        papers, citations, citation_counts, selected = full_records(
            folder, data / "fulltext" / "v1" / "selection.json"
        )
        model = os.environ.get("EMBED_MODEL") or DEFAULT_EMBED_MODEL
        if model not in TOKENIZER_REVISIONS:
            raise ValueError(f"No pinned tokenizer revision for {model}.")
        ids = {paper[0] for paper in papers}
        authors, links, author_counts, bibliographies, chunks, full = s2orc_pass(
            folder, ids, selected, token_offsets(model), args.chunk_tokens, args.overlap
        )
    except (ValueError, KeyError, OSError, ImportError) as error:
        parser.exit(1, f"Full corpus validation failed: {error}\n")
    direction = direction_counts(citations, {paper[0]: paper[3] for paper in papers})
    direction.update(bibliography_counts(citations, bibliographies))
    print(
        f"Citations: {len(citations)} from {citation_counts.get('entries', 0)} list entries; "
        f"dropped {citation_counts.get('outside_corpus', 0)} outside the corpus, "
        f"{citation_counts.get('self_citation', 0)} self-citations, "
        f"{citation_counts.get('duplicate', 0)} duplicates; "
        f"{direction.get('in_bibliography', 0)} in the citing paper's S2ORC bibliography."
    )
    total = full.get("text_chars") or 1
    print(
        f"Full texts: {full.get('sections', 0)} sections from {full.get('papers', 0)} selected papers; "
        f"kept {full.get('kept_chars', 0) / total:.1%} of characters."
    )
    left = full.get("emails_left", 0) + sum(
        count_emails(title) + count_emails(abstract) for _, title, abstract, _ in papers
    )
    print(f"E-mails: {full.get('emails_masked', 0)} masked in full texts; {left} remain.")
    if left:
        parser.exit(1, "Full corpus validation failed: e-mail expressions remain after masking.\n")
    print(
        f"Chunks: {len(chunks)} from {len({chunk[1] for chunk in chunks})} papers; "
        f"{args.chunk_tokens} tokens with {args.overlap} % overlap "
        f"({model} tokenizer, revision {TOKENIZER_REVISIONS[model][:12]})."
    )
    print(
        f"Authors: {len(authors)} with {len(links)} paper links from "
        f"{author_counts.get('author_blocks', 0)} author blocks."
    )
    if args.check_only:
        print(f"Valid corpus: {len(papers)} papers, {len(selected)} with full-text chunks.")
        print(f"Validation: {perf_counter() - started:.0f}s. No database connection.")
        return
    dsn = os.environ.get("INGEST_DATABASE_URL")
    if not dsn:
        parser.error("Set INGEST_DATABASE_URL in .env; the app URL is never used for imports.")
    import psycopg

    try:
        with psycopg.connect(dsn, connect_timeout=5) as connection:
            import_records(connection, papers, citations, authors, links, chunks)
    except ValueError as error:
        parser.exit(1, f"Import refused: {error}\n")
    except psycopg.Error as error:
        parser.exit(
            1,
            f"Database import failed: {database_error_detail(error, dsn)}; transaction rolled back.\n",
        )
    print(
        f"Loaded {len(papers)} papers, {len(citations)} citations, {len(authors)} authors, "
        f"{len(links)} paper links and {len(chunks)} chunks in {perf_counter() - started:.0f}s."
    )
    print("Next: scripts/embed_database.py, then indexes.sql and retrieval_indexes.sql.")


if __name__ == "__main__":
    main()
