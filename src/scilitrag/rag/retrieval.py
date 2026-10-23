from scilitrag.config import load_settings
from scilitrag.contracts import Evidence, Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.evidence.index import allocate_evidence
from scilitrag.rag.scenarios.citation_path import retrieve_s2
from scilitrag.rag.search.backend import default_backend


def paper_evidence(row: dict, *, has_passages: bool) -> Evidence:
    available = bool(row.get("has_indexed_chunks"))
    return Evidence(
        label=0,
        type="paper",
        corpus_id=row["corpus_id"],
        title=row["title"],
        year=row.get("year"),
        author_ids=row.get("author_ids") or [],
        section="Abstract" if has_passages else "Abstract only",
        score=row.get("score"),
        text=(
            f"Indexed passages: {'available' if available else 'unavailable'}. "
            f"Retrieved supporting passages: {'available' if has_passages else 'none'}. "
            f"Abstract: {row.get('abstract') or ''}"
        ),
    )


def retrieve_s1(plan: Plan, backend) -> list[Evidence]:
    papers = backend.hybrid(plan)
    if not papers:
        if plan.scenario == "S3":
            return []
        raise RagUserError(
            "No matching papers were found in the indexed corpus. Try fewer filters."
        )
    paper_ids = {row["corpus_id"] for row in papers}
    chunks = backend.chunks(
        plan, [row["corpus_id"] for row in papers if row.get("has_indexed_chunks")]
    )
    if any(chunk["corpus_id"] not in paper_ids for chunk in chunks):
        raise RagUserError("Supporting passages could not be matched to the selected papers.")
    sources = []
    for row in papers:
        passages = [chunk for chunk in chunks if chunk["corpus_id"] == row["corpus_id"]][:2]
        sources.append(paper_evidence(row, has_passages=bool(passages)))
        sources.extend(
            Evidence(
                label=0,
                type="chunk",
                corpus_id=row["corpus_id"],
                chunk_id=chunk["chunk_id"],
                title=row["title"],
                year=row.get("year"),
                author_ids=row.get("author_ids") or [],
                section=chunk.get("section"),
                text=chunk["text"],
                score=chunk.get("score"),
            )
            for chunk in passages
        )
    return allocate_evidence(sources)


def retrieve_with(plan: Plan, backend) -> list[Evidence]:
    if plan.scenario == "S2":
        return retrieve_s2(plan, backend)
    return retrieve_s1(plan, backend)


def build_component(settings):
    return lambda plan: retrieve_with(plan, default_backend(settings))


def retrieve(plan: Plan) -> list[Evidence]:
    return build_component(load_settings())(plan)
