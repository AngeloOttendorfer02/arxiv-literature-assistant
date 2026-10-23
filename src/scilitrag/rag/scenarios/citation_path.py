"""S2: follow incoming citation edges and retrieve target-specific passages."""

import re

from scilitrag.contracts import Evidence, Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.evidence.index import allocate_evidence
from scilitrag.rag.search.backend import PAPER_COLUMNS

CITING_SQL = (
    "SELECT " + PAPER_COLUMNS + " FROM papers p "
    "WHERE EXISTS (SELECT 1 FROM citations edge WHERE edge.citing_id=p.corpus_id "
    "AND edge.cited_id=%(target_id)s) "
    "AND (%(low)s::integer IS NULL OR p.year >= %(low)s) "
    "AND (%(high)s::integer IS NULL OR p.year <= %(high)s) "
    "AND (%(author_id)s::bigint IS NULL OR EXISTS (SELECT 1 FROM paper_authors pa "
    "WHERE pa.corpus_id=p.corpus_id AND pa.author_id=%(author_id)s)) "
    "ORDER BY p.year DESC NULLS LAST,p.corpus_id LIMIT %(limit)s"
)


def baseline_candidate(text: str, title: str, section: str | None = None) -> bool:
    """Conservative candidate filter, never a replacement for WP3 entailment checks.

    Require an explicit experimental-use expression tied to the target in one
    sentence. Negated and related-work passages are excluded. Human labels are
    required to measure the precision/recall of this heuristic.
    """
    if re.search(r"related work|references|bibliograph", section or "", re.IGNORECASE):
        return False
    aliases = {title.strip(), title.split(":", 1)[0].strip()}
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if re.search(r"\b(?:not|no|never|without|neither)\b", sentence, re.IGNORECASE):
            continue
        for alias in aliases:
            if len(alias) < 4:
                continue
            target = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            patterns = (
                r"\b(?:use|used|using|adopt|adopted|evaluate|evaluated)\b[^.!?]{0,100}"
                + target
                + r"[^.!?]{0,50}\bas (?:a |the |our )?baseline\b",
                r"\bcompar(?:e|ed|ing|ison)\b[^.!?]{0,60}\b(?:with|against|to)\s+" + target,
                target + r"\s+(?:is|was|serves as|served as)\s+(?:a |the |our )?baseline\b",
            )
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns):
                return True
    return False


def retrieve_s2(plan: Plan, backend) -> list[Evidence]:
    if not plan.paper_ref:
        raise RagUserError(
            "Which paper should the citation search target? Provide its title or ID."
        )
    if not 1 <= plan.k <= 20:
        raise RagUserError("Choose a result limit between 1 and 20.")
    target = backend.linker.paper_record(plan.paper_ref)
    target["title"] = target.get("title") or str(target["corpus_id"])
    params = {
        "target_id": target["corpus_id"],
        "low": plan.year_from,
        "high": plan.year_to,
        "author_id": backend.linker.author(plan.author_ref) if plan.author_ref else None,
        "limit": 100 if plan.require_baseline else plan.k,
    }
    papers = backend.reader(CITING_SQL, params)
    if not papers:
        raise RagUserError("No indexed papers cite this target within the requested filters.")
    passage_plan = plan.model_copy(
        update={"topic": f"{target['title']} experimental baseline comparison"}
    )
    chunks = backend.chunks(
        passage_plan, [p["corpus_id"] for p in papers if p.get("has_indexed_chunks")], per_paper=4
    )
    ids = {p["corpus_id"] for p in papers}
    if any(c["corpus_id"] not in ids for c in chunks):
        raise RagUserError("Supporting passages could not be matched to the citing papers.")
    sources = [
        Evidence(
            label=0,
            type="paper",
            corpus_id=target["corpus_id"],
            title=target["title"],
            year=target.get("year"),
            section="Citation target",
            text=f"Citation target, not a citing result. Abstract: {target.get('abstract') or ''}",
        )
    ]
    selected = 0
    for paper in papers:
        if selected >= plan.k:
            break
        passages = [c for c in chunks if c["corpus_id"] == paper["corpus_id"]]
        if plan.require_baseline:
            passages = [
                c
                for c in passages
                if baseline_candidate(c["text"], target["title"], c.get("section"))
            ]
            if not passages:
                continue
        selected += 1
        sources.append(
            Evidence(
                label=0,
                type="paper",
                corpus_id=paper["corpus_id"],
                title=paper["title"],
                year=paper.get("year"),
                author_ids=paper.get("author_ids") or [],
                section="Citing paper" if passages else "Abstract only",
                text=f"Indexed citation edge: this paper cites corpus ID {target['corpus_id']}. "
                "The edge alone does not establish experimental baseline use. "
                f"Abstract: {paper.get('abstract') or ''}",
            )
        )
        sources.extend(
            Evidence(
                label=0,
                type="chunk",
                corpus_id=paper["corpus_id"],
                chunk_id=c["chunk_id"],
                title=paper["title"],
                year=paper.get("year"),
                section=c.get("section"),
                author_ids=paper.get("author_ids") or [],
                text=c["text"],
                score=c.get("score"),
            )
            for c in passages
        )
    if len(sources) == 1:
        raise RagUserError(
            "Citing papers were found, but no explicit baseline-use candidate was found in the "
            "retrieved indexed passages of up to 100 citing candidates. Citation edges or mentions alone do not prove baseline use; "
            "full text may be unavailable. Try asking for citing papers without the baseline filter."
        )
    return allocate_evidence(sources)
