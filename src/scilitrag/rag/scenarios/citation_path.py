"""S2: follow incoming citation edges and retrieve target-specific passages."""

import re

from scilitrag.contracts import Evidence, Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.evidence.index import allocate_evidence
from scilitrag.rag.search.backend import PAPER_COLUMNS
from scilitrag.rag.search.paper_aliases import (
    VERIFIED_ALIASES,
    baseline_search_topic,
    target_aliases,
)

# Select the citing IDs first; PAPER_COLUMNS over thousands of citing papers (e.g. BERT)
# otherwise makes PostgreSQL scan all chunks.
CITING_SQL = (
    "SELECT " + PAPER_COLUMNS + " FROM (SELECT p.corpus_id FROM papers p "
    "WHERE EXISTS (SELECT 1 FROM citations edge WHERE edge.citing_id=p.corpus_id "
    "AND edge.cited_id=%(target_id)s) "
    "AND (%(low)s::integer IS NULL OR p.year >= %(low)s) "
    "AND (%(high)s::integer IS NULL OR p.year <= %(high)s) "
    "AND (%(author_id)s::bigint IS NULL OR EXISTS (SELECT 1 FROM paper_authors pa "
    "WHERE pa.corpus_id=p.corpus_id AND pa.author_id=%(author_id)s)) "
    "ORDER BY p.year DESC NULLS LAST,p.corpus_id LIMIT %(limit)s) top "
    "JOIN papers p USING (corpus_id) ORDER BY p.year DESC NULLS LAST,p.corpus_id"
)


def baseline_candidate(
    text: str, title: str, section: str | None = None, *, target_id: int | None = None
) -> bool:
    """Conservative candidate filter, never a replacement for WP3 entailment checks.

    Require an explicit experimental-use expression tied to the target in one
    sentence. Negated and related-work passages are excluded. Human labels are
    required to measure the precision/recall of this heuristic.
    """
    if re.search(r"related work|references|bibliograph", section or "", re.IGNORECASE):
        return False
    aliases = target_aliases(title, target_id)
    if not aliases:
        return False
    verified = VERIFIED_ALIASES.get(target_id)
    short_names = {name.casefold() for name in verified.short_names} if verified else set()
    # Preserve title-prefix matching for BERT-base/BERT-large even when BERT is reviewed.
    short_names.difference_update(name.casefold() for name in target_aliases(title))
    alias_patterns = []
    for alias in sorted(aliases, key=len, reverse=True):
        # Reviewed short names exclude distinct models such as DPR-PAQ. Preserve
        # main's title/prefix boundaries, which allow BERT-base and BERT-large.
        boundary = r"[\w-]" if alias.casefold() in short_names else r"\w"
        literal = re.escape(alias).replace(r"\ ", r"\s+")
        alias_patterns.append(rf"(?<!{boundary}){literal}(?!{boundary})")
    alias_pattern = re.compile("|".join(alias_patterns), re.IGNORECASE)
    # Protect punctuation in recognized author-year aliases before sentence splitting.
    # Choose a marker absent from source text so literal marker text cannot match.
    marker = "S2TARGETMENTION"
    while marker.casefold() in text.casefold():
        marker += "X"
    normalized, mentions = alias_pattern.subn(marker, text)
    if not mentions:
        return False
    target = re.escape(marker)
    # Only recognized citations may interrupt the target/predicate: DPR (Karpukhin ...)
    # or DPR [12]. Do not skip arbitrary parenthetical claims about other models.
    normalized = re.sub(
        target + r"\s*(?:\(\s*" + target + r"\s*\)|\[\d+(?:\s*[,;-]\s*\d+)*\])",
        marker,
        normalized,
    )
    modifiers = r"(?:(?:pretrained|pre-trained|strong|standard|retrieval|dense)\s+){0,2}"
    baseline = r"(?:(?:a|the|our|its|their)\s+)?" + modifiers + r"baseline\b"
    passive_use = r"(?:is|was|has been)\s+(?:used|adopted|evaluated)\s+as\s+"
    passive_baseline = target + r"\s+" + passive_use + baseline
    study_scope = r"\b(?:in|for|during)\s+our\s+(?:experiments|evaluation|study)\b"
    own_model = r"\bour\s+(?:method|model|approach|system)\b"
    passive_comparison = r"(?:is|was|has been)\s+compared\s+(?:with|against|to)\s+"
    # Passive use needs a directly linked own-study cue, not one elsewhere in the sentence.
    # Preserve the existing active rules and negation/hypothetical exclusions.
    patterns = (
        r"\b(?:we|our (?:experiments|evaluation))\s+"
        r"(?:use|used|are using|adopt|adopted|evaluate|evaluated)\b[^.!?]{0,100}"
        + target
        + r"[^.!?]{0,50}\bas\s+"
        + baseline,
        r"\bwe\s+compar(?:e|ed)\b[^.!?]{0,60}\b(?:with|against|to)\s+" + target,
        r"\bour comparisons?\b[^.!?]{0,60}\b(?:with|against|to)\s+" + target,
        target + r"\s+(?:is|was|serves as|served as)\s+our\s+" + modifiers + r"baseline\b",
    )
    passive_patterns = (
        study_scope + r"\s*,?\s+" + passive_baseline,
        passive_baseline + r"\s+" + study_scope,
        target + r"\s+" + passive_use + r"our\s+" + modifiers + r"baseline\b",
        own_model + r"\s+" + passive_comparison + target,
        target + r"\s+" + passive_comparison + own_model,
    )
    # A citation such as "Smith et al., 2022" is not a sentence boundary.
    normalized = re.sub(r"\bet\s+al\.", "et al", normalized, flags=re.IGNORECASE)
    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        # "Without fine-tuning" describes how a baseline is used, not whether it is used.
        negation_text = re.sub(
            r"\bwithout\s+(?:any\s+)?fine[- ]tuning\b", "", sentence,
            flags=re.IGNORECASE,
        )
        if re.search(
            r"\b(?:not|no|never|without|neither|could|might|would|will)\b|\?",
            negation_text,
            re.IGNORECASE,
        ):
            continue
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns):
            return True
        # A composite baseline can identify its retriever by acronym after a
        # description, e.g. "our baseline ... with a dense passage retriever (DPR; ...)".
        composite = (
            r"\bour baseline\s+is\b[^.!?]{0,200}\bwith\s+(?:a|the)\s+"
            r"(?:dense\s+)?(?:passage\s+)?retriever\s*\(\s*" + target + r"(?=[\s;,)])"
        )
        if re.search(composite, sentence, re.IGNORECASE):
            return True
        # In a baselines section, an explicitly named target-based baseline
        # directly using the target is a comparison candidate, not a mention.
        named_baseline = (
            target + r"\s+\w+\s+directly\s+uses\s+(?:the\s+)?" + target + r"\b"
        )
        if re.search(r"\bbaselines?\b", section or "", re.IGNORECASE) and re.search(
            named_baseline, sentence, re.IGNORECASE
        ):
            return True
        if re.search(
            r"\b(?:earlier|prior|previous)\s+(?:work|research|studies|study|experiments|evaluations)\b",
            sentence,
            re.IGNORECASE,
        ):
            continue
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in passive_patterns):
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
        update={
            "topic": baseline_search_topic(target["title"], target["corpus_id"])
            if plan.require_baseline
            else f"{target['title']} experimental baseline comparison"
        }
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
                if baseline_candidate(
                    c["text"], target["title"], c.get("section"), target_id=target["corpus_id"]
                )
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
