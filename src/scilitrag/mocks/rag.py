import asyncio
import re
from collections.abc import AsyncIterator

from scilitrag.contracts import (
    Citation,
    Evidence,
    EvidenceMapping,
    Plan,
    RenderedSentence,
    SqlFacts,
)

LABEL = re.compile(r"\[(\d+)\]")

EVIDENCE = [
    Evidence(
        label=1,
        type="chunk",
        chunk_id="mock-1-0",
        corpus_id=900001,
        title="MOCK Dense Retrieval for Scientific Search",
        year=2021,
        section="Method",
        text="We train a dual encoder on citation pairs and report recall@20.",
        score=0.91,
    ),
    Evidence(
        label=2,
        type="chunk",
        chunk_id="mock-2-3",
        corpus_id=900002,
        title="MOCK Hybrid Search with BM25 and Embeddings",
        year=2022,
        section="Experiments",
        text="Combining BM25 with dense scores improves results for rare terms.",
        score=0.87,
    ),
    Evidence(
        label=3,
        type="paper",
        corpus_id=900003,
        title="MOCK A Survey of Literature Search",
        year=2023,
        text="This survey compares sparse, dense and hybrid retrieval methods.",
        score=0.80,
    ),
]

ANSWER = (
    "MOCK answer: dense retrieval improves recall on scientific queries [1]. "
    "Hybrid search with BM25 helps for rare terms [2]. "
    "This sentence has no source and is removed by the check. "
    "A recent survey compares both approaches [1][3]."
)


def plan(masked_query: str) -> Plan:
    query = masked_query.lower()
    if "most cited author" in query:
        scenario = "S3"
    elif "cite" in query or "baseline" in query:
        scenario = "S2"
    else:
        scenario = "S1"
    return Plan(scenario=scenario, topic=masked_query, k=3)


def retrieve(plan: Plan) -> list[Evidence]:
    return list(EVIDENCE)


def sql_facts(plan: Plan) -> list[SqlFacts]:
    if plan.scenario != "S3":
        return []
    return [
        SqlFacts(
            query_id="mock_top_cited_authors",
            sql="SELECT author, citations FROM mock_top_cited_authors",
            columns=["author", "citations"],
            rows=[["MOCK Author A", 12], ["MOCK Author B", 9], ["MOCK Author C", 7]],
        )
    ]


class Generate:
    def __init__(self, token_delay: float = 0.02) -> None:
        self.token_delay = token_delay

    async def __call__(self, plan: Plan, evidence: list[Evidence]) -> AsyncIterator[str]:
        for token in re.findall(r"\S+\s*", ANSWER):
            if self.token_delay:
                await asyncio.sleep(self.token_delay)
            yield token


def render_citations(sentence: str, mapping: EvidenceMapping) -> RenderedSentence:
    citations = []
    for label in dict.fromkeys(int(n) for n in LABEL.findall(sentence)):
        evidence = mapping.get(label)
        if evidence is None:
            continue
        citations.append(
            Citation(
                label=label,
                corpus_id=evidence.corpus_id,
                chunk_id=evidence.chunk_id,
                title=evidence.title,
                year=evidence.year,
                query_id=evidence.query_id,
                section=evidence.section,
                text=evidence.text,
            )
        )
    return RenderedSentence(text=sentence, citations=citations)
