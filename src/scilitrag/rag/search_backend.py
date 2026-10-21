import logging
import time

from scilitrag.config import load_settings
from scilitrag.contracts import Plan
from scilitrag.rag.chunk_selection import select_passages
from scilitrag.rag.database import DatabaseReader, vector_literal
from scilitrag.rag.database_contract import validate_database
from scilitrag.rag.entities import EntityLinker
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.ranking import reciprocal_rank_fusion
from scilitrag.rag.search_readiness import validate_search_indexes
from scilitrag.rag.search_trace import SearchTrace

logger = logging.getLogger(__name__)

PAPER_COLUMNS = """
    p.corpus_id, COALESCE(p.title, 'Untitled paper') AS title, p.year, COALESCE(p.abstract, '') AS abstract,
    EXISTS(SELECT 1 FROM chunks indexed WHERE indexed.corpus_id=p.corpus_id
           AND btrim(indexed.text) <> '') AS has_indexed_chunks,
    ARRAY(SELECT pa.author_id FROM paper_authors pa
          WHERE pa.corpus_id=p.corpus_id ORDER BY pa.author_id) AS author_ids
"""
PAPER_FILTERS = """
    (%(low)s::integer IS NULL OR p.year >= %(low)s)
    AND (%(high)s::integer IS NULL OR p.year <= %(high)s)
    AND (%(author_id)s::bigint IS NULL OR EXISTS (
        SELECT 1 FROM paper_authors af WHERE af.corpus_id=p.corpus_id
        AND af.author_id=%(author_id)s))
    AND (%(paper_id)s::bigint IS NULL OR p.corpus_id=%(paper_id)s)
"""
DENSE_SQL = (
    "SELECT " + PAPER_COLUMNS + ", 1-(p.embedding <=> %(vector)s::vector) AS score "
    "FROM papers p WHERE p.embedding IS NOT NULL AND "
    + PAPER_FILTERS
    + " ORDER BY p.embedding <=> %(vector)s::vector, p.corpus_id LIMIT %(limit)s"
)


BM25_SQL = (
    "SELECT " + PAPER_COLUMNS + ", pdb.score(p.corpus_id) AS score "
    "FROM papers p WHERE p.abstract ||| %(topic)s AND "
    + PAPER_FILTERS
    + " ORDER BY pdb.score(p.corpus_id) DESC, p.corpus_id LIMIT %(limit)s"
)


CHUNK_DENSE_SQL = """
    SELECT hit.* FROM unnest(%(paper_ids)s::bigint[]) AS ids(corpus_id)
    CROSS JOIN LATERAL (
        SELECT c.chunk_id,c.corpus_id,c.section,c.text,
               1-(c.embedding <=> %(vector)s::vector) AS score
        FROM chunks c WHERE c.corpus_id=ids.corpus_id AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> %(vector)s::vector,c.chunk_id LIMIT %(candidates)s
    ) hit ORDER BY hit.corpus_id,hit.score DESC,hit.chunk_id
"""
CHUNK_BM25_SQL = """
    SELECT hit.* FROM unnest(%(paper_ids)s::bigint[]) AS ids(corpus_id)
    CROSS JOIN LATERAL (
        SELECT c.chunk_id,c.corpus_id,c.section,c.text,pdb.score(c.chunk_id) AS score
        FROM chunks c WHERE c.corpus_id=ids.corpus_id AND c.text ||| %(topic)s
        ORDER BY pdb.score(c.chunk_id) DESC,c.chunk_id LIMIT %(candidates)s
    ) hit ORDER BY hit.corpus_id,hit.score DESC,hit.chunk_id
"""


class SearchBackend:
    def __init__(self, reader, embed, *, dimension: int = 768):
        self.reader = reader
        self.embed = embed
        self.dimension = dimension
        self.settings = None
        self.linker = EntityLinker(reader)

    def params(self, plan: Plan, limit: int) -> dict:
        if not 1 <= limit <= 100:
            raise ValueError("Search limit must be 1..100.")
        return {
            "topic": (plan.topic or "").strip(),
            "low": plan.year_from,
            "high": plan.year_to,
            "limit": limit,
            "author_id": self.linker.author(
                plan.author_ref, paper_ref=plan.paper_ref if plan.scenario == "S1" else None
            )
            if plan.author_ref and plan.scenario != "S3"
            else None,
            "paper_id": self.linker.paper(plan.paper_ref)
            if plan.paper_ref and plan.scenario == "S1"
            else None,
        }

    def query_vector(self, query: str) -> str:
        if not query.strip():
            raise RagUserError("Provide a topic for the literature search.")
        vectors = self.embed([query])
        if not isinstance(vectors, (list, tuple)) or len(vectors) != 1:
            raise RagUserError("The embedding model returned an invalid batch.")
        return vector_literal(vectors[0], self.dimension)

    def dense(self, plan: Plan, limit: int = 50) -> list[dict]:
        params = self.params(plan, limit)
        params["vector"] = self.query_vector(params["topic"])
        return self.reader(DENSE_SQL, params)

    def bm25(self, plan: Plan, limit: int = 50) -> list[dict]:
        params = self.params(plan, limit)
        if not params["topic"]:
            return []
        return self.reader(BM25_SQL, params)

    def hybrid(self, plan: Plan, limit: int | None = None) -> list[dict]:
        wanted = plan.k if limit is None else limit
        if not 1 <= wanted <= 20:
            raise RagUserError("Choose a result limit between 1 and 20.")
        candidates = max(50, wanted * 5)
        start = time.perf_counter()
        params = self.params(plan, candidates)
        params["vector"] = self.query_vector(params["topic"])
        dense = self.reader(DENSE_SQL, params)
        dense_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        sparse = self.reader(BM25_SQL, params) if params["topic"] else []
        sparse_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        rows = reciprocal_rank_fusion(dense, sparse, limit=wanted, k=60)
        self.last_trace = SearchTrace(
            len(dense),
            len(sparse),
            len(rows),
            dense_ms,
            sparse_ms,
            (time.perf_counter() - start) * 1000,
        )
        logger.info("hybrid_search %s", self.last_trace.public_fields())
        return rows

    def chunks(self, plan: Plan, paper_ids: list[int], per_paper: int = 2) -> list[dict]:
        ids = list(dict.fromkeys(paper_ids))
        if not ids:
            return []
        if len(ids) > 100 or any(item < 1 for item in ids) or not 1 <= per_paper <= 4:
            raise RagUserError(
                "Chunk search requires up to 100 valid paper IDs and 1..4 passages per paper."
            )
        query = f"{plan.topic or ''} methods metrics experimental results".strip()
        params = {
            "paper_ids": ids,
            "topic": query,
            "vector": self.query_vector(query),
            "candidates": 10,
        }
        dense = self.reader(CHUNK_DENSE_SQL, params)
        sparse = self.reader(CHUNK_BM25_SQL, params)
        result = []
        allowed = set(ids)
        if any(row["corpus_id"] not in allowed for row in dense + sparse):
            raise RagUserError("Chunk retrieval returned a source outside the selected papers.")
        for paper_id in ids:
            ranked = reciprocal_rank_fusion(
                [row for row in dense if row["corpus_id"] == paper_id],
                [row for row in sparse if row["corpus_id"] == paper_id],
                limit=20,
                k=60,
                key_field="chunk_id",
            )
            result.extend(select_passages(ranked, limit=per_paper))
        return result


def default_backend(settings=None) -> SearchBackend:
    settings = settings or load_settings()
    reader = DatabaseReader(settings)
    metadata = validate_database(reader, settings.embed_model)
    validate_search_indexes(reader)
    dimension = metadata.embed_dim
    try:
        from scilitrag.data import embedding
    except ImportError:
        raise RagUserError("The search embedding service is not available.") from None
    factory = getattr(embedding, "build_component", None)
    embed = factory(settings) if factory else embedding.embed
    backend = SearchBackend(reader, embed, dimension=dimension)
    backend.settings = settings
    backend.metadata = metadata
    return backend
