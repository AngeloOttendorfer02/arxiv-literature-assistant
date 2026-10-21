from scilitrag.config import load_settings
from scilitrag.contracts import Plan
from scilitrag.rag.database import DatabaseReader, vector_literal
from scilitrag.rag.database_contract import validate_database
from scilitrag.rag.entities import EntityLinker
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.search_readiness import validate_search_indexes

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
