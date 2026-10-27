"""Real embed component (D-14): granite embeddings for papers, chunks and queries (D044)."""

import math
from functools import lru_cache

from scilitrag.config import Settings
from scilitrag.contracts import Embed

# Pinned like D-22 and the D-13 tokenizer, so stored vectors and chunk boundaries match.
MODEL_REVISIONS = {
    "ibm-granite/granite-embedding-english-r2": "47ea694b257b703fee9253d75c2b1f2985180498",
}
# Largest O5 chunk size (D-13); special tokens come on top, so whole chunks are embedded.
CONTENT_TOKENS = 512
DIMENSION = 768


def document_text(title, abstract):
    """Paper text for papers.embedding, as compared in D-22."""
    title, abstract = (title or "").strip(), (abstract or "").strip()
    return f"{title}\n{abstract}" if title and abstract else title or abstract


def fingerprint(model: str) -> str:
    """Everything that changes stored vectors: model, revision, input limit and texts."""
    return (
        f"{model}@{MODEL_REVISIONS[model]};content_tokens={CONTENT_TOKENS};"
        "papers=title+newline+abstract;chunks=text;normalized"
    )


def check_vectors(vectors, count: int) -> None:
    """Reject wrong counts, wrong sizes, non-finite values and non-normalized vectors."""
    if len(vectors) != count:
        raise ValueError(f"The embedding model returned {len(vectors)} vectors for {count} texts.")
    for vector in vectors:
        if len(vector) != DIMENSION or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"Embeddings must be finite {DIMENSION}-dimensional vectors.")
        # A zero vector has no cosine direction and is skipped by the cosine index.
        if not 0.99 <= math.sqrt(sum(value * value for value in vector)) <= 1.01:
            raise ValueError("Embeddings must be normalized; zero vectors are not allowed.")


@lru_cache(maxsize=2)
def load_model(model: str):
    if model not in MODEL_REVISIONS:
        raise ValueError(f"No pinned revision for embedding model {model}.")
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp32 on CPU; granite would otherwise load in bf16, which is slow on most CPUs.
    dtype = torch.float16 if device == "cuda" else torch.float32
    encoder = SentenceTransformer(
        model, revision=MODEL_REVISIONS[model], device=device, model_kwargs={"dtype": dtype}
    )
    encoder.max_seq_length = CONTENT_TOKENS + encoder.tokenizer.num_special_tokens_to_add()
    return encoder


def encode(model: str, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Normalized vectors, so cosine distance (<=>) ranks them."""
    if not texts:
        return []
    vectors = load_model(model).encode(
        list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
    )
    result = [[float(value) for value in vector] for vector in vectors]
    check_vectors(result, len(texts))
    return result


def build_component(settings: Settings) -> Embed:
    def embed(texts: list[str]) -> list[list[float]]:
        return encode(settings.embed_model, texts)

    return embed
