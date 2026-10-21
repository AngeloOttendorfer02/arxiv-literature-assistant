from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SearchTrace:
    dense_candidates: int
    bm25_candidates: int
    returned: int
    dense_ms: float
    bm25_ms: float
    fusion_ms: float
    rrf_k: int = 60

    def public_fields(self) -> dict:
        return asdict(self)
