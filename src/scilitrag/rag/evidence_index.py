from collections.abc import Iterable

from scilitrag.contracts import Evidence, EvidenceMapping


class EvidenceIndex:
    """A defensive per-request snapshot; callers never receive the internal objects."""

    def __init__(self, evidence: Iterable[Evidence]):
        self._items: EvidenceMapping = {}
        for source in evidence:
            item = source.model_copy(deep=True)
            if item.label < 1 or item.label in self._items:
                raise ValueError("Evidence labels must be unique positive integers.")
            if item.type == "chunk" and (not item.chunk_id or item.corpus_id is None):
                raise ValueError("Chunk evidence requires a chunk ID and corpus ID.")
            if item.type == "paper" and item.corpus_id is None:
                raise ValueError("Paper evidence requires a corpus ID.")
            if item.type == "sql" and not item.query_id:
                raise ValueError("SQL evidence requires a query ID.")
            if any(author_id < 1 for author_id in item.author_ids):
                raise ValueError("Author IDs must be positive integers.")
            item.author_ids = sorted(set(item.author_ids))
            self._items[item.label] = item

    def mapping(self) -> EvidenceMapping:
        return {label: item.model_copy(deep=True) for label, item in self._items.items()}

    def resolve(self, label: int) -> Evidence:
        if label not in self._items:
            raise ValueError(f"Unknown citation label [{label}].")
        return self._items[label].model_copy(deep=True)

    def values(self) -> list[Evidence]:
        return list(self.mapping().values())


def allocate_evidence(sources: Iterable[Evidence]) -> list[Evidence]:
    labelled = [
        item.model_copy(update={"label": index}, deep=True) for index, item in enumerate(sources, 1)
    ]
    return EvidenceIndex(labelled).values()
