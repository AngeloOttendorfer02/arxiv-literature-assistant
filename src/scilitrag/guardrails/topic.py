"""Topic gate (S-06): cosine similarity of the query embedding to labelled prototypes."""

import math
import threading

from scilitrag.contracts import Embed

REFUSALS = ("write_intent", "off_topic")


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def decide(scores: dict[str, float], margin: float) -> tuple[str, float]:
    """Category and gap; a refusing class has to beat allow by more than the margin."""
    label = max(REFUSALS, key=scores.__getitem__)
    gap = scores[label] - scores["allow"]
    return (label if gap > margin else "ok"), gap


class TopicGate:
    """Prototype vectors are embedded on first use, so building the component loads nothing."""

    def __init__(self, embed: Embed, prototypes: dict[str, list[str]]) -> None:
        self._embed = embed
        self._prototypes = prototypes
        self._lock = threading.Lock()
        self._vectors: dict[str, list[list[float]]] | None = None

    def scores(self, text: str) -> dict[str, float]:
        """Highest cosine similarity to the prototypes of each class."""
        vectors = self._load()
        query = _unit(self._embed([text])[0])
        return {
            label: max(sum(a * b for a, b in zip(query, vector)) for vector in class_vectors)
            for label, class_vectors in vectors.items()
        }

    def _load(self) -> dict[str, list[list[float]]]:
        with self._lock:
            if self._vectors is None:
                texts = [text for items in self._prototypes.values() for text in items]
                unit = iter(_unit(vector) for vector in self._embed(texts))
                self._vectors = {
                    label: [next(unit) for _ in items] for label, items in self._prototypes.items()
                }
        return self._vectors
