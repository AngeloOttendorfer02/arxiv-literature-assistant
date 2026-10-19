"""Resolve paper/author references through a caller-supplied SQL reader.

The reader must bind named parameters and return mapping rows. Database
adapters must provide compatible tables and use the same name-normalization
rule for ingestion and lookup.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scilitrag.rag.errors import RagUserError

MAX_REFERENCE_CHARS = 512
MAX_DATABASE_ID = 9_223_372_036_854_775_807
MIN_PAPER_SIMILARITY = 0.65
MIN_SIMILARITY_MARGIN = 0.15
MAX_DISPLAYED_CANDIDATES = 5

type DatabaseRow = dict[str, Any]
type ReadRows = Callable[[str, dict[str, Any]], list[DatabaseRow]]
type EntityCacheKey = tuple[str, str] | tuple[str, str, int | None]


def normalize_name(value: str) -> str:
    """Normalize names using the rule that must also be applied during ingestion."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", value).split())


@dataclass(frozen=True)
class PaperCandidate:
    """A candidate with an optional trigram score for fuzzy matches."""

    corpus_id: int
    title: str
    similarity: float | None = None


class EntityLinker:
    """Resolve unambiguous references; instantiate per request to scope the cache."""

    def __init__(self, reader: ReadRows) -> None:
        self.reader = reader
        self._cache: dict[EntityCacheKey, int] = {}

    def paper(self, reference: str) -> int:
        """Try numeric ID, full title, title prefix and finally a confident fuzzy match."""
        reference = reference.strip().strip('"')
        if not reference or len(reference) > MAX_REFERENCE_CHARS:
            raise RagUserError("Provide a paper title or numeric corpus ID.")
        key = ("paper", reference.casefold())
        if key in self._cache:
            return self._cache[key]
        if reference.isdigit():
            if int(reference) > MAX_DATABASE_ID:
                raise RagUserError("The corpus ID is outside the database ID range.")
            rows = self.reader(
                "SELECT corpus_id,title FROM papers WHERE corpus_id=%(id)s", {"id": int(reference)}
            )
        else:
            rows = self.reader(
                "SELECT corpus_id,title FROM papers WHERE lower(title)=lower(%(ref)s) "
                "ORDER BY corpus_id LIMIT 6",
                {"ref": reference},
            )
            if not rows:
                rows = self.reader(
                    "SELECT corpus_id,title FROM papers WHERE "
                    "lower(btrim(split_part(title,':',1)))=lower(%(ref)s) "
                    "ORDER BY corpus_id LIMIT 6",
                    {"ref": reference},
                )
            if not rows:
                # %% is a literal trigram operator when the driver expands
                # %(ref)s placeholders; the user value remains a bound parameter.
                rows = self.reader(
                    "SELECT corpus_id,title,similarity(title,%(ref)s) AS similarity "
                    "FROM papers WHERE title %% %(ref)s "
                    "ORDER BY similarity DESC,corpus_id LIMIT 6",
                    {"ref": reference},
                )
        candidates = [
            PaperCandidate(int(row["corpus_id"]), row["title"], row.get("similarity"))
            for row in rows
        ]
        if not candidates:
            raise RagUserError("Paper not found. Provide its full title or corpus ID.")
        first = candidates[0]
        confident = (
            first.similarity is not None
            and first.similarity >= MIN_PAPER_SIMILARITY
            and (
                len(candidates) == 1
                or first.similarity - (candidates[1].similarity or 0) >= MIN_SIMILARITY_MARGIN
            )
        )
        if (first.similarity is not None and not confident) or (
            len(candidates) > 1 and not confident
        ):
            choices = "; ".join(
                f"{item.corpus_id}: {item.title}" for item in candidates[:MAX_DISPLAYED_CANDIDATES]
            )
            raise RagUserError(f"Which paper do you mean? Use a corpus ID: {choices}")
        self._cache[key] = first.corpus_id
        return first.corpus_id

    def paper_record(self, reference: str) -> DatabaseRow:
        """Load metadata for the resolved identity, detecting stale cached IDs."""
        corpus_id = self.paper(reference)
        rows = self.reader(
            "SELECT corpus_id,title,year,abstract FROM papers WHERE corpus_id=%(id)s",
            {"id": corpus_id},
        )
        if len(rows) != 1:
            raise RagUserError("The selected paper is no longer present in this database.")
        return rows[0]

    def author(self, reference: str, *, paper_ref: str | None = None) -> int:
        """Resolve a normalized name/ID, optionally restricted to a linked paper."""
        raw = reference.strip()
        if not raw or len(raw) > MAX_REFERENCE_CHARS:
            raise RagUserError("Provide the author's complete name or numeric author ID.")
        paper_id = self.paper(paper_ref) if paper_ref else None
        name = normalize_name(raw)
        key = ("author", name, paper_id)
        if key in self._cache:
            return self._cache[key]
        params: dict[str, Any] = {"name": name, "paper_id": paper_id}
        if raw.isdigit():
            params["id"] = int(raw)
            if params["id"] > MAX_DATABASE_ID:
                raise RagUserError("The author ID is outside the database ID range.")
            where = "a.author_id=%(id)s"
        else:
            where = "a.normalized_name=%(name)s"
        rows = self.reader(
            "SELECT a.author_id,a.name FROM authors a WHERE "
            + where
            + " AND (%(paper_id)s::bigint IS NULL OR EXISTS (SELECT 1 FROM paper_authors pa "
            "WHERE pa.author_id=a.author_id AND pa.corpus_id=%(paper_id)s)) "
            "ORDER BY a.author_id LIMIT 6",
            params,
        )
        if not rows:
            raise RagUserError(
                "Author not found for these filters. Use the complete indexed name or author ID."
            )
        if len(rows) > 1:
            choices = "; ".join(
                f"{row['author_id']}: {row['name']}" for row in rows[:MAX_DISPLAYED_CANDIDATES]
            )
            raise RagUserError(f"Which author do you mean? Use an author ID: {choices}")
        result = int(rows[0]["author_id"])
        self._cache[key] = result
        return result
