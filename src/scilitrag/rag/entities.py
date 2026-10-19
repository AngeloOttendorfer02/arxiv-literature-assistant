from dataclasses import dataclass

from scilitrag.rag.errors import RagUserError


@dataclass(frozen=True)
class PaperCandidate:
    corpus_id: int
    title: str
    similarity: float | None = None


class EntityLinker:
    def __init__(self, reader):
        self.reader = reader
        self._cache: dict[tuple, int] = {}

    def paper(self, reference: str) -> int:
        reference = reference.strip().strip('"')
        if not reference or len(reference) > 512:
            raise RagUserError("Provide a paper title or numeric corpus ID.")
        key = ("paper", reference.casefold())
        if key in self._cache:
            return self._cache[key]
        if reference.isdigit():
            if int(reference) > 9223372036854775807:
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
            and first.similarity >= 0.65
            and (len(candidates) == 1 or first.similarity - (candidates[1].similarity or 0) >= 0.15)
        )
        if (first.similarity is not None and not confident) or (
            len(candidates) > 1 and not confident
        ):
            choices = "; ".join(f"{item.corpus_id}: {item.title}" for item in candidates[:5])
            raise RagUserError(f"Which paper do you mean? Use a corpus ID: {choices}")
        self._cache[key] = first.corpus_id
        return first.corpus_id

    def paper_record(self, reference: str) -> dict:
        corpus_id = self.paper(reference)
        rows = self.reader(
            "SELECT corpus_id,title,year,abstract FROM papers WHERE corpus_id=%(id)s",
            {"id": corpus_id},
        )
        if len(rows) != 1:
            raise RagUserError("The selected paper is no longer present in this database.")
        return rows[0]
