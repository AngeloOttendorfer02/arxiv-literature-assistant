"""Reviewed paper aliases shared by user lookup, passage matching and evaluation."""

import re
import unicodedata
from dataclasses import dataclass

from scilitrag.rag.search.filters import YEAR


@dataclass(frozen=True)
class VerifiedAliases:
    short_names: tuple[str, ...]
    citation_author: str
    year: int


# DPR demo identity from D048. Do not guess acronyms or author order from IDs.
VERIFIED_ALIASES = {
    215737187: VerifiedAliases(("DPR",), "Karpukhin et al.", 2020),
}


def target_aliases(title: str, target_id: int | None = None) -> tuple[str, ...]:
    """Return title variants plus identity-specific, reviewed short/citation names."""
    aliases = [name for name in (title.strip(), title.split(":", 1)[0].strip()) if len(name) >= 4]
    verified = VERIFIED_ALIASES.get(target_id)
    if verified:
        author, year = verified.citation_author, verified.year
        aliases.extend(verified.short_names)
        aliases.extend((f"{author}, {year}", f"{author} ({year})", f"{author} {year}"))
    return tuple(dict.fromkeys(aliases))


def baseline_search_topic(title: str, target_id: int) -> str:
    """Include one spelling of each reviewed name in the passage search."""
    terms = [title]
    verified = VERIFIED_ALIASES.get(target_id)
    if verified:
        terms.extend(verified.short_names)
        terms.append(f"{verified.citation_author}, {verified.year}")
    return " ".join([*terms, "experimental baseline comparison"])


def _reference_key(value: str) -> str:
    """Ignore case and citation spacing while preserving spelling and hyphens."""
    value = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    return re.sub(r"\s*([(),])\s*", r"\1", value)


def reviewed_reference_ids(reference: str) -> list[int]:
    """Find reviewed identities; the caller must check DB existence and ambiguity."""
    key = _reference_key(reference)
    matches = []
    for corpus_id, verified in VERIFIED_ALIASES.items():
        aliases = target_aliases("", corpus_id)
        forms = [*aliases]
        forms.extend(
            f"{name} ({citation})"
            for name in verified.short_names
            for citation in aliases
            if citation not in verified.short_names
        )
        if key in {_reference_key(form) for form in forms}:
            matches.append(corpus_id)
    return sorted(matches)


def author_year_reference(reference: str) -> tuple[str, int] | None:
    """Parse citation-shaped references, not arbitrary titles ending in a year.

    Bare surname/year pairs require 'et al.'; single-author forms need parentheses
    or a comma. Multiword surnames are limited to explicit family-name particles.
    """
    surname = (
        r"(?:(?:van|von|de|del|der|da|di|la|le)\s+){0,3}"
        r"[^\W\d_]+(?:[-'’][^\W\d_]+)*"
    )
    match = re.fullmatch(
        rf"(?P<author>{surname})(?P<etal>\s+et\s+al\.?)?\s*"
        rf"(?:\(\s*(?P<parenthesized>{YEAR})\s*\)|,\s*(?P<comma>{YEAR})|\s+(?P<plain>{YEAR}))",
        reference.strip(),
        re.IGNORECASE,
    )
    if not match or (match["plain"] and not match["etal"]):
        return None
    year = match["parenthesized"] or match["comma"] or match["plain"]
    return " ".join(match["author"].split()), int(year)
