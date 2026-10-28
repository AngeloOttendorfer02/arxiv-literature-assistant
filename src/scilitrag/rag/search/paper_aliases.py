"""Reviewed target aliases shared by S2 passage matching and offline evaluation."""

from dataclasses import dataclass


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
