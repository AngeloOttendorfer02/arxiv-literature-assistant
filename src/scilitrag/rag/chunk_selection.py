import re


def overlap(left: str, right: str) -> float:
    a = set(re.findall(r"\w+", left.casefold()))
    b = set(re.findall(r"\w+", right.casefold()))
    return len(a & b) / len(a | b) if a or b else 1.0


def select_passages(ranked: list[dict], *, limit: int) -> list[dict]:
    selected = []
    remaining = list(ranked)
    while remaining and len(selected) < limit:
        sections = {item.get("section") for item in selected}
        # Prefer useful section coverage, then preserve the fused ranking.
        index = next(
            (i for i, item in enumerate(remaining) if item.get("section") not in sections), 0
        )
        candidate = remaining.pop(index)
        if any(overlap(candidate["text"], item["text"]) >= 0.9 for item in selected):
            continue
        selected.append(candidate)
    return selected
