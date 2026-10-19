import re
from dataclasses import dataclass

from scilitrag.rag.errors import RagUserError

YEAR = r"(?:18|19|20|21)\d{2}"


@dataclass(frozen=True)
class ExplicitFilters:
    year_from: int | None = None
    year_to: int | None = None
    k: int | None = None


def parse_filters(query: str) -> ExplicitFilters:
    text = query.casefold()
    lows, highs = [], []
    for match in re.finditer(
        rf"\b(?:between|zwischen)\s+({YEAR})\s+(?:and|und)\s+({YEAR})\b", text
    ):
        lows.append(int(match[1]))
        highs.append(int(match[2]))
    for match in re.finditer(rf"\b({YEAR})\s*[-–]\s*({YEAR})\b", text):
        lows.append(int(match[1]))
        highs.append(int(match[2]))
    for match in re.finditer(rf"\b(since|from|seit|ab|after|nach)\s+({YEAR})\b", text):
        lows.append(int(match[2]) + int(match[1] in {"after", "nach"}))
    for match in re.finditer(rf"\b(until|through|bis|before|vor)\s+({YEAR})\b", text):
        highs.append(int(match[2]) - int(match[1] in {"before", "vor"}))
    for match in re.finditer(rf"\b(?:in|im)\s+({YEAR})\b", text):
        lows.append(int(match[1]))
        highs.append(int(match[1]))
    low = max(lows) if lows else None
    high = min(highs) if highs else None
    if low is not None and high is not None and low > high:
        raise RagUserError("The year filters conflict. Provide a valid start and end year.")
    limits = [
        int(match[1])
        for match in re.finditer(
            r"\b(?:top|best|limit(?:ed)?\s+to|max(?:imum)?|höchstens)\s+(\d+)\b", text
        )
    ]
    if len(set(limits)) > 1 or any(not 1 <= limit <= 20 for limit in limits):
        raise RagUserError("Choose one result limit between 1 and 20.")
    return ExplicitFilters(low, high, limits[0] if limits else None)
