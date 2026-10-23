def reciprocal_rank_fusion(
    *rankings: list[dict], limit: int = 20, k: int = 60, key_field: str = "corpus_id"
) -> list[dict]:
    if k < 1 or limit < 1:
        raise ValueError("RRF parameters must be positive.")
    scores: dict[int | str, float] = {}
    rows: dict[int | str, dict] = {}
    for ranking in rankings:
        seen = set()
        rank = 0
        for row in ranking:
            key = row[key_field]
            if key in seen:
                continue
            seen.add(key)
            rank += 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            rows.setdefault(key, dict(row))
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
    return [{**rows[key], "score": scores[key]} for key in ordered]
