import hashlib

DIM = 8


def embed(texts: list[str]) -> list[list[float]]:
    return [[b / 255 for b in hashlib.sha256(text.encode()).digest()[:DIM]] for text in texts]
