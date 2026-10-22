from scilitrag.rag.errors import LLMClientError


class SentenceBuffer:
    """Bound unfinished model output while preserving the pluggable splitter."""

    def __init__(self, splitter, *, max_chars: int = 32768):
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self.splitter = splitter
        self.max_chars = max_chars
        self.pending = ""

    def feed(self, token: str) -> list[str]:
        if not isinstance(token, str):
            raise LLMClientError("The model stream contained a non-text delta.")
        if len(self.pending) + len(token) > self.max_chars:
            raise LLMClientError("The model produced too much unfinished sentence text.")
        completed, self.pending = self.splitter(self.pending + token)
        return completed

    def finish(self) -> list[str]:
        tail = self.pending.strip()
        self.pending = ""
        if not tail:
            return []
        if tail[-1] not in ".!?":
            raise LLMClientError("The model stream ended with an incomplete sentence.")
        return [tail]
