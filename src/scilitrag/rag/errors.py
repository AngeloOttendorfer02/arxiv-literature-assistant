class LLMClientError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "protocol") -> None:
        super().__init__(message)
        self.kind = kind


class RagUserError(ValueError):
    """A controlled clarification or no-results message for the existing API."""
