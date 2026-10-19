"""Controlled component failures translated into events by the API pipeline."""


class LLMClientError(RuntimeError):
    """A safe model-service message with a machine-readable failure category.

    Provider response bodies must not be included in the public message.
    """

    def __init__(self, message: str, *, kind: str = "protocol") -> None:
        super().__init__(message)
        self.kind = kind


class RagUserError(ValueError):
    """A controlled clarification or no-results message for the API."""
