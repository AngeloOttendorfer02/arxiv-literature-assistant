class LLMClientError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "protocol") -> None:
        super().__init__(message)
        self.kind = kind
