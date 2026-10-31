import threading
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
MAX_TOKENS = 512
# Overlap so an instruction cut by a window border still appears whole in the next one.
OVERLAP_TOKENS = 64
BATCH_SIZE = 8
# The checkpoint's id2label is the generic LABEL_0 / LABEL_1; index 1 is "malicious".
MALICIOUS = 1


def _windows(ids: list[int]) -> list[list[int]]:
    body = MAX_TOKENS - 2
    step = body - OVERLAP_TOKENS
    starts = range(0, max(len(ids) - OVERLAP_TOKENS, 1), step)
    return [ids[start : start + body] for start in starts]


class PromptGuard:
    """Prompt Guard 2 on CPU. The model loads on first use, gated access needs HF_TOKEN."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self._model_id = model_id
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None

    def windows(self, text: str) -> list[list[int]]:
        """Model inputs for the text: overlapping windows with CLS and SEP."""
        tokenizer = self._load_tokenizer()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
        return [[cls, *w, sep] for w in _windows(ids)]

    def score(self, text: str) -> float:
        """Highest malicious probability over all windows of the text."""
        windows = self.windows(text)
        tokenizer, model = self._load()
        worst = 0.0
        for i in range(0, len(windows), BATCH_SIZE):
            batch = windows[i : i + BATCH_SIZE]
            width = max(len(w) for w in batch)
            pad = tokenizer.pad_token_id
            input_ids = torch.tensor([w + [pad] * (width - len(w)) for w in batch])
            mask = torch.tensor([[1] * len(w) + [0] * (width - len(w)) for w in batch])
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=mask).logits
            worst = max(worst, logits.softmax(-1)[:, MALICIOUS].max().item())
        return worst

    def _load_tokenizer(self):
        with self._lock:
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        return self._tokenizer

    def _load(self):
        tokenizer = self._load_tokenizer()
        with self._lock:
            if self._model is None:
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_id
                ).eval()
        return tokenizer, self._model


class RemoteGuard:
    """The same model on the inference server's GPU (D079). Windows are cut here, so the
    server only runs the model; one request per evidence check."""

    def __init__(self, url: str, api_key: str | None, model_id: str = MODEL_ID, transport=None):
        import httpx

        self._url = url
        self._model_id = model_id
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # Modal answers a slow call (cold start, first model load) with a redirect to wait on.
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(30, connect=10),
            follow_redirects=True,
            transport=transport,
        )

    def scores(self, windows: list[list[list[int]]]) -> list[float]:
        """Highest malicious probability per text, given the windows of each text."""
        import httpx

        flat = [w for text_windows in windows for w in text_windows]
        if not flat:
            return []
        body = {"model": self._model_id, "windows": flat}
        try:
            response = self._client.post(self._url, json=body)
        # Scoring has no side effects, so a dropped connection is retried once.
        except httpx.TransportError:
            response = self._client.post(self._url, json=body)
        response.raise_for_status()
        data = response.json()
        if data.get("model") != self._model_id:
            raise RuntimeError(f"guard server runs {data.get('model')}, expected {self._model_id}")
        probs = iter(data["malicious"])
        return [max(next(probs) for _ in text_windows) for text_windows in windows]


@lru_cache(maxsize=1)
def shared_guard() -> PromptGuard:
    """One model instance for the query check and the evidence check."""
    return PromptGuard()
