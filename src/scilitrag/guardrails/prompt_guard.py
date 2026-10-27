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

    def score(self, text: str) -> float:
        """Highest malicious probability over all windows of the text."""
        tokenizer, model = self._load()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
        windows = [[cls, *w, sep] for w in _windows(ids)]
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

    def _load(self):
        with self._lock:
            if self._model is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_id
                ).eval()
        return self._tokenizer, self._model


@lru_cache(maxsize=1)
def shared_guard() -> PromptGuard:
    """One model instance for the query check and the evidence check."""
    return PromptGuard()
