import json
import re
import threading
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scilitrag.config import Settings
from scilitrag.contracts import CheckSentence, Evidence, EvidenceMapping, SentenceCheck

# Checkpoint order; the config.json id2label of some NLI models disagrees with it.
NLI_LABELS = ("contradiction", "entailment", "neutral")
LABEL = re.compile(r"\[\s*\d+\s*\]")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+")


class NliModel:
    """NLI cross-encoder on CPU, loaded on first use."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None

    def probabilities(self, premise: str, hypothesis: str) -> dict[str, float]:
        tokenizer, model = self._load()
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            probs = model(**inputs).logits.softmax(-1)[0].tolist()
        return dict(zip(NLI_LABELS, probs, strict=True))

    def _load(self):
        with self._lock:
            if self._model is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_id
                ).eval()
        return self._tokenizer, self._model


def numbers(text: str) -> set[str]:
    found = set()
    for match in NUMBER.findall(text):
        found.add(match.replace(",", "") if THOUSANDS.fullmatch(match) else match.replace(",", "."))
    return found


def _source_text(evidence: Evidence) -> str:
    if evidence.type != "sql":
        return evidence.text
    try:
        table = json.loads(evidence.text)
        return json.dumps([table.get("rows"), table.get("parameters")])
    except (ValueError, AttributeError):
        return evidence.text


# Stage 1 compares numbers and years with the cited sources, stage 2 runs NLI per cited
# chunk or paper. SQL evidence is only checked by stage 1. One supporting source is enough.
def build_component(settings: Settings) -> CheckSentence:
    nli = NliModel(settings.nli_model)

    def check_sentence(sentence: str, mapping: EvidenceMapping) -> SentenceCheck:
        start = time.perf_counter()

        def result(status: str, scores: dict[str, float] | None = None) -> SentenceCheck:
            return SentenceCheck(
                status=status,
                nli_scores=scores or {},
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        labels = list(dict.fromkeys(int(n) for n in re.findall(r"\[\s*(\d+)\s*\]", sentence)))
        if any(label not in mapping for label in labels):
            return result("unknown_label")
        cited = [mapping[label] for label in labels]
        claim = " ".join(LABEL.sub("", sentence).split())

        known = numbers(" ".join(_source_text(item) for item in cited))
        if not numbers(claim) <= known:
            return result("fact_mismatch")

        texts = [item for item in cited if item.type != "sql"]
        if not texts:
            return result("ok")

        best = {name: 0.0 for name in NLI_LABELS}
        for item in texts:
            probs = nli.probabilities(item.text, claim)
            if probs["entailment"] >= best["entailment"]:
                best = probs
        if best["entailment"] >= settings.nli_entailment_threshold:
            return result("ok", best)
        if best["contradiction"] >= settings.nli_contradiction_threshold:
            return result("fact_mismatch", best)
        return result("unsupported", best)

    return check_sentence
