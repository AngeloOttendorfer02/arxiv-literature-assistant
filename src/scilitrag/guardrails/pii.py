from functools import lru_cache

import spacy
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    IpRecognizer,
    PhoneRecognizer,
)

# PERSON, LOCATION and similar are left out on purpose: author names are search terms.
PLACEHOLDERS = {
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<PHONE>",
    "CREDIT_CARD": "<CARD>",
    "IBAN_CODE": "<IBAN>",
    "IP_ADDRESS": "<IP>",
}
LANGUAGE = "en"
MIN_SCORE = 0.4


class _BlankSpacyEngine(SpacyNlpEngine):
    """Tokenizer only: the enabled recognizers are pattern based and need no model."""

    def __init__(self) -> None:
        super().__init__(models=[{"lang_code": LANGUAGE, "model_name": "blank"}])

    def load(self) -> None:
        self.nlp = {LANGUAGE: spacy.blank(LANGUAGE)}


@lru_cache(maxsize=1)
def _analyzer() -> AnalyzerEngine:
    engine = _BlankSpacyEngine()
    engine.load()
    registry = RecognizerRegistry(supported_languages=[LANGUAGE])
    for recognizer in (
        EmailRecognizer,
        PhoneRecognizer,
        CreditCardRecognizer,
        IbanRecognizer,
        IpRecognizer,
    ):
        registry.add_recognizer(recognizer(supported_language=LANGUAGE))
    return AnalyzerEngine(registry=registry, nlp_engine=engine, supported_languages=[LANGUAGE])


def mask_pii(text: str) -> str:
    if not text:
        return text
    results = _analyzer().analyze(
        text=text, language=LANGUAGE, entities=list(PLACEHOLDERS), score_threshold=MIN_SCORE
    )
    # Replace from the end so earlier offsets stay valid; skip spans nested in a kept one.
    masked, floor = text, len(text) + 1
    for hit in sorted(results, key=lambda r: (r.start, -(r.end - r.start)), reverse=True):
        if hit.end > floor:
            continue
        masked = masked[: hit.start] + PLACEHOLDERS[hit.entity_type] + masked[hit.end :]
        floor = hit.start
    return masked
