import logging

import pytest
from pydantic import BaseModel, ConfigDict

from scilitrag.config import load_settings
from scilitrag.rag.llm_client import LLMClientError
from scilitrag.rag.structured import complete_json, generation_metadata


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accepted: bool
    count: int


class Client:
    def __init__(self, raw):
        self.raw = raw

    def complete(self, messages, *, schema):
        assert schema["additionalProperties"] is False
        return self.raw


def test_structured_result_and_safe_diagnostics(caplog):
    with caplog.at_level(logging.INFO):
        value = complete_json(Client('{"accepted":true,"count":2}'), [], Decision)
    assert value.count == 2 and value.accepted is True
    assert "schema=Decision valid=True" in caplog.text
    assert '"accepted"' not in caplog.text


@pytest.mark.parametrize(
    "raw", ["no JSON", '{"accepted":"true","count":2}', '{"accepted":true,"count":2,"other":1}']
)
def test_strict_schema_is_enforced(raw):
    with pytest.raises(LLMClientError) as error:
        complete_json(Client(raw), [], Decision)
    assert error.value.kind == "structured"


def test_metadata_contains_only_reproducible_public_settings():
    settings = load_settings(
        {"USE_MOCKS": "1", "LLM_MODEL": "candidate", "LLM_BASE_URL": "http://private-host/v1"}
    )
    metadata = generation_metadata(settings)
    assert metadata["model"] == "candidate" and metadata["max_output_tokens"] == 1024
    assert "private-host" not in str(metadata)
