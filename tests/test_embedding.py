"""D-14 embedding checks with a fake encoder, never real models or LitSearch."""

import math
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import embed_database
import numpy as np
import pytest
from embed_database import PENDING, embed_rows, publish_model, start_run, vector_text

from scilitrag.config import load_settings
from scilitrag.data import embedding

GRANITE = "ibm-granite/granite-embedding-english-r2"
UNIT = [1 / math.sqrt(768)] * 768


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.ones((len(texts), 768), dtype=np.float32) / np.sqrt(768)


def test_paper_text_matches_the_d22_comparison():
    assert embedding.document_text(" Title ", " Abstract ") == "Title\nAbstract"
    assert embedding.document_text(None, "Abstract") == "Abstract"
    assert embedding.document_text("Title", "") == "Title"


def test_component_returns_normalized_768_vectors(monkeypatch):
    fake = FakeEncoder()
    monkeypatch.setattr(embedding, "load_model", lambda model: fake)
    embed = embedding.build_component(load_settings({"USE_MOCKS": "1", "EMBED_MODEL": GRANITE}))
    vectors = embed(["a query", "another"])
    assert len(vectors) == 2 and all(len(vector) == 768 for vector in vectors)
    assert all(type(value) is float for value in vectors[0])
    assert fake.calls[0][1]["normalize_embeddings"] is True
    assert embed([]) == []


def test_unpinned_model_is_refused():
    with pytest.raises(ValueError, match="pinned revision"):
        embedding.load_model.__wrapped__("someone/unpinned-model")


def test_whole_chunk_fits_after_special_tokens(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model, **kwargs):
            self.kwargs = kwargs
            self.tokenizer = SimpleNamespace(num_special_tokens_to_add=lambda: 2)

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), float16="fp16", float32="fp32"
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    encoder = embedding.load_model.__wrapped__(GRANITE)
    assert encoder.max_seq_length == 512 + 2
    assert encoder.kwargs["revision"] == embedding.MODEL_REVISIONS[GRANITE]
    assert encoder.kwargs["model_kwargs"] == {"dtype": "fp32"}


@pytest.mark.parametrize(
    "vector",
    [
        [0.0] * 768,
        [float("nan")] + UNIT[1:],
        [float("inf")] + UNIT[1:],
        [value * 3 for value in UNIT],
        UNIT[:767],
    ],
)
def test_zero_non_finite_unnormalized_and_short_vectors_are_rejected(vector):
    with pytest.raises(ValueError):
        embedding.check_vectors([vector], 1)


def test_rows_are_embedded_in_batches_and_committed():
    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = [
        (1, "T1", "A1"),
        (2, "T2", None),
        (3, None, "A3"),
    ]
    seen = []

    def fake_encode(model, texts):
        seen.append(texts)
        return [UNIT for _ in texts]

    count = embed_rows(
        connection,
        GRANITE,
        fake_encode,
        embed_database.PAPERS_TODO,
        embed_database.PAPER_UPDATE,
        lambda row: embedding.document_text(row[1], row[2]),
        batch_size=2,
    )
    assert count == 3 and seen == [["T1\nA1", "T2"], ["A3"]]
    cursor = connection.cursor.return_value.__enter__.return_value
    first = cursor.executemany.call_args_list[0].args
    assert first[0] == embed_database.PAPER_UPDATE and first[1][0][1] == 1
    assert first[1][0][0].startswith("[0.03608439,")
    assert connection.commit.call_count == 2


def test_zero_vector_stops_before_writing():
    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = [("1-0", "text")]
    with pytest.raises(ValueError, match="normalized"):
        embed_rows(
            connection,
            GRANITE,
            lambda model, texts: [[0.0] * 768],
            embed_database.CHUNKS_TODO,
            embed_database.CHUNK_UPDATE,
            lambda row: row[1],
            batch_size=32,
        )
    connection.cursor.assert_not_called()


def metadata_connection(*fetches):
    connection = MagicMock()
    connection.execute.return_value.fetchone.side_effect = list(fetches)
    return connection


def test_first_run_records_the_pending_fingerprint():
    connection = metadata_connection((None, None), (0,))
    start_run(connection, GRANITE)
    expected = embedding.fingerprint(GRANITE)
    marker = PENDING + expected
    assert connection.execute.call_args.args == (embed_database.SET_RUN, (marker, expected))
    assert "47ea694b257b" in marker and "content_tokens=512" in marker
    connection.commit.assert_called_once()


def test_resume_needs_the_same_fingerprint_and_known_vectors():
    expected = embedding.fingerprint(GRANITE)
    start_run(metadata_connection((PENDING + expected, expected)), GRANITE)
    with pytest.raises(ValueError, match="unknown origin"):
        start_run(metadata_connection((None, None), (5,)), GRANITE)
    with pytest.raises(ValueError, match="pending:other"):
        start_run(metadata_connection(("pending:other@abc", expected)), GRANITE)


@pytest.mark.parametrize("published", [GRANITE, PENDING + embedding.fingerprint(GRANITE)])
@pytest.mark.parametrize("saved_fingerprint", [None, "different-revision-or-preprocessing"])
def test_existing_vectors_without_matching_provenance_are_refused(published, saved_fingerprint):
    connection = metadata_connection((published, saved_fingerprint))
    with pytest.raises(ValueError, match="fingerprint"):
        start_run(connection, GRANITE)
    assert connection.execute.call_count == 1
    connection.commit.assert_not_called()


@pytest.mark.parametrize("change", ["limit", "revision"])
def test_completed_run_refuses_changed_embedding_settings(monkeypatch, change):
    saved_fingerprint = embedding.fingerprint(GRANITE)
    if change == "limit":
        monkeypatch.setattr(embedding, "CONTENT_TOKENS", 256)
    else:
        monkeypatch.setitem(embedding.MODEL_REVISIONS, GRANITE, "different-revision")
    connection = metadata_connection((GRANITE, saved_fingerprint))
    with pytest.raises(ValueError, match="fingerprint"):
        start_run(connection, GRANITE)
    connection.commit.assert_not_called()


def test_completed_run_can_reuse_vectors_with_matching_provenance():
    expected = embedding.fingerprint(GRANITE)
    connection = metadata_connection((GRANITE, expected))
    start_run(connection, GRANITE)
    assert connection.execute.call_args.args == (
        embed_database.SET_RUN,
        (PENDING + expected, expected),
    )


def test_model_reset_requires_clearing_the_fingerprint_too():
    connection = metadata_connection((None, embedding.fingerprint(GRANITE)))
    with pytest.raises(ValueError, match="fingerprint exists without a model"):
        start_run(connection, GRANITE)
    connection.commit.assert_not_called()


def test_model_is_published_only_after_a_complete_run():
    expected = embedding.fingerprint(GRANITE)
    marker = PENDING + expected
    with pytest.raises(ValueError, match="5 rows"):
        publish_model(metadata_connection((marker, expected), (5,)), GRANITE)
    with pytest.raises(ValueError, match="Unexpected"):
        publish_model(metadata_connection((None, None)), GRANITE)
    connection = metadata_connection((marker, expected), (0,))
    publish_model(connection, GRANITE)
    assert connection.execute.call_args.args == (embed_database.SET_MODEL, (GRANITE,))


@pytest.mark.parametrize("saved_fingerprint", [None, "different-revision-or-preprocessing"])
def test_publication_refuses_missing_or_changed_fingerprint(saved_fingerprint):
    marker = PENDING + embedding.fingerprint(GRANITE)
    connection = metadata_connection((marker, saved_fingerprint))
    with pytest.raises(ValueError, match="Unexpected"):
        publish_model(connection, GRANITE)
    assert connection.execute.call_count == 1
    connection.commit.assert_not_called()


def test_published_fingerprint_survives_restarts_and_rejects_changed_settings(monkeypatch):
    # Exercise the actual metadata SQL across commits; vector math is tested separately.
    with sqlite3.connect(":memory:") as database:
        database.executescript(
            "CREATE TABLE schema_metadata (id int, embed_model text, embed_fingerprint text);"
            "INSERT INTO schema_metadata VALUES (1, NULL, NULL);"
            "CREATE TABLE papers (embedding text);"
            "CREATE TABLE chunks (embedding text);"
        )
        connection = SimpleNamespace(
            execute=lambda statement, params=(): database.execute(
                statement.replace(" FOR UPDATE", "").replace("%s", "?"), params
            ),
            commit=database.commit,
        )
        expected = embedding.fingerprint(GRANITE)
        start_run(connection, GRANITE)
        database.execute("INSERT INTO papers VALUES ('saved-vector')")
        publish_model(connection, GRANITE)
        assert connection.execute(embed_database.METADATA).fetchone() == (GRANITE, expected)

        start_run(connection, GRANITE)
        publish_model(connection, GRANITE)
        assert connection.execute(embed_database.METADATA).fetchone() == (GRANITE, expected)

        monkeypatch.setattr(embedding, "CONTENT_TOKENS", 256)
        with pytest.raises(ValueError, match="fingerprint"):
            start_run(connection, GRANITE)
        assert connection.execute(embed_database.METADATA).fetchone() == (GRANITE, expected)
        assert database.execute("SELECT embedding FROM papers").fetchone() == ("saved-vector",)


def test_vector_literal_is_pgvector_text():
    assert vector_text([0.25, -1.0, 1e-9]) == "[0.25,-1,1e-09]"
