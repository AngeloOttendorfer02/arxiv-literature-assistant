"""D-13 chunker checks with a whitespace tokenizer, never real LitSearch or model downloads."""

import re

import pytest
from chunker import chunk_sections


def words(text):
    return [match.span() for match in re.finditer(r"\S+", text)]


def section(count, prefix="w"):
    return " ".join(f"{prefix}{index}" for index in range(count))


def test_long_section_uses_fixed_windows_with_overlap():
    chunks = chunk_sections(7, [("1 Intro", section(600))], words, size=256, overlap=20)
    starts = [int(text.split()[0][1:]) for *_, text in chunks]
    assert starts == [0, 205, 410]
    assert [len(text.split()) for *_, text in chunks] == [256, 256, 190]
    assert [chunk[:4] for chunk in chunks] == [
        ("7-0", 7, "1 Intro", 0),
        ("7-1", 7, "1 Intro", 1),
        ("7-2", 7, "1 Intro", 2),
    ]


def test_chunks_never_cross_sections_and_seq_no_runs_per_paper():
    sections = [("1 Intro", section(10, "a")), (None, section(300, "b")), ("2 Method", "")]
    chunks = chunk_sections(3, sections, words, size=256, overlap=0)
    assert [(chunk[2], chunk[3]) for chunk in chunks] == [("1 Intro", 0), (None, 1), (None, 2)]
    assert chunks[0][4] == section(10, "a")
    assert all(not ("a" in text and "b" in text) for *_, text in chunks)
    assert len({chunk[0] for chunk in chunks}) == 3


@pytest.mark.parametrize("size", [256, 384, 512])
@pytest.mark.parametrize("overlap", [0, 10, 20])
def test_no_chunk_exceeds_the_window_and_all_text_is_covered(size, overlap):
    text = section(1300)
    chunks = chunk_sections(1, [("1 A", text)], words, size=size, overlap=overlap)
    assert max(len(body.split()) for *_, body in chunks) <= size
    covered = {token for *_, body in chunks for token in body.split()}
    assert covered == set(text.split())


def test_only_o5_sizes_and_overlaps_are_allowed():
    with pytest.raises(ValueError, match="Chunk size"):
        chunk_sections(1, [("1 A", "x")], words, size=500)
    with pytest.raises(ValueError, match="Chunk size"):
        chunk_sections(1, [("1 A", "x")], words, overlap=15)


def multi_token_symbols(text):
    """Like a subword tokenizer, a rare symbol becomes two tokens with the same span."""
    spans = []
    for start, end in words(text):
        spans.extend([(start, end)] * (2 if text[start:end] == "§" else 1))
    return spans


@pytest.mark.parametrize("overlap", [0, 10, 20])
def test_windows_shrink_until_the_chunk_re_encodes_within_the_size(overlap):
    text = " ".join(["w"] * 255 + ["§"] + ["w"] * 300)
    chunks = chunk_sections(1, [("1 A", text)], multi_token_symbols, size=256, overlap=overlap)
    assert all(len(multi_token_symbols(body)) <= 256 for *_, body in chunks)
    assert chunks[0][4].split() == ["w"] * 255
    assert sum(body.count("§") for *_, body in chunks) >= 1


def test_real_tokenizer_keeps_unicode_chunks_within_the_size():
    pytest.importorskip("transformers")
    from chunker import token_offsets

    try:
        offsets = token_offsets("ibm-granite/granite-embedding-english-r2", local_files_only=True)
    except OSError:
        pytest.skip("Pinned granite tokenizer is not cached locally.")
    text = " ".join(["Ünïcödé", "naïve", "数学", "∑∫√", "🙂🚀", "Ωmega", "ﬁne"] * 400)
    for size in (256, 384, 512):
        chunks = chunk_sections(1, [("1 A", text)], offsets, size=size, overlap=10)
        assert max(len(offsets(body)) for *_, body in chunks) <= size


def test_chunk_text_has_no_leading_or_trailing_space():
    def spaced(text):
        # Like byte-level tokenizers, a token span includes the space before the word.
        return [(max(start - 1, 0), end) for start, end in words(text)]

    chunks = chunk_sections(1, [("1 A", section(600))], spaced, size=256, overlap=10)
    assert all(body == body.strip() for *_, body in chunks)
    assert chunks[1][4].startswith("w231")
