"""D-13 section-based chunks with a fixed token window and overlap (O5; 256/10 chosen in D-20)."""

from scilitrag.data.embedding import MODEL_REVISIONS as TOKENIZER_REVISIONS

SIZES = (256, 384, 512)
OVERLAPS = (0, 10, 20)


def chunk_sections(corpus_id, sections, offsets, *, size=256, overlap=10):
    """Split (section, text) pairs into (chunk_id, corpus_id, section, seq_no, text) rows.

    offsets(text) returns the character span of every token. Chunks never cross a section
    boundary; seq_no runs across the whole paper.
    """
    if size not in SIZES or overlap not in OVERLAPS:
        raise ValueError(f"Chunk size must be one of {SIZES} and overlap one of {OVERLAPS} %.")
    overlap_tokens = size * overlap // 100
    chunks = []
    for section, text in sections:
        spans = offsets(text)
        start = 0
        while start < len(spans):
            end = min(start + size, len(spans))
            # Token spans can include the preceding space.
            body = text[spans[start][0] : spans[end - 1][1]].strip()
            # A character can map to several tokens, so the cut text may re-encode longer.
            while end - start > 1 and len(offsets(body)) > size:
                end -= 1
                body = text[spans[start][0] : spans[end - 1][1]].strip()
            seq_no = len(chunks)
            chunks.append((f"{corpus_id}-{seq_no}", corpus_id, section, seq_no, body))
            if end >= len(spans):
                break
            start = max(end - overlap_tokens, start + 1)
    return chunks


def token_offsets(model, *, local_files_only=False):
    """Character spans of the embedding model's tokens, without special tokens."""
    from transformers import AutoTokenizer

    if model not in TOKENIZER_REVISIONS:
        raise ValueError(f"No pinned tokenizer revision for {model}.")
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=TOKENIZER_REVISIONS[model], local_files_only=local_files_only
    )

    def offsets(text):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(start, end) for start, end in encoded["offset_mapping"] if end > start]

    return offsets
