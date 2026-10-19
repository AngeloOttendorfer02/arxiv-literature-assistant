from scilitrag.contracts import Citation, EvidenceMapping, RenderedSentence
from scilitrag.rag.citation_labels import citation_labels


def render_citations(sentence: str, mapping: EvidenceMapping) -> RenderedSentence:
    if any(key != value.label or key < 1 for key, value in mapping.items()):
        raise ValueError("Invalid request evidence mapping.")
    citations = []
    for label in citation_labels(sentence):
        if label not in mapping:
            raise ValueError(f"Unknown citation label [{label}].")
        evidence = mapping[label]
        citations.append(
            Citation(
                label=label,
                corpus_id=evidence.corpus_id,
                chunk_id=evidence.chunk_id,
                author_ids=list(evidence.author_ids),
                title=evidence.title,
                year=evidence.year,
                query_id=evidence.query_id,
                section=evidence.section,
                text=evidence.text,
            )
        )
    return RenderedSentence(text=sentence, citations=citations)
