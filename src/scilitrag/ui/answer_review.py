"""Explicit, source-bound human review of answers shown in the chat."""

import json

import streamlit as st

OPTIONS = ("Not reviewed", "Correct", "Incorrect")
VALUES = {"Not reviewed": None, "Correct": True, "Incorrect": False}


def summarize(values: list[bool | None]) -> dict:
    reviewed = [value for value in values if value is not None]
    return {
        "correct": sum(reviewed),
        "reviewed": len(reviewed),
        "total": len(values),
        "rate": sum(reviewed) / len(reviewed) if reviewed else None,
    }


def render_answer_review(answer, index: int) -> None:
    if not answer.done or answer.done.refused or answer.error or not answer.sentences:
        return
    with st.expander("Review answer quality"):
        st.caption(
            "Check each citation against its source. Review years and counts only where they "
            "appear; compare counts against independently checked SQL results. Unreviewed "
            "items are excluded from rates."
        )
        citations, years, counts, records = [], [], [], []
        for number, sentence in enumerate(answer.sentences):
            with st.container(border=True):
                st.write(sentence.text)
                checks = []
                for citation in sentence.citations:
                    st.caption(f"[{citation.label}] {citation.title or 'Untitled'}")
                    if citation.text:
                        st.text(citation.text)
                    selected = st.selectbox(
                        "Does this source support the claim?",
                        OPTIONS,
                        key=f"review-{index}-{number}-citation-{citation.label}",
                    )
                    value = VALUES[selected]
                    citations.append(value)
                    checks.append({"source": citation.model_dump(), "supported": value})
                year = VALUES[st.selectbox(
                    "Are all years in this sentence correct?", OPTIONS,
                    key=f"review-{index}-{number}-year",
                )]
                count = VALUES[st.selectbox(
                    "Are all counts in this sentence correct?", OPTIONS,
                    key=f"review-{index}-{number}-count",
                )]
                years.append(year)
                counts.append(count)
                records.append({"sentence": sentence.text, "citations": checks,
                                "year_exact": year, "count_exact": count})
        results = {
            "citation_precision": summarize(citations),
            "year_exact": summarize(years),
            "count_exact": summarize(counts),
        }
        st.json(results)
        st.download_button(
            "Download answer review",
            data=json.dumps({"request_id": answer.request_id, "sentences": records,
                             "metrics": results}, ensure_ascii=False, indent=2) + "\n",
            file_name=f"answer-review-{index}.json",
            mime="application/json",
            key=f"review-download-{index}",
        )
