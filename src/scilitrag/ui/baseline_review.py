"""Review an existing private reference set without changing its source passages."""

import copy
import hashlib
import json

import streamlit as st

from scilitrag.evaluation.datasets import ReferenceSet

OPTIONS = ("Uncertain", "Yes", "No")
VALUES = {"Uncertain": None, "Yes": True, "No": False}


def reviewed_reference(source: dict, decisions: list[tuple[str, str]]) -> dict:
    """Only labels and notes may change; retain and validate the source fingerprints."""
    ReferenceSet.model_validate(source)
    result = copy.deepcopy(source)
    annotations = result.get("baseline_annotations", [])
    if len(annotations) != len(decisions):
        raise ValueError("A review decision is required for each passage")
    for item, (label, note) in zip(annotations, decisions, strict=True):
        if label not in VALUES:
            raise ValueError("Unknown review label")
        item["baseline_use"] = VALUES[label]
        item["reviewer_note"] = note
    ReferenceSet.model_validate(result)
    return result


def render_baseline_review() -> None:
    with st.expander("Review baseline passages"):
        st.caption(
            "Upload a reference set exported from your database. Review whether each passage "
            "shows the target used as an experimental baseline; a mention alone is not enough. "
            "The file and its passages may contain private dataset text."
        )
        uploaded = st.file_uploader("Reference set (JSON)", type="json", key="baseline_upload")
        if not uploaded:
            return
        raw = uploaded.getvalue()
        try:
            source = json.loads(raw.decode("utf-8-sig"))
            ReferenceSet.model_validate(source)
        except (UnicodeError, ValueError, TypeError) as exc:
            st.error(f"Invalid reference set: {exc}")
            return
        annotations = source.get("baseline_annotations", [])
        if not annotations:
            st.info("This file has no baseline passages to review.")
            return
        file_key = hashlib.sha256(raw).hexdigest()[:16]
        decisions = []
        for index, item in enumerate(annotations):
            with st.container(border=True):
                st.markdown(f"**Passage {index + 1}: citing paper {item['corpus_id']}**")
                st.write(f"Target: {item.get('target_title') or item['target_id']}")
                st.caption(f"Section: {item.get('section') or 'not specified'}")
                st.text(item["text"])
                current = {None: "Uncertain", True: "Yes", False: "No"}[item.get("baseline_use")]
                label = st.radio(
                    "Does this passage support experimental baseline use?",
                    OPTIONS,
                    index=OPTIONS.index(current),
                    key=f"baseline-{file_key}-{index}",
                    horizontal=True,
                )
                note = st.text_input(
                    "Review note",
                    value=item.get("reviewer_note", ""),
                    key=f"baseline-note-{file_key}-{index}",
                )
                decisions.append((label, note))
        reviewed = reviewed_reference(source, decisions)
        complete = sum(label != "Uncertain" for label, _ in decisions)
        st.caption(f"{complete} of {len(decisions)} passages labeled. Uncertain remains null.")
        st.download_button(
            "Download reviewed reference",
            data=json.dumps(reviewed, ensure_ascii=False, indent=2) + "\n",
            file_name="reviewed-reference.json",
            mime="application/json",
            key=f"baseline-download-{file_key}",
        )
