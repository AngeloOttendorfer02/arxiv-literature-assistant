"""D-08 author extraction and loading checks with synthetic rows, never real LitSearch."""

import json
from unittest.mock import MagicMock

import pytest
from author_names import extract_authors, is_excluded
from load_fixture import AUTHOR_COPY, PAPER_AUTHOR_COPY, import_authors

from scilitrag.rag.search.entities import normalize_name


def paper(corpus_id, blocks, numeric_strings=False):
    """Build an S2ORC-like row; each block is (first name parts, last name parts, trailing text)."""
    text, spans = "", {"author": [], "authorfirstname": [], "authorlastname": []}

    def add(value, field):
        nonlocal text
        start = len(text)
        text += value
        bounds = (str(start), str(len(text))) if numeric_strings else (start, len(text))
        spans[field].append({"start": bounds[0], "end": bounds[1]})

    for firsts, lasts, trailing in blocks:
        block_start = len(text)
        for part in firsts:
            add(part, "authorfirstname")
            text += " "
        for part in lasts:
            add(part, "authorlastname")
            text += " "
        text += trailing + "\n"
        spans["author"].append({"start": block_start, "end": len(text)})
    annotations = {field: json.dumps(entries) for field, entries in spans.items()}
    return {"corpusid": corpus_id, "content": {"text": text, "annotations": annotations}}


def test_names_are_paired_per_block_filtered_and_deduplicated():
    rows = [
        paper(
            1,
            [
                (["Athul", "Paul"], ["Jacob"], "ap.jacob@example.org University X"),
                (["Google"], ["Brain"], ""),
                (["Che"], [], ""),
                ([], ["Tong"], ""),
                (["Jane"], ["Doe"], ""),
                (["Jane"], ["Doe"], "second block of the same person"),
                (["* -Amir"], ["Abura&apos;ed,"], ""),
            ],
        ),
    ]
    authors, links, counts = extract_authors(rows)
    assert authors == [
        ("Amir", "Abura'ed", "Amir Abura'ed", "amir abura ed"),
        ("Athul Paul", "Jacob", "Athul Paul Jacob", "athul paul jacob"),
        ("Jane", "Doe", "Jane Doe", "jane doe"),
    ]
    assert links == [("amir abura ed", 1), ("athul paul jacob", 1), ("jane doe", 1)]
    assert normalize_name("Amir Abura'ed") == authors[0][3]
    assert counts == {
        "author_blocks": 7,
        "affiliation_or_header": 1,
        "missing_first_or_last": 2,
        "duplicate_in_paper": 1,
        "authors": 3,
        "paper_authors": 3,
    }


def test_same_normalized_name_is_one_author_with_most_frequent_spelling():
    rows = [
        paper(1, [(["Noah", "A."], ["Smith"], "")]),
        paper(2, [(["Noah", "A."], ["Smith"], "")], numeric_strings=True),
        paper(3, [(["NOAH", "A"], ["SMITH"], "")]),
    ]
    authors, links, _ = extract_authors(rows)
    assert authors == [("Noah A.", "Smith", "Noah A. Smith", "noah a smith")]
    assert links == [("noah a smith", 1), ("noah a smith", 2), ("noah a smith", 3)]
    assert normalize_name("Noah A. Smith") == authors[0][3]


def test_exclusion_list_keeps_real_people():
    excluded = (
        "google research",
        "meta ai",
        "alexa ai",
        "bruno kessler",
        "eth zürich",
        "a i bytedance",
        "a a i i language language",
        "國立台北科技大學電子工程系 kung",
    )
    for key in excluded:
        assert is_excluded(key)
    kept = (
        "ai ti aw",
        "alexa y pan",
        "dan roth",
        "r sangal",
        "antal van den bosch",
        "蕭博元 po yuan",
    )
    for key in kept:
        assert not is_excluded(key)


def test_invalid_span_is_rejected():
    row = paper(1, [(["Jane"], ["Doe"], "")])
    row["content"]["annotations"]["authorlastname"] = json.dumps([{"start": 3, "end": 999}])
    with pytest.raises(ValueError, match="authorlastname"):
        extract_authors([row])


def test_rows_without_annotations_have_no_authors():
    authors, links, counts = extract_authors([{"corpusid": 1, "content": {"text": "Body"}}])
    assert (authors, links, counts["authors"]) == ([], [], 0)


def test_paper_links_use_database_ids_in_sorted_insert_order():
    connection, cursor = MagicMock(), MagicMock()
    connection.execute.return_value.fetchall.return_value = [(1, "jane doe"), (2, "noah a smith")]
    authors = [
        ("Jane", "Doe", "Jane Doe", "jane doe"),
        ("Noah A.", "Smith", "Noah A. Smith", "noah a smith"),
    ]
    import_authors(connection, cursor, authors, [("jane doe", 7), ("noah a smith", 7)])
    copies = [call.args[0] for call in cursor.copy.call_args_list]
    assert copies == [AUTHOR_COPY, PAPER_AUTHOR_COPY]
    rows = [
        call.args[0]
        for call in cursor.copy.return_value.__enter__.return_value.write_row.call_args_list
    ]
    assert rows[-2:] == [(1, 7), (2, 7)]


def test_reused_identity_sequence_is_refused():
    connection, cursor = MagicMock(), MagicMock()
    connection.execute.return_value.fetchall.return_value = [(5, "jane doe")]
    with pytest.raises(ValueError, match="RESTART IDENTITY"):
        import_authors(connection, cursor, [("Jane", "Doe", "Jane Doe", "jane doe")], [])
