from scilitrag.contracts import CONTRACT_VERSION, Citation, Evidence


def test_additive_author_ids_round_trip():
    e = Evidence(label=1, type="paper", text="abstract", author_ids=[7, 8])
    assert Evidence.model_validate_json(e.model_dump_json()).author_ids == [7, 8]
    assert Evidence(label=2, type="paper", text="old payload").author_ids == []
    assert Citation(label=1).author_ids == []
    assert CONTRACT_VERSION == "0.4"
