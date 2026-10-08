"""Unit tests for the graph compiler (no model needed)."""


def test_subject_match_and_walk():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from editbench.methods_graph import build_store, compile

    recs = [
        {"subject": "Leonardo DiCaprio", "relation": "COUNTRY_OF_CITIZENSHIP",
         "new_object": "Syria", "edit_id": "a"},
        {"subject": "Syria", "relation": "CURRENCY",
         "new_object": "Syrian pound", "edit_id": "b"},
    ]
    store = build_store(recs)
    triples, covered = compile(store, "What is the currency of the country of citizenship of Leonardo DiCaprio?")
    assert covered
    assert ("Syria", "CURRENCY", "Syrian pound") in triples
    _, covered2 = compile(store, "Who painted the Mona Lisa?")
    assert not covered2
