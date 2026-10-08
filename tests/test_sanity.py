"""Scaffold sanity: the common record schema every dataset loader must produce."""


def test_schema_keys():
    required = {"edit_id", "subject", "relation", "old_object", "new_object",
                "paraphrases", "ripple_questions", "locality_questions", "source"}
    assert len(required) == 9
