"""Converter tests on tiny inline fixtures (no data downloads needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from editbench.datasets_common import match  # noqa: E402
from editbench.datasets_rippleedits import LOCALITY, RIPPLE  # noqa: E402


def test_match_uses_aliases():
    assert match("Irmelin DiCaprio", [{"value": "Irmelin DiCaprio", "aliases": ["Irmelin Indenbirken"]}])
    assert match("born Irmelin Indenbirken", [{"value": "Irmelin DiCaprio", "aliases": ["Irmelin Indenbirken"]}])
    assert not match("Leonardo DiCaprio", [{"value": "Irmelin DiCaprio", "aliases": []}])


def test_criteria_split():
    assert RIPPLE == {"Compositionality_I", "Compositionality_II", "Logical_Generalization"}
    assert LOCALITY == {"Relation_Specifity", "Subject_Aliasing", "Forgetfulness"}
