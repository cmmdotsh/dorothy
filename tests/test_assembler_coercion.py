"""Regression: 1B-class models return ordering entries in many shapes."""
from src.synthesis.assembler import _entry_index, _entry_transition, assemble_article

GRAPH = {
    "corroborated": [
        {"sources": [{"text": "Fact zero text here.", "source_name": "AP"}]},
        {"sources": [{"text": "Fact one text here.", "source_name": "BBC"}]},
    ],
    "unique_details": [],
}


def test_entry_index_shapes():
    assert _entry_index({"cluster": 1}) == 1
    assert _entry_index({"fact": "Fact 1"}) == 1
    assert _entry_index({"index": 0}) == 0
    assert _entry_index("Fact 1") == 1
    assert _entry_index("0") == 0
    assert _entry_index(1) == 1
    assert _entry_index(1.0) == 1
    assert _entry_index("no digits") == -1
    assert _entry_index(None) == -1
    assert _entry_index(True) == -1
    assert _entry_index({"cluster": "Fact 0"}) == 0


def test_entry_transition_shapes():
    assert _entry_transition({"transition": "Meanwhile,"}) == "Meanwhile,"
    assert _entry_transition({"transition": 3}) == ""
    assert _entry_transition("Fact 1") == ""


def test_assemble_article_with_string_entries():
    ordering = {"headline": "H", "ordering": ["Fact 0", {"cluster": 1}]}
    article = assemble_article(GRAPH, ordering)
    assert "Fact zero text here." in article
    assert "Fact one text here." in article


def test_assemble_article_out_of_range_skipped():
    ordering = {"headline": "H", "ordering": [{"cluster": 99}, "Fact 0"]}
    article = assemble_article(GRAPH, ordering)
    assert "Fact zero text here." in article
