"""Wire copies are one source, not corroboration."""
from src.claim_graph.graph_builder import count_independent_sources
from src.claim_graph.models import Chunk


def _chunk(article_id, text, source="src"):
    return Chunk(article_id=article_id, source_name=source, source_slug=source,
                 source_bias="center", source_region="", source_perspective="",
                 column="politics", chunk_index=0, text=text)


WIRE = ('Trump met with the reclusive North Korean leader three times during his '
        'first term to discuss the country\u2019s nuclear program, most recently in 2019.')
WIRE_GLYPHS = WIRE.replace('\u2019', "'")
OWN_WORDING = ('The president has previously held three summits with Kim Jong Un, '
               'the last of them in 2019, focused on denuclearization talks.')


def test_verbatim_wire_copy_is_one_source():
    assert count_independent_sources([_chunk("a1", WIRE, "ap"),
                                      _chunk("a2", WIRE_GLYPHS, "fox")]) == 1


def test_distinct_wordings_are_independent():
    assert count_independent_sources([_chunk("a1", WIRE, "ap"),
                                      _chunk("a2", OWN_WORDING, "bbc")]) == 2


def test_wire_pair_plus_independent_is_two():
    chunks = [_chunk("a1", WIRE, "ap"), _chunk("a2", WIRE_GLYPHS, "fox"),
              _chunk("a3", OWN_WORDING, "bbc")]
    assert count_independent_sources(chunks) == 2


def test_two_wordings_one_article_capped_by_articles():
    assert count_independent_sources([_chunk("a1", WIRE, "ap"),
                                      _chunk("a1", OWN_WORDING, "ap")]) == 1
