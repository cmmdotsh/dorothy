"""Tests for the markdown-aware article chunker."""

from src.claim_graph.chunker import (
    _drop_boilerplate,
    _drop_leading_title,
    _is_boilerplate_line,
    chunk_article,
    strip_markdown,
)


def test_strip_markdown_removes_headers():
    assert strip_markdown("## Section Title") == "Section Title"
    assert strip_markdown("### Where does Mami go from here?") == "Where does Mami go from here?"
    assert strip_markdown("# H1\n## H2\n### H3") == "H1\nH2\nH3"


def test_strip_markdown_removes_inline_header_runs():
    # NBC/ABC index pages emit timestamp directly fused with header marker
    out = strip_markdown("01:30## Deadly shooting at Mexico tourist site")
    assert "##" not in out
    assert "Deadly shooting at Mexico tourist site" in out
    out2 = strip_markdown("Apr 25, 2026### NBA 1st round playoff matchups")
    assert "###" not in out2
    assert "NBA 1st round playoff matchups" in out2


def test_strip_markdown_removes_bold_and_italic():
    assert strip_markdown("**bold**") == "bold"
    assert strip_markdown("***bold-italic***") == "bold-italic"
    assert strip_markdown("*italic*") == "italic"
    assert strip_markdown("_italic_") == "italic"
    # trailing space inside bold (from extracted articles)
    assert strip_markdown("**Bron Breakker **") == "Bron Breakker "


def test_strip_markdown_handles_inline_emphasis():
    out = strip_markdown(
        "Heyman brought out **Bron Breakker** for his first appearance."
    )
    assert "**" not in out
    assert "Bron Breakker" in out


def test_strip_markdown_keeps_underscores_inside_words():
    # snake_case identifiers must not be mangled
    assert strip_markdown("variable_name") == "variable_name"


def test_strip_markdown_removes_links_keep_text():
    assert strip_markdown("See [the report](https://example.com).") == "See the report."
    assert strip_markdown("![alt text](https://img.example/x.png)") == "alt text"


def test_strip_markdown_removes_list_markers():
    text = "- item one\n- item two\n* item three\n1. ordered one"
    out = strip_markdown(text)
    assert "item one" in out
    assert "item two" in out
    assert "item three" in out
    assert "ordered one" in out
    assert "- " not in out
    assert "* " not in out


def test_strip_markdown_removes_blockquotes_and_hrules():
    assert strip_markdown("> quoted line").strip() == "quoted line"
    assert strip_markdown("---").strip() == ""
    assert strip_markdown("***").strip() == ""


def test_chunk_article_strips_markdown_before_splitting():
    body = (
        "## WrestleMania 42 Results\n\n"
        "**Saturday**\n\n"
        "Roman Reigns defeated CM Punk in a hard-fought main event "
        "that lasted nearly thirty minutes and featured multiple "
        "false finishes that drew the crowd into the action.\n\n"
        "### Where does Mami go from here?\n\n"
        "Rhea Ripley defeats Jade Cargill with the help of Iyo Sky "
        "to win the WWE Women's Championship at Night 2 of WrestleMania 42, "
        "setting up a likely feud with Iyo over the coming months."
    )
    chunks = chunk_article({
        "id": "a1",
        "source_name": "Test",
        "source_slug": "test",
        "source_bias": "center",
        "column": "sports",
        "body": body,
    })
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "##" not in chunk.text
        assert "**" not in chunk.text
        assert not chunk.text.lstrip().startswith("#")


def test_is_boilerplate_line_matches_known_patterns():
    assert _is_boilerplate_line("Sign up here.")
    assert _is_boilerplate_line("Sign up")
    assert _is_boilerplate_line("Subscribe")
    assert _is_boilerplate_line("Subscribe now")
    assert _is_boilerplate_line("Read more: Some Headline About Anything")
    assert _is_boilerplate_line("READ MORE: A Loud Headline")
    assert _is_boilerplate_line("Also Read: Foo Bar")
    assert _is_boilerplate_line("Reporting by Mali Newsroom; Writing by Bate Felix")
    assert _is_boilerplate_line("Editing by Alexander Smith")
    assert _is_boilerplate_line("Our Standards: The Thomson Reuters Trust Principles.")
    assert _is_boilerplate_line("Available to UK users only.")
    assert _is_boilerplate_line("More from Sports")
    assert _is_boilerplate_line("Follow us on Twitter for updates")
    assert _is_boilerplate_line("Continue reading below")
    assert _is_boilerplate_line("Tags: Politics, Election, 2026")
    assert _is_boilerplate_line("© 2026 Reuters")
    assert _is_boilerplate_line("Copyright 2026 Associated Press")


def test_is_boilerplate_line_does_not_match_real_prose():
    assert not _is_boilerplate_line("The president signed the bill into law on Tuesday.")
    assert not _is_boilerplate_line("Sources told reporters they expected a vote.")
    assert not _is_boilerplate_line("She continued reading the document late into the night.")
    # Make sure "for more than" prose isn't mistaken for a "For more..." footer
    assert not _is_boilerplate_line("She has worked there for more than ten years.")


def test_is_boilerplate_line_matches_extended_patterns():
    # Trailing teasers / footers
    assert _is_boilerplate_line("For more NFL news, visit Newsweek Sports.")
    assert _is_boilerplate_line("For more about the Los Angeles Rams and the NFL, visit Newsweek Sports.")
    assert _is_boilerplate_line("For more on the 49ers and the NFL, head to Newsweek Sports.")
    # Tweet attribution lines
    assert _is_boilerplate_line("— Mario Nawfal (@MarioNawfal) April 26, 2026")
    assert _is_boilerplate_line("— TheBlaze (@theblaze) April 26, 2026")
    assert _is_boilerplate_line("— ACLU (@aclu.org) April 23, 2026 at 7:12 AM")
    # Commerce / affiliate
    assert _is_boilerplate_line("If you buy something through a link in this article, we may earn commission.")
    # Subscribe footers
    assert _is_boilerplate_line("Subscribe for free here.")
    # USA Today email+handle byline
    assert _is_boilerplate_line("Reach her at nalund@usatoday.com and follow her on X @nataliealund.")
    # Contributing reporter credit
    assert _is_boilerplate_line("Contributing: Bill Poehler, Salem Statesman Journal")
    assert _is_boilerplate_line("Contributing from London: Kim Hjelmgaard")


def test_drop_boilerplate_removes_leading_signup():
    p = "Sign up here.\n\nIn a televised statement, Charles said his mother had shaped the world."
    out = _drop_boilerplate(p)
    assert "Sign up" not in out
    assert "Charles said" in out


def test_drop_boilerplate_removes_trailing_byline():
    p = (
        "Islamic militants attacked locations across Mali on Saturday.\n"
        "Reporting by Mali Newsroom; Writing by Bate Felix\n"
        "Our Standards: The Thomson Reuters Trust Principles."
    )
    out = _drop_boilerplate(p)
    assert "Islamic militants" in out
    assert "Reporting by" not in out
    assert "Our Standards" not in out


def test_drop_leading_title_strips_h1_from_body_start():
    body = "# Asos demands £7m from US as firms rush to claim tariff refunds\n\nOnline fashion retailer Asos is seeking refunds."
    out = _drop_leading_title(body)
    assert out.startswith("Online fashion retailer")


def test_drop_leading_title_no_op_without_h1():
    body = "Online fashion retailer Asos is seeking refunds."
    assert _drop_leading_title(body) == body


def test_drop_leading_title_only_strips_h1_not_h2():
    # H2 might be a real subsection header — leave it alone (our regex requires
    # exactly one '#' followed by a space at the start of the body)
    body = "## Section Header\n\nContent follows."
    assert _drop_leading_title(body) == body


def test_chunk_article_does_not_leak_article_title():
    body = (
        "# Judge sides with Arizona election official in ruling that has implications for midterms voting\n\n"
        "PHOENIX (AP) — The top election official in Arizona's most populous "
        "county will get more authority in running elections after a judge "
        "sided with his office in a prolonged legal fight with the local "
        "board that shares responsibility for overseeing the vote."
    )
    chunks = chunk_article({
        "id": "a1",
        "source_name": "AP",
        "source_slug": "ap",
        "source_bias": "center",
        "column": "politics",
        "headline": "Judge sides with Arizona election official in ruling that has implications for midterms voting",
        "body": body,
    })
    assert len(chunks) >= 1
    combined = "\n".join(c.text for c in chunks)
    assert "Judge sides with Arizona election official" not in combined
    assert "PHOENIX (AP)" in combined


def test_chunk_article_drops_boilerplate_only_paragraphs():
    body = (
        "Read more: Cowboys Rumors Shed Light on NFL Draft Trade Possibilities\n\n"
        "The Rams selected a new quarterback in the first round of the 2026 NFL "
        "Draft, capping a long offseason of speculation and creating a new "
        "succession plan for the position.\n\n"
        "For more about the Los Angeles Rams and the NFL, visit Newsweek Sports."
    )
    chunks = chunk_article({
        "id": "a1",
        "source_name": "Newsweek",
        "source_slug": "newsweek",
        "source_bias": "center",
        "column": "sports",
        "body": body,
    })
    combined = "\n".join(c.text for c in chunks)
    assert "Cowboys Rumors" not in combined
    assert "For more about" not in combined
    assert "Rams selected" in combined


def test_chunk_article_falls_back_to_headline_summary():
    chunks = chunk_article({
        "id": "a1",
        "source_name": "Test",
        "source_slug": "test",
        "source_bias": "center",
        "column": "politics",
        "headline": "Headline",
        "summary": "Summary text.",
    })
    assert len(chunks) == 1
    assert "Headline" in chunks[0].text
