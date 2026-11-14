"""Tests for node source-position observation.

These cover the cases the feature exists for: missing closing tags, implicit
html/head/body, table foster parenting and adoption-agency repairs all need
stable source positions, while nodes produced after parsing must report that
they cannot be located. Positions are byte offsets into the UTF-8 input.
"""

import pytest

from selectolax.lexbor import LexborHTMLParser, NodeSourcePosition


def _all_nodes(parser):
    """Every node in document order, including text and comment nodes."""
    seen = []

    def walk(node):
        while node is not None:
            seen.append(node)
            walk(node.first_child)
            node = node.next

    # Start from the document node so doctype/comments are included.
    root = parser.root
    top = root
    while top.parent is not None:
        top = top.parent
    walk(top)
    return seen


def test_disabled_by_default_returns_none():
    parser = LexborHTMLParser(b"<div><span>x</span></div>")
    assert parser.css_first("div").source_position is None
    assert parser.css_first("span").first_child.source_position is None


def test_position_is_value_object():
    parser = LexborHTMLParser(b"<div></div>", source_positions=True)
    position = parser.css_first("div").source_position
    assert isinstance(position, NodeSourcePosition)
    assert position.origin == "document"
    # A fresh object is returned each time, never a cached C pointer.
    assert parser.css_first("div").source_position == position
    assert parser.css_first("div").source_position is not position


def test_missing_closing_tag_offsets():
    html = b"<div><span>x"
    parser = LexborHTMLParser(html, source_positions=True)

    div = parser.css_first("div")
    span = parser.css_first("span")
    text = span.first_child

    assert html[div.source_position.start:div.source_position.end] == b"<div>"
    assert html[span.source_position.start:span.source_position.end] == b"<span>"
    assert html[text.source_position.start:text.source_position.end] == b"x"

    # Half-open ranges are ordered and point inside the input.
    assert 0 <= div.source_position.start < div.source_position.end <= len(html)


def test_element_span_includes_attributes():
    html = b'<a href="h" id="l">text</a>'
    parser = LexborHTMLParser(html, source_positions=True)
    anchor = parser.css_first("a")
    assert html[anchor.source_position.start:anchor.source_position.end] == (
        b'<a href="h" id="l">'
    )


def test_implicit_html_head_body_get_positions():
    html = b"<title>t</title><p>hi"
    parser = LexborHTMLParser(html, source_positions=True)

    for tag in ("html", "head", "body"):
        node = parser.css_first(tag)
        position = node.source_position
        assert position is not None
        # Synthesized nodes attribute themselves to the real token that forced
        # their creation, so their offsets stay inside the input.
        assert 0 <= position.start <= position.end <= len(html)
        assert position.origin == "document"


def test_doctype_and_comment_spans():
    html = b"<!DOCTYPE html><!-- hi --><p>x</p>"
    parser = LexborHTMLParser(html, source_positions=True)

    nodes = {node.tag: node for node in _all_nodes(parser)}
    doctype = nodes["-doctype"].source_position
    comment = nodes["-comment"].source_position

    assert html[doctype.start:doctype.end] == b"<!DOCTYPE html>"
    assert html[comment.start:comment.end] == b"<!-- hi -->"


def test_table_foster_parenting_keeps_position():
    # A <div> directly in a table row is moved out by foster parenting; its
    # recorded position must still describe its original markup.
    html = b"<table><tr><div>foster</div></tr></table>"
    parser = LexborHTMLParser(html, source_positions=True)

    div = parser.css_first("div")
    position = div.source_position
    assert html[position.start:position.end] == b"<div>"

    text = div.first_child
    text_position = text.source_position
    assert html[text_position.start:text_position.end] == b"foster"


def test_adoption_agency_rebuild_positions():
    html = b'<a href="1"><b><a href="2">inner</a></b></a>'
    parser = LexborHTMLParser(html, source_positions=True)

    anchors = parser.css("a")
    assert len(anchors) >= 2
    for anchor in anchors:
        position = anchor.source_position
        assert position is not None
        span = html[position.start:position.end]
        assert span.startswith(b"<a")


def test_fragment_origin_and_offsets():
    fragment_input = b"<div>a</div><b>b</b>"
    fragment = LexborHTMLParser(
        fragment_input, is_fragment=True, source_positions=True
    )
    assert fragment.root.source_position.origin == "fragment"
    # Fragment offsets index the fragment input from zero.
    assert fragment.root.source_position.start == 0

    document = LexborHTMLParser(fragment_input, source_positions=True)
    assert document.css_first("div").source_position.origin == "document"


def test_raw_html_slice_round_trip():
    html = b"<div><p>Hello</p><p>World</p></div>"
    parser = LexborHTMLParser(html, source_positions=True)
    for node in parser.css("p"):
        position = node.source_position
        assert html[position.start:position.end] == b"<p>"


def test_unlinked_node_keeps_history():
    html = b"<div><span>a</span><b>c</b></div>"
    parser = LexborHTMLParser(html, source_positions=True)
    span = parser.css_first("span")
    before = span.source_position

    span.unwrap()
    # Unlinked, not destroyed: the historical position survives.
    assert span.source_position == before


def test_created_and_cloned_nodes_not_located():
    parser = LexborHTMLParser(b"<div><i>x</i></div>", source_positions=True)

    assert parser.create_node("span").source_position is None

    original = parser.css_first("i")
    assert original.clone().source_position is None


def test_replace_with_keeps_old_locates_not_new():
    parser = LexborHTMLParser(b"<div><i>x</i></div>", source_positions=True)
    italic = parser.css_first("i")
    history = italic.source_position
    replacement = parser.create_node("em")

    italic.replace_with(replacement)

    assert italic.source_position == history
    assert replacement.source_position is None


def test_inserted_nodes_not_located():
    parser = LexborHTMLParser(b"<div></div>", source_positions=True)
    div = parser.css_first("div")
    div.insert_child("plain text")
    div.insert_child(parser.create_node("span"))

    assert div.first_child.source_position is None
    assert div.css_first("span").source_position is None
    # The original div itself remains locatable.
    assert div.source_position is not None


def test_inner_html_synthetic_origin_document():
    parser = LexborHTMLParser(b"<div></div>", source_positions=True)
    div = parser.css_first("div")
    content = b"<p>hello</p>"
    div.inner_html = content.decode()

    paragraph = parser.css_first("p")
    position = paragraph.source_position
    assert position.origin == "synthetic"
    # Offsets index the newly supplied content, not raw_html.
    assert content[position.start:position.end] == b"<p>"
    assert div.source_position.origin == "document"


def test_inner_html_synthetic_origin_fragment():
    parser = LexborHTMLParser(
        b"<div></div>", is_fragment=True, source_positions=True
    )
    parser.root.inner_html = "<b>z</b>"
    position = parser.css_first("b").source_position
    assert position.origin == "synthetic"


def test_origin_does_not_leak_between_parses():
    # A synthetic inner_html parse must not change the origin of an unrelated
    # later fragment parse.
    parser = LexborHTMLParser(b"<div></div>", source_positions=True)
    parser.css_first("div").inner_html = "<p>x</p>"

    other = LexborHTMLParser(b"<i>q</i>", is_fragment=True, source_positions=True)
    assert other.root.source_position.origin == "fragment"


def test_selector_and_serialization_unaffected():
    html = b"<div><span>x</span></div>"
    enabled = LexborHTMLParser(html, source_positions=True)
    disabled = LexborHTMLParser(html)

    assert enabled.css_first("span").html == disabled.css_first("span").html
    assert enabled.html == disabled.html
    assert [n.tag for n in enabled.css("*")] == [n.tag for n in disabled.css("*")]


def test_text_node_offsets():
    html = b"<p>ab</p>cd"
    parser = LexborHTMLParser(html, source_positions=True)

    paragraph_text = parser.css_first("p").first_child.source_position
    assert html[paragraph_text.start:paragraph_text.end] == b"ab"

    # Top-level text after the paragraph is its own text node.
    trailing = parser.css_first("p").next
    trailing_text = trailing.source_position
    assert html[trailing_text.start:trailing_text.end] == b"cd"


def test_multibyte_offsets_are_byte_based():
    html = "<p>héllo</p>".encode("utf-8")
    parser = LexborHTMLParser(html, source_positions=True)
    text = parser.css_first("p").first_child
    position = text.source_position
    # 'é' is two UTF-8 bytes; offsets count bytes, not characters.
    assert html[position.start:position.end] == "héllo".encode("utf-8")
    assert position.end - position.start == len("héllo".encode("utf-8"))
