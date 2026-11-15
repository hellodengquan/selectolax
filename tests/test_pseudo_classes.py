"""Pseudo-class matching across the three parsing scopes selectolax exposes.

The selectors exercised here are the ones a field-extraction pipeline reaches
for most often:

* structural pseudo-classes -- ``:first-child``, ``:last-child``,
  ``:nth-child()``, ``:nth-last-child()``, ``:only-child`` and the
  ``*-of-type`` family, plus ``:empty``;
* the negation pseudo-class ``:not()`` (simple, compound and chained);
* the lexbor text extension ``:lexbor-contains()`` together with the
  Python-side ``LexborSelector.text_contains()`` filter.

Every selector is evaluated against the same markup in three scopes:

1. a complete HTML document (parser-inserted ``html/head/body`` included);
2. an HTML fragment parsed with ``is_fragment=True``, whose top-level nodes
   live under a wrapper element the parser creates but never exposes;
3. the ``DocumentFragment`` held by a ``<template>`` element.

A second test module (:mod:`tests.test_pseudo_browser`) replays the same
expectations inside a real browser engine to prove that selectolax and the
browser understand these structures identically.
"""

import pytest

from selectolax.lexbor import LexborHTMLParser, SelectolaxError

# Markup reused for every scope. Keeping one canonical structure means the
# three scopes differ only in *where* the nodes live, never in their shape.
MARKUP = """
<section id="articles">
  <!-- a comment between siblings must not change child indices -->
  <article class="post" data-n="a1">
    <h2>Alpha release</h2>
    <p class="summary">First summary</p>
    <p class="summary">Second summary</p>
    <a class="link" href="/a1">read more</a>
  </article>
  <article class="post" data-n="a2">
    <h2>Beta release</h2>
    <p class="summary">Beta summary</p>
    <a class="link" href="/a2">read more</a>
  </article>
  <article class="post sponsored" data-n="a3">
    <h2>Gamma release</h2>
    <p class="summary">Gamma summary</p>
    <span></span>
  </article>
</section>
<section id="cards">
  <div class="card" data-n="c1"><span class="tag">news</span><p>Card one</p></div>
  <div class="card" data-n="c2"><span class="tag">blog</span><p>Card two</p></div>
  <div class="card sponsored" data-n="c3"><span class="tag">ad</span><p>Card three</p></div>
</section>
"""

TEMPLATE_MARKUP = (
    "<template id='tpl'>"
    "<ul class='tpl-list'>"
    "<li data-n='t1'>one</li>"
    "<li data-n='t2'>two</li>"
    "<li data-n='t3'>three</li>"
    "<li data-n='t4'>four<span></span></li>"
    "</ul>"
    "</template>"
)

# (selector, expected data-n values in document order). The expectations are
# browser semantics: indices count element children only, text/comment nodes
# are invisible to ``nth-child`` and friends.
STRUCTURAL_CASES = [
    ("article:first-child", ["a1"]),
    ("article:last-child", ["a3"]),
    ("article:nth-child(1)", ["a1"]),
    ("article:nth-child(2)", ["a2"]),
    ("article:nth-child(2n+1)", ["a1", "a3"]),
    ("article:nth-child(even)", ["a2"]),
    ("article:nth-last-child(1)", ["a3"]),
    ("article:nth-last-child(2)", ["a2"]),
    (".post:first-of-type", ["a1"]),
    (".post:last-of-type", ["a3"]),
    # section/div/p carry no data-n; their expectations are checked separately
    # through their own attributes in test_of_type_pseudo_classes_read_ids.
    ("h2:only-child", []),
    ("p:empty", []),
]

# Compact markup keeps element children element-only, so only-child and empty
# are not disturbed by the inter-element whitespace the pretty MARKUP contains.
ONLY_CHILD_MARKUP = (
    "<div id='compact'>"
    "<div class='card' data-n='c1'><span class='tag'>news</span></div>"
    "<div class='card' data-n='c2'><span class='tag'>blog</span></div>"
    "<div class='card' data-n='c3'><span></span><b>x</b></div>"
    "</div>"
)

NEGATION_CASES = [
    ("article:not(.sponsored)", ["a1", "a2"]),
    ("article:not(:first-child)", ["a2", "a3"]),
    ("article:not(.post)", []),
    (".card:not(.sponsored)", ["c1", "c2"]),
    ("article:not([data-n='a2'])", ["a1", "a3"]),
    # chained negation and negation of a compound selector
    ("div.card:not(.sponsored):not([data-n='c1'])", ["c2"]),
    ("article:not(article.sponsored)", ["a1", "a2"]),
    # a selector list inside :not()
    (".card:not(.sponsored, [data-n='c2'])", ["c1"]),
]


def _data_n(nodes):
    return [node.attributes.get("data-n") for node in nodes]


def _section_ids(nodes):
    return [node.attributes.get("id") for node in nodes]


@pytest.fixture
def document():
    return LexborHTMLParser(MARKUP)


@pytest.fixture
def fragment():
    return LexborHTMLParser(MARKUP, is_fragment=True)


@pytest.fixture
def template_fragment():
    tree = LexborHTMLParser(TEMPLATE_MARKUP)
    return tree.template_fragments()[0]


# ---------------------------------------------------------------------------
# Structural pseudo-classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("selector, expected", STRUCTURAL_CASES)
def test_structural_pseudo_classes_on_a_full_document(document, selector, expected):
    assert _data_n(document.css(selector)) == expected


@pytest.mark.parametrize("selector, expected", STRUCTURAL_CASES)
def test_structural_pseudo_classes_on_a_fragment(fragment, selector, expected):
    # The fragment wrapper is invisible, so the two <section> elements are
    # sibling top-level nodes, exactly as they are sibling children of <main>
    # in a full document.
    assert _data_n(fragment.css(selector)) == expected


def test_nth_child_ignores_comments_and_text_nodes_in_every_scope():
    markup = (
        "<ul><!-- lead comment -->"
        "<li data-n='1'>a</li> whitespace"
        "<li data-n='2'>b</li>"
        "<!-- tail comment --></ul>"
    )
    expected = {
        "li:first-child": ["1"],
        "li:nth-child(1)": ["1"],
        "li:nth-child(2)": ["2"],
        "li:last-child": ["2"],
        "li:nth-last-child(1)": ["2"],
    }
    for scope in (
        LexborHTMLParser(markup),
        LexborHTMLParser(markup, is_fragment=True),
    ):
        for selector, ids in expected.items():
            assert _data_n(scope.css(selector)) == ids

    fragment = LexborHTMLParser(
        "<!-- c --><li data-n='1'>a</li><li data-n='2'>b</li>",
        is_fragment=True,
    )
    assert _data_n(fragment.css("li:first-child")) == ["1"]
    assert _data_n(fragment.css("li:nth-child(2)")) == ["2"]
    # The comment must not be reported as an element either.
    assert [node.tag for node in fragment.css("*")] == ["li", "li"]


def test_of_type_pseudo_classes_count_per_tag_name_across_fragment_siblings():
    tree = LexborHTMLParser(
        "<p data-n='1'>one</p><span>s</span><p data-n='2'>two</p><p data-n='3'>three</p>",
        is_fragment=True,
    )
    assert _data_n(tree.css("p:nth-of-type(2)")) == ["2"]
    assert _data_n(tree.css("p:last-of-type")) == ["3"]
    assert _data_n(tree.css("p:first-of-type")) == ["1"]
    assert tree.css("span:only-of-type") and tree.css("span:only-child") == []
    # <p> is never the last child while <span> follows the first one...
    assert _data_n(tree.css("p:last-child")) == ["3"]
    assert tree.css("span:first-child") == []


def test_of_type_pseudo_classes_read_ids_and_text(document, fragment):
    # These nodes carry identifying attributes other than data-n.
    assert _section_ids(document.css("section:nth-of-type(2)")) == ["cards"]
    assert _section_ids(fragment.css("section:nth-of-type(2)")) == ["cards"]
    assert [node.text() for node in document.css("div:nth-of-type(2) > p")] == [
        "Card two"
    ]
    assert [node.text() for node in fragment.css("div:nth-of-type(2) > p")] == [
        "Card two"
    ]


def test_nth_child_of_selector_filter_matches_browser_semantics(document):
    # CSS4 "nth-child(An+B of S)" filters the candidate set before indexing.
    assert _data_n(document.css("article:nth-child(1 of .sponsored)")) == ["a3"]
    assert document.css("article:nth-child(2 of .sponsored)") == []
    assert _data_n(document.css(".card:nth-child(2 of .card)")) == ["c2"]


@pytest.mark.parametrize("is_fragment", [False, True])
def test_only_child_and_empty_on_compact_markup(is_fragment):
    tree = LexborHTMLParser(ONLY_CHILD_MARKUP, is_fragment=is_fragment)
    # the first two cards hold a single (text-bearing) span child
    assert _data_n(tree.css("div.card:has(> span:only-child)")) == ["c1", "c2"]
    assert [node.tag for node in tree.css("span:only-child")] == ["span", "span"]
    # only the content-free span is :empty; whitespace text also breaks :empty
    assert [node.tag for node in tree.css("span:empty")] == ["span"]
    assert tree.css("b:only-child") == []
    whitespace = LexborHTMLParser("<div><p>   </p><span></span></div>")
    assert [node.tag for node in whitespace.css(":empty")] == ["head", "span"]


def test_only_child_inside_template_content(template_fragment):
    assert _data_n(template_fragment.css("li:only-child")) == []
    # the <ul> is the single top-level node of the template's DocumentFragment
    assert [node.tag for node in template_fragment.css("ul:only-child")] == ["ul"]
    # span is the sole child of the last <li>
    assert [node.tag for node in template_fragment.css("span:only-child")] == ["span"]
    assert [node.tag for node in template_fragment.css("span:empty")] == ["span"]


@pytest.mark.parametrize(
    "selector, expected",
    [
        ("li:first-child", ["t1"]),
        ("li:last-child", ["t4"]),
        ("li:nth-child(2n+1)", ["t1", "t3"]),
        ("li:nth-last-child(2)", ["t3"]),
        ("li:first-of-type", ["t1"]),
        ("li:last-of-type", ["t4"]),
        ("li:nth-of-type(3)", ["t3"]),
    ],
)
def test_structural_pseudo_classes_inside_template_content(
    template_fragment, selector, expected
):
    assert _data_n(template_fragment.css(selector)) == expected


def test_template_content_with_several_top_level_nodes_uses_fragment_indexing():
    tree = LexborHTMLParser(
        "<template><p data-n='1'>one</p><p data-n='2'>two</p><i>x</i></template>"
    )
    fragment = tree.template_fragments()[0]

    assert _data_n(fragment.css("p:first-child")) == ["1"]
    assert fragment.css("p:last-child") == []  # <i> is the last child
    assert _data_n(fragment.css("p:nth-child(2)")) == ["2"]
    assert fragment.css("p:only-child") == []
    assert fragment.css("i:first-child") == []
    assert fragment.css("i:only-of-type") and fragment.css("i:last-child")
    assert [node.tag for node in fragment.css("*:first-child")] == ["p"]
    assert [node.tag for node in fragment.css("*:last-child")] == ["i"]


# ---------------------------------------------------------------------------
# Negation pseudo-class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("selector, expected", NEGATION_CASES)
def test_negation_pseudo_class_on_a_full_document(document, selector, expected):
    assert _data_n(document.css(selector)) == expected


@pytest.mark.parametrize("selector, expected", NEGATION_CASES)
def test_negation_pseudo_class_on_a_fragment(fragment, selector, expected):
    assert _data_n(fragment.css(selector)) == expected


def test_negation_combined_with_structural_pseudo_classes(template_fragment):
    assert _data_n(template_fragment.css("li:not(:first-child):not(:last-child)")) == [
        "t2",
        "t3",
    ]
    assert _data_n(template_fragment.css("li:not([data-n='t2'])")) == [
        "t1",
        "t3",
        "t4",
    ]


def test_nested_not_is_rejected_or_evaluated_consistently():
    tree = LexborHTMLParser("<div><p class='a'>1</p><p>2</p></div>")
    # :not() nesting is supported by the engine and keeps ordinary semantics.
    assert [node.text() for node in tree.css("p:not(:not(.a))")] == ["1"]
    # An empty :not() is not a valid selector.
    with pytest.raises(SelectolaxError, match="Can't parse CSS selector"):
        tree.css(":not()")


# ---------------------------------------------------------------------------
# Text-containing extension pseudo-class
# ---------------------------------------------------------------------------


def test_lexbor_contains_is_case_sensitive_by_default():
    tree = LexborHTMLParser(
        "<div>"
        "<p data-n='1'>AwesOme text</p>"
        "<p data-n='2'>awesome text</p>"
        "<p data-n='3'>nothing here</p>"
        "</div>"
    )
    assert _data_n(tree.css('p:lexbor-contains("awesome")')) == ["2"]
    assert _data_n(tree.css('p:lexbor-contains("AWESOME")')) == []
    assert _data_n(tree.css('p:lexbor-contains("AwesOme")')) == ["1"]


def test_lexbor_contains_case_insensitive_flag():
    tree = LexborHTMLParser(
        "<div><p data-n='1'>AwesOme text</p><p data-n='2'>awesome</p></div>"
    )
    assert _data_n(tree.css('p:lexbor-contains("AWESOME" i)')) == ["1", "2"]


def test_lexbor_contains_matches_substrings_not_only_whole_words():
    tree = LexborHTMLParser(
        "<div><p data-n='1'>hello world</p><p data-n='2'>say helloworld</p></div>"
    )
    assert _data_n(tree.css('p:lexbor-contains("llo")')) == ["1", "2"]
    assert _data_n(tree.css('p:lexbor-contains("hello w")')) == ["1"]
    assert tree.css('p:lexbor-contains("goodbye")') == []


def test_lexbor_contains_only_inspects_direct_text_children():
    # The C engine walks the element's own text children; text nested in
    # descendants does not count. This pins the own-text-only semantics so a
    # future engine change cannot silently switch to deep-text matching.
    tree = LexborHTMLParser(
        "<div>"
        "<p data-n='1'>before <b>bold</b> after</p>"
        "<p data-n='2'>plain bold text</p>"
        "</div>"
    )
    assert _data_n(tree.css('p:lexbor-contains("bold")')) == ["2"]
    assert _data_n(tree.css('p:lexbor-contains("before")')) == ["1"]
    assert _data_n(tree.css('p:lexbor-contains("after")')) == ["1"]


def test_lexbor_contains_inside_every_scope():
    selector = 'li:lexbor-contains("two")'
    doc = LexborHTMLParser("<ul><li>one</li><li>two</li><li>twofold</li></ul>")
    frag = LexborHTMLParser(
        "<li>one</li><li>two</li><li>twofold</li>", is_fragment=True
    )
    tpl = LexborHTMLParser(
        "<template><li>one</li><li>two</li><li>twofold</li></template>"
    ).template_fragments()[0]

    for scope in (doc, frag, tpl):
        assert [node.text() for node in scope.css(selector)] == [
            "two",
            "twofold",
        ]
        assert scope.css_matches(selector) is True
        assert scope.css_matches('li:lexbor-contains("absent")') is False


def test_python_text_contains_complements_the_pseudo_class_with_deep_text():
    tree = LexborHTMLParser(
        "<div>"
        "<p data-n='1'>before <b>bold</b> after</p>"
        "<p data-n='2'>plain text</p>"
        "</div>"
    )
    # The Python filter uses deep text extraction, so it reaches nested text.
    deep = tree.root.select("p").text_contains("bold")
    assert _data_n(deep.matches) == ["1"]
    assert deep.any_matches is True
    assert tree.root.select("p").any_text_contains("bold") is True
    assert tree.root.select("p").any_text_contains("missing") is False


def test_lexbor_contains_respects_scope_and_result_order(fragment):
    nodes = fragment.css('article:lexbor-contains("summary")')
    # "summary" only appears on descendant <p> nodes, never as direct text of
    # <article>, so the extension matches nothing -- and never the wrapper.
    assert nodes == []
    ordered = fragment.css('p:lexbor-contains("summary")')
    assert [node.text() for node in ordered] == [
        "First summary",
        "Second summary",
        "Beta summary",
        "Gamma summary",
    ]


# ---------------------------------------------------------------------------
# Result ordering
# ---------------------------------------------------------------------------


def test_results_follow_tree_order_not_selector_order(document):
    # querySelectorAll returns nodes in tree order even for a selector list.
    expected = ["a1", "a2", "a3", "c1", "c2", "c3"]
    assert _data_n(document.css("div, article")) == expected
    assert _data_n(document.css("article, div")) == expected
    assert _data_n(document.css(".post, .card")) == expected
    # Same guarantee through the selector wrapper.
    assert _data_n(document.root.select("article, div").matches) == expected


def test_fragment_and_template_share_the_document_ordering(fragment, template_fragment):
    assert _section_ids(fragment.css("section")) == ["articles", "cards"]
    assert _data_n(fragment.css(".post, .card")) == [
        "a1",
        "a2",
        "a3",
        "c1",
        "c2",
        "c3",
    ]
    assert _data_n(template_fragment.css("li")) == ["t1", "t2", "t3", "t4"]
    # tree order: every <li> precedes the <span> nested in the last one;
    # span carries no data-n, which itself proves it appears after the <li>s.
    mixed = template_fragment.css("li, span")
    assert _data_n(mixed) == ["t1", "t2", "t3", "t4", None]
    assert [node.tag for node in mixed] == ["li", "li", "li", "li", "span"]


def test_css_first_returns_the_first_tree_order_match(document):
    node = document.css_first("article:nth-child(2n+1)")
    assert node.attributes["data-n"] == "a1"
    assert document.css_first("article.sponsored").attributes["data-n"] == "a3"
    assert document.css_first("table") is None
    assert document.css_first("table", default="missing") == "missing"


def test_results_are_stable_across_repeated_queries(template_fragment):
    first = _data_n(template_fragment.css("li:nth-child(2n+1)"))
    for _ in range(5):
        assert _data_n(template_fragment.css("li:nth-child(2n+1)")) == first


# ---------------------------------------------------------------------------
# The parser-inserted fragment wrapper must never match
# ---------------------------------------------------------------------------


def test_fragment_wrapper_is_invisible_to_structural_pseudo_classes():
    tree = LexborHTMLParser(
        "<article data-n='a1'>1</article><article data-n='a2'>2</article>",
        is_fragment=True,
    )

    # No <html> wrapper in any answer, including universal/root queries.
    assert tree.css("html") == []
    assert tree.css(":root") == []
    assert [node.tag for node in tree.css("*")] == ["article", "article"]
    assert [node.tag for node in tree.css("*:first-child")] == ["article"]
    assert [node.tag for node in tree.css("*:last-child")] == ["article"]
    assert _data_n(tree.css("article:first-child")) == ["a1"]
    assert _data_n(tree.css("article:last-child")) == ["a2"]
    assert tree.css_matches("html") is False
    assert tree.any_css_matches(("html", ":root")) is False
    assert tree.select("html").matches == []
    assert tree.root.select("*:first-child").matches[0].tag == "article"


def test_fragment_wrapper_survives_table_context_fragment_tag():
    tree = LexborHTMLParser(
        "<td data-n='1'>a</td><td data-n='2'>b</td>",
        is_fragment=True,
        fragment_tag="tr",
    )
    assert [node.tag for node in tree.css("*")] == ["td", "td"]
    assert _data_n(tree.css("td:first-child")) == ["1"]
    assert _data_n(tree.css("td:nth-child(2)")) == ["2"]
    assert _data_n(tree.css("td:last-child")) == ["2"]
    assert tree.css("tr") == []


def test_full_document_still_matches_its_real_html_element():
    # Only the *fragment* wrapper is artificial; a document's <html> is real.
    tree = LexborHTMLParser("<div>x</div>")
    assert [node.tag for node in tree.css(":root")] == ["html"]
    assert [node.tag for node in tree.css("html:first-child")] == ["html"]
    assert tree.css_matches("html") is True


def test_template_fragment_does_not_leak_its_host_into_results(template_fragment):
    # The content is a separate fragment: the host <template>/<ul> of the
    # outer document is never reachable from a content-scoped query.
    assert template_fragment.css("template") == []
    assert [node.tag for node in template_fragment.css("*")] == [
        "ul",
        "li",
        "li",
        "li",
        "li",
        "span",
    ]
    # And a document-level query cannot see the content either.
    tree = LexborHTMLParser(TEMPLATE_MARKUP)
    assert tree.css(".tpl-list") == []
    assert tree.css('li:lexbor-contains("two")') == []


# ---------------------------------------------------------------------------
# Invalid selectors fail deterministically
# ---------------------------------------------------------------------------

INVALID_SELECTORS = [
    " ",
    "###",
    "p..",
    "p:",
    "::not-a-pseudo",
    ":nth-child()",
    ":nth-child(b)",
    ":not()",
    ":not( )",
    "p:not(:not())",
    ":lexbor-contains()",
    ":unknown-pseudo",
    "p::before",
    "p,,p",
    "p { color: red }",
    "[id=",
    "div >> p",
    "(",
    "*[",
]


# Every genuinely unparseable selector must raise through css()/css_matches().
@pytest.mark.parametrize("bad_selector", INVALID_SELECTORS)
def test_invalid_selectors_raise_selectolax_error(document, bad_selector):
    with pytest.raises(SelectolaxError, match="Can't parse CSS selector"):
        document.css(bad_selector)


@pytest.mark.parametrize("bad_selector", INVALID_SELECTORS)
def test_invalid_selectors_fail_through_every_entry_point(document, bad_selector):
    root = document.root
    with pytest.raises(SelectolaxError):
        root.css_first(bad_selector)
    with pytest.raises(SelectolaxError):
        root.select(bad_selector)
    with pytest.raises(SelectolaxError):
        root.css_matches(bad_selector)
    with pytest.raises(SelectolaxError):
        root.any_css_matches((bad_selector,))


def test_empty_and_blank_selectors_have_stable_well_defined_behaviour(document):
    # A blank selector is unparseable everywhere.
    with pytest.raises(SelectolaxError):
        document.css(" ")
    with pytest.raises(SelectolaxError):
        document.root.select(" ")
    # An empty query is the documented "no query" shortcut for select(): it
    # answers with the scope node itself, rather than reaching the CSS parser.
    root = document.root
    assert root.select("").matches == [root]
    assert root.select().matches == [root]
    # css() has no such shortcut, so the empty string is a parse error there.
    with pytest.raises(SelectolaxError):
        document.css("")
    with pytest.raises(SelectolaxError):
        document.css_matches("")


@pytest.mark.parametrize("bad_selector", INVALID_SELECTORS)
def test_invalid_selectors_fail_in_fragments_and_templates(
    fragment, template_fragment, bad_selector
):
    with pytest.raises(SelectolaxError):
        fragment.css(bad_selector)
    with pytest.raises(SelectolaxError):
        fragment.css_matches(bad_selector)
    with pytest.raises(SelectolaxError):
        template_fragment.css(bad_selector)
    with pytest.raises(SelectolaxError):
        template_fragment.css_first("ul").select(bad_selector)


def test_parser_remains_usable_after_a_parse_error(document):
    with pytest.raises(SelectolaxError):
        document.css("p..")
    # The shared CSS parser must not be left in a broken state.
    assert _data_n(document.css("article")) == ["a1", "a2", "a3"]
    with pytest.raises(SelectolaxError):
        document.css_matches("###")
    assert document.css_matches("article.sponsored") is True
    # any_css_matches does not swallow invalid selectors either; a valid one
    # still answers once the bad entry is removed.
    with pytest.raises(SelectolaxError):
        document.any_css_matches(("###", "article"))
    assert document.any_css_matches(("table", "article")) is True
    # Repeated failures followed by repeated successes must stay deterministic.
    for _ in range(3):
        with pytest.raises(SelectolaxError):
            document.css("[id=")
        assert _data_n(document.css("article.sponsored")) == ["a3"]


def test_non_string_query_is_a_type_error(document):
    # Cython enforces the declared str type before the CSS parser is touched.
    with pytest.raises(TypeError):
        document.css(123)
    with pytest.raises(TypeError):
        document.css_matches(None)
