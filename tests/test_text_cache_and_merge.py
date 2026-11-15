"""Repeated extraction and merge behavior on an edited page.

These cover the guarantees behind reusing a ``text()`` result and skipping a
repeated ``merge_text_nodes()``: the answers always match a fresh extraction,
merging restores the same text with the documented boundaries, and edits made
between calls are reflected whether the next read starts from the whole page or
from a subtree reference obtained earlier.
"""

import pytest

from selectolax.lexbor import LexborHTMLParser


def _reference_text(node, deep, separator, strip, skip_empty):
    """Recompute ``text()`` in pure Python as ``separator.join(parts)``."""
    parts = []

    def add(candidate):
        content = candidate.text_content
        if content is None:
            return
        if skip_empty and candidate.is_empty_text_node:
            return
        parts.append(content.strip() if strip else content)

    def descendants(root):
        """Every descendant of ``root`` in lexbor's document order."""
        stack = list(reversed(list(root.iter(include_text=True))))
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(list(current.iter(include_text=True))))

    if node.is_text_node:
        add(node)
    if deep:
        for descendant in descendants(node):
            if descendant.is_text_node:
                add(descendant)
    else:
        for child in node.iter(include_text=True):
            if child.is_text_node:
                add(child)
    return separator.join(parts)


# --- text() reuse stays correct across edits ---------------------------------


def test_repeated_text_reflects_later_insertions():
    tree = LexborHTMLParser("<div><p>alpha</p><p>beta</p></div>")
    div = tree.css_first("div")

    assert div.text(separator="|") == "alpha|beta"

    div.css_first("p").insert_child("!")
    assert div.text(separator="|") == "alpha|!|beta"

    tail = tree.create_node("p")
    tail.insert_child("gamma")
    div.insert_child(tail)
    assert div.text(separator="|") == "alpha|!|beta|gamma"


def test_repeated_text_reflects_deletions():
    tree = LexborHTMLParser("<div><p>keep</p><p>drop</p></div>")
    div = tree.css_first("div")

    assert div.text(separator="|") == "keep|drop"

    div.css("p")[-1].decompose()
    assert div.text(separator="|") == "keep"


def test_edit_elsewhere_keeps_a_siblings_cached_text():
    """A subtree untouched by the edit must keep answering the same text."""
    tree = LexborHTMLParser(
        "<div id='a'><span>A</span></div><div id='b'><span>B</span></div>"
    )
    a = tree.css_first("#a")

    assert a.text() == "A"

    tree.css_first("#b").decompose()
    # The sibling subtree was not edited, so its text must be unchanged.
    assert a.text() == "A"
    assert tree.text() == "A"


def test_text_with_changed_options_returns_the_new_option_result():
    tree = LexborHTMLParser("<div>a<span>b</span>c</div>")
    div = tree.css_first("div")

    assert div.text(deep=True, separator="") == "abc"
    # Changing only the options must not reuse the previous answer.
    assert div.text(deep=False, separator="") == "ac"
    assert div.text(deep=True, separator="|") == "a|b|c"
    assert div.text(deep=True, separator="|", strip=True) == "a|b|c"


def test_text_options_do_not_cross_contaminate():
    tree = LexborHTMLParser("<div>\n  <span>\t</span><span>keep</span></div>")
    div = tree.css_first("div")

    assert div.text(skip_empty=False) == "\n  \tkeep"
    assert div.text(skip_empty=True) == "keep"
    # Switching back reproduces the first answer rather than the second.
    assert div.text(skip_empty=False) == "\n  \tkeep"


def test_text_on_a_subtree_reference_survives_page_edits():
    tree = LexborHTMLParser("<div id='a'><span>A</span></div><div id='b'>B</div>")
    a = tree.css_first("#a")
    assert a.text() == "A"

    tree.css_first("#b").unwrap()
    assert a.text() == "A"
    assert tree.text() == "AB"


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize("separator", ["", "|"])
@pytest.mark.parametrize("strip", [True, False])
@pytest.mark.parametrize("skip_empty", [True, False])
def test_reused_text_matches_reference_after_each_edit(
    deep, separator, strip, skip_empty
):
    """Every option/JSON combination agrees with a fresh reference read."""
    tree = LexborHTMLParser("<div>a<span>b</span>c<p> </p>d</div>")
    div = tree.css_first("div")
    options = dict(deep=deep, separator=separator, strip=strip, skip_empty=skip_empty)

    # Establish a cached answer under one tree state ...
    assert div.text(**options) == _reference_text(
        div, deep, separator, strip, skip_empty
    )

    # ... edit, then check the next read against a fresh reference.
    div.css_first("span").unwrap()
    assert div.text(**options) == _reference_text(
        div, deep, separator, strip, skip_empty
    )

    div.insert_child("e")
    assert div.text(**options) == _reference_text(
        div, deep, separator, strip, skip_empty
    )


# --- merge_text_nodes semantics ----------------------------------------------


def test_merge_rejoins_words_split_by_inline_tags():
    html = "<div><p><strong>J</strong>ohn <em>D</em>oe</p></div>"
    tree = LexborHTMLParser(html)
    tree.unwrap_tags(["strong", "em"])
    div = tree.css_first("div")

    # Before merging the run is still several text nodes, so separator insertion
    # shows up as extra spaces.
    assert div.text(separator=" ", strip=True) == "J ohn D oe"

    div.merge_text_nodes()

    assert div.html == "<div><p>John Doe</p></div>"
    assert div.text(separator=" ", strip=True) == "John Doe"


def test_merge_keeps_block_and_comment_boundaries():
    tree = LexborHTMLParser("<div>a<p>b</p>c<!--x-->d</div>")
    div = tree.css_first("div")

    div.merge_text_nodes()

    # The <p> and the comment still separate the runs; only same-parent siblings
    # collapse.
    assert div.html == "<div>a<p>b</p>c<!--x-->d</div>"
    assert div.text(separator="|") == "a|b|c|d"


def test_merge_does_not_pull_nested_text_up():
    tree = LexborHTMLParser("<div>outer<b>inner</b></div>")
    div = tree.css_first("div")

    div.merge_text_nodes()

    # "outer" and "inner" are not siblings, so they must stay separate nodes.
    assert div.html == "<div>outer<b>inner</b></div>"
    children = list(div.iter(include_text=True))
    assert [c.tag for c in children] == ["-text", "b"]


def test_merge_keeps_script_content_within_its_own_parent():
    tree = LexborHTMLParser("<div>ok<script>var x=1;</script></div>")
    div = tree.css_first("div")

    div.merge_text_nodes()

    assert div.css_first("script").text() == "var x=1;"
    assert div.text() == "okvar x=1;"


def test_merge_is_idempotent():
    tree = LexborHTMLParser("<div>He<b>ll</b>o</div>")
    tree.unwrap_tags(["b"])
    div = tree.css_first("div")

    div.merge_text_nodes()
    after_first = div.html
    div.merge_text_nodes()
    assert div.html == after_first
    assert div.text() == "Hello"


def test_merge_after_an_edit_runs_again():
    """A merge that was followed by an edit must not be treated as done."""
    tree = LexborHTMLParser("<div><p>a<b>x</b>c</p></div>")
    div = tree.css_first("div")

    div.merge_text_nodes()
    tree.unwrap_tags(["b"])
    div.merge_text_nodes()

    assert div.html == "<div><p>axc</p></div>"


def test_merge_invalidates_a_cached_extraction_that_predates_it():
    tree = LexborHTMLParser("<div>He<b>ll</b>o</div>")
    tree.unwrap_tags(["b"])
    div = tree.css_first("div")

    # Cache a pre-merge read (separated fragments).
    assert div.text(separator=" ", strip=True) == "He ll o"

    div.merge_text_nodes()

    # The cached read described three text nodes; the tree now has one.
    assert div.text(separator=" ", strip=True) == "Hello"


def test_merge_on_a_scope_already_merged_by_an_ancestor_is_a_noop():
    tree = LexborHTMLParser("<div>a<b>c</b>d</div>")
    tree.unwrap_tags(["b"])
    div = tree.css_first("div")

    div.merge_text_nodes()
    html = div.html

    # A node inside the merged scope was already covered.
    div.merge_text_nodes()
    assert div.html == html


def test_merge_covers_the_whole_fragment_from_its_root():
    parser = LexborHTMLParser("<p>a<b>c</b>e</p><p>f</p>", is_fragment=True)
    parser.unwrap_tags(["b"])

    parser.root.merge_text_nodes()

    assert parser.html == "<p>ace</p><p>f</p>"


def test_merge_returns_none():
    tree = LexborHTMLParser("<div>a<b>c</b></div>")
    tree.unwrap_tags(["b"])
    assert tree.css_first("div").merge_text_nodes() is None


# --- structural cost and bounded extra space --------------------------------


def test_long_adjacent_text_run_merges_in_linear_time():
    """A long run of adjacent text nodes folds without quadratic copying."""
    count = 5000
    html = "<div>" + "".join(f"<b>w{i}</b>" for i in range(count)) + "</div>"
    tree = LexborHTMLParser(html)
    tree.unwrap_tags(["b"])
    div = tree.css_first("div")

    assert sum(1 for _ in div.iter(include_text=True)) == count

    div.merge_text_nodes()

    children = list(div.iter(include_text=True))
    assert len(children) == 1
    assert children[0].text_content == "".join(f"w{i}" for i in range(count))


def test_deeply_nested_tree_merges_without_recursion_error():
    depth = 5000
    html = "<div>" * depth + "core" + "</div>" * depth
    tree = LexborHTMLParser(html)

    tree.root.merge_text_nodes()

    assert tree.text() == "core"


def test_repeated_edits_keep_bookkeeping_bounded():
    """Rereading after many edits stays correct and does not grow unbounded."""
    tree = LexborHTMLParser("<div></div>")
    div = tree.css_first("div")

    expected = []
    for i in range(300):
        div.insert_child(str(i % 10))
        expected.append(str(i % 10))
        assert div.text() == "".join(expected)

    assert div.text() == "".join(expected)


def test_text_cache_is_bounded_by_character_cap():
    """Many distinct extractions must not retain unbounded characters."""
    tree = LexborHTMLParser(
        "<div>" + "".join(f"<p>{i}</p>" for i in range(400)) + "</div>"
    )
    div = tree.css_first("div")

    for node in div.css("p"):
        node.text(separator="|")

    cache = tree._text_cache_size_for_tests()
    assert cache[0] <= 1 << 16
    assert cache[1] <= 1 << 12


def test_text_cache_is_bounded_by_entry_cap_for_empty_results():
    """Zero-length results still count against the entry cap, not just chars."""
    tree = LexborHTMLParser(
        "<div>" + "".join("<p></p>" for _ in range(5000)) + "</div>"
    )
    div = tree.css_first("div")

    for node in div.css("p"):
        assert node.text() == ""

    cache = tree._text_cache_size_for_tests()
    assert cache[0] <= 1 << 16
    assert cache[1] <= 1 << 12


# --- cache never diverges from a cold read -----------------------------------


def test_cached_read_equals_a_cold_read_after_edits():
    tree = LexborHTMLParser("<div id='r'>a<span>b</span>c<p>d</p></div>")
    root = tree.css_first("#r")

    # Warm the cache for several option sets.
    for options in (
        {},
        {"separator": "|"},
        {"deep": False},
        {"strip": True},
        {"skip_empty": True},
    ):
        root.text(**options)

    root.css_first("span").unwrap()
    root.insert_child("e")
    tree.css_first("p").insert_before("x")

    option_sets = [
        {},
        {"separator": "|"},
        {"deep": False},
        {"strip": True, "separator": "|"},
        {"skip_empty": True},
    ]
    warm = [root.text(**opts) for opts in option_sets]

    tree._clear_text_cache_for_tests()
    cold = [root.text(**opts) for opts in option_sets]

    assert warm == cold


def test_selectedcontent_mirroring_is_not_hidden_by_the_cache():
    """A mutation event can fill a subtree outside the edited one.

    Lexbor's selectedness algorithm copies an option's content into
    ``<selectedcontent>`` when an option is inserted. Whatever that leaves in
    the tree, a cached read of the ``<selectedcontent>`` must equal a cold read
    of the same tree - the cache must reflect the post-event structure.
    """
    html = (
        "<select><button><selectedcontent></selectedcontent></button>"
        "<option selected>a</option></select>"
    )
    tree = LexborHTMLParser(html)
    select = tree.css_first("select")
    selectedcontent = tree.css_first("selectedcontent")

    # Warm both caches before the mutation event runs.
    assert selectedcontent.text() == "a"
    assert select.text() == "aa"

    option = tree.create_node("option")
    option.insert_child("B")
    select.insert_child(option)

    warm_sc = selectedcontent.text()
    warm_select = select.text()

    tree._clear_text_cache_for_tests()

    assert selectedcontent.text() == warm_sc
    assert select.text() == warm_select


def test_recursive_decompose_invalidates_a_descendant_view():
    """``decompose(recursive=True)`` empties descendants, not just unlinks them.

    A view of a descendant that was read before the removal must not keep
    answering with its old text once the subtree has been torn down.
    """
    tree = LexborHTMLParser("<div id='top'><section><span>keep</span></section></div>")
    top = tree.css_first("#top")
    span = tree.css_first("span")

    assert span.text() == "keep"

    top.decompose(recursive=True)

    assert span.text() == ""


def test_recursive_strip_tags_invalidates_a_descendant_view():
    tree = LexborHTMLParser("<div><section><span>keep</span></section></div>")
    span = tree.css_first("span")

    assert span.text() == "keep"

    tree.strip_tags(["section"], recursive=True)

    assert span.text() == ""


def test_non_recursive_decompose_leaves_a_descendant_view_intact():
    """Removing a node without recursing does not touch its subtree."""
    tree = LexborHTMLParser("<div id='top'><section><span>keep</span></section></div>")
    top = tree.css_first("#top")
    span = tree.css_first("span")

    assert span.text() == "keep"

    top.decompose(recursive=False)

    assert span.text() == "keep"
    assert span.html == "<span>keep</span>"
