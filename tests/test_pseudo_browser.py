"""Cross-check selectolax extraction against a real browser engine.

The browser fixture is a static HTML page under ``tests/fixtures`` that
contains the same structures the unit tests build inline: sibling groups
separated by comments, ``:empty`` elements, negation targets and a
``<template>`` with its own ``DocumentFragment``. A headless Chrome instance
loads it through a ``file://`` URL, evaluates ``querySelectorAll`` in three
scopes and prints a JSON document that this module compares node-for-node with
selectolax's answers.

Why a subprocess instead of a Selenium/Playwright dependency: the project's
test requirements are intentionally just pytest, and the verification only
needs the DOM API every browser already ships. When Chrome is not installed
the module skips itself, so CI without a browser keeps passing.

The lexbor-only ``:lexbor-contains()`` extension is not a browser selector, so
the harness emulates it in JavaScript -- using the same own-text-only
semantics the C implementation has -- and the Python side compares it against
both the pseudo-class and the deep ``text_contains()`` filter.
"""

import json
import os
import shutil
import subprocess

import pytest

from selectolax.lexbor import LexborHTMLParser

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "field_extraction.html")

# Selectors every conforming engine must answer identically. They cover the
# structural family, negation (simple, chained, selector-list) and order.
SHARED_SELECTORS = [
    "article:first-child",
    "article:last-child",
    "article:nth-child(1)",
    "article:nth-child(2)",
    "article:nth-child(2n+1)",
    "article:nth-child(even)",
    "article:nth-last-child(1)",
    "article:nth-last-child(2)",
    "article:first-of-type",
    "article:last-of-type",
    "section:nth-of-type(2)",
    "article:not(.sponsored)",
    "article:not(:first-child)",
    ".card:not(.sponsored)",
    "article:not([data-n='a2'])",
    "div.card:not(.sponsored):not([data-n='c1'])",
    ".card:not(.sponsored, [data-n='c2'])",
    "span:empty",
    "p:empty",
    "article, div",
    "div, article",
    ".post, .card",
]

# (selector, needle, case_insensitive). The harness mirrors lexbor's own
# direct-text-child semantics so both engines judge the same nodes.
CONTAINS_CASES = [
    ('p:lexbor-contains("summary")', "summary", False),
    ('p:lexbor-contains("SUMMARY" i)', "SUMMARY", True),
    ('h2:lexbor-contains("release")', "release", False),
    ('article:lexbor-contains("Alpha")', "Alpha", False),
]

HARNESS_SCRIPT = """
<script>
(function () {
    const shared = %(shared)s;
    const contains = %(contains)s;
    const out = {document: {}, fragment: {}, template: {}, contains: {}};

    const dataN = (el) => el.getAttribute("data-n");
    const ids = (nodeList) => [...nodeList].map(dataN);

    for (const q of shared) {
        out.document[q] = ids(document.querySelectorAll(q));
    }

    // Scope 2: a fragment built from the articles section's outer HTML, the
    // same markup selectolax parses with is_fragment=True.
    const sectionHTML = document.getElementById("articles").outerHTML;
    const fragment = document.createElement("template");
    fragment.innerHTML = sectionHTML;
    const fragRoot = fragment.content;
    for (const q of shared) {
        out.fragment[q] = ids(fragRoot.querySelectorAll(q));
    }
    out.fragment["*"] = [...fragRoot.querySelectorAll("*")].map((el) => el.tagName.toLowerCase());
    out.fragment["html"] = ids(fragRoot.querySelectorAll("html"));

    // Scope 3: the real <template id="card-template"> content.
    const tpl = document.getElementById("card-template");
    const tplRoot = tpl.content;
    out.template["article:first-child"] =
        [...tplRoot.querySelectorAll("article:first-child")].map((el) => el.className);
    out.template["article:last-child"] =
        [...tplRoot.querySelectorAll("article:last-child")].map((el) => el.className);
    out.template["article:only-child"] =
        [...tplRoot.querySelectorAll("article:only-child")].map((el) => el.className);
    out.template["a:only-of-type"] =
        [...tplRoot.querySelectorAll("a:only-of-type")].map((el) => el.textContent);
    out.template["*"] =
        [...tplRoot.querySelectorAll("*")].map((el) => el.tagName.toLowerCase());
    out.template["documentCannotSeeContent"] =
        document.querySelectorAll(".desc").length;

    // Emulated :lexbor-contains(): own direct text children only, substring,
    // optional ASCII case-insensitive flag -- matching lexbor's C engine.
    function lexborContains(root, selector, needle, insensitive) {
        const hay = (text) => insensitive ? text.toLowerCase() : text;
        const target = hay(needle);
        return [...root.querySelectorAll(selector.split(":")[0])]
            .filter((el) => [...el.childNodes].some((child) =>
                child.nodeType === Node.TEXT_NODE
                && hay(child.data).includes(target)))
            .map(dataN);
    }
    for (const [q, needle, insensitive] of contains) {
        const base = q.split(":")[0];
        out.contains[q] = {
            document: lexborContains(document, base, needle, insensitive),
            fragment: lexborContains(fragRoot, base, needle, insensitive),
        };
    }

    const pre = document.createElement("pre");
    pre.id = "browser-results";
    pre.textContent = JSON.stringify(out);
    document.body.appendChild(pre);
})();
</script>
"""


def _chrome_executable():
    for candidate in (
        os.environ.get("CHROME_BIN"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


CHROME = _chrome_executable()
pytestmark = pytest.mark.skipif(
    CHROME is None, reason="headless Chrome/Chromium is not installed"
)


@pytest.fixture(scope="module")
def browser_results(tmp_path_factory):
    """Run the harness once and share its JSON answer across the module."""
    tmp_dir = tmp_path_factory.mktemp("browser_harness")
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        fixture_html = fh.read()

    script = HARNESS_SCRIPT % {
        "shared": json.dumps(SHARED_SELECTORS),
        "contains": json.dumps(CONTAINS_CASES),
    }
    page = fixture_html.replace("</body>", script + "</body>")
    page_path = tmp_dir / "harness.html"
    page_path.write_text(page, encoding="utf-8")

    # A private user-data-dir keeps repeated runs (and parallel pytest
    # workers) from fighting over Chrome's singleton profile lock.
    profile_dir = tmp_path_factory.mktemp("chrome_profile")
    cmd = [
        CHROME,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile_dir}",
        "--virtual-time-budget=5000",
        "--dump-dom",
        page_path.as_uri(),
    ]
    # Current Chrome builds exit with code 2 on macOS (CVDisplayLink noise)
    # even when --dump-dom produced the full page, so the payload -- not the
    # exit code -- is the success criterion.
    completed = subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=60
    )

    marker = '<pre id="browser-results">'
    start = completed.stdout.find(marker)
    if start == -1:
        pytest.fail(
            "harness did not emit results "
            f"(Chrome rc={completed.returncode}).\nstderr:\n"
            f"{completed.stderr[:1000]}"
        )
    start += len(marker)
    end = completed.stdout.find("</pre>", start)
    return json.loads(completed.stdout[start:end])


@pytest.fixture(scope="module")
def selectolax_scopes():
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        fixture_html = fh.read()
    document = LexborHTMLParser(fixture_html)
    section = document.css_first("#articles")
    fragment = LexborHTMLParser(section.html, is_fragment=True)
    template = document.template_fragments()[0]
    return document, fragment, template


def _data_n(nodes):
    return [node.attributes.get("data-n") for node in nodes]


@pytest.mark.parametrize("selector", SHARED_SELECTORS)
def test_document_matches_browser(browser_results, selectolax_scopes, selector):
    document, _, _ = selectolax_scopes
    assert _data_n(document.css(selector)) == browser_results["document"][selector]


@pytest.mark.parametrize("selector", SHARED_SELECTORS)
def test_fragment_matches_browser(browser_results, selectolax_scopes, selector):
    _, fragment, _ = selectolax_scopes
    assert _data_n(fragment.css(selector)) == browser_results["fragment"][selector]


def test_fragment_wrapper_is_invisible_in_both_engines(
    browser_results, selectolax_scopes
):
    _, fragment, _ = selectolax_scopes
    assert browser_results["fragment"]["html"] == []
    assert fragment.css("html") == []
    browser_universal = browser_results["fragment"]["*"]
    assert [node.tag for node in fragment.css("*")] == browser_universal
    assert "html" not in browser_universal


def test_template_content_matches_browser(browser_results, selectolax_scopes):
    _, _, template = selectolax_scopes
    browser = browser_results["template"]

    assert [
        n.attributes.get("class") for n in template.css("article:first-child")
    ] == browser["article:first-child"]
    assert [
        n.attributes.get("class") for n in template.css("article:last-child")
    ] == browser["article:last-child"]
    assert [
        n.attributes.get("class") for n in template.css("article:only-child")
    ] == browser["article:only-child"]
    assert [n.text() for n in template.css("a:only-of-type")] == browser[
        "a:only-of-type"
    ]
    assert [node.tag for node in template.css("*")] == browser["*"]


def test_template_content_is_invisible_to_document_queries(
    browser_results, selectolax_scopes
):
    document, _, _ = selectolax_scopes
    # The browser also keeps template content out of document.querySelectorAll.
    assert browser_results["template"]["documentCannotSeeContent"] == 0
    assert document.css(".desc") == []
    assert document.css_matches(".desc") is False


@pytest.mark.parametrize("selector, needle, insensitive", CONTAINS_CASES)
def test_contains_extension_matches_emulated_browser_semantics(
    browser_results, selectolax_scopes, selector, needle, insensitive
):
    document, fragment, _ = selectolax_scopes
    expected = browser_results["contains"][selector]

    assert _data_n(document.css(selector)) == expected["document"]
    assert _data_n(fragment.css(selector)) == expected["fragment"]
    # The extension agrees with the harness on both scopes; also prove its
    # case behaviour explicitly once, independent of the emulated answer.
    hits = document.css(selector)
    for node in hits:
        own_text = "".join(
            child.text(deep=False)
            for child in node.iter(include_text=True)
            if child.is_text_node
        )
        haystack = own_text.lower() if insensitive else own_text
        assert (needle.lower() if insensitive else needle) in haystack


def test_deep_python_filter_goes_beyond_the_browser_emulation(selectolax_scopes):
    # The browser/lexbor extension inspects own text only; the Python helper
    # deep-walks, so nested text is reachable through it on the same nodes.
    document, _, _ = selectolax_scopes
    # "Alpha release" is direct text of the <h2>, nested inside <article>:
    # the extension misses it at article level...
    assert document.css('article:lexbor-contains("Alpha")') == []
    # ...while the deep filter finds the article that contains it.
    deep = document.root.select("article").text_contains("Alpha release")
    assert _data_n(deep.matches) == ["a1"]
    # text_contains() mutates the selector, so build a fresh one per query.
    assert document.root.select("article").any_text_contains("Beta release") is True
    assert document.root.select("article").any_text_contains("missing") is False


def test_result_order_is_identical_to_browser(browser_results, selectolax_scopes):
    document, fragment, _ = selectolax_scopes
    for selector in ("article, div", "div, article", ".post, .card"):
        assert _data_n(document.css(selector)) == browser_results["document"][selector]
        assert _data_n(fragment.css(selector)) == browser_results["fragment"][selector]


def test_fixture_file_is_local_and_self_contained():
    # The whole point of the fixture is offline, repeatable verification.
    assert os.path.isfile(FIXTURE_PATH)
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        content = fh.read()
    assert "<template" in content and "data-n=" in content
    assert "http://" not in content and "https://" not in content
