import pytest

import selectolax.lexbor as lexbor_module
from selectolax.lexbor import LexborHTMLParser

# Encoded documents whose declared encoding is not UTF-8.
CYRILLIC = "Привет"
JAPANESE = "日本語"


def test_bytes_are_parsed_as_utf8_by_default():
    # The default has to stay exactly as it was: no detection, no transcoding,
    # and a declaration in the document means nothing.
    raw = f'<meta charset="windows-1251"><p>{CYRILLIC}</p>'.encode("windows-1251")

    parser = LexborHTMLParser(raw)

    assert parser.raw_html == raw
    assert parser.css_first("meta").attributes["charset"] == "windows-1251"
    assert parser.text() != CYRILLIC
    # The bytes are not repaired, but reading them substitutes U+FFFD rather
    # than raising: a declaration in the document still means nothing here.
    assert parser.html == (
        '<html><head><meta charset="windows-1251"></head>'
        f"<body><p>{'�' * len(CYRILLIC)}</p></body></html>"
    )


def test_meta_charset_is_honored():
    raw = f'<meta charset="windows-1251"><p>{CYRILLIC}</p>'.encode("windows-1251")

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.text() == CYRILLIC
    assert parser.html == (
        '<html><head><meta charset="windows-1251"></head>'
        f"<body><p>{CYRILLIC}</p></body></html>"
    )


def test_http_equiv_content_type_is_honored():
    raw = (
        '<meta http-equiv="Content-Type" content="text/html; charset=windows-1251">'
        f"<p>{CYRILLIC}</p>"
    ).encode("windows-1251")

    assert LexborHTMLParser(raw, encoding=True).text() == CYRILLIC


@pytest.mark.parametrize(
    "declaration",
    [
        '<meta charset="Shift_JIS">',
        "<meta charset=shift_jis>",
        "<meta charset='shift_jis'>",
        '<meta charset=" shift_jis ">',
        '<meta http-equiv="content-type" content="text/html;charset=shift_jis">',
    ],
)
def test_declaration_forms(declaration):
    raw = f"<p>{JAPANESE}</p>{declaration}".encode("shift_jis")

    assert LexborHTMLParser(raw, encoding=True).text() == JAPANESE


def test_declaration_wins_over_a_bom_free_default():
    # UTF-8 is what an undeclared document is decoded as, so a declaration that
    # says otherwise has to actually be read.
    raw = '<meta charset="iso-8859-1"><p>café</p>'.encode("iso-8859-1")

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.text() == "café"


@pytest.mark.parametrize(
    "declarations",
    [
        ["<meta charset='iso-8859-1'>"],
        [
            "<meta charset='iso-8859-1'>",
            "<meta http-equiv=content-type content='charset=shift_jis'>",
        ],
        ["<meta name=x content='yo'>", "<meta charset='iso-8859-1'>"],
        # More declarations than one prescan array holds, so the entries have to
        # grow past their initial size.
        ["<meta name=a content=b>"] * 40 + ["<meta charset='iso-8859-1'>"],
        ["<meta charset=iso-8859-1><meta charset=shift_jis>"],
    ],
)
def test_declaration_is_found_among_many_entries(declarations):
    # Each declaration lands an entry in the prescan's arrays. Getting the
    # right answer out of a document with many of them is what keeps that
    # collection honest: the arrays have to be sized before they are pushed to,
    # or the write past the block they hand out corrupts the entries.
    raw = "".join(declarations).encode("ascii") + "<p>café</p>".encode("iso-8859-1")

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.text() == "café"
    # Where the declaration sits among the other meta tags is up to the case;
    # what matters is that it is the one that was read.
    assert any(
        meta.attributes.get("charset") == "iso-8859-1" for meta in parser.css("meta")
    )


def test_utf8_bom_is_skipped():
    raw = b"\xef\xbb\xbf" + "<p>café</p>".encode()

    assert LexborHTMLParser(raw, encoding=True).text() == "café"
    assert LexborHTMLParser(raw, encoding=True).html.startswith("<html>")


@pytest.mark.parametrize(
    "encoding,bom",
    [("utf-16-le", b"\xff\xfe"), ("utf-16-be", b"\xfe\xff")],
)
def test_utf16_bom_is_decoded(encoding, bom):
    raw = bom + "<p>café 中</p>".encode(encoding)

    assert LexborHTMLParser(raw, encoding=True).text() == "café 中"


def test_utf16_without_a_bom():
    # The XML declaration is what identifies these; the standard prescan looks
    # for it at the very start of the stream.
    raw = '<?xml version="1.0" encoding="UTF-16"?><p>café</p>'.encode("utf-16-le")

    assert LexborHTMLParser(raw, encoding=True).text() == "café"


def test_declaration_after_the_prescan_window_is_ignored():
    # The standard stops looking for a declaration after 1024 bytes, and lexbor
    # scans as far as it is allowed to. Honouring a later one would let page
    # content decide the encoding it is decoded with.
    padding = "<!--{}-->".format("x" * 1100)
    raw = f"{padding}<meta charset='windows-1251'><p>{CYRILLIC}</p>".encode(
        "windows-1251"
    )

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.text() != CYRILLIC


def test_unknown_declaration_falls_back_to_utf8():
    raw = '<meta charset="definitely-not-an-encoding"><p>café</p>'.encode()

    assert LexborHTMLParser(raw, encoding=True).text() == "café"


@pytest.mark.parametrize(
    "label",
    [
        # Transforms from the codec module that cannot read text. Each of these
        # used to raise from inside the codec rather than fall back: base64 and
        # friends an AssertionError out of binascii, rot13 a TypeError about
        # str.translate, idna a UnicodeError about its error handler.
        "base64",
        "base32",
        "base85",
        "hex",
        "binascii",
        "zlib",
        "bz2",
        "uu",
        "quopri",
        "rot13",
        "rot_13",
        "idna",
    ],
)
def test_declaration_naming_a_non_encoding_falls_back_to_utf8(label):
    raw = f'<meta charset="{label}"><p>Hello world</p>'.encode()

    assert LexborHTMLParser(raw, encoding=True).text() == "Hello world"


def test_ignoring_an_unusable_label_keeps_the_rest_of_the_document_intact():
    # The label is dropped, but the declaration itself is ordinary markup and
    # has to survive like any other attribute.
    raw = b'<meta charset="base64" name="x"><p id="a">Hello</p>'

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.text() == "Hello"
    assert parser.css_first("meta").attributes == {"charset": "base64", "name": "x"}
    assert parser.css_first("#a").text() == "Hello"


def test_x_user_defined_is_read_as_windows_1252():
    # 0x80 is a euro sign in windows-1252, which is what the standard maps the
    # x-user-defined label to. Python has no codec by that name.
    raw = '<meta charset="x-user-defined"><p>a\x80b</p>'.encode("latin-1")

    assert LexborHTMLParser(raw, encoding=True).text() == "a€b"


def test_invalid_bytes_become_replacement_characters():
    # 0x81 starts a two-byte sequence in gb18030 and nothing follows it, so it
    # cannot be decoded. The standard has those bytes become U+FFFD rather than
    # being kept, which would break every accessor that reads the tree back out.
    raw = b'<meta charset="gb18030"><p>ok\x81</p>'

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.text() == "ok�"
    assert parser.html == (
        '<html><head><meta charset="gb18030"></head><body><p>ok�</p></body></html>'
    )


def test_raw_html_holds_the_parsed_bytes():
    raw = f'<meta charset="windows-1251"><p>{CYRILLIC}</p>'.encode("windows-1251")

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.raw_html == f'<meta charset="windows-1251"><p>{CYRILLIC}</p>'.encode()
    assert parser.clone().raw_html == parser.raw_html


def test_text_input_is_untouched():
    html = f'<meta charset="windows-1251"><p>{CYRILLIC}</p>'

    for encoding in (False, True):
        parser = LexborHTMLParser(html, encoding=encoding)
        assert parser.text() == CYRILLIC
        assert parser.raw_html == html.encode()


def test_fragments_are_decoded_too():
    raw = f'<meta charset="windows-1251"><p>{CYRILLIC}</p>'.encode("windows-1251")

    parser = LexborHTMLParser(raw, is_fragment=True, encoding=True)

    assert parser.text() == CYRILLIC
    assert parser.root.tag == "meta"


def test_multibyte_character_across_a_chunk_boundary():
    # Transcoding walks the input in slices, so a character whose bytes land on
    # either side of a slice boundary must not come out broken.
    padding = "a" * (lexbor_module._TRANSCODE_CHUNK_SIZE - 1)
    raw = f'<meta charset="shift_jis"><p>{padding}{JAPANESE}</p>'.encode("shift_jis")

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.text() == padding + JAPANESE


def test_transcoded_input_is_size_limited(monkeypatch):
    # Decoding can grow the input, so the limit applies to the result too.
    monkeypatch.setattr(lexbor_module, "MAX_HTML_INPUT_SIZE", 16)
    raw = f'<meta charset="utf-8"><p>{"x" * 64}</p>'.encode()

    with pytest.raises(ValueError, match="too large"):
        LexborHTMLParser(raw, encoding=True)


def test_undeclared_input_keeps_its_bytes():
    # Nothing is declared, so the document is read as UTF-8 and bytes that are
    # not valid UTF-8 are left alone rather than repaired. Reading them
    # substitutes U+FFFD.
    raw = "<p>café</p>".encode("iso-8859-1")

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.raw_html == raw
    assert parser.html == "<html><head></head><body><p>caf�</p></body></html>"


# --- Pipeline stages -------------------------------------------------------
#
# The helpers below are internal, but each one is a stage the parser relies on.
# Testing them directly pins the rule each stage owns, so a later change cannot
# silently move a decision from one stage to another while the end-to-end tests
# still pass for the exact labels they happen to use.


def test_bom_takes_priority_over_a_declaration():
    raw = (
        b"\xef\xbb\xbf"
        + '<meta charset="windows-1251"><p>café</p>'.encode()
    )

    assert lexbor_module._detect_encoding(raw) == ("utf-8", len(b"\xef\xbb\xbf"))
    assert LexborHTMLParser(raw, encoding=True).text() == "café"


@pytest.mark.parametrize(
    "bom,codec",
    [(b"\xef\xbb\xbf", "utf-8"), (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be")],
)
def test_sniff_bom_reports_codec_and_length(bom, codec):
    assert lexbor_module._sniff_bom(bom + b"<p>x</p>") == (codec, len(bom))
    assert lexbor_module._sniff_bom(b"\x00<p>x</p>") == (None, 0)
    assert lexbor_module._sniff_bom(b"") == (None, 0)


def test_bom_of_a_non_utf8_stream_is_not_parsed_as_text():
    # The mark must be consumed, not turned into a U+FEFF text node.
    raw = b"\xef\xbb\xbf<p>caf\xc3\xa9</p>"

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.text() == "café"
    assert "﻿" not in parser.html
    assert parser.raw_html == "<p>café</p>".encode()


def test_detect_encoding_prefers_bom_then_label_then_utf8():
    assert lexbor_module._detect_encoding(b"\xff\xfe" + "<p>a</p>".encode("utf-16-le")) == (
        "utf-16-le",
        2,
    )
    assert lexbor_module._detect_encoding(b'<meta charset="shift_jis"><p>a</p>') == (
        "shift_jis",
        0,
    )
    assert lexbor_module._detect_encoding(b"<p>a</p>") == ("utf-8", 0)


def test_label_aliases_are_applied_before_validation():
    # `x-user-defined` is already aliased to windows-1252 by lexbor's prescan,
    # so it reaches the resolver under its standard name.
    assert lexbor_module._prescan_encoding_label(
        b'<meta charset="x-user-defined">'
    ) == b"windows-1252"
    assert lexbor_module._resolve_label_codec(b"windows-1252") == "windows-1252"
    # These reach the resolver under a name Python does not know and are what
    # `_CODEC_ALIASES` is for.
    assert lexbor_module._resolve_label_codec(b"iso-8859-8-i") == "iso-8859-8"
    assert lexbor_module._resolve_label_codec(b"replacement") == "ascii"
    assert lexbor_module._resolve_label_codec(b"x-mac-cyrillic") == "mac_cyrillic"


@pytest.mark.parametrize(
    "label",
    [
        None,
        b"",
        b"definitely-not-an-encoding",
        # Binary and text transforms are not encodings.
        b"base64",
        b"rot13",
        b"idna",
        b"undefined",
        # A label whose decoder cannot be handed a byte stream.
        b"utf-32",
        # A label whose decoder consumes the document and returns nothing.
        b"punycode",
        # Labels whose decoders emit lone surrogates, which UTF-8 cannot hold.
        b"unicode_escape",
        b"raw_unicode_escape",
        # The prescan only hands back ASCII, but a bogus one must still fall
        # through to UTF-8 rather than raise. (`codecs.lookup` ignores trailing
        # punctuation, so the junk comes before a real name here.)
        b"utf-8\xffx",
    ],
)
def test_unusable_labels_resolve_to_nothing(label):
    assert lexbor_module._resolve_label_codec(label) is None


def test_declared_encoding_is_resolved_before_the_parse_routine():
    # Document and fragment parsing share the pipeline: the bytes are resolved
    # by `preprocess_input` before the constructor chooses either parse routine.
    raw = f'<meta charset="windows-1251"><p>{CYRILLIC}</p>'.encode("windows-1251")

    document = LexborHTMLParser(raw, encoding=True)
    fragment = LexborHTMLParser(raw, is_fragment=True, encoding=True)

    assert document.text() == CYRILLIC
    assert fragment.text() == CYRILLIC
    assert document.raw_html == fragment.raw_html
    assert LexborHTMLParser(raw, encoding=True).raw_html == document.raw_html


# --- Fixture matrix --------------------------------------------------------
#
# Multi-encoding news-style pages, delayed declarations, conflicting
# declarations and invalid sequences, each checked through every public
# accessor so the tree, the text, the attributes and the non-ASCII characters
# are all pinned at once.

NEWS_TEXTS = [
    ("windows-1251", "Привет, мир"),
    ("shift_jis", "日本語のニュース"),
    ("gb18030", "中文新闻"),
    ("big5", "繁體中文新聞"),
    ("euc-kr", "한국어 뉴스"),
    ("euc-jp", "日本語ニュース"),
    ("iso-8859-1", "café société"),
    ("iso-8859-2", "žluťoučký kůň"),
    ("koi8-r", "Новости"),
    ("windows-1252", "café — “quotes” €"),
]


@pytest.mark.parametrize("encoding,text", NEWS_TEXTS)
def test_news_page_round_trips_through_every_accessor(encoding, text):
    raw = (
        f'<html><head><meta charset="{encoding}"><title>{text}</title></head>'
        f'<body><div class="story" data-lang="{text}"><p>{text}</p>'
        f'<a href="/{encoding}">{text}</a></div></body></html>'
    ).encode(encoding)

    parser = LexborHTMLParser(raw, encoding=True)
    expected = raw.decode(encoding)

    assert parser.text() == text * 3
    assert parser.html == expected
    assert parser.css_first("div").attributes == {
        "class": "story",
        "data-lang": text,
    }
    assert parser.css_first("a").attributes["href"] == f"/{encoding}"
    assert parser.css_first("title").text() == text
    assert parser.raw_html == expected.encode()

    # A fragment resolves the same declaration the same way; it just has no
    # head/body wrapper. Only the markup it actually contains is in the tree,
    # so the fragment has to carry its own declaration to be transcoded.
    fragment = LexborHTMLParser(
        f'<meta charset="{encoding}"><div data-lang="{text}"><p>{text}</p></div>'.encode(
            encoding
        ),
        is_fragment=True,
        encoding=True,
    )
    assert fragment.text() == text
    assert fragment.css_first("div").attributes == {"data-lang": text}


@pytest.mark.parametrize("encoding,text", NEWS_TEXTS)
def test_conflicting_declaration_wins_and_the_tree_stays_complete(encoding, text):
    # Without a BOM the declaration is the only instruction there is, so a page
    # that declares gb18030 is decoded as gb18030 even when it was written in
    # something else. The point of the fixture is not that the text is right -
    # it cannot be - but that the wrong choice still yields a whole tree.
    raw = f'<meta charset="gb18030"><p>{text}</p>'.encode(encoding)

    parser = LexborHTMLParser(raw, encoding=True)

    assert lexbor_module._detect_encoding(raw) == ("gb18030", 0)
    assert parser.css_first("meta").attributes["charset"] == "gb18030"
    assert parser.css_first("p") is not None
    assert parser.html.startswith("<html><head>")
    assert parser.html.endswith("</body></html>")
    if encoding != "gb18030":
        # The bytes were not gb18030, so the text cannot have survived intact.
        assert parser.text() != text
    else:
        assert parser.text() == text


def test_conflicting_meta_declarations_use_the_first_one():
    # The standard's prescan keeps the first declaration it finds; later ones
    # are data, not instructions. Both remain ordinary markup in the tree.
    raw = (
        '<meta charset="windows-1251"><meta charset="shift_jis">'
        "<p>Привет</p>"
    ).encode("windows-1251")

    parser = LexborHTMLParser(raw, encoding=True)

    assert lexbor_module._detect_encoding(raw) == ("windows-1251", 0)
    assert parser.text() == "Привет"
    assert [meta.attributes["charset"] for meta in parser.css("meta")] == [
        "windows-1251",
        "shift_jis",
    ]


@pytest.mark.parametrize("offset", [0, 500, 995, 996])
def test_declaration_inside_the_window_is_honoured(offset):
    raw = (
        b" " * offset
        + b'<meta charset="windows-1251"><p>'
        + CYRILLIC.encode("windows-1251")
        + b"</p>"
    )

    assert LexborHTMLParser(raw, encoding=True).text() == CYRILLIC


@pytest.mark.parametrize("offset", [997, 1024, 2048])
def test_declaration_whose_label_ends_past_the_window_is_ignored(offset):
    # Lexbor stops reading a tag at the end of the window, so a label that
    # straddles the boundary is not completed and the stream stays UTF-8.
    raw = (
        b" " * offset
        + b'<meta charset="windows-1251"><p>'
        + CYRILLIC.encode("windows-1251")
        + b"</p>"
    )

    parser = LexborHTMLParser(raw, encoding=True)

    assert lexbor_module._detect_encoding(raw)[0] == "utf-8"
    assert parser.text() != CYRILLIC
    # The declaration is still markup and still readable.
    assert parser.css_first("meta").attributes["charset"] == "windows-1251"


def test_a_bom_is_honoured_even_when_a_late_declaration_disagrees():
    raw = (
        b"\xff\xfe"
        + '<meta charset="gb18030"><p>Привет</p>'.encode("utf-16-le")
    )

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.text() == "Привет"
    assert parser.css_first("meta").attributes["charset"] == "gb18030"


@pytest.mark.parametrize(
    "encoding,label,payload",
    [
        ("shift_jis", "gb18030", "日本語"),
        ("windows-1251", "iso-8859-1", "Привет"),
        ("gb18030", "shift_jis", "中文测试"),
    ],
)
def test_declaration_that_disagrees_with_the_bytes_still_builds_a_tree(
    encoding, label, payload
):
    # A page can lie about its encoding. Whatever comes out, every accessor has
    # to keep working: the tree is complete, the declaration survives, and any
    # byte the chosen decoding cannot use is U+FFFD - never a dropped node.
    raw = f'<meta charset="{label}"><p>{payload}</p>'.encode(encoding)

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.css_first("meta").attributes["charset"] == label
    assert parser.html.startswith("<html><head>")
    assert parser.css_first("p") is not None
    # Whatever the bytes were deciphered into, all accessors agree on it and it
    # is encodable text, not a decode failure or a surrogate.
    paragraph = parser.css_first("p").text()
    assert isinstance(paragraph, str)
    assert paragraph in parser.text()
    assert all(not ("\ud800" <= ch <= "\udfff") for ch in parser.text())


def test_a_declaration_can_reinterpret_bytes_into_invalid_sequences():
    # `\xc0 ` is a letter in windows-1251 but a truncated sequence in gb18030,
    # which the page declares. The bad byte has to become U+FFFD and the rest
    # of the paragraph has to survive, together with the declaration itself.
    raw = b'<meta charset="gb18030"><p>ok\xc0 </p>'

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.css_first("p").text() == "ok� "
    assert parser.css_first("meta").attributes["charset"] == "gb18030"
    assert "</p>" in parser.html


@pytest.mark.parametrize(
    "raw",
    [
        b'<meta charset="gb18030"><p>ok\x81</p>',
        b'<meta charset="shift_jis"><p>ok\x82 </p>',
        b'<meta charset="windows-1251"><p>ok\x98 </p>',
        b'<meta charset="utf-8"><p>ok\xff\xfe</p>',
        b'<meta charset="euc-jp"><p>ok\x8f</p>',
        b'<meta charset="big5"><p>ok\x81 </p>',
    ],
)
def test_invalid_sequences_become_replacement_characters_without_truncating(raw):
    parser = LexborHTMLParser(raw, encoding=True)

    text = parser.css_first("p").text()
    assert text.startswith("ok")
    assert "�" in text
    # Nothing was silently cut: the closing tag and the rest of the document
    # are still in the tree.
    assert "</p>" in parser.html
    assert parser.css_first("p") is not None
    assert all(not ("\ud800" <= ch <= "\udfff") for ch in parser.text())


def test_text_after_an_invalid_sequence_is_kept():
    raw = b'<meta charset="gb18030"><p>ok\x81</p><span>tail</span>'

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.css_first("p").text() == "ok�"
    assert parser.css_first("span").text() == "tail"


def test_invalid_utf16_units_become_replacement_characters():
    raw = (
        b"\xff\xfe"
        + b"<\x00p\x00>\x00"
        + b"a\x00"
        + b"\x00\xd8"  # lone high surrogate
        + b"b\x00"
        + b"<\x00/\x00p\x00>\x00"
    )

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.css_first("p").text() == "a�b"
    assert parser.raw_html == "<p>a�b</p>".encode()


def test_incomplete_trailing_sequence_is_not_dropped():
    # A multi-byte character cut off at the end of the stream has nothing to
    # complete it, so it becomes U+FFFD; dropping it would lose input.
    raw = b'<meta charset="gb18030"><p>ok\x81'

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.text() == "ok�"


def test_repeated_parses_of_the_same_input_agree():
    raw = (
        b'<meta charset="big5"><p>'
        + "繁體中文新聞".encode("big5")
        + b"</p>"
    )

    parsers = [LexborHTMLParser(raw, encoding=True) for _ in range(5)]

    assert len({parser.raw_html for parser in parsers}) == 1
    assert len({parser.html for parser in parsers}) == 1
    assert len({parser.text() for parser in parsers}) == 1
    assert parsers[0].text() == "繁體中文新聞"


def test_repeated_parses_of_invalid_bytes_agree():
    # 0x81 is followed by a space, which cannot complete it in gb18030, so it
    # is invalid both in text and in an attribute value.
    raw = b'<meta charset="gb18030"><div a="\x81 ">x\x81 y</div>'

    parsers = [LexborHTMLParser(raw, encoding=True) for _ in range(5)]

    assert {parser.html for parser in parsers} == {
        '<html><head><meta charset="gb18030"></head>'
        '<body><div a="� ">x� y</div></body></html>'
    }
    assert parsers[0].css_first("div").attributes == {"a": "� "}


def test_rewrapping_a_transcoded_parser_keeps_the_same_tree():
    # Reading back `raw_html` and parsing it again has to yield the same result,
    # since `raw_html` is documented as the bytes that were parsed. Re-reading
    # with detection off is the meaningful check: `raw_html` is already UTF-8,
    # so no second declaration may reinterpret it.
    raw = b'<meta charset="shift_jis"><p>' + JAPANESE.encode("shift_jis") + b"</p>"

    parser = LexborHTMLParser(raw, encoding=True)
    again = LexborHTMLParser(parser.raw_html)

    assert again.html == parser.html
    assert again.text() == parser.text() == JAPANESE


def test_fragment_and_document_agree_on_invalid_bytes():
    raw = b'<div a="\xff">x\xffy</div>'

    as_document = LexborHTMLParser(raw, encoding=True)
    as_fragment = LexborHTMLParser(raw, is_fragment=True, encoding=True)

    assert as_document.css_first("div").html == as_fragment.html
    assert as_fragment.css_first("div").attributes == {"a": "�"}
    assert as_fragment.text() == "x�y"


# --- Labels that used to break a parse -------------------------------------
#
# Each of these names a codec the registry offers but that cannot be used to
# read a document the way the pipeline needs to. A page can declare any of them,
# so they have to fall back to UTF-8 rather than raise or silently empty the
# document.


@pytest.mark.parametrize(
    "label,reason",
    [
        ("utf-32", "its decoder raises when the stream has no BOM"),
        ("punycode", "it decodes ordinary markup to nothing"),
        ("unicode_escape", "it turns escapes into lone surrogates"),
        ("raw_unicode_escape", "it turns escapes into lone surrogates"),
    ],
)
def test_unusable_declaration_falls_back_without_losing_the_page(label, reason):
    raw = f'<meta charset="{label}"><p>Hello world</p><span id="tail">end</span>'.encode()

    parser = LexborHTMLParser(raw, encoding=True)

    assert lexbor_module._encoding_codec(raw) == "utf-8", reason
    assert parser.text() == "Hello worldend"
    assert parser.css_first("span#tail").text() == "end"
    assert parser.css_first("span#tail").attributes == {"id": "tail"}
    assert parser.css_first("meta").attributes["charset"] == label


def test_a_page_whose_label_was_unusable_keeps_its_invalid_bytes_readable():
    # The label is dropped but the bytes stay as they arrived, so a byte that
    # is invalid UTF-8 is still reported as U+FFFD by every accessor.
    raw = b'<meta charset="punycode"><p>a\xffb</p>'

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.css_first("p").text() == "a�b"
    assert parser.raw_html == raw


def test_utf8_bom_with_an_unusable_declaration_is_still_utf8():
    raw = b'\xef\xbb\xbf<meta charset="utf-32"><p>caf\xc3\xa9</p>'

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.text() == "café"
    assert not parser.raw_html.startswith(b'\xef\xbb\xbf')


def test_replacement_label_replaces_high_bytes_and_keeps_the_markup():
    # The standard's "replacement" encoding is implemented as ASCII with
    # `errors='replace'`, so every non-ASCII byte is U+FFFD and the markup
    # still parses.
    raw = b'<meta charset="replacement"><p id="a">a\x80b\xffc</p>'

    parser = LexborHTMLParser(raw, encoding=True)

    assert parser.css_first("p").text() == "a�b�c"
    assert parser.css_first("p").attributes == {"id": "a"}
