# Turn parser input into the UTF-8 bytes Lexbor is fed with.
#
# Byte input that is read with ``encoding=True`` goes through four stages, in
# this order, and nothing else may make the decision for it:
#
# 1. A leading byte-order mark (`_sniff_bom`). It wins over everything else,
#    because the bytes themselves announce their encoding and no markup does.
# 2. The HTML Standard's prescan of the first 1024 bytes
#    (`_prescan_encoding_label`). It finds a ``<meta charset>``,
#    ``<meta http-equiv="content-type">`` or UTF-16 XML declaration. Lexbor
#    implements that step, including the "first declaration wins" rule, so the
#    window is enforced here by handing it no more than 1024 bytes.
# 3. Resolution of the declared label to a Python codec
#    (`_resolve_label_codec`). The standard's aliases are applied, then the
#    registry is asked whether the name really is a text encoding that can
#    stream bytes (`_text_codec_name`). Anything else - an unknown name, a
#    binary transform such as ``base64``, a decoder that cannot be handed a
#    byte stream - resolves to nothing and the stream falls back to UTF-8.
# 4. Transcoding to UTF-8 with ``errors='replace'`` (`_transcode_to_utf8`), so
#    invalid bytes become U+FFFD instead of raising or being dropped. The BOM,
#    if any, is removed before this step.
#
# The default when nothing usable is declared is UTF-8, not the HTML Standard's
# windows-1252: every other selectolax path assumes UTF-8, so a different
# default would reinterpret input that already parsed correctly. A declaration
# is only ever allowed to *choose* an encoding for bytes that were not already
# readable.
#
# Non-UTF-8 text always leaves this module as UTF-8, which is the only encoding
# the parser and every accessor assume. Text input (``str``) never enters the
# pipeline: it is already decoded and only needs encoding.
#
# The stages are separate functions so that a caller can inspect one of them
# without repeating the whole pipeline, but `_decode_to_utf8` and
# `preprocess_input` are the only entry points the parser uses, and document and
# fragment parsing both go through them.

import codecs

# 2500 MB
MAX_HTML_INPUT_SIZE = 2_500_000_000

# The HTML Standard only looks for a <meta> encoding declaration within the
# first 1024 bytes of a byte stream.
_MAX_ENCODING_PRESCAN_SIZE = 1024

# How much input is decoded per step when transcoding. Transcoding walks the
# whole document, and `MAX_HTML_INPUT_SIZE` allows for very large ones, so the
# work is done in slices to keep the extra memory close to a single slice.
_TRANSCODE_CHUNK_SIZE = 1_048_576

# Transcoding must never abort a parse over a malformed byte sequence: the
# HTML Standard replaces what it cannot decode with U+FFFD and keeps going.
_DECODING_ERRORS = 'replace'

_BOM_UTF8 = b'\xef\xbb\xbf'
_BOM_UTF16_BE = b'\xfe\xff'
_BOM_UTF16_LE = b'\xff\xfe'

# Byte-order marks in the order they are checked, with the codec each announces.
# The signatures cannot overlap, but UTF-8 comes first because it is the common
# case and UTF-16 LE/BE are distinguishable from each other by their first byte.
_BOM_SIGNATURES = (
    (_BOM_UTF8, 'utf-8'),
    (_BOM_UTF16_LE, 'utf-16-le'),
    (_BOM_UTF16_BE, 'utf-16-be'),
)

# Labels the HTML Standard resolves to an encoding Python's codec registry does
# not know under that name. `replacement` maps to `ascii` deliberately: with
# `errors='replace'` that turns every non-ASCII byte into U+FFFD and leaves the
# rest alone, which is what the standard's "replacement" encoding prescribes.
_CODEC_ALIASES = {
    'iso-8859-8-i': 'iso-8859-8',
    'replacement': 'ascii',
    'x-mac-cyrillic': 'mac_cyrillic',
}

# Bytes a candidate codec is asked to decode before it is trusted.
#
# A document can name any registered codec, and the registry also holds things
# that are not encodings at all. The probe is ordinary ASCII markup, which every
# text encoding must be able to read, plus an escaped surrogate (`\\ud800`),
# which only the escape-style codecs turn into a character. Decoding it has to
# produce non-empty text that is encodable as UTF-8. That rejects the binary
# transforms (`base64`, `zlib`, ...), the text transforms (`rot13`, ...),
# `punycode` (which swallows ordinary text and returns nothing) and
# `unicode_escape`/`raw_unicode_escape` (which turn escapes into lone
# surrogates). See :func:`_text_codec_name`.
_CODEC_PROBE = b'<meta charset="probe">\\ud800'


def _sniff_bom(bytes html_bytes):
    """Return the codec a leading byte-order mark announces, and its length.

    A BOM is checked before the ``<meta>`` prescan because it is the one
    declaration that is made by the bytes themselves, so the HTML Standard lets
    it win over any markup. For a stream with no BOM no bytes are consumed.

    Parameters
    ----------
    html_bytes : bytes
        Raw HTML to inspect.

    Returns
    -------
    tuple of (str or None, int)
        The codec name, or ``None`` when there is no mark; and the number of
        leading bytes the mark occupies (``0`` when there is none).
    """
    for bom, codec in _BOM_SIGNATURES:
        if html_bytes.startswith(bom):
            return codec, len(bom)
    return None, 0


def _prescan_encoding_label(bytes html_bytes):
    """Return the encoding label declared at the start of a byte stream, if any.

    Delegates to lexbor's ``lxb_html_encoding_prescan()``, an implementation of
    the HTML Standard's "prescan a byte stream to determine its encoding" step.
    It recognises both ``<meta charset>`` and
    ``<meta http-equiv="content-type" content="...charset=...">``, and it
    already applies the standard's label aliasing, so a ``utf-16*`` label comes
    back as ``UTF-8`` and ``x-user-defined`` as ``windows-1252``. A UTF-16 XML
    declaration at the very start of the stream identifies its byte order the
    same way.

    Only the first ``_MAX_ENCODING_PRESCAN_SIZE`` bytes are looked at. Lexbor
    scans as far as it is allowed to, and the standard is what draws the line at
    1024 bytes: honouring a declaration found further into the document would
    let page content decide the encoding it is decoded with. A label that
    straddles the end of the window therefore only wins if lexbor can finish
    reading it inside the window; that boundary is the standard's, not ours.

    Parameters
    ----------
    html_bytes : bytes
        Raw HTML to scan.

    Returns
    -------
    bytes or None
        The declared label, or ``None`` when the stream declares nothing.

    Raises
    ------
    MemoryError
        If the prescan object cannot be allocated.
    """
    cdef lxb_html_encoding_t* encoding
    cdef lxb_status_t status
    cdef const lxb_char_t* data
    cdef const lxb_char_t* label
    cdef size_t label_len = 0
    cdef size_t scan_len = min(len(html_bytes), _MAX_ENCODING_PRESCAN_SIZE)

    data = <const lxb_char_t *> PyBytes_AS_STRING(html_bytes)

    encoding = lxb_html_encoding_create()
    if encoding == NULL:
        raise MemoryError("Can't allocate memory for the encoding prescan.")

    try:
        status = lxb_html_encoding_init(encoding)
        if status != LXB_STATUS_OK:
            raise SelectolaxError("Can't initialize the encoding prescan.")

        with nogil:
            label = lxb_html_encoding_prescan(encoding, data,
                                              data + scan_len, &label_len)

        if label == NULL:
            return None

        return PyBytes_FromStringAndSize(<const char *> label, label_len)
    finally:
        lxb_html_encoding_destroy(encoding, True)


def _text_codec_name(str name):
    """Return ``name`` when bytes can be decoded with it, ``None`` otherwise.

    ``codecs`` also holds transforms that are not encodings, such as the binary
    ``base64`` and the text transform ``rot13``. A document can name one in
    ``<meta charset>``, so such a label has to be rejected like an unknown one
    rather than raise from inside the codec module. The registry flags those
    with ``_is_text_encoding``; the rest are caught by decoding ``_CODEC_PROBE``.

    The probe is decoded twice, because that is how transcoding reads the
    document. A one-shot decode catches codecs that refuse the error handler;
    an incremental decode catches codecs that cannot be handed a stream at all
    (``utf-32`` and a BOM-less ``utf-16`` raise on their first non-empty chunk).
    The result has to be non-empty - a codec that consumes the document and
    returns nothing would truncate the page - and has to encode back to UTF-8,
    which rejects decoders that hand out lone surrogates. The probe asks for
    ``errors='replace'`` because that is what transcoding uses, and some codecs
    accept only some of the error handlers.

    Parameters
    ----------
    name : str
        Lower-cased encoding label, after the standard's aliases have been
        applied.

    Returns
    -------
    str or None
        ``name`` when it names a codec that can decode a byte stream to UTF-8
        text, ``None`` otherwise.
    """
    if not name:
        return None

    try:
        codec_info = codecs.lookup(name)
    except (LookupError, ValueError):
        return None

    if not getattr(codec_info, '_is_text_encoding', False):
        return None

    try:
        text = _CODEC_PROBE.decode(name, _DECODING_ERRORS)
    except Exception:
        return None
    if not text:
        return None
    try:
        text.encode('UTF-8')
    except UnicodeEncodeError:
        return None

    try:
        decoder = codecs.getincrementaldecoder(name)(errors=_DECODING_ERRORS)
    except LookupError:
        # A codec without an incremental decoder is decoded in one piece by
        # `_transcode_to_utf8`; the one-shot probe above already vetted it.
        return name
    except Exception:
        return None

    try:
        text = decoder.decode(_CODEC_PROBE) + decoder.decode(b'', True)
    except Exception:
        return None
    if not text:
        return None
    try:
        text.encode('UTF-8')
    except UnicodeEncodeError:
        return None

    return name


def _resolve_label_codec(label):
    """Turn a declared label into the name of a codec that can decode a stream.

    Applies the HTML Standard's aliases for labels Python spells differently,
    then validates the result with :func:`_text_codec_name`. Both an unknown
    label and one that names something unusable come back as ``None``, which is
    how the caller learns to fall back to UTF-8. The label is treated as ASCII
    (the prescan only returns ASCII): anything else becomes U+FFFD and then
    fails the codec lookup, exactly like an unknown label.

    Parameters
    ----------
    label : bytes or None
        Label returned by :func:`_prescan_encoding_label`.

    Returns
    -------
    str or None
        A codec name, or ``None`` when the label cannot be used.
    """
    if label is None:
        return None

    name = label.decode('ascii', errors=_DECODING_ERRORS).strip().lower()

    return _text_codec_name(_CODEC_ALIASES.get(name, name))


def _detect_encoding(bytes html_bytes):
    """Resolve a byte stream to the codec it should be transcoded with.

    Applies the pipeline's order of preference: a byte-order mark, then the
    declaration found in the first 1024 bytes, then UTF-8.

    Parameters
    ----------
    html_bytes : bytes
        Raw HTML to inspect.

    Returns
    -------
    tuple of (str, int)
        The codec name and the number of leading byte-order-mark bytes to skip.
        When nothing usable is declared this is ``('utf-8', 0)``.
    """
    codec, bom_len = _sniff_bom(html_bytes)
    if codec is not None:
        return codec, bom_len

    codec = _resolve_label_codec(_prescan_encoding_label(html_bytes))
    if codec is None:
        # An undeclared, unknown or unusable label is not worth failing a parse
        # over. The stream is read as UTF-8, the encoding every other code path
        # already assumes.
        return 'utf-8', 0

    return codec, 0


def _encoding_codec(bytes html_bytes):
    """Return the name of the codec a byte stream should be decoded with.

    This is :func:`_detect_encoding` without the BOM length, kept as the single
    answer to "which encoding was chosen". The preference order and the UTF-8
    fallback are documented there.

    Parameters
    ----------
    html_bytes : bytes
        Raw HTML to inspect.

    Returns
    -------
    str
        Name of a codec known to Python's codec registry that decodes bytes.
    """
    return _detect_encoding(html_bytes)[0]


def _utf8_from_text(str text):
    """Encode decoded text as UTF-8, turning lone surrogates into U+FFFD.

    A codec that passes :func:`_text_codec_name` does not normally produce
    surrogates, but ``errors='replace'`` is a decoding policy, not a promise
    about every input a codec might see. UTF-8 cannot represent a surrogate, and
    the HTML Standard maps one to U+FFFD, so the rare unencodable character is
    replaced instead of raising out of a parse.

    Parameters
    ----------
    text : str
        Decoded document text.

    Returns
    -------
    bytes
        ``text`` as UTF-8.
    """
    try:
        return text.encode('UTF-8')
    except UnicodeEncodeError:
        # Round-trip through UTF-16 with `surrogatepass`, then decode with
        # `replace`: a lone surrogate becomes U+FFFD while everything else,
        # including astral characters, is preserved.
        return (
            text.encode('utf-16-le', 'surrogatepass')
            .decode('utf-16-le', _DECODING_ERRORS)
            .encode('UTF-8')
        )


def _transcode_to_utf8(bytes html_bytes, str codec):
    """Decode a byte stream with ``codec`` and return the text as UTF-8 bytes.

    Bytes that do not form a valid sequence in the source encoding become
    U+FFFD, as the HTML Standard prescribes, instead of being dropped or kept
    as-is to blow up on the way out of the parser. The decoder is incremental so
    that a character split across two slices is not mistaken for invalid input,
    and it is flushed at the end so an incomplete trailing sequence also becomes
    U+FFFD rather than disappearing.

    Parameters
    ----------
    html_bytes : bytes
        Raw HTML to transcode.
    codec : str
        Name of the codec to decode with, as returned by
        :func:`_detect_encoding`.

    Returns
    -------
    bytes
        The decoded document, encoded as UTF-8.

    Raises
    ------
    ValueError
        If the transcoded document exceeds ``MAX_HTML_INPUT_SIZE``.
    """
    cdef bytearray out

    if not html_bytes:
        return b''

    try:
        decoder = codecs.getincrementaldecoder(codec)(errors=_DECODING_ERRORS)
    except LookupError:
        # A codec without an incremental decoder has to be handed all at once.
        out = bytearray(_utf8_from_text(
            html_bytes.decode(codec, _DECODING_ERRORS)
        ))
        _check_input_size(len(out))
        return bytes(out)

    out = bytearray()
    for start in range(0, len(html_bytes), _TRANSCODE_CHUNK_SIZE):
        out += _utf8_from_text(decoder.decode(
            html_bytes[start:start + _TRANSCODE_CHUNK_SIZE]
        ))
        _check_input_size(len(out))

    out += _utf8_from_text(decoder.decode(b'', True))
    _check_input_size(len(out))

    return bytes(out)


def _decode_to_utf8(bytes html_bytes):
    """Return a byte stream as UTF-8, using the encoding it declares.

    The stream is split at the byte-order mark (if any) before transcoding, so
    the mark itself never reaches the parser and never becomes a text node. A
    stream that ends up read as UTF-8 is returned without a mark; any other
    stream is transcoded.

    A declared codec that cannot consume the stream, or that consumes all of it
    without producing text, is treated as unusable and the bytes are kept as
    they are: the parser still builds a tree and accessors substitute U+FFFD for
    invalid bytes. Silently returning an empty document in that case would lose
    the page body, which is worse than an imperfect but complete tree.

    Parameters
    ----------
    html_bytes : bytes
        Raw HTML to decode.

    Returns
    -------
    bytes
        The document as UTF-8, without a leading byte-order mark.
    """
    cdef str codec
    cdef Py_ssize_t bom_len
    cdef bytes body
    cdef object transcoded

    codec, bom_len = _detect_encoding(html_bytes)
    body = html_bytes[bom_len:] if bom_len else html_bytes

    if codec == 'utf-8':
        return body

    try:
        transcoded = _transcode_to_utf8(body, codec)
    except (LookupError, UnicodeError):
        # A decoder that refuses the stream has already been rejected by
        # `_text_codec_name`; this guard is for inputs it could not foresee.
        # Keeping the bytes still yields a tree, while raising here would turn
        # a malformed page into no parse at all.
        return body

    if not transcoded and body:
        # The codec consumed the whole document and produced no text. That is
        # silent truncation, so keep the bytes and let the parser and its
        # accessors report what they cannot decode as U+FFFD.
        return body

    return transcoded


def _check_input_size(html_len):
    """Reject an input that is too large to be processed.

    Parameters
    ----------
    html_len : int
        Size of the input in bytes.

    Raises
    ------
    ValueError
        If the input exceeds ``MAX_HTML_INPUT_SIZE``.
    """
    if html_len > MAX_HTML_INPUT_SIZE:
        raise ValueError("The specified HTML input is too large to be processed (%d bytes)" % html_len)


def preprocess_input(html, decode_errors='ignore', encoding=False):
    """Normalise parser input into the UTF-8 bytes lexbor is fed with.

    This is the single entry point for both document and fragment parsing: the
    parser resolves bytes here before it decides which of the two parse routines
    to use, so an encoding declaration means the same thing in a whole page and
    in a fragment.

    Parameters
    ----------
    html : str or bytes
        Content to parse.
    decode_errors : str, default 'ignore'
        Passed to ``str.encode`` when ``html`` is text. It has no effect on
        bytes: those are repaired by the pipeline's own ``errors='replace'``
        policy.
    encoding : bool, default False
        When ``True`` and ``html`` is bytes, run the detection pipeline and
        transcode the document to UTF-8 first, so that a leading byte-order
        mark and a declaration within the first 1024 bytes are honoured. Text
        input needs none of this and is unaffected.

    Returns
    -------
    tuple of (bytes, int)
        The UTF-8 bytes to parse and their length.

    Raises
    ------
    TypeError
        If ``html`` is neither text nor bytes.
    ValueError
        If the input, or its transcoded form, exceeds ``MAX_HTML_INPUT_SIZE``.
    """
    if isinstance(html, (str, unicode)):
        bytes_html = html.encode('UTF-8', errors=decode_errors)
    elif isinstance(html, bytes):
        bytes_html = _decode_to_utf8(html) if encoding and html else html
    else:
        raise TypeError("Expected a string, but %s found" % type(html).__name__)
    html_len = len(bytes_html)
    _check_input_size(html_len)
    return bytes_html, html_len
