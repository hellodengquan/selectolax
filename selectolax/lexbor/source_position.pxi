"""Source position objects exposed to callers.

The C parser records byte offsets into the input it was fed with. This module
turns a C record into a small, self-contained Python object so callers never
hold a C pointer that could outlive the parse.
"""


class NodeSourcePosition:
    """Byte range of a node inside the input its parser was given.

    Offsets are zero-based byte positions into the UTF-8 byte stream that was
    parsed, with a half-open range: ``raw[start:end]`` is the exact span the
    node originated from. Offsets - never line/column and never raw pointers -
    are all that is stored, so the object stays valid after the input buffer or
    even the node it describes is released.

    Attributes
    ----------
    origin : str
        Which input stream the offsets refer to:

        * ``"document"`` - a full-document parse; offsets index
          ``LexborHTMLParser.raw_html``.
        * ``"fragment"`` - an ``is_fragment=True`` parse; offsets index that
          fragment input from zero.
        * ``"synthetic"`` - content parsed into the tree afterwards, such as
          an ``inner_html`` assignment; offsets index that new content.
    start : int
        Byte offset of the first byte of the node's source span (inclusive).
    end : int
        Byte offset one past the last byte of the node's source span
        (exclusive).
    """

    __slots__ = ("origin", "start", "end")

    def __init__(self, origin, start, end):
        self.origin = origin
        self.start = start
        self.end = end

    def __repr__(self):
        return (
            "NodeSourcePosition(origin=%r, start=%d, end=%d)"
            % (self.origin, self.start, self.end)
        )

    def __eq__(self, other):
        if not isinstance(other, NodeSourcePosition):
            return NotImplemented
        return (
            self.origin == other.origin
            and self.start == other.start
            and self.end == other.end
        )

    def __hash__(self):
        return hash((self.origin, self.start, self.end))


# Origin enum values exposed to Python as stable strings rather than integers.
_SOURCE_ORIGIN_NAMES = (
    "document",
    "fragment",
    "synthetic",
)


cdef inline object _source_position_from_node(lxb_dom_node_t *node):
    """Build a NodeSourcePosition for a node, or None when it is not locatable.

    A node is not locatable when it was not produced by a position-aware parse:
    freshly created nodes, clones and imported nodes carry no record. The
    accessor also rejects a ``user`` pointer whose magic does not match, so a
    recycled node address can never surface a bogus position.
    """
    cdef lxb_html_source_origin_t origin
    cdef size_t start = 0
    cdef size_t end = 0
    cdef bint found

    if node == NULL:
        return None

    found = selectolax_source_position_get(node, &origin, &start, &end)
    if not found:
        return None

    if <int> origin < 0 or <int> origin >= len(_SOURCE_ORIGIN_NAMES):
        return None

    return NodeSourcePosition(_SOURCE_ORIGIN_NAMES[<int> origin], start, end)
