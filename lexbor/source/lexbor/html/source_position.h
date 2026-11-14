/*
 * Copyright (C) 2026 Alexander Borisov
 *
 * Author: Alexander Borisov <borisov@lexbor.com>
 */

#ifndef LEXBOR_HTML_SOURCE_POSITION_H
#define LEXBOR_HTML_SOURCE_POSITION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "lexbor/core/base.h"
#include "lexbor/core/dobject.h"

#include "lexbor/dom/interfaces/node.h"
#include "lexbor/html/interface.h"


/*
 * Signature written into every record so a recycled or foreign ``user``
 * pointer cannot be read as a position record. Chosen outside the range of
 * small allocation-size and node-type values the field otherwise holds.
 */
#define LXB_HTML_SOURCE_POSITION_MAGIC 0x53504F53U /* 'SPOS' */

/*
 * Where a node's position is measured against.
 *
 * A full-document parse and a fragment parse feed different byte streams to
 * the tokenizer, so the same numeric offset means different things in each.
 * The origin records which stream a record belongs to.
 */
typedef enum {
    /* Positions are offsets into the full document input. */
    LXB_HTML_SOURCE_ORIGIN_DOCUMENT  = 0x00,
    /* Positions are offsets into an is_fragment parse input. */
    LXB_HTML_SOURCE_ORIGIN_FRAGMENT  = 0x01,
    /* Positions come from content re-parsed later, e.g. an inner_html set. */
    LXB_HTML_SOURCE_ORIGIN_SYNTHETIC = 0x02,
    LXB_HTML_SOURCE_ORIGIN_LAST_ENTRY
}
lxb_html_source_origin_t;

/*
 * A single source position. Offsets are byte positions into the input the
 * owning parse was fed with; the range is half-open, [start, end).
 *
 * Only offsets are stored, never input pointers: the input buffer may be
 * released once parsing finishes, whereas these records stay alive for as
 * long as the document that owns the registry does.
 */
typedef struct {
    /* Marks records handed out by this module. */
    uint32_t                    magic;
    lxb_html_source_origin_t    origin;
    size_t                      start;
    size_t                      end;
}
lxb_html_source_position_t;

/*
 * Holds every position record created while parsing one document.
 *
 * The records live in a dobject rather than inside document nodes: nodes are
 * allocated from size-keyed pools and a freed address can be handed out again
 * to an unrelated node, so anything referenced from a node has to outlive that
 * recycling. A dobject never moves an allocated object, so a node may keep a
 * stable pointer to its record even after sibling nodes are freed.
 */
typedef struct {
    lexbor_dobject_t *positions;

    /*
     * Which input stream the offsets in this registry refer to. Set when the
     * registry is enabled and left untouched by fragment child documents, which
     * share the owner's registry through their inherited ``user`` field.
     *
     * A re-parse into an existing tree (inner_html) temporarily overrides this
     * with another origin; ``origin_saved`` / ``origin_overridden`` restore the
     * default when the re-parse finishes.
     */
    lxb_html_source_origin_t origin;
    lxb_html_source_origin_t origin_saved;
    bool                    origin_overridden;
}
lxb_html_source_position_registry_t;


LXB_API lxb_html_source_position_registry_t *
lxb_html_source_position_registry_create(void);

LXB_API lxb_status_t
lxb_html_source_position_registry_init(lxb_html_source_position_registry_t *registry);

LXB_API lxb_html_source_position_registry_t *
lxb_html_source_position_registry_destroy(
    lxb_html_source_position_registry_t *registry);

LXB_API lxb_html_source_position_t *
lxb_html_source_position_append(lxb_html_source_position_registry_t *registry,
                                lxb_html_source_origin_t origin,
                                size_t start, size_t end);

/*
 * Attach a freshly created registry to ``document``'s ``user`` field.
 *
 * No-op when a registry is already present, so a fragment parse that inherits
 * the owner document's field keeps sharing one registry. Returns the registry,
 * or NULL on allocation failure or when the field is occupied by foreign data.
 */
LXB_API lxb_html_source_position_registry_t *
lxb_html_document_source_position_enable(lxb_html_document_t *document,
                                         lxb_html_source_origin_t origin);

/*
 * Destroy the registry stored on an owner (non-fragment) document and clear
 * its ``user`` field. Safe to call on a document without a registry.
 */
LXB_API void
lxb_html_document_source_position_disable(lxb_html_document_t *document);

/*
 * Temporarily mark positions recorded during an inner_html-style re-parse.
 *
 * ``document`` is any document sharing the registry (a fragment child document
 * works too): the override is stored on the registry itself. Every begin()
 * between push and pop records the given origin; pop restores the default.
 * Balanced use is required: push before the parse, pop after it.
 */
LXB_API lxb_status_t
lxb_html_document_push_source_origin(lxb_html_document_t *document,
                                     lxb_html_source_origin_t origin);

LXB_API void
lxb_html_document_pop_source_origin(lxb_html_document_t *document);

/*
 * Widen an existing record's right edge while keeping its start. Used when
 * adjacent character tokens are merged into one text node.
 */
LXB_API void
lxb_html_source_position_extend_end(lxb_html_source_position_t *position,
                                    size_t end);

/*
 * Return the registry a document stores in its ``user`` field, or NULL when
 * the field holds something other than a registry (including nothing at all).
 */
lxb_inline lxb_html_source_position_registry_t *
lxb_html_source_position_registry(void *user)
{
    lxb_html_source_position_registry_t *registry = user;

    if (registry == NULL || registry->positions == NULL) {
        return NULL;
    }

    return registry;
}

/*
 * Return the position record a node points at through its ``user`` field, or
 * NULL when the node was not produced by a position-aware parse. The magic
 * check keeps a foreign or recycled pointer from being mistaken for a record.
 */
lxb_inline lxb_html_source_position_t *
lxb_html_source_position_get(const lxb_dom_node_t *node)
{
    lxb_html_source_position_t *position;

    if (node == NULL || node->user == NULL) {
        return NULL;
    }

    position = node->user;

    if (position->magic != LXB_HTML_SOURCE_POSITION_MAGIC) {
        return NULL;
    }

    return position;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LEXBOR_HTML_SOURCE_POSITION_H */
