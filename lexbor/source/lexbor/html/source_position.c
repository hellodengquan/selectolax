/*
 * Copyright (C) 2026 Alexander Borisov
 *
 * Author: Alexander Borisov <borisov@lexbor.com>
 */

#include "lexbor/html/source_position.h"
#include "lexbor/html/interfaces/document.h"
#include "lexbor/dom/interfaces/document.h"
#include "lexbor/core/mem.h"


lxb_html_source_position_registry_t *
lxb_html_source_position_registry_create(void)
{
    return lexbor_calloc(1, sizeof(lxb_html_source_position_registry_t));
}

lxb_status_t
lxb_html_source_position_registry_init(
    lxb_html_source_position_registry_t *registry)
{
    lxb_status_t status;

    if (registry == NULL) {
        return LXB_STATUS_ERROR_OBJECT_IS_NULL;
    }

    registry->positions = lexbor_dobject_create();
    if (registry->positions == NULL) {
        return LXB_STATUS_ERROR_MEMORY_ALLOCATION;
    }

    /*
     * Fixed-size objects: a dobject grows by appending chunks and never moves
     * an object it has already handed out, so node->user may keep pointing at
     * a record while later records are allocated.
     */
    status = lexbor_dobject_init(registry->positions, 4096,
                                 sizeof(lxb_html_source_position_t));
    if (status != LXB_STATUS_OK) {
        lexbor_dobject_destroy(registry->positions, true);
        registry->positions = NULL;

        return status;
    }

    return LXB_STATUS_OK;
}

lxb_html_source_position_registry_t *
lxb_html_source_position_registry_destroy(
    lxb_html_source_position_registry_t *registry)
{
    if (registry == NULL) {
        return NULL;
    }

    if (registry->positions != NULL) {
        lexbor_dobject_destroy(registry->positions, true);
    }

    return lexbor_free(registry);
}

lxb_html_source_position_t *
lxb_html_source_position_append(lxb_html_source_position_registry_t *registry,
                                lxb_html_source_origin_t origin,
                                size_t start, size_t end)
{
    lxb_html_source_position_t *position;

    if (registry == NULL || registry->positions == NULL) {
        return NULL;
    }

    position = lexbor_dobject_calloc(registry->positions);
    if (position == NULL) {
        return NULL;
    }

    position->magic = LXB_HTML_SOURCE_POSITION_MAGIC;
    position->origin = origin;
    position->start = start;
    position->end = end;

    return position;
}

void
lxb_html_source_position_extend_end(lxb_html_source_position_t *position,
                                    size_t end)
{
    if (position == NULL || end < position->end) {
        return;
    }

    position->end = end;
}

lxb_html_source_position_registry_t *
lxb_html_document_source_position_enable(lxb_html_document_t *document,
                                         lxb_html_source_origin_t origin)
{
    lxb_status_t status;
    lxb_dom_document_t *dom_document;
    lxb_html_source_position_registry_t *registry;

    if (document == NULL) {
        return NULL;
    }

    dom_document = lxb_dom_interface_document(document);

    /*
     * A fragment parse builds a child document that inherits the owner's
     * ``user`` field in lxb_dom_document_init(). Sharing one registry is what
     * lets fragment nodes point at records owned by the parser's document.
     */
    registry = lxb_html_source_position_registry(dom_document->user);
    if (registry != NULL) {
        return registry;
    }

    if (dom_document->user != NULL) {
        /* Do not overwrite foreign data stored by an embedding application. */
        return NULL;
    }

    registry = lxb_html_source_position_registry_create();
    if (registry == NULL) {
        return NULL;
    }

    status = lxb_html_source_position_registry_init(registry);
    if (status != LXB_STATUS_OK) {
        lxb_html_source_position_registry_destroy(registry);
        return NULL;
    }

    registry->origin = origin;
    dom_document->user = registry;

    return registry;
}

void
lxb_html_document_source_position_disable(lxb_html_document_t *document)
{
    lxb_dom_document_t *dom_document;
    lxb_html_source_position_registry_t *registry;

    if (document == NULL) {
        return;
    }

    dom_document = lxb_dom_interface_document(document);
    registry = lxb_html_source_position_registry(dom_document->user);

    if (registry == NULL) {
        return;
    }

    dom_document->user = NULL;
    lxb_html_source_position_registry_destroy(registry);
}

lxb_status_t
lxb_html_document_push_source_origin(lxb_html_document_t *document,
                                     lxb_html_source_origin_t origin)
{
    lxb_html_source_position_registry_t *registry;
    lxb_dom_document_t *dom_document;

    if (document == NULL) {
        return LXB_STATUS_ERROR_OBJECT_IS_NULL;
    }

    dom_document = lxb_dom_interface_document(document);
    registry = lxb_html_source_position_registry(dom_document->user);
    if (registry == NULL) {
        return LXB_STATUS_ERROR;
    }

    if (!registry->origin_overridden) {
        registry->origin_saved = registry->origin;
        registry->origin_overridden = true;
    }
    /* Nested pushes keep the outermost saved value and just change the active
     * origin, mirroring a stack of depth two. */
    registry->origin = origin;

    return LXB_STATUS_OK;
}

void
lxb_html_document_pop_source_origin(lxb_html_document_t *document)
{
    lxb_html_source_position_registry_t *registry;
    lxb_dom_document_t *dom_document;

    if (document == NULL) {
        return;
    }

    dom_document = lxb_dom_interface_document(document);
    registry = lxb_html_source_position_registry(dom_document->user);
    if (registry == NULL || !registry->origin_overridden) {
        return;
    }

    registry->origin = registry->origin_saved;
    registry->origin_overridden = false;
}
