"""Reusable UI building blocks.

Inclusion tags render a partial with a known data shape; block tags wrap
arbitrary template content. Templates stay declarative — anything that
needs a decision lives in Python here.
"""
from __future__ import annotations

import os
import re

from django import template
from django.conf import settings
from django.contrib.staticfiles import finders
from django.templatetags.static import static
from django.utils.html import format_html
from django.utils.safestring import mark_safe

register = template.Library()


@register.inclusion_tag("synapse_web/components/kv_table.html")
def kv_table(rows, key_label="", value_label=""):
    """Two-column key/value table.

    `rows` may be a dict or any iterable of (key, value) pairs.
    """
    if hasattr(rows, "items"):
        rows = list(rows.items())
    else:
        rows = list(rows)
    return {
        "rows": rows,
        "key_label": key_label,
        "value_label": value_label,
        "has_header": bool(key_label or value_label),
    }


@register.inclusion_tag("synapse_web/components/thesis_link.html")
def thesis_link(chapter, lead="See more in the"):
    """Link to a chapter of the thesis PDF.

    Use this anywhere the frontend references the thesis text — rule 7
    requires chapter-name + PDF-link, never repo file paths.
    """
    return {
        "chapter": chapter,
        "lead": lead,
        "thesis_pdf_url": settings.THESIS_PDF_URL,
    }


@register.inclusion_tag("synapse_web/components/file_link.html")
def file_link(file_field, label="View"):
    """Render a FileField as a link, or an em-dash when empty."""
    return {"file_field": file_field, "label": label}


@register.inclusion_tag("synapse_web/components/file_thumb.html")
def file_thumb(file_field, label="image"):
    """Render an ImageField as an inline thumbnail linked to full size."""
    return {"file_field": file_field, "label": label}


@register.inclusion_tag("synapse_web/components/status_badge.html")
def status_badge(status):
    return {"status": status or "unknown"}


@register.simple_tag
def versioned_static(path):
    """Return the static URL with a ?v=<mtime> cache-buster appended.

    Survives runserver's no-cache-headers policy so CSS edits always
    take effect on the next page load.
    """
    url = static(path)
    abs_path = finders.find(path)
    if abs_path and os.path.isfile(abs_path):
        version = int(os.path.getmtime(abs_path))
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}v={version}"
    return url


_KWARG_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+)$")


class _BlockNode(template.Node):
    def __init__(self, nodelist, kwargs, render_fn, name):
        self.nodelist = nodelist
        self.kwargs = kwargs
        self.render_fn = render_fn
        self.name = name

    def render(self, context):
        resolved = {k: v.resolve(context) for k, v in self.kwargs.items()}
        body = self.nodelist.render(context).strip()
        return self.render_fn(body, **resolved)


def _make_block_tag(name, end_name, render_fn):
    def parse(parser, token):
        bits = token.split_contents()[1:]
        kwargs = {}
        for bit in bits:
            match = _KWARG_RE.match(bit)
            if not match:
                raise template.TemplateSyntaxError(
                    f"{{% {name} %}} only accepts keyword arguments, got {bit!r}"
                )
            kwargs[match.group(1)] = parser.compile_filter(match.group(2))
        nodelist = parser.parse((end_name,))
        parser.delete_first_token()
        return _BlockNode(nodelist, kwargs, render_fn, name)

    parse.__name__ = name
    register.tag(name)(parse)


def _style(**declarations) -> str:
    parts = [f"{k.replace('_', '-')}: {v}" for k, v in declarations.items() if v]
    return "; ".join(parts)


def _render_card(body, title=None, margin=None, max_width=None, centered=False):
    style = _style(
        margin_top=margin,
        max_width=max_width,
        margin_left="auto" if centered else None,
        margin_right="auto" if centered else None,
    )
    style_attr = format_html(' style="{};"', style) if style else ""
    title_html = format_html("<h2>{}</h2>", title) if title else ""
    return format_html(
        '<div class="card"{}>{}{}</div>',
        style_attr,
        mark_safe(title_html),
        mark_safe(body),
    )


def _render_section(body, title=None, level=2, margin=None):
    style = _style(margin_top=margin)
    style_attr = format_html(' style="{};"', style) if style else ""
    title_html = (
        format_html("<h{0}>{1}</h{0}>", level, title) if title else ""
    )
    return format_html(
        "<section{}>{}{}</section>",
        style_attr,
        mark_safe(title_html),
        mark_safe(body),
    )


def _render_empty_state(body):
    return format_html('<div class="empty-state">{}</div>', mark_safe(body))


_GRAPH_PLACEHOLDER = mark_safe(
    '<p style="color: var(--text-secondary);">'
    "Chart placeholder &mdash; visualization will be added here."
    "</p>"
)


def _render_graph_box(body, title=None, caption=None):
    title_html = format_html("<h2>{}</h2>", title) if title else ""
    inner = mark_safe(body) if body else _GRAPH_PLACEHOLDER
    caption_html = (
        format_html(
            '<p style="color: var(--text-secondary); '
            'font-size: 0.85rem; margin-top: 0.5rem;">{}</p>',
            caption,
        )
        if caption
        else ""
    )
    return format_html(
        '<div class="card">{}<div class="plot-container">{}</div>{}</div>',
        mark_safe(title_html),
        inner,
        mark_safe(caption_html),
    )


_make_block_tag("card", "endcard", _render_card)
_make_block_tag("section_block", "endsection_block", _render_section)
_make_block_tag("empty_state", "endempty_state", _render_empty_state)
_make_block_tag("graph_box", "endgraph_box", _render_graph_box)
