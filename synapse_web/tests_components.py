"""Tests for the reusable UI building blocks in synapse_web.templatetags.ui."""
from django.template import Context, Template, TemplateSyntaxError
from django.test import TestCase, override_settings


def _render(src: str, ctx: dict | None = None) -> str:
    return Template("{% load ui %}" + src).render(Context(ctx or {}))


class CardTagTests(TestCase):
    def test_renders_card_with_title_and_body(self):
        out = _render(
            '{% card title="Hello" %}<p>body</p>{% endcard %}'
        )
        self.assertIn('<div class="card"', out)
        self.assertIn("<h2>Hello</h2>", out)
        self.assertIn("<p>body</p>", out)

    def test_card_without_title_omits_heading(self):
        out = _render("{% card %}<p>body</p>{% endcard %}")
        self.assertIn('<div class="card"', out)
        self.assertNotIn("<h2>", out)
        self.assertIn("<p>body</p>", out)

    def test_card_title_is_escaped(self):
        out = _render(
            '{% card title=t %}x{% endcard %}',
            {"t": "<script>alert(1)</script>"},
        )
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)

    def test_card_margin_kwarg_becomes_inline_style(self):
        out = _render('{% card title="X" margin="2rem" %}body{% endcard %}')
        self.assertIn("margin-top: 2rem", out)

    def test_card_positional_argument_is_rejected(self):
        with self.assertRaises(TemplateSyntaxError):
            _render('{% card "Hello" %}body{% endcard %}')


class EmptyStateTagTests(TestCase):
    def test_wraps_body_in_empty_state_div(self):
        out = _render("{% empty_state %}<p>nothing</p>{% endempty_state %}")
        self.assertIn('<div class="empty-state">', out)
        self.assertIn("<p>nothing</p>", out)


class GraphBoxTagTests(TestCase):
    def test_shows_placeholder_when_empty(self):
        out = _render(
            '{% graph_box title="Density" %}{% endgraph_box %}'
        )
        self.assertIn('<div class="card">', out)
        self.assertIn("<h2>Density</h2>", out)
        self.assertIn('<div class="plot-container">', out)
        self.assertIn("Chart placeholder", out)

    def test_renders_body_when_present(self):
        out = _render(
            '{% graph_box title="Density" %}<svg id="chart"></svg>'
            "{% endgraph_box %}"
        )
        self.assertIn("<svg id=\"chart\"></svg>", out)
        self.assertNotIn("Chart placeholder", out)

    def test_caption_is_rendered_below_plot(self):
        out = _render(
            '{% graph_box title="X" caption="A nice note" %}'
            "{% endgraph_box %}"
        )
        self.assertIn("A nice note", out)


class KvTableTagTests(TestCase):
    def test_renders_dict(self):
        out = _render(
            "{% kv_table data %}",
            {"data": {"alpha": 1, "beta": 2}},
        )
        self.assertIn('<table class="data-table">', out)
        self.assertIn("alpha", out)
        self.assertIn("beta", out)
        self.assertIn(">1<", out)
        self.assertIn(">2<", out)
        self.assertNotIn("<thead>", out)

    def test_renders_list_of_pairs_with_header(self):
        out = _render(
            '{% kv_table data key_label="Key" value_label="Value" %}',
            {"data": [("a", "x"), ("b", "y")]},
        )
        self.assertIn("<thead>", out)
        self.assertIn("Key", out)
        self.assertIn("Value", out)
        self.assertIn(">a<", out)
        self.assertIn(">x<", out)


@override_settings(THESIS_PDF_URL="https://example.test/thesis.pdf")
class ThesisLinkTagTests(TestCase):
    def test_uses_settings_url_and_chapter(self):
        out = _render('{% thesis_link chapter="Data" %}')
        self.assertIn("https://example.test/thesis.pdf", out)
        self.assertIn("<em>Data</em>", out)
        self.assertIn("See more in the", out)

    def test_custom_lead(self):
        out = _render('{% thesis_link chapter="Methods" lead="Defined in the" %}')
        self.assertIn("Defined in the", out)


class FileLinkTagTests(TestCase):
    def test_shows_dash_when_no_file(self):
        out = _render("{% file_link f %}", {"f": None})
        self.assertNotIn("<a ", out)
        self.assertIn("&mdash;", out)

    def test_shows_link_when_file_present(self):
        class FakeFile:
            url = "/media/foo.png"

            def __bool__(self):
                return True

        out = _render(
            '{% file_link f label="Open" %}', {"f": FakeFile()}
        )
        self.assertIn('href="/media/foo.png"', out)
        self.assertIn(">Open</a>", out)


class StatusBadgeTagTests(TestCase):
    def test_renders_status_class_and_text(self):
        out = _render('{% status_badge "completed" %}')
        self.assertIn("badge badge-completed", out)
        self.assertIn(">completed</span>", out)

    def test_unknown_when_blank(self):
        out = _render('{% status_badge "" %}')
        self.assertIn("badge badge-unknown", out)


class VersionedStaticTagTests(TestCase):
    def test_appends_mtime_query_to_existing_static_file(self):
        out = _render("{% versioned_static 'style.css' %}")
        self.assertTrue(out.startswith("/static/style.css?v="))
        version = out.split("?v=", 1)[1]
        self.assertTrue(version.isdigit())

    def test_returns_plain_url_for_missing_file(self):
        out = _render("{% versioned_static 'no_such_file.css' %}")
        self.assertEqual(out, "/static/no_such_file.css")
