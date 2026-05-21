"""Tests for services.artifacts and the extended run_detail view."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from django.test import TestCase, override_settings
from django.urls import reverse

from synapse_web.models import AnalysisRun, SourceImageStats
from synapse_web.services import artifacts


def _png_bytes() -> bytes:
    # 1×1 transparent PNG; smallest valid image.
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfa\xcf\x00"
        b"\x00\x00\x02\x00\x01\xe5\x27\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82"
    )


class _MediaTmpMixin:
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.media_root = Path(self._tmp.name)
        self._override = override_settings(
            MEDIA_ROOT=str(self.media_root),
            MEDIA_URL="/media/",
        )
        self._override.enable()

    def tearDown(self):
        self._override.disable()
        self._tmp.cleanup()
        super().tearDown()

    def _make_run(self, output_relpath="runs/abc/output", **kwargs) -> AnalysisRun:
        out_dir = self.media_root / output_relpath
        out_dir.mkdir(parents=True, exist_ok=True)
        defaults = dict(
            status="completed",
            run_kind="cluster_only",
            output_dir=str(out_dir),
            n_clusters=4,
            input_manifest=[],
        )
        defaults.update(kwargs)
        return AnalysisRun.objects.create(**defaults)

    def _add_plot(self, run: AnalysisRun, name: str) -> Path:
        p = Path(run.output_dir) / "plots" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_png_bytes())
        return p


# ---------------------------------------------------------------------------
# media_url_for
# ---------------------------------------------------------------------------

class MediaUrlForTests(_MediaTmpMixin, TestCase):
    def test_served_path_returns_url(self):
        p = self.media_root / "runs" / "x" / "out" / "plots" / "u.png"
        p.parent.mkdir(parents=True)
        p.write_bytes(_png_bytes())
        url = artifacts.media_url_for(p)
        self.assertEqual(url, "/media/runs/x/out/plots/u.png")

    def test_outside_media_root_returns_none(self):
        outside = Path(tempfile.gettempdir()) / "phaseF-outside.png"
        outside.write_bytes(_png_bytes())
        try:
            self.assertIsNone(artifacts.media_url_for(outside))
        finally:
            outside.unlink()

    def test_media_url_without_trailing_slash(self):
        p = self.media_root / "a.png"
        p.write_bytes(_png_bytes())
        with override_settings(MEDIA_URL="/static-media"):
            url = artifacts.media_url_for(p)
        self.assertEqual(url, "/static-media/a.png")

    def test_returns_url_even_for_missing_file_within_media_root(self):
        # media_url_for is pure URL mapping; existence is a caller concern.
        self.assertEqual(
            artifacts.media_url_for(self.media_root / "missing.png"),
            "/media/missing.png",
        )


# ---------------------------------------------------------------------------
# list_plots
# ---------------------------------------------------------------------------

class ListPlotsTests(_MediaTmpMixin, TestCase):
    def test_empty_when_no_output_dir(self):
        run = AnalysisRun.objects.create(status="completed", output_dir="")
        self.assertEqual(artifacts.list_plots(run), [])

    def test_empty_when_plots_dir_missing(self):
        run = self._make_run()
        self.assertEqual(artifacts.list_plots(run), [])

    def test_sorted_and_known_labels(self):
        run = self._make_run()
        self._add_plot(run, "umap_clusters.png")
        self._add_plot(run, "resolution_stability.png")
        self._add_plot(run, "extra_diagnostic.png")
        plots = artifacts.list_plots(run)
        names = [p["name"] for p in plots]
        # Sorted by filename (lower-case).
        self.assertEqual(names, [
            "extra_diagnostic", "resolution_stability", "umap_clusters",
        ])
        labels = {p["name"]: p["label"] for p in plots}
        self.assertEqual(labels["umap_clusters"], "UMAP coloured by cluster")
        # Generic title-case fallback for unknown names.
        self.assertEqual(labels["extra_diagnostic"], "Extra Diagnostic")

    def test_skips_non_png_and_hidden(self):
        run = self._make_run()
        self._add_plot(run, "umap_clusters.png")
        (Path(run.output_dir) / "plots" / "junk.txt").write_text("x")
        (Path(run.output_dir) / "plots" / ".hidden.png").write_bytes(_png_bytes())
        names = [p["name"] for p in artifacts.list_plots(run)]
        self.assertEqual(names, ["umap_clusters"])

    def test_rejects_symlinked_plots_dir(self):
        run = self._make_run()
        real_plots = Path(run.output_dir) / "real_plots"
        real_plots.mkdir()
        (real_plots / "u.png").write_bytes(_png_bytes())
        try:
            os.symlink(real_plots, Path(run.output_dir) / "plots",
                       target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable on this platform")
        self.assertEqual(artifacts.list_plots(run), [])

    def test_url_none_when_output_dir_outside_media_root(self):
        outside = Path(tempfile.mkdtemp())
        try:
            (outside / "plots").mkdir()
            (outside / "plots" / "u.png").write_bytes(_png_bytes())
            run = AnalysisRun.objects.create(
                status="completed", output_dir=str(outside),
            )
            [entry] = artifacts.list_plots(run)
            self.assertIsNone(entry["url"])
            self.assertEqual(entry["name"], "u")
            self.assertEqual(entry["path"], str(outside / "plots" / "u.png"))
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_url_has_cache_buster(self):
        run = self._make_run()
        self._add_plot(run, "umap_clusters.png")
        [entry] = artifacts.list_plots(run)
        self.assertIn("?v=", entry["url"])


# ---------------------------------------------------------------------------
# cluster_palette / cluster_color_map
# ---------------------------------------------------------------------------

class ClusterColorMapTests(TestCase):
    def _stat(self, cid):
        s = SourceImageStats()
        s.dominant_cluster = cid
        return s

    def test_noise_always_grey(self):
        m = artifacts.cluster_color_map([self._stat(-1)])
        self.assertEqual(m[-1], "#9e9e9e")

    def test_distinct_ids_get_distinct_colors_until_cycle(self):
        stats = [self._stat(i) for i in range(5)]
        m = artifacts.cluster_color_map(stats)
        self.assertEqual(len(set(m.values())), 5)

    def test_stable_for_same_inputs(self):
        s = [self._stat(0), self._stat(1), self._stat(2)]
        self.assertEqual(
            artifacts.cluster_color_map(s),
            artifacts.cluster_color_map(s),
        )

    def test_ignores_none(self):
        m = artifacts.cluster_color_map([self._stat(None), self._stat(0)])
        self.assertEqual(set(m.keys()), {0})

    def test_palette_cycles(self):
        pal = artifacts.cluster_palette(15)
        self.assertEqual(len(pal), 15)
        # First entry returns after 12 cycles.
        self.assertEqual(pal[0], pal[12])


# ---------------------------------------------------------------------------
# Run detail view
# ---------------------------------------------------------------------------

class RunDetailWithArtifactsTests(_MediaTmpMixin, TestCase):
    def test_completed_run_with_plots_renders_images(self):
        run = self._make_run(run_kind="cluster_only", n_clusters=3)
        self._add_plot(run, "umap_clusters.png")
        self._add_plot(run, "resolution_stability.png")
        resp = self.client.get(reverse("synapse_web:run_detail", args=[run.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Cluster space")
        self.assertContains(resp, "umap_clusters.png?v=")
        self.assertContains(resp, "UMAP projection of patch embeddings")
        self.assertContains(resp, "Resolution stability")

    def test_group_card_omitted_when_no_group_plots(self):
        run = self._make_run(n_clusters=3)
        self._add_plot(run, "umap_clusters.png")
        resp = self.client.get(reverse("synapse_web:run_detail", args=[run.id]))
        self.assertNotContains(resp, "Group analysis")

    def test_group_card_present_when_at_least_one_group_plot(self):
        run = self._make_run(n_clusters=3)
        self._add_plot(run, "umap_clusters.png")
        self._add_plot(run, "umap_groups.png")
        resp = self.client.get(reverse("synapse_web:run_detail", args=[run.id]))
        self.assertContains(resp, "Group analysis")
        self.assertContains(resp, "UMAP coloured by treatment group")

    def test_not_web_served_message(self):
        outside = Path(tempfile.mkdtemp())
        try:
            (outside / "plots").mkdir()
            (outside / "plots" / "umap_clusters.png").write_bytes(_png_bytes())
            run = AnalysisRun.objects.create(
                status="completed",
                run_kind="cluster_only",
                n_clusters=3,
                output_dir=str(outside),
                input_manifest=[],
            )
            resp = self.client.get(
                reverse("synapse_web:run_detail", args=[run.id])
            )
        finally:
            shutil.rmtree(outside, ignore_errors=True)
        self.assertContains(resp, "Plot files were generated but are not web-served")
        self.assertNotContains(resp, "<img")

    def test_output_dir_missing_alert(self):
        run = self._make_run(n_clusters=3)
        shutil.rmtree(run.output_dir, ignore_errors=True)
        resp = self.client.get(reverse("synapse_web:run_detail", args=[run.id]))
        self.assertContains(resp, "Output directory missing")

    def test_failed_run_with_partial_plots(self):
        run = self._make_run(status="failed", n_clusters=None)
        self._add_plot(run, "umap_clusters.png")
        resp = self.client.get(reverse("synapse_web:run_detail", args=[run.id]))
        self.assertContains(resp, "Partial artifacts from failed run")

    def test_per_source_table_has_color_chip(self):
        run = self._make_run(n_clusters=3)
        SourceImageStats.objects.create(
            analysis_run=run, source_image="A.vsi",
            treatment_group="BAEO", n_patches=3,
            cluster_counts={"0": 3}, dominant_cluster=0,
        )
        resp = self.client.get(reverse("synapse_web:run_detail", args=[run.id]))
        self.assertContains(resp, "background:#")
