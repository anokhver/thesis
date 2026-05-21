"""Tests for the real ML pipeline orchestrator (services.ml_jobs).

Heavy dependencies (torch, monai, etc.) are NOT imported here. The
extraction and clustering internals are mocked; we exercise the JSON
sanitiser, the dominant-cluster picker, the ingestion of synthetic
summary.json + patch_labels.csv fixtures, and the do_work wiring.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from unittest import mock

from django.test import TestCase, TransactionTestCase, override_settings
from django.urls import reverse

from synapse_web.models import AnalysisRun, SourceImageStats
from synapse_web.services import jobs, ml_jobs


# ---------------------------------------------------------------------------
# _to_json_safe
# ---------------------------------------------------------------------------

class ToJsonSafeTests(TestCase):
    def test_basic_types_pass_through(self):
        self.assertEqual(ml_jobs._to_json_safe(None), None)
        self.assertEqual(ml_jobs._to_json_safe(True), True)
        self.assertEqual(ml_jobs._to_json_safe(7), 7)
        self.assertEqual(ml_jobs._to_json_safe("hi"), "hi")
        self.assertEqual(ml_jobs._to_json_safe(1.5), 1.5)

    def test_replaces_builtin_nan_and_inf(self):
        self.assertIsNone(ml_jobs._to_json_safe(float("nan")))
        self.assertIsNone(ml_jobs._to_json_safe(float("inf")))
        self.assertIsNone(ml_jobs._to_json_safe(float("-inf")))

    def test_replaces_numpy_nan_and_converts_scalars(self):
        import numpy as np
        self.assertIsNone(ml_jobs._to_json_safe(np.float32("nan")))
        self.assertIsNone(ml_jobs._to_json_safe(np.inf))
        self.assertEqual(ml_jobs._to_json_safe(np.float64(2.5)), 2.5)
        self.assertEqual(ml_jobs._to_json_safe(np.int64(9)), 9)
        self.assertEqual(ml_jobs._to_json_safe(np.bool_(True)), True)

    def test_recurses_dict_list_array(self):
        import numpy as np
        payload = {
            "a": [1, np.int64(2), float("nan")],
            "b": {"x": np.array([1.0, 2.0])},
            "c": (np.nan, "ok"),
        }
        out = ml_jobs._to_json_safe(payload)
        self.assertEqual(out, {
            "a": [1, 2, None],
            "b": {"x": [1.0, 2.0]},
            "c": [None, "ok"],
        })
        # Must be JSON-serialisable.
        json.dumps(out)


# ---------------------------------------------------------------------------
# _dominant_cluster
# ---------------------------------------------------------------------------

class DominantClusterTests(TestCase):
    def test_picks_majority(self):
        self.assertEqual(ml_jobs._dominant_cluster({"0": 1, "1": 5, "2": 3}), 1)

    def test_excludes_noise_unless_only_noise(self):
        self.assertEqual(ml_jobs._dominant_cluster({"-1": 10, "0": 1}), 0)
        self.assertEqual(ml_jobs._dominant_cluster({"-1": 4}), -1)

    def test_tie_breaks_on_lowest_cluster_id(self):
        self.assertEqual(ml_jobs._dominant_cluster({"3": 5, "1": 5, "2": 5}), 1)

    def test_empty(self):
        self.assertIsNone(ml_jobs._dominant_cluster({}))


# ---------------------------------------------------------------------------
# _ingest_results — uses synthetic fixtures, no torch / clustering
# ---------------------------------------------------------------------------

def _write_summary(out_dir: Path, **overrides) -> None:
    summary = {
        "schema_version": 1,
        "data": {
            "n_patches": 10,
            "n_images": 3,
            "embedding_dim": 768,
            "pca_dim": 32,
            "n_clusters": 4,
        },
        "resolution_pick": {
            "resolution": 0.42,
            "mean_ari": 0.87,
            "n_clusters": 4,
        },
        "junk_nan": float("nan"),
    }
    summary.update(overrides)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            summary, f,
            default=lambda x: None if isinstance(x, float) else str(x),
        )


def _write_patch_labels(out_dir: Path, rows: list[tuple[str, str, int]]) -> None:
    import csv as _csv
    with open(out_dir / "patch_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["filename", "source_image", "cluster"])
        for r in rows:
            w.writerow(r)


class IngestResultsTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.run = AnalysisRun.objects.create(
            status="running",
            run_kind="cluster_only",
            input_manifest=[
                {"source_image": "A.vsi", "treatment_group": "BAEO",
                 "n_patches": 3},
                {"source_image": "B.vsi", "treatment_group": "PSI",
                 "n_patches": 2},
            ],
            output_dir=str(self.out),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_happy_path_populates_run_and_per_source(self):
        _write_summary(self.out)
        _write_patch_labels(self.out, [
            ("0.npy", "A.vsi", 0),
            ("1.npy", "A.vsi", 0),
            ("2.npy", "A.vsi", 1),
            ("3.npy", "B.vsi", 1),
            ("4.npy", "B.vsi", -1),
        ])
        ml_jobs._ingest_results(self.run, self.out)

        self.run.refresh_from_db()
        self.assertEqual(self.run.n_clusters, 4)
        self.assertEqual(self.run.embedding_dim, 768)
        self.assertEqual(self.run.pca_dim, 32)
        self.assertAlmostEqual(self.run.picked_resolution, 0.42)
        self.assertAlmostEqual(self.run.mean_ari, 0.87)
        # NaN sanitised away in the JSON payload.
        self.assertIsNone(self.run.clustering_summary.get("junk_nan"))

        stats = {s.source_image: s for s in self.run.source_stats.all()}
        self.assertEqual(stats["A.vsi"].n_patches, 3)
        self.assertEqual(stats["A.vsi"].cluster_counts, {"0": 2, "1": 1})
        self.assertEqual(stats["A.vsi"].dominant_cluster, 0)
        self.assertEqual(stats["A.vsi"].treatment_group, "BAEO")
        self.assertEqual(stats["B.vsi"].cluster_counts, {"1": 1, "-1": 1})
        # -1 is noise and is excluded when a non-noise cluster exists.
        self.assertEqual(stats["B.vsi"].dominant_cluster, 1)
        self.assertEqual(stats["B.vsi"].treatment_group, "PSI")

    def test_unmatched_source_image_falls_back_to_unknown(self):
        _write_summary(self.out)
        _write_patch_labels(self.out, [
            ("0.npy", "ghost.vsi", 0),
            ("1.npy", "ghost.vsi", 0),
        ])
        ml_jobs._ingest_results(self.run, self.out)
        stats = self.run.source_stats.get(source_image="ghost.vsi")
        self.assertEqual(stats.treatment_group, "UNKNOWN")

    def test_missing_summary_raises(self):
        _write_patch_labels(self.out, [("0.npy", "A.vsi", 0)])
        with self.assertRaises(RuntimeError):
            ml_jobs._ingest_results(self.run, self.out)

    def test_missing_patch_labels_raises(self):
        _write_summary(self.out)
        with self.assertRaises(RuntimeError):
            ml_jobs._ingest_results(self.run, self.out)

    def test_invalid_patch_labels_columns_raises(self):
        _write_summary(self.out)
        with open(self.out / "patch_labels.csv", "w", encoding="utf-8") as f:
            f.write("wrong,cols\nfoo,bar\n")
        with self.assertRaises(RuntimeError):
            ml_jobs._ingest_results(self.run, self.out)


# ---------------------------------------------------------------------------
# _run_clustering — verify the logger handler snapshot/restore
# ---------------------------------------------------------------------------

class RunClusteringHandlerLeakTests(TestCase):
    def test_handlers_added_by_run_clustering_are_removed(self):
        root = logging.getLogger()
        pre = list(root.handlers)

        class _FakeMod:
            @staticmethod
            def parse_args(argv):
                return argv

            @staticmethod
            def run(args):
                root.addHandler(logging.FileHandler(
                    Path(args[1]).parent / "noop.log", mode="w"
                ))
                return 0

        tmp = tempfile.TemporaryDirectory()
        try:
            bundle = Path(tmp.name) / "b"
            outd = Path(tmp.name) / "o"
            bundle.mkdir()
            with mock.patch.object(ml_jobs, "_load_run_clustering",
                                   return_value=_FakeMod):
                ml_jobs._run_clustering(bundle, outd)
                ml_jobs._run_clustering(bundle, outd)
        finally:
            tmp.cleanup()
            # Defensive: remove any handler leaked despite the snapshot.
            for h in list(root.handlers):
                if h not in pre:
                    root.removeHandler(h)
                    try: h.close()
                    except Exception: pass

        self.assertEqual(list(root.handlers), pre)


# ---------------------------------------------------------------------------
# do_work wiring (cluster_only, both internals mocked)
# ---------------------------------------------------------------------------

@override_settings(
    ANALYSIS_JOBS_SYNC=True,
    ANALYSIS_WORK_FN="synapse_web.services.ml_jobs.do_work",
)
class DoWorkClusterOnlyWiringTests(TestCase):
    def test_cluster_only_calls_run_clustering_and_ingest(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        bundle = Path(tmp.name) / "bundle"
        out = Path(tmp.name) / "out"
        bundle.mkdir()

        run = AnalysisRun.objects.create(
            status="pending",
            run_kind="cluster_only",
            bundle_dir=str(bundle),
            output_dir=str(out),
            input_manifest=[
                {"source_image": "A.vsi", "treatment_group": "BAEO",
                 "n_patches": 2},
            ],
        )

        def _fake_run_clustering(bd, od):
            od.mkdir(parents=True, exist_ok=True)
            _write_summary(od)
            _write_patch_labels(od, [
                ("0.npy", "A.vsi", 0),
                ("1.npy", "A.vsi", 1),
            ])

        with mock.patch.object(
            ml_jobs, "_run_clustering", side_effect=_fake_run_clustering
        ), mock.patch.object(
            ml_jobs, "_extract_and_write_bundle"
        ) as ext_mock:
            jobs.submit_run(str(run.id))

        ext_mock.assert_not_called()
        run.refresh_from_db()
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.n_clusters, 4)
        self.assertEqual(run.source_stats.count(), 1)

    def test_extract_and_cluster_calls_extraction(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        bundle = Path(tmp.name) / "bundle"
        out = Path(tmp.name) / "out"
        bundle.mkdir()

        run = AnalysisRun.objects.create(
            status="pending",
            run_kind="extract_and_cluster",
            bundle_dir=str(bundle),
            output_dir=str(out),
            input_dir=str(Path(tmp.name) / "input"),
            checkpoint_path="ignored",
            input_manifest=[
                {"source_image": "A.vsi", "treatment_group": "BAEO",
                 "n_patches": 1},
            ],
        )

        def _fake_run_clustering(bd, od):
            od.mkdir(parents=True, exist_ok=True)
            _write_summary(od)
            _write_patch_labels(od, [("0.npy", "A.vsi", 0)])

        with mock.patch.object(
            ml_jobs, "_extract_and_write_bundle"
        ) as ext_mock, mock.patch.object(
            ml_jobs, "_run_clustering", side_effect=_fake_run_clustering
        ):
            jobs.submit_run(str(run.id))

        ext_mock.assert_called_once()
        run.refresh_from_db()
        self.assertEqual(run.status, "completed")


# ---------------------------------------------------------------------------
# Checkpoint guardrail
# ---------------------------------------------------------------------------

class ValidateCheckpointTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_rejects_empty(self):
        with self.assertRaises(RuntimeError):
            ml_jobs._validate_checkpoint("")

    def test_rejects_missing(self):
        with self.assertRaises(RuntimeError):
            ml_jobs._validate_checkpoint(str(self.tmp / "nope.pt"))

    def test_rejects_oversize(self):
        ck = self.tmp / "big.pt"
        ck.write_bytes(b"x" * 1000)
        with override_settings(MAX_CHECKPOINT_UPLOAD_BYTES=100):
            with self.assertRaises(RuntimeError):
                ml_jobs._validate_checkpoint(str(ck))

    def test_accepts_regular_file(self):
        ck = self.tmp / "ok.pt"
        ck.write_bytes(b"w" * 10)
        with override_settings(MAX_CHECKPOINT_UPLOAD_BYTES=100):
            self.assertEqual(ml_jobs._validate_checkpoint(str(ck)), ck)
