"""Tests for the background-job runner, run_detail page, stub work,
and the run-artifacts cleanup signal."""
from __future__ import annotations

import time
import uuid as _uuid
from io import StringIO

from django.conf import settings
from django.core.management import call_command
from django.test import TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from synapse_web.models import AnalysisRun, SourceImageStats
from synapse_web.services import jobs


def _manifest_item(source_image="2_5_BAEO_1.vsi", group="BAEO", n_patches=4) -> dict:
    return {
        "source_image": source_image,
        "treatment_group": group,
        "n_patches": n_patches,
    }


def _make_run(items, *, checkpoint="ckpt/x.pt", kind="extract_and_cluster") -> AnalysisRun:
    return AnalysisRun.objects.create(
        status="pending",
        run_kind=kind,
        checkpoint_path=checkpoint,
        config_snapshot={"checkpoint_path": checkpoint, "image_count": len(items)},
        input_manifest=items,
    )


@override_settings(ANALYSIS_JOBS_SYNC=True)
class JobRunnerSyncTests(TestCase):
    def test_stub_run_completes_and_creates_source_stats(self):
        items = [_manifest_item("a.vsi", "BAEO"), _manifest_item("b.vsi", "PSI", 7)]
        run = _make_run(items)
        jobs.submit_run(str(run.id))

        run.refresh_from_db()
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.progress, 100)
        self.assertIsNotNone(run.started_at)
        self.assertIsNotNone(run.finished_at)
        self.assertEqual(run.source_stats.count(), 2)
        by_name = {s.source_image: s for s in run.source_stats.all()}
        self.assertEqual(by_name["a.vsi"].treatment_group, "BAEO")
        self.assertEqual(by_name["b.vsi"].n_patches, 7)

    def test_idempotent_rerun_does_not_duplicate_stats(self):
        run = _make_run([_manifest_item("a.vsi")])
        jobs.submit_run(str(run.id))
        jobs.submit_run(str(run.id))
        self.assertEqual(SourceImageStats.objects.filter(analysis_run=run).count(), 1)

    @override_settings(
        ANALYSIS_WORK_FN="synapse_web.tests_jobs._raising_work"
    )
    def test_failure_records_short_message_and_traceback(self):
        run = _make_run([_manifest_item("a.vsi")])
        jobs.submit_run(str(run.id))
        run.refresh_from_db()
        self.assertEqual(run.status, "failed")
        self.assertIsNotNone(run.finished_at)
        self.assertIn("boom", run.error_message)
        self.assertLessEqual(len(run.error_message), 500)
        self.assertIn("Traceback", run.error_traceback)
        self.assertNotEqual(run.progress, 100)


def _raising_work(**kwargs):
    raise RuntimeError("boom")


class ProgressReporterTests(TestCase):
    def test_throttles_no_op_writes(self):
        run = _make_run([_manifest_item()])
        reporter = jobs.ProgressReporter(str(run.id))

        reporter.update(progress=10, message="hi")
        reporter.update(progress=10, message="hi")
        run.refresh_from_db()
        self.assertEqual(run.progress, 10)
        self.assertEqual(run.progress_message, "hi")

        reporter.update(progress=37)
        run.refresh_from_db()
        self.assertEqual(run.progress, 37)
        self.assertEqual(run.progress_message, "hi")

    def test_clamps_progress_to_0_100(self):
        run = _make_run([_manifest_item()])
        reporter = jobs.ProgressReporter(str(run.id))
        reporter.update(progress=-5)
        run.refresh_from_db()
        self.assertEqual(run.progress, 0)
        reporter.update(progress=999)
        run.refresh_from_db()
        self.assertEqual(run.progress, 100)


@override_settings(ANALYSIS_JOBS_SYNC=True)
class RunDetailViewTests(TestCase):
    def test_run_detail_renders_for_completed_run(self):
        run = _make_run([_manifest_item("a.vsi"), _manifest_item("b.vsi", "PSI")])
        jobs.submit_run(str(run.id))
        resp = self.client.get(reverse("synapse_web:run_detail", args=[run.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Per-source-image stats")
        self.assertContains(resp, "a.vsi")
        self.assertContains(resp, "b.vsi")

    def test_run_detail_renders_for_pending_run(self):
        run = _make_run([_manifest_item()])
        resp = self.client.get(reverse("synapse_web:run_detail", args=[run.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "poll()")

    def test_run_detail_404_for_unknown_id(self):
        resp = self.client.get(
            reverse("synapse_web:run_detail", args=[_uuid.uuid4()])
        )
        self.assertEqual(resp.status_code, 404)


class RunStatusJsonTests(TestCase):
    def test_status_shape_and_no_cache(self):
        run = _make_run([_manifest_item(), _manifest_item("b.vsi")])
        resp = self.client.get(reverse("synapse_web:run_status", args=[run.id]))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for key in (
            "status", "progress", "progress_message",
            "result_count", "expected_count", "finished", "error_message",
        ):
            self.assertIn(key, data)
        self.assertEqual(data["status"], "pending")
        self.assertEqual(data["expected_count"], 2)
        self.assertEqual(data["result_count"], 0)
        self.assertFalse(data["finished"])
        cache = resp.get("Cache-Control", "")
        self.assertTrue(
            "no-store" in cache or "no-cache" in cache,
            f"expected no-store/no-cache header, got {cache!r}",
        )

    @override_settings(ANALYSIS_JOBS_SYNC=True)
    def test_finished_flag_flips_after_completion(self):
        run = _make_run([_manifest_item()])
        jobs.submit_run(str(run.id))
        data = self.client.get(
            reverse("synapse_web:run_status", args=[run.id])
        ).json()
        self.assertEqual(data["status"], "completed")
        self.assertTrue(data["finished"])
        self.assertEqual(data["result_count"], data["expected_count"])


class RecoverStaleRunsCommandTests(TestCase):
    def test_only_fails_running_rows_older_than_threshold(self):
        old = AnalysisRun.objects.create(
            status="running",
            started_at=timezone.now() - timezone.timedelta(hours=2),
        )
        recent = AnalysisRun.objects.create(
            status="running", started_at=timezone.now()
        )
        completed = AnalysisRun.objects.create(
            status="completed",
            started_at=timezone.now() - timezone.timedelta(hours=2),
            finished_at=timezone.now() - timezone.timedelta(hours=2),
        )

        out = StringIO()
        call_command("recover_stale_runs", stdout=out)

        old.refresh_from_db(); recent.refresh_from_db(); completed.refresh_from_db()
        self.assertEqual(old.status, "failed")
        self.assertEqual(old.error_message, "Server restart — job orphaned.")
        self.assertIsNotNone(old.finished_at)
        self.assertEqual(recent.status, "running")
        self.assertEqual(completed.status, "completed")
        self.assertIn("Recovered 1", out.getvalue())

    def test_custom_threshold(self):
        run = AnalysisRun.objects.create(
            status="running",
            started_at=timezone.now() - timezone.timedelta(minutes=5),
        )
        call_command("recover_stale_runs", "--older-than=1")
        run.refresh_from_db()
        self.assertEqual(run.status, "failed")


class RunHistoryClickableTests(TestCase):
    def test_history_links_to_run_detail(self):
        run = _make_run([_manifest_item()])
        resp = self.client.get(reverse("synapse_web:results"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, reverse("synapse_web:run_detail", args=[run.id]))


class RunArtifactsCleanupTests(TestCase):
    def test_delete_run_removes_filesystem_artifacts(self):
        from synapse_web.services import run_artifacts as ra

        run = _make_run([_manifest_item()])
        input_dir, bundle_dir, output_dir = ra.ensure_run_dirs(run)
        (output_dir / "marker.txt").write_text("hi", encoding="utf-8")
        self.assertTrue(ra.run_root(run).exists())

        run.delete()
        self.assertFalse(ra.run_root(run).exists())

    def test_runs_root_path_traversal_guard(self):
        from synapse_web.services import run_artifacts as ra

        class FakeRun:
            id = "../../etc"

        self.assertFalse(ra.delete_run_artifacts(FakeRun()))


class ThreadedSubmitSmokeTest(TransactionTestCase):
    """Spin a real ThreadPoolExecutor job and verify it finishes via polling.

    Uses TransactionTestCase because the worker thread needs to see the
    committed AnalysisRun row — Django's default TestCase wraps every test
    in a transaction that other threads can't read.
    """

    def test_threaded_run_completes(self):
        run = _make_run([_manifest_item()])
        future = jobs.submit_run(str(run.id))
        self.assertIsNotNone(future, "expected a Future in async mode")

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            run.refresh_from_db()
            if run.status == "completed":
                break
            time.sleep(0.05)
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.source_stats.count(), 1)
