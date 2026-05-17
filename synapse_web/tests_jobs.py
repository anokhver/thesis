"""Tests for the background-job runner, run_detail page, and stub work."""
from __future__ import annotations

import time
from io import StringIO

from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from synapse_web.models import AnalysisRun, ImageResult, MicroscopyImage
from synapse_web.services import jobs


def _image(name="2_5_BAEO_1.vsi", group="BAEO") -> MicroscopyImage:
    f = SimpleUploadedFile(name, b"fake", content_type="application/octet-stream")
    img = MicroscopyImage(original_filename=name, treatment_group=group)
    img.file.save(name, f, save=True)
    return img


def _make_run(images, checkpoint="ckpt/x.pt") -> AnalysisRun:
    manifest = [
        {
            "image_id": str(img.id),
            "original_filename": img.original_filename,
            "treatment_group": img.treatment_group,
            "path": img.file.name,
        }
        for img in images
    ]
    return AnalysisRun.objects.create(
        status="pending",
        checkpoint_path=checkpoint,
        config_snapshot={"checkpoint_path": checkpoint, "image_count": len(manifest)},
        image_manifest=manifest,
    )


@override_settings(ANALYSIS_JOBS_SYNC=True)
class JobRunnerSyncTests(TestCase):
    def test_stub_run_completes_and_creates_results(self):
        imgs = [_image("a.vsi", "BAEO"), _image("b.vsi", "PSI")]
        run = _make_run(imgs)
        jobs.submit_run(str(run.id))

        run.refresh_from_db()
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.progress, 100)
        self.assertIsNotNone(run.started_at)
        self.assertIsNotNone(run.finished_at)
        self.assertEqual(run.results.count(), 2)
        for r in run.results.all():
            self.assertEqual(r.puncta_count, 0)
            self.assertEqual(r.puncta_density, 0.0)

    def test_idempotent_rerun_does_not_duplicate_results(self):
        imgs = [_image("a.vsi")]
        run = _make_run(imgs)
        jobs.submit_run(str(run.id))
        jobs.submit_run(str(run.id))
        self.assertEqual(ImageResult.objects.filter(analysis_run=run).count(), 1)

    @override_settings(
        ANALYSIS_WORK_FN="synapse_web.tests_jobs._raising_work"
    )
    def test_failure_records_short_message_and_traceback(self):
        run = _make_run([_image("a.vsi")])
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
        img = _image()
        run = _make_run([img])
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
        run = _make_run([_image()])
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
        run = _make_run([_image("a.vsi"), _image("b.vsi")])
        jobs.submit_run(str(run.id))
        resp = self.client.get(reverse("synapse_web:run_detail", args=[run.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Per-image results")
        self.assertContains(resp, "a.vsi")
        self.assertContains(resp, "b.vsi")
        self.assertContains(resp, "Summary by treatment group")

    def test_run_detail_renders_for_pending_run(self):
        run = _make_run([_image()])
        resp = self.client.get(reverse("synapse_web:run_detail", args=[run.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "poll()")

    def test_run_detail_404_for_unknown_id(self):
        import uuid as _uuid
        resp = self.client.get(
            reverse("synapse_web:run_detail", args=[_uuid.uuid4()])
        )
        self.assertEqual(resp.status_code, 404)


class RunStatusJsonTests(TestCase):
    def test_status_shape_and_no_cache(self):
        run = _make_run([_image(), _image("b.vsi")])
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
        run = _make_run([_image()])
        jobs.submit_run(str(run.id))
        data = self.client.get(
            reverse("synapse_web:run_status", args=[run.id])
        ).json()
        self.assertEqual(data["status"], "completed")
        self.assertTrue(data["finished"])
        self.assertEqual(data["result_count"], data["expected_count"])


@override_settings(ANALYSIS_JOBS_SYNC=True)
class RunInferenceSubmissionTests(TestCase):
    def test_submission_creates_run_with_manifest_and_redirects_to_detail(self):
        img = _image("a.vsi", "PSI")
        resp = self.client.post(
            reverse("synapse_web:run_inference"),
            data={
                "image_ids": [str(img.id)],
                "checkpoint_path": "ckpt/best.pt",
            },
        )
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/analysis/runs/", resp.url)

        run = AnalysisRun.objects.get()
        self.assertEqual(len(run.image_manifest), 1)
        self.assertEqual(run.image_manifest[0]["original_filename"], "a.vsi")
        self.assertEqual(run.image_manifest[0]["treatment_group"], "PSI")
        self.assertEqual(run.checkpoint_path, "ckpt/best.pt")
        self.assertEqual(run.status, "completed")

    def test_submission_rejects_unknown_image_ids(self):
        import uuid as _uuid
        resp = self.client.post(
            reverse("synapse_web:run_inference"),
            data={"image_ids": [str(_uuid.uuid4())]},
            follow=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(AnalysisRun.objects.count(), 0)
        self.assertContains(resp, "no longer exist")

    def test_submission_warns_when_no_image_selected(self):
        resp = self.client.post(
            reverse("synapse_web:run_inference"),
            data={},
            follow=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "No images selected.")


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
        img = _image()
        run = _make_run([img])
        resp = self.client.get(reverse("synapse_web:results"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, reverse("synapse_web:run_detail", args=[run.id]))


class ThreadedSubmitSmokeTest(TransactionTestCase):
    """Spin a real ThreadPoolExecutor job and verify it finishes via polling.

    Uses TransactionTestCase because the worker thread needs to see the
    committed AnalysisRun row — Django's default TestCase wraps every test
    in a transaction that other threads can't read.
    """

    def test_threaded_run_completes(self):
        img = _image()
        run = _make_run([img])
        future = jobs.submit_run(str(run.id))
        self.assertIsNotNone(future, "expected a Future in async mode")

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            run.refresh_from_db()
            if run.status == "completed":
                break
            time.sleep(0.05)
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.results.count(), 1)
