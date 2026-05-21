"""Tests for the checkpoint registry: service, form, and views."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from synapse_web.forms import CheckpointUploadForm
from synapse_web.services import checkpoints as svc


def _upload(name: str, content: bytes = b"weights") -> SimpleUploadedFile:
    return SimpleUploadedFile(
        name, content, content_type="application/octet-stream"
    )


class _CheckpointDirMixin:
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self._override = override_settings(
            CHECKPOINT_DIR=self.tmp_path,
            MAX_CHECKPOINT_UPLOAD_BYTES=1024 * 1024,
        )
        self._override.enable()

    def tearDown(self):
        self._override.disable()
        self._tmp.cleanup()
        super().tearDown()


class ListCheckpointsTests(_CheckpointDirMixin, TestCase):
    def test_returns_empty_when_dir_missing(self):
        self._tmp.cleanup()  # remove the dir on disk
        # Re-create the TemporaryDirectory machinery so tearDown's
        # cleanup() doesn't blow up.
        self._tmp = tempfile.TemporaryDirectory()
        with override_settings(CHECKPOINT_DIR=Path(self._tmp.name) / "nope"):
            self.assertEqual(svc.list_checkpoints(), [])

    def test_lists_only_pt_and_pth_sorted_by_mtime_desc(self):
        (self.tmp_path / "old.pt").write_bytes(b"a")
        time.sleep(0.02)
        (self.tmp_path / "new.pth").write_bytes(b"b")
        (self.tmp_path / "ignored.bin").write_bytes(b"c")
        names = [c["name"] for c in svc.list_checkpoints()]
        self.assertEqual(names, ["new.pth", "old.pt"])

    def test_skips_hidden_partial_and_non_pt(self):
        (self.tmp_path / ".hidden.pt").write_bytes(b"a")
        (self.tmp_path / "x.pt.partial").write_bytes(b"a")
        (self.tmp_path / "model.txt").write_bytes(b"a")
        (self.tmp_path / "kept.pt").write_bytes(b"a")
        names = [c["name"] for c in svc.list_checkpoints()]
        self.assertEqual(names, ["kept.pt"])

    def test_skips_symlinks(self):
        target = self.tmp_path / "real.pt"
        target.write_bytes(b"a")
        link = self.tmp_path / "alias.pt"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable on this platform")
        names = [c["name"] for c in svc.list_checkpoints()]
        self.assertEqual(names, ["real.pt"])

    def test_contract_keys_and_size_display(self):
        (self.tmp_path / "model.pt").write_bytes(b"x" * 2048)
        [entry] = svc.list_checkpoints()
        self.assertEqual(
            set(entry.keys()),
            {"name", "relative_path", "size_bytes", "size_display", "modified_at"},
        )
        self.assertEqual(entry["size_bytes"], 2048)
        self.assertIn("KB", entry["size_display"])


class ValidateUploadTests(_CheckpointDirMixin, TestCase):
    def test_accepts_pt_and_pth_and_uppercase(self):
        self.assertEqual(svc.validate_checkpoint_upload(_upload("a.pt")), "a.pt")
        self.assertEqual(svc.validate_checkpoint_upload(_upload("b.PTH")), "b.pth")

    def test_rejects_wrong_suffix(self):
        with self.assertRaises(ValidationError):
            svc.validate_checkpoint_upload(_upload("model.bin"))

    def test_rejects_path_separators(self):
        class _Stub:
            size = 4
            def __init__(self, name): self.name = name

        with self.assertRaises(ValidationError):
            svc.validate_checkpoint_upload(_Stub("../escape.pt"))
        with self.assertRaises(ValidationError):
            svc.validate_checkpoint_upload(_Stub("dir\\evil.pt"))
        with self.assertRaises(ValidationError):
            svc.validate_checkpoint_upload(_Stub("dir/inner.pt"))

    def test_rejects_empty_stem(self):
        with self.assertRaises(ValidationError):
            svc.validate_checkpoint_upload(_upload(".pt"))
        with self.assertRaises(ValidationError):
            svc.validate_checkpoint_upload(_upload("***.pt"))

    def test_rejects_oversize(self):
        with override_settings(MAX_CHECKPOINT_UPLOAD_BYTES=10):
            with self.assertRaises(ValidationError):
                svc.validate_checkpoint_upload(_upload("a.pt", b"x" * 100))

    def test_rejects_duplicate_name(self):
        (self.tmp_path / "a.pt").write_bytes(b"x")
        with self.assertRaises(ValidationError):
            svc.validate_checkpoint_upload(_upload("A.pt"))

    def test_slugifies_stem(self):
        name = svc.validate_checkpoint_upload(_upload("My Model v1.pt"))
        self.assertEqual(name, "my-model-v1.pt")

    def test_no_filesystem_side_effects(self):
        before = list(self.tmp_path.iterdir())
        svc.validate_checkpoint_upload(_upload("a.pt"))
        after = list(self.tmp_path.iterdir())
        self.assertEqual(before, after)


class SaveUploadedTests(_CheckpointDirMixin, TestCase):
    def test_happy_path_writes_slugified_name(self):
        path = svc.save_uploaded_checkpoint(_upload("Best Model.pt", b"WEIGHT"))
        self.assertEqual(path.name, "best-model.pt")
        self.assertEqual(path.read_bytes(), b"WEIGHT")

    def test_creates_missing_checkpoint_dir(self):
        nested = self.tmp_path / "subdir"
        with override_settings(CHECKPOINT_DIR=nested):
            path = svc.save_uploaded_checkpoint(_upload("x.pt", b"a"))
        self.assertTrue(path.exists())
        self.assertEqual(path.parent, nested)

    def test_partial_temp_cleaned_on_validation_error(self):
        (self.tmp_path / "dup.pt").write_bytes(b"orig")
        with self.assertRaises(ValidationError):
            svc.save_uploaded_checkpoint(_upload("dup.pt", b"new"))
        # Nothing left over from the (never-started) partial write.
        leftovers = [
            p for p in self.tmp_path.iterdir() if p.name.endswith(".partial")
        ]
        self.assertEqual(leftovers, [])
        # And the original is untouched.
        self.assertEqual((self.tmp_path / "dup.pt").read_bytes(), b"orig")


class DeleteCheckpointTests(_CheckpointDirMixin, TestCase):
    def test_deletes_existing_file(self):
        (self.tmp_path / "a.pt").write_bytes(b"x")
        self.assertTrue(svc.delete_checkpoint("a.pt"))
        self.assertFalse((self.tmp_path / "a.pt").exists())

    def test_refuses_path_traversal(self):
        outside = self.tmp_path.parent / "outside.pt"
        outside.write_bytes(b"x")
        try:
            self.assertFalse(svc.delete_checkpoint("../outside.pt"))
            self.assertFalse(svc.delete_checkpoint("..\\outside.pt"))
            self.assertTrue(outside.exists())
        finally:
            outside.unlink()

    def test_refuses_hidden_and_specials(self):
        (self.tmp_path / ".hidden.pt").write_bytes(b"x")
        self.assertFalse(svc.delete_checkpoint(".hidden.pt"))
        self.assertFalse(svc.delete_checkpoint(""))
        self.assertFalse(svc.delete_checkpoint(".."))

    def test_returns_false_for_missing(self):
        self.assertFalse(svc.delete_checkpoint("nope.pt"))


class CheckpointFormTests(_CheckpointDirMixin, TestCase):
    def test_invalid_file_surfaces_error(self):
        form = CheckpointUploadForm(
            data={}, files={"file": _upload("bad.bin")}
        )
        self.assertFalse(form.is_valid())
        self.assertIn("file", form.errors)

    def test_valid_file_does_not_write_to_disk(self):
        form = CheckpointUploadForm(
            data={}, files={"file": _upload("ok.pt")}
        )
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(list(self.tmp_path.iterdir()), [])


class CheckpointsViewTests(_CheckpointDirMixin, TestCase):
    def test_get_renders_empty_state(self):
        resp = self.client.get(reverse("synapse_web:checkpoints"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "No checkpoints uploaded yet")

    def test_get_lists_existing_checkpoint(self):
        (self.tmp_path / "saved.pt").write_bytes(b"x" * 1024)
        resp = self.client.get(reverse("synapse_web:checkpoints"))
        self.assertContains(resp, "saved.pt")

    def test_post_upload_succeeds_and_redirects(self):
        resp = self.client.post(
            reverse("synapse_web:checkpoints"),
            data={"file": _upload("new.pt", b"weights")},
            follow=False,
        )
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.url, reverse("synapse_web:checkpoints"))
        self.assertTrue((self.tmp_path / "new.pt").exists())

    def test_post_upload_rejects_bad_suffix(self):
        resp = self.client.post(
            reverse("synapse_web:checkpoints"),
            data={"file": _upload("model.bin")},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Unsupported file type")
        self.assertEqual(list(self.tmp_path.iterdir()), [])

    def test_post_delete_removes_file_and_redirects(self):
        (self.tmp_path / "kill.pt").write_bytes(b"x")
        resp = self.client.post(
            reverse("synapse_web:delete_checkpoint"),
            data={"name": "kill.pt"},
        )
        self.assertEqual(resp.status_code, 302)
        self.assertFalse((self.tmp_path / "kill.pt").exists())

    def test_post_delete_rejects_traversal(self):
        resp = self.client.post(
            reverse("synapse_web:delete_checkpoint"),
            data={"name": "../escape.pt"},
        )
        self.assertEqual(resp.status_code, 302)
        # No crash; the redirect goes back to the list page.
        self.assertEqual(resp.url, reverse("synapse_web:checkpoints"))

    def test_delete_requires_post(self):
        resp = self.client.get(reverse("synapse_web:delete_checkpoint"))
        self.assertEqual(resp.status_code, 405)
