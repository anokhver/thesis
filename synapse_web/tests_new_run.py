"""Tests for the run-creation flow: safe ZIP extraction, inspection,
form, and the /runs/new/ view."""
from __future__ import annotations

import csv
import io
import os
import tempfile
import zipfile
from pathlib import Path

from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from synapse_web.forms import NewRunForm
from synapse_web.models import AnalysisRun
from synapse_web.services import run_uploads as ru


# ---------------------------------------------------------------------------
# Zip helpers
# ---------------------------------------------------------------------------

def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _make_backslash_zip() -> bytes:
    """Reserved for future use; Python's zipfile normalises backslashes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("sub/evil.txt", b"X")
    return buf.getvalue().replace(b"sub/evil.txt", b"sub\\evil.txt")


def _make_encrypted_flag_zip() -> bytes:
    """Build a normal ZIP, then flip bit 0 of the general-purpose flag on
    both the local file header and central directory entry so zipfile
    sees the entry as encrypted (writestr can't actually encrypt)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("secret.txt", b"X")
    data = bytearray(buf.getvalue())
    if data[:4] == b"PK\x03\x04":
        data[6] |= 0x1
    cd_off = data.find(b"PK\x01\x02")
    if cd_off >= 0:
        data[cd_off + 8] |= 0x1
    return bytes(data)
    """Build a normal ZIP, then flip bit 0 of the general-purpose flag on
    both the local file header and central directory entry so zipfile
    sees the entry as encrypted (writestr can't actually encrypt)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("secret.txt", b"X")
    data = bytearray(buf.getvalue())
    if data[:4] == b"PK\x03\x04":
        data[6] |= 0x1  # local file header flag (LE, low byte)
    cd_off = data.find(b"PK\x01\x02")
    if cd_off >= 0:
        data[cd_off + 8] |= 0x1  # central directory flag
    return bytes(data)


def _make_symlink_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.create_system = 3  # Unix
        info.external_attr = (0o120777 & 0xFFFF) << 16
        zf.writestr(info, b"target")
    return buf.getvalue()


class _MixinTmp:
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()


# ---------------------------------------------------------------------------
# safe_extract_zip
# ---------------------------------------------------------------------------

class SafeExtractZipTests(_MixinTmp, TestCase):
    def _extract(self, data: bytes) -> Path:
        dest = self.tmp_path / "out"
        ru.safe_extract_zip(io.BytesIO(data), dest)
        return dest

    def test_happy_path_strips_common_top_level(self):
        data = _make_zip({"root/a.txt": b"A", "root/sub/b.txt": b"B"})
        dest = self._extract(data)
        self.assertTrue((dest / "a.txt").exists())
        self.assertTrue((dest / "sub" / "b.txt").exists())
        self.assertFalse((dest / "root").exists())

    def test_keeps_root_when_entries_have_different_tops(self):
        data = _make_zip({"a.txt": b"A", "b.txt": b"B"})
        dest = self._extract(data)
        self.assertTrue((dest / "a.txt").exists())
        self.assertTrue((dest / "b.txt").exists())

    def test_rejects_corrupted_zip(self):
        with self.assertRaises(ValidationError):
            ru.safe_extract_zip(io.BytesIO(b"not a zip"), self.tmp_path / "out")

    def test_rejects_traversal(self):
        data = _make_zip({"../escape.txt": b"X"})
        with self.assertRaises(ValidationError):
            self._extract(data)

    # Note: Python's zipfile silently normalises backslashes to forward
    # slashes on read, so the backslash guard in safe_extract_zip cannot
    # be exercised via stdlib zipfile. The guard is kept as defense in
    # depth against archives produced by other libraries.

    def test_rejects_colon_ads(self):
        data = _make_zip({"safe.txt:evil": b"X"})
        with self.assertRaises(ValidationError):
            self._extract(data)

    def test_rejects_drive_prefix(self):
        data = _make_zip({"C:/evil.txt": b"X"})
        with self.assertRaises(ValidationError):
            self._extract(data)

    def test_rejects_absolute_path(self):
        data = _make_zip({"/abs.txt": b"X"})
        with self.assertRaises(ValidationError):
            self._extract(data)

    def test_rejects_symlink_entry(self):
        with self.assertRaises(ValidationError):
            self._extract(_make_symlink_zip())

    def test_rejects_encrypted_entries(self):
        with self.assertRaises(ValidationError):
            self._extract(_make_encrypted_flag_zip())

    def test_rejects_too_many_members(self):
        data = _make_zip({f"f{i}.txt": b"" for i in range(5)})
        with override_settings(MAX_ZIP_MEMBERS=3):
            with self.assertRaises(ValidationError):
                self._extract(data)

    def test_rejects_over_uncompressed_budget(self):
        data = _make_zip({"big.bin": b"X" * 5000})
        with override_settings(MAX_ZIP_UNCOMPRESSED_BYTES=1000):
            with self.assertRaises(ValidationError):
                self._extract(data)

    def test_skips_macosx_metadata(self):
        data = _make_zip(
            {"__MACOSX/foo": b"junk", "real.txt": b"R"}
        )
        dest = self._extract(data)
        self.assertTrue((dest / "real.txt").exists())
        self.assertFalse((dest / "__MACOSX").exists())

    def test_rejects_case_collision(self):
        data = _make_zip({"A.txt": b"A", "a.txt": b"a"})
        with self.assertRaises(ValidationError):
            self._extract(data)

    def test_rejects_empty_zip(self):
        data = _make_zip({})
        with self.assertRaises(ValidationError):
            self._extract(data)


# ---------------------------------------------------------------------------
# inspect_patches
# ---------------------------------------------------------------------------

def _write_index_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


class InspectPatchesTests(_MixinTmp, TestCase):
    def test_flat_layout(self):
        _write_index_csv(
            self.tmp_path / "index.csv",
            [
                {"filename": "a.npy", "source_image": "2_5_BAEO_1.vsi"},
                {"filename": "b.npy", "source_image": "2_5_BAEO_1.vsi"},
                {"filename": "c.npy", "source_image": "1_3_PSI_2.vsi"},
            ],
        )
        manifest = ru.inspect_patches(self.tmp_path)
        by_src = {m["source_image"]: m for m in manifest}
        self.assertEqual(by_src["2_5_BAEO_1.vsi"]["n_patches"], 2)
        self.assertEqual(by_src["2_5_BAEO_1.vsi"]["treatment_group"], "BAEO")
        self.assertEqual(by_src["1_3_PSI_2.vsi"]["n_patches"], 1)
        self.assertEqual(by_src["1_3_PSI_2.vsi"]["treatment_group"], "PSI")

    def test_nested_layout(self):
        _write_index_csv(
            self.tmp_path / "day1" / "index.csv",
            [{"filename": "x.npy", "source_image": "1_BAEO.vsi"}],
        )
        _write_index_csv(
            self.tmp_path / "day2" / "index.csv",
            [{"filename": "y.npy", "source_image": "1_BAEO.vsi"}],
        )
        manifest = ru.inspect_patches(self.tmp_path)
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["n_patches"], 2)

    def test_unknown_token_gives_none(self):
        _write_index_csv(
            self.tmp_path / "index.csv",
            [{"filename": "a.npy", "source_image": "mystery.vsi"}],
        )
        manifest = ru.inspect_patches(self.tmp_path)
        self.assertIsNone(manifest[0]["treatment_group"])

    def test_missing_index_raises(self):
        with self.assertRaises(ValidationError):
            ru.inspect_patches(self.tmp_path)

    def test_missing_source_image_column_raises(self):
        _write_index_csv(
            self.tmp_path / "index.csv",
            [{"filename": "a.npy"}],
        )
        with self.assertRaises(ValidationError):
            ru.inspect_patches(self.tmp_path)


# ---------------------------------------------------------------------------
# inspect_bundle
# ---------------------------------------------------------------------------

class InspectBundleTests(_MixinTmp, TestCase):
    def _make_bundle(self, dest: Path, rows: list[dict] | None = None,
                     with_emb: bool = True, with_meta: bool = True) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        if with_emb:
            (dest / "embeddings.npy").write_bytes(b"FAKE")
        if with_meta:
            _write_index_csv(
                dest / "metadata.csv",
                rows or [
                    {"filename": "0.npy", "source_image": "x_PSI.vsi"},
                    {"filename": "1.npy", "source_image": "x_PSI.vsi"},
                    {"filename": "2.npy", "source_image": "y_BAEO.vsi"},
                ],
            )

    def test_flat_layout(self):
        self._make_bundle(self.tmp_path)
        manifest, root = ru.inspect_bundle(self.tmp_path)
        self.assertEqual(root, self.tmp_path)
        self.assertEqual(sum(m["n_patches"] for m in manifest), 3)

    def test_nested_layout_returns_actual_root(self):
        nested = self.tmp_path / "bundle_v1"
        self._make_bundle(nested)
        manifest, root = ru.inspect_bundle(self.tmp_path)
        self.assertEqual(root, nested)
        self.assertEqual(len(manifest), 2)

    def test_missing_files_raise(self):
        self._make_bundle(self.tmp_path, with_emb=False)
        with self.assertRaises(ValidationError):
            ru.inspect_bundle(self.tmp_path)

    def test_missing_source_image_column_raises(self):
        self._make_bundle(self.tmp_path, rows=[{"filename": "0.npy"}])
        with self.assertRaises(ValidationError):
            ru.inspect_bundle(self.tmp_path)


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------

class _CheckpointMixin:
    def setUp(self):
        super().setUp()
        self._ck = tempfile.TemporaryDirectory()
        self.ck_path = Path(self._ck.name)
        self._override = override_settings(
            CHECKPOINT_DIR=self.ck_path,
            MAX_CHECKPOINT_UPLOAD_BYTES=1024 * 1024,
        )
        self._override.enable()

    def tearDown(self):
        self._override.disable()
        self._ck.cleanup()
        super().tearDown()


def _zip_upload(name: str, data: bytes) -> SimpleUploadedFile:
    return SimpleUploadedFile(name, data, content_type="application/zip")


class NewRunFormTests(_CheckpointMixin, TestCase):
    def _patches_zip(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "index.csv",
                "filename,source_image\na.npy,1_BAEO.vsi\n",
            )
            zf.writestr("a.npy", b"FAKE")
        return buf.getvalue()

    def test_rejects_non_zip(self):
        form = NewRunForm(
            data={"run_kind": "cluster_only"},
            files={"archive": _zip_upload("x.tar", b"x")},
        )
        self.assertFalse(form.is_valid())
        self.assertIn("archive", form.errors)

    def test_requires_checkpoint_for_extract_and_cluster(self):
        form = NewRunForm(
            data={"run_kind": "extract_and_cluster"},
            files={"archive": _zip_upload("x.zip", self._patches_zip())},
        )
        self.assertFalse(form.is_valid())
        self.assertIn("checkpoint", form.errors)

    def test_rejects_unknown_checkpoint(self):
        form = NewRunForm(
            data={"run_kind": "extract_and_cluster", "checkpoint": "ghost.pt"},
            files={"archive": _zip_upload("x.zip", self._patches_zip())},
        )
        self.assertFalse(form.is_valid())
        self.assertIn("checkpoint", form.errors)

    def test_accepts_cluster_only_without_checkpoint(self):
        form = NewRunForm(
            data={"run_kind": "cluster_only"},
            files={"archive": _zip_upload("x.zip", self._patches_zip())},
        )
        self.assertTrue(form.is_valid(), form.errors)


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------

class _RunsViewMixin(_CheckpointMixin, _MixinTmp):
    def setUp(self):
        super().setUp()
        self._runs_override = override_settings(
            MEDIA_ROOT=str(self.tmp_path),
            ANALYSIS_JOBS_SYNC=True,
        )
        self._runs_override.enable()

    def tearDown(self):
        self._runs_override.disable()
        super().tearDown()


class NewRunViewTests(_RunsViewMixin, TestCase):
    def _patches_zip(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "index.csv",
                "filename,source_image\n"
                "a.npy,1_BAEO.vsi\n"
                "b.npy,1_BAEO.vsi\n"
                "c.npy,2_PSI.vsi\n",
            )
            for n in ("a.npy", "b.npy", "c.npy"):
                zf.writestr(n, b"FAKE")
        return buf.getvalue()

    def _bundle_zip(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("embeddings.npy", b"FAKE")
            zf.writestr(
                "metadata.csv",
                "filename,source_image\n"
                "0.npy,1_BAEO.vsi\n"
                "1.npy,2_PSI.vsi\n",
            )
        return buf.getvalue()

    def test_get_renders_form(self):
        resp = self.client.get(reverse("synapse_web:new_run"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Start a new clustering run")
        self.assertContains(resp, "Run mode")

    def test_get_warns_when_no_checkpoints(self):
        resp = self.client.get(reverse("synapse_web:new_run"))
        self.assertContains(resp, "No encoder checkpoints are available")

    def test_post_cluster_only_creates_run_and_dispatches(self):
        resp = self.client.post(
            reverse("synapse_web:new_run"),
            data={
                "run_kind": "cluster_only",
                "archive": _zip_upload("b.zip", self._bundle_zip()),
            },
        )
        self.assertEqual(resp.status_code, 302)
        run = AnalysisRun.objects.get()
        self.assertEqual(run.run_kind, "cluster_only")
        self.assertEqual(run.n_source_images, 2)
        self.assertEqual(run.n_patches, 2)
        self.assertEqual(run.status, "completed")
        self.assertTrue(Path(run.input_dir).exists())
        self.assertTrue(Path(run.bundle_dir).exists())
        # The stub work creates SourceImageStats per manifest item.
        self.assertEqual(run.source_stats.count(), 2)

    def test_post_extract_and_cluster_creates_run(self):
        (self.ck_path / "model.pt").write_bytes(b"weights")
        resp = self.client.post(
            reverse("synapse_web:new_run"),
            data={
                "run_kind": "extract_and_cluster",
                "checkpoint": "model.pt",
                "archive": _zip_upload("p.zip", self._patches_zip()),
            },
        )
        self.assertEqual(resp.status_code, 302)
        run = AnalysisRun.objects.get()
        self.assertEqual(run.run_kind, "extract_and_cluster")
        self.assertEqual(
            Path(run.checkpoint_path).resolve(),
            (self.ck_path / "model.pt").resolve(),
        )
        self.assertEqual(run.n_source_images, 2)
        self.assertEqual(run.n_patches, 3)

    def test_post_bad_zip_does_not_leave_orphan_run(self):
        resp = self.client.post(
            reverse("synapse_web:new_run"),
            data={
                "run_kind": "cluster_only",
                "archive": _zip_upload("x.zip", b"not a zip"),
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(AnalysisRun.objects.count(), 0)
        # And the runs directory has no leftover children for our (deleted) row.
        runs_root = self.tmp_path / "runs"
        if runs_root.exists():
            self.assertEqual(list(runs_root.iterdir()), [])

    def test_post_with_unknown_source_image_warns(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("embeddings.npy", b"FAKE")
            zf.writestr(
                "metadata.csv",
                "filename,source_image\n0.npy,mystery.vsi\n",
            )
        resp = self.client.post(
            reverse("synapse_web:new_run"),
            data={
                "run_kind": "cluster_only",
                "archive": _zip_upload("b.zip", buf.getvalue()),
            },
            follow=True,
        )
        self.assertContains(resp, "treatment-group token")
