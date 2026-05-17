import shutil
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from synapse_web.models import MicroscopyImage
from synapse_web.services.naming import KNOWN_GROUPS, derive_treatment_group


class DeriveTreatmentGroupTests(TestCase):
    def test_simple_known_tokens(self):
        cases = {
            "1_8_KONTROLA.vsi": "KONTROLA",
            "2_2_PSI.vsi": "PSI",
            "2_3_PSY.vsi": "PSY",
            "2_5_BAEO.vsi": "BAEO",
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(derive_treatment_group(filename), expected)

    def test_long_tokens_win_over_short_prefixes(self):
        self.assertEqual(derive_treatment_group("2_3_PSYHARMIN.vsi"), "PSYHARMIN")
        self.assertEqual(derive_treatment_group("2_4_NORPSIHARMIN.vsi"), "NORPSIHARMIN")
        self.assertEqual(derive_treatment_group("2_5_BAEOHARMIN.vsi"), "BAEOHARMIN")

    def test_norpsi_is_not_psi(self):
        self.assertEqual(derive_treatment_group("2_4_NORPSI.vsi"), "NORPSI")

    def test_real_cellsens_filename(self):
        self.assertEqual(
            derive_treatment_group(
                "2_3_PSYHARMIN_7_Multichannel Z-Stack_20251030_72.vsi"
            ),
            "PSYHARMIN",
        )

    def test_unknown_filename_returns_none(self):
        self.assertIsNone(derive_treatment_group("random_image.tif"))
        self.assertIsNone(derive_treatment_group("test.vsi"))
        self.assertIsNone(derive_treatment_group("1_2_FOOBAR.vsi"))

    def test_case_insensitive(self):
        self.assertEqual(derive_treatment_group("2_2_psi.vsi"), "PSI")
        self.assertEqual(derive_treatment_group("1_8_Kontrola.VSI"), "KONTROLA")

    def test_handles_full_paths(self):
        self.assertEqual(
            derive_treatment_group("/data/microscopy/2_5_BAEO.vsi"), "BAEO"
        )
        self.assertEqual(
            derive_treatment_group("C:\\data\\2_5_BAEO.vsi"), "BAEO"
        )

    def test_handles_multi_part_extensions(self):
        self.assertEqual(derive_treatment_group("2_3_PSY.ome.tif"), "PSY")
        self.assertEqual(derive_treatment_group("1_8_KONTROLA.ome.tiff"), "KONTROLA")

    def test_empty_and_garbage(self):
        self.assertIsNone(derive_treatment_group(""))
        self.assertIsNone(derive_treatment_group(".vsi"))
        self.assertIsNone(derive_treatment_group("___.vsi"))

    def test_known_groups_are_uppercase(self):
        for g in KNOWN_GROUPS:
            self.assertEqual(g, g.upper())


class FolderUploadViewTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._media_root = tempfile.mkdtemp(prefix="synapse_test_media_")
        cls._media_override = override_settings(MEDIA_ROOT=cls._media_root)
        cls._media_override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._media_override.disable()
        shutil.rmtree(cls._media_root, ignore_errors=True)
        super().tearDownClass()

    def _vsi(self, name: str) -> SimpleUploadedFile:
        return SimpleUploadedFile(name, b"fake-vsi-bytes", content_type="application/octet-stream")

    def test_folder_upload_registers_only_vsi_files(self):
        vsi_name = "2_3_PSYHARMIN_7_Multichannel Z-Stack_20251030_72.vsi"
        files = [
            self._vsi(vsi_name),
            SimpleUploadedFile("plan.oex", b"x", content_type="text/xml"),
            SimpleUploadedFile("frame_t.ets", b"y"),
            self._vsi("2_5_BAEO_1.vsi"),
            SimpleUploadedFile("readme.txt", b"hello"),
        ]
        rel_paths = "\n".join([
            f"session/{vsi_name}",
            "session/2_3_PSYHARMIN_7_Multichannel Z-Stack_20251030_72.oex",
            "session/_2_3_PSYHARMIN_7_Multichannel Z-Stack_20251030_72_/frame_t.ets",
            "session/2_5_BAEO_1.vsi",
            "session/readme.txt",
        ])

        resp = self.client.post(
            reverse("synapse_web:upload"),
            data={
                "action": "upload",
                "files": files,
                "relative_paths": rel_paths,
                "treatment_group": "",
            },
        )
        self.assertEqual(resp.status_code, 302)

        rows = list(MicroscopyImage.objects.order_by("original_filename"))
        self.assertEqual(len(rows), 2)
        names = {r.original_filename for r in rows}
        self.assertEqual(names, {vsi_name, "2_5_BAEO_1.vsi"})
        groups = {r.original_filename: r.treatment_group for r in rows}
        self.assertEqual(groups[vsi_name], "PSYHARMIN")
        self.assertEqual(groups["2_5_BAEO_1.vsi"], "BAEO")

    def test_fallback_group_used_when_filename_does_not_match(self):
        resp = self.client.post(
            reverse("synapse_web:upload"),
            data={
                "action": "upload",
                "files": [self._vsi("random_name.vsi")],
                "relative_paths": "session/random_name.vsi",
                "treatment_group": "manual_batch",
            },
        )
        self.assertEqual(resp.status_code, 302)
        row = MicroscopyImage.objects.get()
        self.assertEqual(row.treatment_group, "manual_batch")

    def test_warns_when_no_vsi_in_upload(self):
        resp = self.client.post(
            reverse("synapse_web:upload"),
            data={
                "action": "upload",
                "files": [SimpleUploadedFile("notes.txt", b"x")],
                "relative_paths": "session/notes.txt",
            },
            follow=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(MicroscopyImage.objects.count(), 0)
        self.assertContains(resp, "No .vsi files found")

    def test_delete_removes_image_and_file(self):
        self.client.post(
            reverse("synapse_web:upload"),
            data={
                "action": "upload",
                "files": [self._vsi("2_5_BAEO_1.vsi")],
                "relative_paths": "session/2_5_BAEO_1.vsi",
            },
        )
        img = MicroscopyImage.objects.get()
        stored = img.file.name

        resp = self.client.post(
            reverse("synapse_web:upload"),
            data={"action": "delete", "image_id": img.id},
            follow=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(MicroscopyImage.objects.count(), 0)
        from django.core.files.storage import default_storage
        self.assertFalse(default_storage.exists(stored))
        self.assertContains(resp, "Image deleted.")

    def test_delete_action_does_not_use_nested_forms(self):
        self.client.post(
            reverse("synapse_web:upload"),
            data={
                "action": "upload",
                "files": [self._vsi("2_5_BAEO_1.vsi")],
                "relative_paths": "session/2_5_BAEO_1.vsi",
            },
        )
        img = MicroscopyImage.objects.get()

        resp = self.client.get(reverse("synapse_web:upload"))
        html = resp.content.decode("utf-8")

        self.assertIn(f'form="delete-form-{img.id}"', html)
        self.assertIn(f'id="delete-form-{img.id}"', html)

        depth = 0
        i = 0
        while i < len(html):
            if html.startswith("<form", i) and (
                len(html) == i + 5 or html[i + 5] in " >\n\t"
            ):
                self.assertEqual(
                    depth, 0,
                    f"nested <form> at offset {i}: {html[max(0,i-60):i+60]!r}",
                )
                depth += 1
                i += 5
                continue
            if html.startswith("</form>", i):
                depth -= 1
                i += 7
                continue
            i += 1
        self.assertEqual(depth, 0, "unbalanced <form> tags")


class OverviewPageTests(TestCase):
    def test_page_renders_with_real_config(self):
        resp = self.client.get(reverse("synapse_web:overview"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Pretraining configuration")
        self.assertContains(resp, "Preprocessing defaults")
        self.assertContains(resp, "Treatment groups")
        self.assertContains(resp, "PSYHARMIN")
        self.assertContains(resp, "Psilocybin + Harmine")
        self.assertContains(resp, "in_channels")
        self.assertContains(resp, "epochs")

    def test_navbar_links_to_overview(self):
        resp = self.client.get(reverse("home:index"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'href="/analysis/overview/"')


class TrainingConfigLoaderTests(TestCase):
    def test_grouped_sections_have_expected_keys(self):
        from synapse_web.services.training_config import grouped_pretrain_config

        sections = grouped_pretrain_config()
        if not sections:
            self.skipTest("pretrain_ae_config.json not present in this checkout")

        titles = [title for title, _ in sections]
        self.assertIn("Architecture", titles)
        self.assertIn("Training", titles)
        self.assertIn("Reproducibility", titles)

        seen: set[str] = set()
        for _, rows in sections:
            for key, _ in rows:
                self.assertNotIn(key, seen, f"duplicated key {key!r}")
                seen.add(key)


class TrainingConfigSchemaTests(TestCase):
    def _load_via_path(self, tmp_path, payload):
        import json
        from synapse_web.services import training_config as tc

        cfg = tmp_path / "pretrain.json"
        cfg.write_text(json.dumps(payload), encoding="utf-8")
        old = tc.PRETRAIN_CONFIG_PATH
        tc.PRETRAIN_CONFIG_PATH = cfg
        try:
            return tc.grouped_pretrain_config()
        finally:
            tc.PRETRAIN_CONFIG_PATH = old

    def test_flat_schema_groups_by_keyset(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            sections = self._load_via_path(
                Path(d),
                {"in_channels": 3, "epochs": 10, "seed": 42, "mystery": "x"},
            )
        titles = [t for t, _ in sections]
        self.assertEqual(
            titles, ["Architecture", "Training", "Reproducibility", "Other"]
        )
        other = dict(sections[-1][1])
        self.assertEqual(other["mystery"], "x")

    def test_nested_schema_uses_top_level_dicts_as_sections(self):
        import tempfile
        from pathlib import Path
        payload = {
            "run_sanity": True,
            "base": {"seed": 42, "tag": "moby"},
            "model": {"in_channels": 3, "img_size": 128},
            "train": {"epochs": 200, "warmup_epochs": 15},
            "ssl": {"mask_ratio": 0.6, "loss_kind": "l1"},
            "unfreeze_schedule": [[1, ["layers4"]], [3, ["layers3"]]],
        }
        with tempfile.TemporaryDirectory() as d:
            sections = self._load_via_path(Path(d), payload)
        titles = [t for t, _ in sections]
        self.assertIn("Overview", titles)
        self.assertIn("Base", titles)
        self.assertIn("Architecture", titles)
        self.assertIn("Training", titles)
        self.assertIn("SSL Objective", titles)

        overview_rows = dict(dict(sections)["Overview"])
        self.assertIn("run_sanity", overview_rows)
        self.assertIn("unfreeze_schedule", overview_rows)

        model_rows = dict(dict(sections)["Architecture"])
        self.assertEqual(model_rows["in_channels"], 3)

    def test_missing_config_returns_empty_list(self):
        import tempfile
        from pathlib import Path
        from synapse_web.services import training_config as tc

        with tempfile.TemporaryDirectory() as d:
            old = tc.PRETRAIN_CONFIG_PATH
            tc.PRETRAIN_CONFIG_PATH = Path(d) / "nope.json"
            try:
                self.assertEqual(tc.grouped_pretrain_config(), [])
            finally:
                tc.PRETRAIN_CONFIG_PATH = old
