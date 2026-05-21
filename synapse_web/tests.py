from django.test import TestCase
from django.urls import reverse

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
    def test_real_config_renders_at_least_one_section_with_no_dupes(self):
        from synapse_web.services.training_config import grouped_pretrain_config

        sections = grouped_pretrain_config()
        if not sections:
            self.skipTest("no pretrain config present in this checkout")

        self.assertGreater(len(sections), 0)
        for title, rows in sections:
            self.assertIsInstance(title, str)
            self.assertGreater(len(rows), 0, f"section {title!r} is empty")

        seen: set[tuple[str, str]] = set()
        for title, rows in sections:
            for key, _ in rows:
                pair = (title, key)
                self.assertNotIn(pair, seen, f"duplicated row {pair!r}")
                seen.add(pair)


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
