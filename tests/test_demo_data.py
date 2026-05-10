"""Tests for bundled demo data assets."""

from __future__ import annotations

from pathlib import Path
import unittest

from src.data_loader import load_csv
from src.geo import load_pjm_zone_geojson
from src.ingestion import ExportSchema, parse_flexworks_export
from app import DEMO_DATA_DIR, DEMO_FILE_PATHS, is_demo_mode, is_public_demo_only


DEMO_DIR = Path("demo_data")
DEMO_FLEXWORKS_EXPORT = DEMO_DIR / "flexworks_export.csv"
DEMO_DEVICE_MAPPING = DEMO_DIR / "device_to_zone_mapping.csv"
DEMO_ZONES_GEOJSON = DEMO_DIR / "zones.geojson"


class DemoDataTests(unittest.TestCase):
    def test_public_demo_only_enabled_by_flag(self) -> None:
        self.assertTrue(is_public_demo_only(environ={"PUBLIC_DEMO_ONLY": "true"}, secrets={}))

    def test_public_demo_only_forces_demo_mode(self) -> None:
        self.assertTrue(is_demo_mode(environ={"PUBLIC_DEMO_ONLY": "true"}, secrets={}))

    def test_force_demo_mode_enables_demo_mode(self) -> None:
        self.assertTrue(is_demo_mode(force_demo_mode=True, environ={}, secrets={}))

    def test_demo_mode_enabled_by_app_mode(self) -> None:
        self.assertTrue(is_demo_mode(environ={"APP_MODE": "demo"}, secrets={}))

    def test_demo_mode_enabled_by_demo_mode_flag(self) -> None:
        self.assertTrue(is_demo_mode(environ={"DEMO_MODE": "true"}, secrets={}))

    def test_demo_mode_enabled_by_streamlit_secret_style_mapping(self) -> None:
        self.assertTrue(is_demo_mode(environ={}, secrets={"APP_MODE": "demo"}))

    def test_demo_mode_disabled_by_default(self) -> None:
        self.assertFalse(is_demo_mode(environ={}, secrets={}))

    def test_demo_files_exist_at_relative_paths(self) -> None:
        for path in (DEMO_FLEXWORKS_EXPORT, DEMO_DEVICE_MAPPING, DEMO_ZONES_GEOJSON):
            with self.subTest(path=str(path)):
                self.assertFalse(path.is_absolute())
                self.assertTrue(path.exists(), f"Missing bundled demo file: {path}")

    def test_app_demo_file_constants_resolve_under_demo_data(self) -> None:
        for path in DEMO_FILE_PATHS:
            with self.subTest(path=str(path)):
                self.assertEqual(path.parent, DEMO_DATA_DIR)
                self.assertEqual(path.parent.name, "demo_data")
                self.assertTrue(path.exists())

    def test_demo_files_load_without_absolute_paths(self) -> None:
        monthly = parse_flexworks_export(load_csv(DEMO_FLEXWORKS_EXPORT))
        mapping = parse_flexworks_export(load_csv(DEMO_DEVICE_MAPPING))
        zones = load_pjm_zone_geojson(DEMO_ZONES_GEOJSON)

        self.assertEqual(monthly.schema, ExportSchema.MONTHLY_WIDE)
        self.assertEqual(mapping.schema, ExportSchema.DEVICE_SUMMARY)
        self.assertGreater(len(monthly.monthly_dataframe), 0)
        self.assertGreater(len(mapping.node_dataframe), 0)
        self.assertEqual(zones.zone_count, 21)


if __name__ == "__main__":
    unittest.main()
