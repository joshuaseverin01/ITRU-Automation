"""Tests for bundled demo data assets."""

from __future__ import annotations

from pathlib import Path
import unittest

from src.data_loader import load_csv
from src.geo import load_pjm_zone_geojson
from src.ingestion import ExportSchema, parse_flexworks_export


DEMO_DIR = Path("demo_data")
DEMO_FLEXWORKS_EXPORT = DEMO_DIR / "flexworks_export.csv"
DEMO_DEVICE_MAPPING = DEMO_DIR / "device_to_zone_mapping.csv"
DEMO_ZONES_GEOJSON = DEMO_DIR / "zones.geojson"


class DemoDataTests(unittest.TestCase):
    def test_demo_files_exist_at_relative_paths(self) -> None:
        for path in (DEMO_FLEXWORKS_EXPORT, DEMO_DEVICE_MAPPING, DEMO_ZONES_GEOJSON):
            with self.subTest(path=str(path)):
                self.assertFalse(path.is_absolute())
                self.assertTrue(path.exists(), f"Missing bundled demo file: {path}")

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

