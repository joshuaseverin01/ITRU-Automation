"""Validation checks for real FlexWorks-style CSV exports."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from app import _build_choropleth_unavailable_message
from src.analysis import (
    ALL_REVENUE_CATEGORIES,
    add_analysis_columns,
    compute_zone_monthly_revenue,
    filter_zone_revenue_to_month,
    rank_nodes,
)
from src.cleaning import clean_flexworks_export
from src.data_loader import load_csv
from src.geo import load_pjm_zone_geojson, prepare_pjm_zone_choropleth_data
from src.ingestion import ExportSchema, join_monthly_to_device_summary, parse_flexworks_export
from src.visualization import (
    build_monthly_revenue_bar,
    build_monthly_revenue_chart,
    build_pjm_cumulative_revenue_map_bars,
    build_pjm_zone_choropleth,
)


DEVICE_EXPORT_PATH = Path("/Users/joshuaseverin/Desktop/test data/PJM Run Dec 16 Device.csv")
MONTHLY_EXPORT_PATH = Path("/Users/joshuaseverin/Desktop/test data/PJM Run Dec 16.csv")
PJM_GEOJSON_PATH = Path("/Users/joshuaseverin/Desktop/internship/PJM/pjm_zones.geojson")


@unittest.skipUnless(
    DEVICE_EXPORT_PATH.exists() and MONTHLY_EXPORT_PATH.exists(),
    "Real FlexWorks test files are not present on this machine.",
)
class RealFlexWorksExportTests(unittest.TestCase):
    def test_device_summary_export_normalizes_to_node_schema(self) -> None:
        parsed = parse_flexworks_export(load_csv(DEVICE_EXPORT_PATH))

        self.assertEqual(parsed.schema, ExportSchema.DEVICE_SUMMARY)
        self.assertIsNotNone(parsed.node_dataframe)
        self.assertEqual(len(parsed.node_dataframe), 21)

        first = parsed.node_dataframe.iloc[0]
        self.assertEqual(first["Node_ID"], "Device 1")
        self.assertEqual(first["Device_ID"], "Device 1")
        self.assertEqual(first["Zone"], "BGE")
        self.assertEqual(first["ISO_Region"], "PJM")
        self.assertAlmostEqual(first["Annualized_Revenue"], 498.69, places=2)
        self.assertAlmostEqual(first["Revenue_per_kW"], 149.61, places=2)

    def test_monthly_wide_export_reshapes_to_long_format(self) -> None:
        parsed = parse_flexworks_export(load_csv(MONTHLY_EXPORT_PATH))

        self.assertEqual(parsed.schema, ExportSchema.MONTHLY_WIDE)
        self.assertIsNotNone(parsed.monthly_dataframe)
        self.assertEqual(len(parsed.monthly_dataframe), 2268)
        self.assertEqual(set(parsed.monthly_dataframe["Revenue_Category"].unique()), {"Energy", "Ancillary", "FCP"})
        self.assertEqual(parsed.monthly_dataframe["Device"].nunique(), 21)

    def test_monthly_export_joins_to_device_summary(self) -> None:
        device = parse_flexworks_export(load_csv(DEVICE_EXPORT_PATH))
        monthly = parse_flexworks_export(load_csv(MONTHLY_EXPORT_PATH))
        cleaned, _ = clean_flexworks_export(device.node_dataframe)
        joined, notes = join_monthly_to_device_summary(monthly.monthly_dataframe, cleaned)

        self.assertIsNotNone(joined)
        self.assertIn("Joined monthly revenue data to device summary metadata using Device.", notes)
        self.assertEqual(len(joined), 2268)
        self.assertFalse(joined["Zone"].isna().any())

        analyzed = add_analysis_columns(cleaned)
        ranked = rank_nodes(analyzed)
        self.assertEqual(ranked.iloc[0]["Opportunity_Score"], 100.0)
        self.assertIsNotNone(build_monthly_revenue_chart(joined, group_by="Revenue_Category").figure)
        self.assertIsNotNone(build_monthly_revenue_bar(joined, group_by="Zone").figure)


@unittest.skipUnless(
    DEVICE_EXPORT_PATH.exists() and MONTHLY_EXPORT_PATH.exists() and PJM_GEOJSON_PATH.exists(),
    "Real FlexWorks device/monthly files or PJM GeoJSON are not present on this machine.",
)
class PjmGeoJsonTests(unittest.TestCase):
    def test_pjm_geojson_loads_and_joins_to_flexworks_zones(self) -> None:
        parsed = parse_flexworks_export(load_csv(DEVICE_EXPORT_PATH))
        cleaned, _ = clean_flexworks_export(parsed.node_dataframe)
        analyzed = add_analysis_columns(cleaned)
        pjm_geojson = load_pjm_zone_geojson(PJM_GEOJSON_PATH)

        self.assertEqual(pjm_geojson.zone_count, 21)
        self.assertEqual(pjm_geojson.zone_property, "zoneName")

        join_result = prepare_pjm_zone_choropleth_data(analyzed, pjm_geojson, "Revenue_per_kW")
        diagnostics = join_result.diagnostics
        self.assertTrue(diagnostics.is_available, diagnostics.to_dict())
        self.assertEqual(diagnostics.geojson_zone_count, 21)
        self.assertEqual(diagnostics.flexworks_zone_count, 21)
        self.assertEqual(diagnostics.matched_zone_count, 21)
        self.assertEqual(diagnostics.unmatched_flexworks_zones, [])
        self.assertEqual(diagnostics.unmatched_geojson_zones, [])

        chart, chart_diagnostics = build_pjm_zone_choropleth(analyzed, pjm_geojson, "Revenue_per_kW")
        self.assertTrue(chart_diagnostics.is_available)
        self.assertIsNotNone(chart.figure)
        figure_dict = chart.figure.to_dict()
        self.assertEqual(figure_dict["layout"]["title"]["text"], "PJM Zone Choropleth: Revenue per kW ($/kW)")
        self.assertEqual(len(figure_dict["data"][0]["locations"]), 21)
        self.assertIn("Annualized Revenue", figure_dict["data"][0]["hovertemplate"])
        self.assertIn("Revenue per kW", figure_dict["data"][0]["hovertemplate"])
        self.assertEqual(figure_dict["layout"]["coloraxis"]["colorbar"]["tickprefix"], "$")
        self.assertEqual(figure_dict["layout"]["coloraxis"]["colorbar"]["ticksuffix"], "/kW")

    def test_non_pjm_active_dataset_gets_dataset_fallback_message(self) -> None:
        parsed = parse_flexworks_export(load_csv(Path("sample_data/sample_flexworks_export.csv")))
        cleaned, _ = clean_flexworks_export(parsed.node_dataframe)
        analyzed = add_analysis_columns(cleaned.loc[cleaned["ISO_Region"].isin(["ERCOT", "CAISO"])])
        pjm_geojson = load_pjm_zone_geojson(PJM_GEOJSON_PATH)

        join_result = prepare_pjm_zone_choropleth_data(analyzed, pjm_geojson, "Revenue_per_kW")
        diagnostics = join_result.diagnostics
        self.assertFalse(diagnostics.is_available)
        self.assertEqual(diagnostics.matched_zone_count, 0)
        self.assertEqual(diagnostics.geojson_zone_count, 21)

        message = _build_choropleth_unavailable_message(diagnostics, ["ERCOT", "CAISO"])
        self.assertIn(
            "Zone choropleth is available only when the current filtered dataset contains PJM zone-level data.",
            message,
        )
        self.assertIn("Your GeoJSON loaded correctly", message)
        self.assertIn("- Active ISO filters: ERCOT, CAISO", message)
        self.assertIn("- PJM zones found in current data: 0", message)
        self.assertIn("- GeoJSON zones loaded: 21", message)

    def test_pjm_monthly_zone_revenue_cumulative_calculation(self) -> None:
        device = parse_flexworks_export(load_csv(DEVICE_EXPORT_PATH))
        monthly = parse_flexworks_export(load_csv(MONTHLY_EXPORT_PATH))
        cleaned, _ = clean_flexworks_export(device.node_dataframe)
        joined, _ = join_monthly_to_device_summary(monthly.monthly_dataframe, cleaned)

        zone_monthly = compute_zone_monthly_revenue(joined, revenue_category=ALL_REVENUE_CATEGORIES)
        self.assertEqual(zone_monthly["Zone_Normalized"].nunique(), 21)
        self.assertEqual(zone_monthly["Month"].nunique(), 36)
        final_month = zone_monthly["Month"].max()
        final_month_rows = filter_zone_revenue_to_month(zone_monthly, final_month)
        raw_zone_totals = joined.groupby("Zone", as_index=False)["Revenue"].sum().rename(columns={"Revenue": "Raw_Revenue"})
        final_comparison = final_month_rows.merge(raw_zone_totals, on="Zone")
        self.assertAlmostEqual(
            (final_comparison["Cumulative_Revenue"] - final_comparison["Raw_Revenue"]).abs().max(),
            0.0,
            places=6,
        )

        bge = zone_monthly.loc[zone_monthly["Zone_Normalized"] == "BGE"].sort_values("Month")
        self.assertAlmostEqual(bge.iloc[0]["Monthly_Revenue"], 79.71, places=2)
        self.assertAlmostEqual(bge.iloc[0]["Cumulative_Revenue"], 79.71, places=2)
        self.assertAlmostEqual(bge.iloc[1]["Cumulative_Revenue"], 95.95, places=2)

    def test_selected_month_filter_and_cumulative_map_bars(self) -> None:
        device = parse_flexworks_export(load_csv(DEVICE_EXPORT_PATH))
        monthly = parse_flexworks_export(load_csv(MONTHLY_EXPORT_PATH))
        cleaned, _ = clean_flexworks_export(device.node_dataframe)
        joined, _ = join_monthly_to_device_summary(monthly.monthly_dataframe, cleaned)
        zone_monthly = compute_zone_monthly_revenue(joined, revenue_category="Energy")
        selected = filter_zone_revenue_to_month(zone_monthly, "2022-02")
        pjm_geojson = load_pjm_zone_geojson(PJM_GEOJSON_PATH)

        self.assertEqual(len(selected), 21)
        self.assertEqual(selected["Month"].dt.strftime("%Y-%m").unique().tolist(), ["2022-02"])

        chart, diagnostics = build_pjm_cumulative_revenue_map_bars(
            selected,
            pjm_geojson,
            metric_column="Cumulative_Revenue",
            sort_order="Top zones",
        )
        self.assertTrue(diagnostics["is_available"], diagnostics)
        self.assertEqual(diagnostics["matched_zone_count"], 21)
        self.assertIsNotNone(chart.figure)
        figure = chart.figure.to_dict()
        self.assertIn("PJM Battery Revenue", figure["layout"]["title"]["text"])
        self.assertIn("Metric: Cumulative Revenue", figure["layout"]["title"]["text"])
        self.assertIn("Category: Energy", figure["layout"]["title"]["text"])
        self.assertGreaterEqual(len(figure["data"]), 2)

    def test_cumulative_revenue_missing_monthly_and_geojson_fallbacks(self) -> None:
        empty = compute_zone_monthly_revenue(None)
        self.assertTrue(empty.empty)

        selected = filter_zone_revenue_to_month(empty, "2022-01")
        chart, diagnostics = build_pjm_cumulative_revenue_map_bars(selected, None)
        self.assertIsNone(chart.figure)
        self.assertEqual(chart.message, "PJM zone map requires the PJM GeoJSON file.")
        self.assertFalse(diagnostics["is_available"])

    def test_category_filter_changes_revenue_values(self) -> None:
        monthly = pd.DataFrame(
            {
                "Device": ["A", "A", "A", "A"],
                "Zone": ["BGE", "BGE", "BGE", "BGE"],
                "ISO_Region": ["PJM", "PJM", "PJM", "PJM"],
                "Revenue_Category": ["Energy", "Ancillary", "Energy", "Ancillary"],
                "Month": ["2022-01", "2022-01", "2022-02", "2022-02"],
                "Revenue": [100, 20, 50, 5],
            }
        )
        all_categories = compute_zone_monthly_revenue(monthly, revenue_category=ALL_REVENUE_CATEGORIES)
        energy = compute_zone_monthly_revenue(monthly, revenue_category="Energy")
        ancillary = compute_zone_monthly_revenue(monthly, revenue_category="Ancillary")

        self.assertEqual(all_categories["Monthly_Revenue"].tolist(), [120, 55])
        self.assertEqual(energy["Monthly_Revenue"].tolist(), [100, 50])
        self.assertEqual(ancillary["Monthly_Revenue"].tolist(), [20, 5])
        self.assertEqual(all_categories["Cumulative_Revenue"].tolist(), [120, 175])
        self.assertEqual(energy["Cumulative_Revenue"].tolist(), [100, 150])

    def test_monthly_vs_cumulative_negative_revenue_and_bar_sorting(self) -> None:
        monthly = pd.DataFrame(
            {
                "Device": ["A", "A", "A", "B", "B", "B"],
                "Zone": ["BGE", "BGE", "BGE", "DPL", "DPL", "DPL"],
                "ISO_Region": ["PJM"] * 6,
                "Revenue_Category": ["Energy"] * 6,
                "Month": ["2022-01", "2022-02", "2022-03", "2022-01", "2022-02", "2022-03"],
                "Revenue": [100, -40, 30, -10, -5, 20],
            }
        )
        zone_monthly = compute_zone_monthly_revenue(monthly, revenue_category="Energy")
        selected = filter_zone_revenue_to_month(zone_monthly, "2022-02")
        bge = selected.loc[selected["Zone"] == "BGE"].iloc[0]
        dpl = selected.loc[selected["Zone"] == "DPL"].iloc[0]
        self.assertEqual(bge["Monthly_Revenue"], -40)
        self.assertEqual(bge["Cumulative_Revenue"], 60)
        self.assertEqual(dpl["Monthly_Revenue"], -5)
        self.assertEqual(dpl["Cumulative_Revenue"], -15)

        pjm_geojson = load_pjm_zone_geojson(PJM_GEOJSON_PATH)
        top_chart, _ = build_pjm_cumulative_revenue_map_bars(
            selected,
            pjm_geojson,
            metric_column="Monthly_Revenue",
            sort_order="Top zones",
        )
        bottom_chart, _ = build_pjm_cumulative_revenue_map_bars(
            selected,
            pjm_geojson,
            metric_column="Monthly_Revenue",
            sort_order="Bottom zones",
        )
        top_bar = next(trace for trace in top_chart.figure.to_dict()["data"] if trace["type"] == "bar")
        bottom_bar = next(trace for trace in bottom_chart.figure.to_dict()["data"] if trace["type"] == "bar")
        self.assertEqual(list(top_bar["y"]), ["DPL", "BGE"])
        self.assertEqual(list(top_bar["text"]), ["-$5", "-$40"])
        self.assertEqual(list(bottom_bar["y"]), ["BGE", "DPL"])
        self.assertIn("Metric: Monthly Revenue", top_chart.figure.to_dict()["layout"]["title"]["text"])
        self.assertIn("Category: Energy", top_chart.figure.to_dict()["layout"]["title"]["text"])

    def test_only_one_required_flexworks_file_fails_gracefully_for_cumulative_feature(self) -> None:
        monthly = parse_flexworks_export(load_csv(MONTHLY_EXPORT_PATH))
        monthly_without_metadata, notes = join_monthly_to_device_summary(monthly.monthly_dataframe, None)
        self.assertIn("Monthly revenue data was loaded without device summary metadata.", notes)
        self.assertTrue(compute_zone_monthly_revenue(monthly_without_metadata).empty)

        no_monthly_join, no_monthly_notes = join_monthly_to_device_summary(None, pd.DataFrame({"Device": ["Device 1"]}))
        self.assertIsNone(no_monthly_join)
        self.assertEqual(no_monthly_notes, [])


if __name__ == "__main__":
    unittest.main()
