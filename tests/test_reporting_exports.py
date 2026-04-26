"""Tests for export and executive-summary helpers."""

from __future__ import annotations

import unittest

import pandas as pd
import plotly.graph_objects as go

from src.reporting import (
    build_executive_summary,
    build_zone_kpi_overview,
    export_dataframe_csv,
    plotly_figure_to_html_bytes,
    plotly_figures_to_html_bytes,
    safe_plotly_png_bytes,
)


class ReportingExportTests(unittest.TestCase):
    def test_zone_kpi_overview_calculates_zone_cards(self) -> None:
        kpis = build_zone_kpi_overview(_zone_data(), metric_column="Selected_Metric")

        self.assertEqual(kpis.zone_count, 5)
        self.assertEqual(kpis.metric_average, 106)
        self.assertEqual(kpis.top_zone, "BGE")
        self.assertEqual(kpis.spread, 110)

    def test_zone_kpi_overview_empty_dataframe(self) -> None:
        kpis = build_zone_kpi_overview(pd.DataFrame(), metric_column="Selected_Metric")

        self.assertEqual(kpis.zone_count, 0)
        self.assertIsNone(kpis.metric_average)
        self.assertIsNone(kpis.top_zone)
        self.assertIsNone(kpis.spread)

    def test_zone_kpi_overview_missing_metric_still_counts_zones(self) -> None:
        kpis = build_zone_kpi_overview(pd.DataFrame({"Zone": ["BGE", "DPL", "BGE"]}), metric_column="Missing")

        self.assertEqual(kpis.zone_count, 2)
        self.assertIsNone(kpis.metric_average)
        self.assertIsNone(kpis.top_zone)
        self.assertIsNone(kpis.spread)

    def test_executive_summary_top_bottom_and_spread(self) -> None:
        summary = build_executive_summary(
            _zone_data(),
            selected_iso="PJM",
            selected_metric="Revenue per kW",
            selected_period="January 2024 to March 2024",
        )

        self.assertEqual(summary.top_zones, ["BGE", "DOM", "DPL"])
        self.assertEqual(summary.bottom_zones, ["JCPL", "PECO", "DPL"])
        self.assertEqual(summary.spread, 110)
        self.assertIn("PJM shows meaningful locational variation", summary.text)
        self.assertIn("Revenue per kW", summary.markdown)

    def test_executive_summary_empty_dataframe(self) -> None:
        summary = build_executive_summary(
            pd.DataFrame(),
            selected_iso="PJM",
            selected_metric="Cumulative Revenue",
            selected_period="selected range",
        )

        self.assertEqual(summary.top_zones, [])
        self.assertEqual(summary.bottom_zones, [])
        self.assertIsNone(summary.spread)
        self.assertIn("No zone performance data is available", summary.text)

    def test_summary_markdown_and_text_are_generated(self) -> None:
        summary = build_executive_summary(
            _zone_data(),
            selected_iso="PJM",
            selected_metric="Cumulative Revenue",
            selected_period="March 2024",
        )

        self.assertTrue(summary.markdown.startswith("# ISO Zone Performance Executive Summary"))
        self.assertTrue(summary.text.startswith("ISO Zone Performance Executive Summary"))
        self.assertIn("- Top zones:", summary.markdown)
        self.assertIn("Top zones:", summary.text)

    def test_export_dataframe_csv_returns_bytes(self) -> None:
        payload = export_dataframe_csv(_zone_data())

        self.assertIsInstance(payload, bytes)
        self.assertIn(b"Zone", payload)
        self.assertIn(b"BGE", payload)

    def test_plotly_html_export_returns_bytes(self) -> None:
        figure = go.Figure(go.Bar(x=[1, 2], y=["A", "B"], orientation="h"))

        payload = plotly_figure_to_html_bytes(figure)
        multi_payload = plotly_figures_to_html_bytes([figure, figure], title="Two figures")

        self.assertIsInstance(payload, bytes)
        self.assertIn(b"<html", payload.lower())
        self.assertIn(b"Plotly", payload)
        self.assertIn(b"Two figures", multi_payload)

    def test_safe_png_export_never_raises(self) -> None:
        figure = go.Figure(go.Bar(x=[1], y=["A"], orientation="h"))

        png_bytes, message = safe_plotly_png_bytes(figure)

        self.assertTrue(png_bytes is None or isinstance(png_bytes, bytes))
        if png_bytes is None:
            self.assertEqual(message, "PNG export requires Kaleido. HTML export is still available.")


def _zone_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Zone": ["BGE", "DPL", "PECO", "DOM", "JCPL"],
            "Selected_Metric": [150, 120, 80, 140, 40],
        }
    )


if __name__ == "__main__":
    unittest.main()
