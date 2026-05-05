"""Tests for export and executive-summary helpers."""

from __future__ import annotations

import unittest
import zipfile
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure

from src.reporting import (
    build_blog_post_draft,
    build_executive_summary,
    build_zone_kpi_overview,
    export_dataframe_csv,
    plotly_figure_to_html_bytes,
    plotly_figures_to_html_bytes,
    safe_plotly_png_bytes,
)
from src.visualization import animation_frames_to_html_player, gif_bytes_to_html_img, matplotlib_figure_to_png_bytes, matplotlib_figures_to_zip_bytes


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

    def test_matplotlib_png_export_returns_bytes(self) -> None:
        figure = _matplotlib_figure()

        payload = matplotlib_figure_to_png_bytes(figure)

        self.assertIsInstance(payload, bytes)
        self.assertTrue(payload.startswith(b"\x89PNG"))

    def test_matplotlib_figures_zip_export_contains_pngs(self) -> None:
        payload = matplotlib_figures_to_zip_bytes(
            [_matplotlib_figure(), _matplotlib_figure()],
            ["snapshot one", "snapshot_two.png"],
        )

        with zipfile.ZipFile(BytesIO(payload), mode="r") as archive:
            names = archive.namelist()
            self.assertEqual(names, ["snapshot_one.png", "snapshot_two.png"])
            self.assertTrue(archive.read("snapshot_one.png").startswith(b"\x89PNG"))

    def test_gif_html_fallback_returns_base64_img_tag(self) -> None:
        html = gif_bytes_to_html_img(b"GIF89a-demo", alt_text='PJM "demo"')

        self.assertIn('<img src="data:image/gif;base64,', html)
        self.assertIn('alt="PJM &quot;demo&quot;"', html)

    def test_animation_frame_player_returns_controls_and_scrubber(self) -> None:
        html = animation_frames_to_html_player([b"png-frame-1", b"png-frame-2"], ["January 2024", "February 2024"])

        self.assertIn("data:image/png;base64,", html)
        self.assertIn("Play", html)
        self.assertIn("Pause", html)
        self.assertIn('type="range"', html)
        self.assertIn("February 2024", html)

    def test_blog_post_draft_returns_markdown_with_required_sections(self) -> None:
        draft = build_blog_post_draft(
            _zone_data(),
            monthly_df=_monthly_blog_data(),
            iso="PJM",
            metric="Selected_Metric",
            start_date="2024-01",
            end_date="2024-03",
            audience="Battery developers",
            title_style="Location is everything",
            cta_text="Book a Flexworks walkthrough.",
            cta_link="https://example.com/demo",
        )

        self.assertTrue(draft.startswith("# Location Is Everything"))
        self.assertIn("## Simulation setup", draft)
        self.assertIn("## Results", draft)
        self.assertIn("## What the revenue data shows", draft)
        self.assertIn("## The power of Flexworks", draft)
        self.assertIn("BGE", draft)
        self.assertIn("DOM", draft)
        self.assertIn("JCPL", draft)
        self.assertIn("Percent spread: 275%", draft)
        self.assertIn("[Book a Flexworks walkthrough.](https://example.com/demo)", draft)

    def test_blog_post_draft_handles_empty_dataframe_gracefully(self) -> None:
        draft = build_blog_post_draft(pd.DataFrame(), iso="PJM", metric="Revenue_per_kW")

        self.assertIn("Not enough processed data to generate a blog draft", draft)

    def test_blog_post_draft_handles_missing_metric_gracefully(self) -> None:
        draft = build_blog_post_draft(pd.DataFrame({"Zone": ["BGE"], "Other": [1]}), iso="PJM", metric="Missing_Metric")

        self.assertIn("Missing_Metric", draft)
        self.assertIn("not available", draft)

    def test_blog_post_draft_handles_missing_cta_link(self) -> None:
        draft = build_blog_post_draft(
            _zone_data(),
            iso="PJM",
            metric="Selected_Metric",
            cta_text="Talk to the Flexworks team.",
            cta_link="",
        )

        self.assertIn("Talk to the Flexworks team.", draft)
        self.assertNotIn("[Talk to the Flexworks team.](", draft)

    def test_blog_post_draft_includes_review_note_when_assumptions_missing(self) -> None:
        draft = build_blog_post_draft(_zone_data(), iso="PJM", metric="Selected_Metric")

        self.assertIn("Review note: confirm final asset specifications and market assumptions before publication.", draft)


def _zone_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Zone": ["BGE", "DPL", "PECO", "DOM", "JCPL"],
            "Selected_Metric": [150, 120, 80, 140, 40],
        }
    )


def _matplotlib_figure() -> Figure:
    figure = Figure(figsize=(2, 2), dpi=100)
    axis = figure.add_subplot(111)
    axis.plot([1, 2], [3, 4])
    return figure


def _monthly_blog_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Zone": ["BGE", "DOM", "BGE", "DPL", "BGE", "JCPL"],
            "Month": ["2024-01", "2024-01", "2024-02", "2024-02", "2024-03", "2024-03"],
            "Revenue": [100, 80, 140, 30, 60, 10],
        }
    )


if __name__ == "__main__":
    unittest.main()
