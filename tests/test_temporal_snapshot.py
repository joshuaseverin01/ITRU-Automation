"""Tests for ISO zone snapshot time-series helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.analysis import (
    ALL_REVENUE_CATEGORIES,
    SNAPSHOT_METRIC_CUMULATIVE_REVENUE,
    SNAPSHOT_METRIC_MONTHLY_REVENUE,
    SNAPSHOT_METRIC_REVENUE_PER_KW,
    aggregate_zone_metric_over_range,
    aggregate_zone_metric,
    compute_cumulative_by_zone,
)
from src.geo import PjmZoneGeoJson
from src.temporal import (
    TIME_GRANULARITY_MONTHLY,
    TIME_GRANULARITY_NONE,
    TIME_GRANULARITY_TIMESTAMP,
    default_frame_count_for_range,
    detect_time_granularity,
    filter_time_range,
    select_evenly_spaced_snapshots,
)
from src.visualization import (
    _build_interpolated_animation_frames,
    _metric_color_hex,
    build_iso_zone_snapshot_map_bars,
    create_animated_zone_performance_figure,
    create_pjm_animation_gif_bytes,
    create_pjm_matplotlib_figure,
    map_value_to_rank_color,
)


class TemporalSnapshotTests(unittest.TestCase):
    def test_monthly_granularity_detection(self) -> None:
        dataframe = pd.DataFrame({"Month": ["2024-01", "2024-02"], "Revenue": [10, 20]})

        self.assertEqual(detect_time_granularity(dataframe), TIME_GRANULARITY_MONTHLY)

    def test_timestamp_granularity_detection_prefers_timestamp(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Month": ["2024-01", "2024-01"],
                "Timestamp": ["2024-01-01 01:00", "2024-01-01 02:00"],
                "Revenue": [10, 20],
            }
        )

        self.assertEqual(detect_time_granularity(dataframe), TIME_GRANULARITY_TIMESTAMP)

    def test_missing_time_column_detection(self) -> None:
        dataframe = pd.DataFrame({"Zone": ["BGE"], "Revenue": [10]})

        self.assertEqual(detect_time_granularity(dataframe), TIME_GRANULARITY_NONE)
        self.assertTrue(filter_time_range(dataframe, "2024-01", "2024-02").empty)

    def test_time_range_filtering_is_inclusive(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Month": ["2024-01", "2024-02", "2024-03"],
                "Revenue": [10, 20, 30],
            }
        )

        filtered = filter_time_range(dataframe, "2024-02", "2024-03")

        self.assertEqual(filtered["Revenue"].tolist(), [20, 30])

    def test_invalid_time_range_returns_empty(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Month": ["2024-01", "2024-02", "2024-03"],
                "Revenue": [10, 20, 30],
            }
        )

        filtered = filter_time_range(dataframe, "2024-03", "2024-01")

        self.assertTrue(filtered.empty)

    def test_compute_cumulative_by_zone_handles_negative_months(self) -> None:
        dataframe = _monthly_zone_frame()

        result = compute_cumulative_by_zone(dataframe, revenue_category=ALL_REVENUE_CATEGORIES)
        bge = result.loc[result["Zone"] == "BGE"].sort_values("Time")
        dpl = result.loc[result["Zone"] == "DPL"].sort_values("Time")

        self.assertEqual(bge["Monthly_Revenue"].tolist(), [100, -40, 30])
        self.assertEqual(bge["Cumulative_Revenue"].tolist(), [100, 60, 90])
        self.assertEqual(dpl["Cumulative_Revenue"].tolist(), [-10, -15, 5])

    def test_aggregate_zone_metric_selected_time_and_category(self) -> None:
        dataframe = _monthly_zone_frame()

        snapshot = aggregate_zone_metric(
            dataframe,
            metric=SNAPSHOT_METRIC_MONTHLY_REVENUE,
            category="Energy",
            time_point="2024-02",
        )

        self.assertEqual(snapshot["Zone"].tolist(), ["DPL", "BGE"])
        self.assertEqual(snapshot["Selected_Metric"].tolist(), [-5, -40])
        self.assertEqual(snapshot["Time_Label"].unique().tolist(), ["February 2024"])

    def test_revenue_per_kw_snapshot_uses_zone_average(self) -> None:
        dataframe = _monthly_zone_frame()

        snapshot = aggregate_zone_metric(
            dataframe,
            metric=SNAPSHOT_METRIC_REVENUE_PER_KW,
            category="Energy",
            time_point="2024-01",
        )

        self.assertEqual(snapshot.loc[snapshot["Zone"] == "BGE", "Selected_Metric"].iloc[0], 150)
        self.assertEqual(snapshot.loc[snapshot["Zone"] == "DPL", "Selected_Metric"].iloc[0], 90)

    def test_no_data_selected_range(self) -> None:
        dataframe = _monthly_zone_frame()

        filtered = filter_time_range(dataframe, "2025-01", "2025-02")
        snapshot = aggregate_zone_metric(
            filtered,
            metric=SNAPSHOT_METRIC_CUMULATIVE_REVENUE,
            category="Energy",
            time_point="2025-01",
        )

        self.assertTrue(filtered.empty)
        self.assertTrue(snapshot.empty)

    def test_aggregate_zone_metric_over_range_sums_revenue_metrics(self) -> None:
        dataframe = _monthly_zone_frame()

        monthly_range = aggregate_zone_metric_over_range(
            dataframe,
            metric=SNAPSHOT_METRIC_MONTHLY_REVENUE,
            category="Energy",
            start_time="2024-02",
            end_time="2024-03",
        )
        cumulative_range = aggregate_zone_metric_over_range(
            dataframe,
            metric=SNAPSHOT_METRIC_CUMULATIVE_REVENUE,
            category="Energy",
            start_time="2024-02",
            end_time="2024-03",
        )

        self.assertEqual(monthly_range["Zone"].tolist(), ["DPL", "BGE"])
        self.assertEqual(monthly_range["Selected_Metric"].tolist(), [15, -10])
        self.assertEqual(cumulative_range["Selected_Metric"].tolist(), [15, -10])
        self.assertEqual(monthly_range["Time_Label"].unique().tolist(), ["February 2024 to March 2024"])

    def test_aggregate_zone_metric_over_range_averages_revenue_per_kw(self) -> None:
        dataframe = _monthly_zone_frame()

        range_values = aggregate_zone_metric_over_range(
            dataframe,
            metric=SNAPSHOT_METRIC_REVENUE_PER_KW,
            category="Energy",
            start_time="2024-02",
            end_time="2024-03",
        )

        self.assertEqual(range_values.loc[range_values["Zone"] == "BGE", "Selected_Metric"].iloc[0], 150)
        self.assertEqual(range_values.loc[range_values["Zone"] == "DPL", "Selected_Metric"].iloc[0], 105)

    def test_aggregate_zone_metric_over_range_invalid_or_no_data_range(self) -> None:
        dataframe = _monthly_zone_frame()

        invalid = aggregate_zone_metric_over_range(
            dataframe,
            metric=SNAPSHOT_METRIC_MONTHLY_REVENUE,
            category="Energy",
            start_time="2024-03",
            end_time="2024-01",
        )
        no_data = aggregate_zone_metric_over_range(
            dataframe,
            metric=SNAPSHOT_METRIC_MONTHLY_REVENUE,
            category="Energy",
            start_time="2025-01",
            end_time="2025-02",
        )

        self.assertTrue(invalid.empty)
        self.assertTrue(no_data.empty)

    def test_select_evenly_spaced_snapshots_caps_to_available_points(self) -> None:
        dataframe = _monthly_zone_frame()

        snapshots = select_evenly_spaced_snapshots(dataframe, "2024-01", "2024-03", 10)

        self.assertEqual([snapshot.strftime("%Y-%m") for snapshot in snapshots], ["2024-01", "2024-02", "2024-03"])

    def test_select_evenly_spaced_snapshots_returns_exact_requested_number(self) -> None:
        dataframe = _long_monthly_frame()

        snapshots = select_evenly_spaced_snapshots(dataframe, "2024-01", "2024-06", 4)

        self.assertEqual(len(snapshots), 4)
        self.assertEqual(snapshots[0].strftime("%Y-%m"), "2024-01")
        self.assertEqual(snapshots[-1].strftime("%Y-%m"), "2024-06")

    def test_select_evenly_spaced_snapshots_invalid_range_returns_empty(self) -> None:
        dataframe = _long_monthly_frame()

        snapshots = select_evenly_spaced_snapshots(dataframe, "2024-06", "2024-01", 4)

        self.assertEqual(snapshots, [])

    def test_select_evenly_spaced_snapshots_single_available_timestamp(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Timestamp": ["2024-01-01 14:00"],
                "Zone": ["BGE"],
                "ISO_Region": ["PJM"],
                "Revenue": [25],
            }
        )

        snapshots = select_evenly_spaced_snapshots(dataframe, "2024-01-01 14:00", "2024-01-01 14:00", 6)

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].strftime("%Y-%m-%d %H:%M"), "2024-01-01 14:00")

    def test_select_evenly_spaced_snapshots_no_data_range(self) -> None:
        dataframe = _long_monthly_frame()

        snapshots = select_evenly_spaced_snapshots(dataframe, "2025-01", "2025-03", 3)

        self.assertEqual(snapshots, [])

    def test_animation_frame_selection_includes_all_months_when_frame_count_matches_range(self) -> None:
        dataframe = _monthly_n_month_frame(36)

        snapshots = select_evenly_spaced_snapshots(dataframe, "2022-01", "2024-12", 36)

        self.assertEqual(len(snapshots), 36)
        self.assertEqual([snapshot.strftime("%Y-%m") for snapshot in snapshots], pd.date_range("2022-01-01", periods=36, freq="MS").strftime("%Y-%m").tolist())

    def test_animation_frame_selection_downsamples_evenly_when_requested_is_smaller(self) -> None:
        dataframe = _monthly_n_month_frame(36)

        snapshots = select_evenly_spaced_snapshots(dataframe, "2022-01", "2024-12", 24)

        self.assertEqual(len(snapshots), 24)
        self.assertEqual(len(set(snapshots)), 24)
        self.assertEqual(snapshots, sorted(snapshots))
        self.assertEqual(snapshots[0].strftime("%Y-%m"), "2022-01")
        self.assertEqual(snapshots[-1].strftime("%Y-%m"), "2024-12")

    def test_default_animation_frame_count_uses_available_points_under_cap(self) -> None:
        dataframe = _monthly_n_month_frame(24)

        self.assertEqual(default_frame_count_for_range(dataframe, "2022-01", "2023-12", 60), 24)

    def test_default_animation_frame_count_returns_one_for_one_month_range(self) -> None:
        dataframe = _monthly_n_month_frame(36)

        snapshots = select_evenly_spaced_snapshots(dataframe, "2022-05", "2022-05", 60)

        self.assertEqual(default_frame_count_for_range(dataframe, "2022-05", "2022-05", 60), 1)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].strftime("%Y-%m"), "2022-05")

    def test_default_animation_frame_count_returns_zero_for_empty_range(self) -> None:
        dataframe = _monthly_n_month_frame(36)

        self.assertEqual(default_frame_count_for_range(dataframe, "2027-01", "2027-02", 60), 0)

    def test_missing_geojson_returns_fallback_message(self) -> None:
        snapshot = aggregate_zone_metric(
            _monthly_zone_frame(),
            metric=SNAPSHOT_METRIC_MONTHLY_REVENUE,
            category="Energy",
            time_point="2024-01",
        )

        chart, diagnostics = build_iso_zone_snapshot_map_bars(snapshot, "PJM", None)

        self.assertIsNone(chart.figure)
        self.assertEqual(chart.message, "PJM zone map requires the PJM GeoJSON file.")
        self.assertFalse(diagnostics["is_available"])

    def test_pjm_matplotlib_renderer_creates_static_map_and_bars(self) -> None:
        snapshot = aggregate_zone_metric(
            _monthly_zone_frame(),
            metric=SNAPSHOT_METRIC_MONTHLY_REVENUE,
            category="Energy",
            time_point="2024-01",
        )

        chart, diagnostics = create_pjm_matplotlib_figure(
            snapshot,
            _tiny_pjm_geojson(),
            metric="Selected_Metric",
            metric_label=SNAPSHOT_METRIC_MONTHLY_REVENUE,
            time_selection="January 2024",
            category_label="Energy",
        )

        self.assertTrue(diagnostics["is_available"])
        self.assertIsNotNone(chart.figure)
        self.assertTrue(hasattr(chart.figure, "savefig"))
        self.assertEqual(len(chart.figure.axes), 3)
        header_text = chart.figure.axes[0].texts[0].get_text()
        self.assertEqual(header_text, "PJM Zone Performance")
        bar_labels = [label.get_text() for label in chart.figure.axes[2].get_yticklabels()]
        self.assertEqual(set(bar_labels), {"BGE", "DPL"})

    def test_green_color_scale_maps_low_to_lighter_and_high_to_darker_green(self) -> None:
        low_color = _metric_color_hex(0, 0, 100)
        mid_color = _metric_color_hex(50, 0, 100)
        high_color = _metric_color_hex(100, 0, 100)

        self.assertEqual(low_color.lower(), "#d9fad7")
        self.assertEqual(high_color.lower(), "#1cb51c")
        self.assertGreater(_hex_luminance(low_color), _hex_luminance(mid_color))
        self.assertGreater(_hex_luminance(mid_color), _hex_luminance(high_color))
        self.assertGreater(_hex_luminance(low_color) - _hex_luminance(high_color), 0.25)

    def test_rank_color_scale_spreads_close_values_monotonically(self) -> None:
        values = [886, 893, 899, 901, 905, 913, 920, 924, 932, 938, 948]
        colors = [map_value_to_rank_color(value, values) for value in values]
        palette_order = ["#D9FAD7", "#B1F3AE", "#72E972", "#4AE34A", "#1CB51C"]
        indices = [palette_order.index(color) for color in colors]

        self.assertEqual(colors[0], "#D9FAD7")
        self.assertEqual(colors[-1], "#1CB51C")
        self.assertGreaterEqual(len(set(colors)), 5)
        self.assertEqual(indices, sorted(indices))

    def test_interpolated_animation_frames_increase_frame_count_and_preserve_endpoints(self) -> None:
        start_frame = pd.DataFrame({"Zone_Normalized": ["BGE"], "Zone": ["BGE"], "Selected_Metric": [10.0]})
        end_frame = pd.DataFrame({"Zone_Normalized": ["BGE"], "Zone": ["BGE"], "Selected_Metric": [70.0]})

        frames = _build_interpolated_animation_frames(
            [start_frame, end_frame],
            ["January 2024", "February 2024"],
            transition_frames_between_keyframes=5,
            max_rendered_frames=20,
        )
        values = [float(frame.loc[frame["Zone"] == "BGE", "Selected_Metric"].iloc[0]) for frame, _, _ in frames]

        self.assertEqual(len(frames), 7)
        self.assertEqual(values[0], 10)
        self.assertEqual(values[-1], 70)
        self.assertEqual(values[1:-1], [20, 30, 40, 50, 60])

    def test_animation_builder_creates_frames_and_controls(self) -> None:
        dataframe = _monthly_zone_frame()
        snapshots = [
            aggregate_zone_metric(
                dataframe,
                metric=SNAPSHOT_METRIC_CUMULATIVE_REVENUE,
                category="Energy",
                time_point=month,
            )
            for month in ["2024-01", "2024-02"]
        ]

        chart, diagnostics = create_animated_zone_performance_figure(
            snapshots,
            iso_region="PJM",
            pjm_geojson=_tiny_pjm_geojson(),
            metric_label=SNAPSHOT_METRIC_CUMULATIVE_REVENUE,
            category_label="Energy",
            frame_labels=["January 2024", "February 2024"],
        )

        self.assertTrue(diagnostics["is_available"])
        self.assertIsNotNone(chart.figure)
        self.assertEqual(len(chart.figure.frames), 2)
        self.assertEqual(chart.figure.frames[0].name, "January 2024")
        layout = chart.figure.to_dict()["layout"]
        self.assertIn("updatemenus", layout)
        self.assertIn("sliders", layout)

    def test_animation_builder_missing_geojson_returns_message(self) -> None:
        snapshot = aggregate_zone_metric(
            _monthly_zone_frame(),
            metric=SNAPSHOT_METRIC_CUMULATIVE_REVENUE,
            category="Energy",
            time_point="2024-01",
        )

        chart, diagnostics = create_animated_zone_performance_figure([snapshot], "PJM", None)

        self.assertIsNone(chart.figure)
        self.assertEqual(chart.message, "PJM zone map requires the PJM GeoJSON file for animation.")
        self.assertFalse(diagnostics["is_available"])

    def test_pjm_animation_gif_generation_returns_bytes(self) -> None:
        result = create_pjm_animation_gif_bytes(
            _monthly_zone_frame(),
            _tiny_pjm_geojson(),
            metric=SNAPSHOT_METRIC_CUMULATIVE_REVENUE,
            category="Energy",
            start_time="2024-01",
            end_time="2024-02",
            frame_count=2,
        )

        self.assertIsNone(result.message)
        self.assertIsNotNone(result.gif_bytes)
        self.assertTrue(result.gif_bytes.startswith(b"GIF"))
        self.assertEqual(result.frame_labels, ["January 2024", "February 2024"])

    def test_pjm_animation_cumulative_frames_start_from_selected_range(self) -> None:
        result = create_pjm_animation_gif_bytes(
            _positive_monthly_zone_frame(),
            _tiny_pjm_geojson(),
            metric=SNAPSHOT_METRIC_CUMULATIVE_REVENUE,
            category="Energy",
            start_time="2024-02",
            end_time="2024-04",
            frame_count=3,
        )

        bge_values = [
            float(frame.loc[frame["Zone"] == "BGE", "Selected_Metric"].iloc[0])
            for frame in result.frame_dataframes
        ]

        self.assertEqual(result.frame_labels, ["February 2024", "March 2024", "April 2024"])
        self.assertEqual(bge_values[0], 20)
        self.assertEqual(bge_values[-1], 90)
        self.assertEqual(bge_values, sorted(bge_values))

    def test_pjm_animation_gif_empty_dataframe_returns_message(self) -> None:
        result = create_pjm_animation_gif_bytes(
            pd.DataFrame(),
            _tiny_pjm_geojson(),
            metric=SNAPSHOT_METRIC_MONTHLY_REVENUE,
            category="Energy",
            start_time="2024-01",
            end_time="2024-02",
            frame_count=2,
        )

        self.assertIsNone(result.gif_bytes)
        self.assertEqual(result.message, "Animation requires time-series revenue data.")

    def test_pjm_animation_gif_supports_one_frame(self) -> None:
        result = create_pjm_animation_gif_bytes(
            _monthly_zone_frame(),
            _tiny_pjm_geojson(),
            metric=SNAPSHOT_METRIC_MONTHLY_REVENUE,
            category="Energy",
            start_time="2024-01",
            end_time="2024-01",
            frame_count=1,
        )

        self.assertIsNotNone(result.gif_bytes)
        self.assertTrue(result.gif_bytes.startswith(b"GIF"))
        self.assertEqual(result.frame_labels, ["January 2024"])


def _monthly_zone_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Device": ["A", "A", "A", "B", "B", "B"],
            "Zone": ["BGE", "BGE", "BGE", "DPL", "DPL", "DPL"],
            "ISO_Region": ["PJM"] * 6,
            "Revenue_Category": ["Energy"] * 6,
            "Month": ["2024-01", "2024-02", "2024-03", "2024-01", "2024-02", "2024-03"],
            "Revenue": [100, -40, 30, -10, -5, 20],
            "Revenue_per_kW": [150, 170, 130, 90, 100, 110],
        }
    )


def _long_monthly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Zone": ["BGE"] * 6,
            "ISO_Region": ["PJM"] * 6,
            "Revenue_Category": ["Energy"] * 6,
            "Month": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
            "Revenue": [10, 20, 30, 40, 50, 60],
        }
    )


def _monthly_n_month_frame(month_count: int) -> pd.DataFrame:
    months = pd.date_range("2022-01-01", periods=month_count, freq="MS")
    return pd.DataFrame(
        {
            "Zone": ["BGE"] * month_count,
            "ISO_Region": ["PJM"] * month_count,
            "Revenue_Category": ["Energy"] * month_count,
            "Month": months.strftime("%Y-%m").tolist(),
            "Revenue": list(range(month_count)),
        }
    )


def _positive_monthly_zone_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Device": ["A", "A", "A", "A"],
            "Zone": ["BGE", "BGE", "BGE", "BGE"],
            "ISO_Region": ["PJM"] * 4,
            "Revenue_Category": ["Energy"] * 4,
            "Month": ["2024-01", "2024-02", "2024-03", "2024-04"],
            "Revenue": [10, 20, 30, 40],
        }
    )


def _hex_luminance(hex_color: str) -> float:
    cleaned = hex_color.lstrip("#")
    red = int(cleaned[0:2], 16) / 255
    green = int(cleaned[2:4], 16) / 255
    blue = int(cleaned[4:6], 16) / 255
    return (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)


def _tiny_pjm_geojson() -> PjmZoneGeoJson:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"Zone": "BGE", "Zone_Normalized": "BGE"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-77.0, 39.0], [-76.0, 39.0], [-76.0, 40.0], [-77.0, 40.0], [-77.0, 39.0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"Zone": "DPL", "Zone_Normalized": "DPL"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-76.0, 38.0], [-75.0, 38.0], [-75.0, 39.0], [-76.0, 39.0], [-76.0, 38.0]]],
                },
            },
        ],
    }
    return PjmZoneGeoJson(geojson=geojson, zone_property="Zone", zone_count=2, zones=["BGE", "DPL"])


if __name__ == "__main__":
    unittest.main()
