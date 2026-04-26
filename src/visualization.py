"""Defensive visualization builders for Plotly and matplotlib charts."""

from __future__ import annotations

import math
import os
import tempfile
import base64
import hashlib
import json
import zipfile
from io import BytesIO
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig_flexworks"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from plotly.subplots import make_subplots

from .geo import PjmZoneGeoJson, detect_coordinate_status, prepare_pjm_zone_choropleth_data


@dataclass(frozen=True)
class ChartResult:
    """Container for a figure or a user-facing skip message."""

    figure: object | None
    message: str | None = None


@dataclass(frozen=True)
class GifAnimationResult:
    """Container for a rendered GIF animation and its source frames."""

    gif_bytes: bytes | None
    message: str | None
    diagnostics: dict[str, object]
    frame_dataframes: list[pd.DataFrame]
    frame_labels: list[str]
    frame_png_bytes: list[bytes] | None = None
    rendered_frame_labels: list[str] | None = None


@dataclass(frozen=True)
class MatplotlibZoneGeometry:
    """Prepared PJM polygon geometry for high-fidelity matplotlib rendering."""

    zone_name: str
    normalized_zone: str
    polygons: tuple[tuple[np.ndarray, ...], ...]
    path: MplPath
    label_point: tuple[float, float]


@dataclass(order=True)
class _PolylabelCell:
    sort_index: float
    x: float
    y: float
    h: float
    distance: float
    max_distance: float


CHOROPLETH_METRIC_LABELS = {
    "Revenue_per_kW": "Revenue per kW ($/kW)",
    "Annualized_Revenue": "Annualized Revenue",
    "Opportunity_Score": "Opportunity Score",
    "Risk_Adjusted_Score": "Risk Adjusted Score",
}
ISO_ZONE_SNAPSHOT_METRIC_LABELS = {
    "Monthly_Revenue": "Monthly Revenue",
    "Cumulative_Revenue": "Cumulative Revenue",
    "Revenue_per_kW": "Revenue per kW",
    "Selected_Metric": "Selected Metric",
}
REVENUE_GREEN_COLORSCALE = [
    (0.0, "#D9FAD7"),
    (0.09, "#D0F8CC"),
    (0.18, "#C6F7C2"),
    (0.27, "#BCF5B8"),
    (0.36, "#B1F3AE"),
    (0.45, "#A7F2A4"),
    (0.55, "#9BF09A"),
    (0.64, "#90EE90"),
    (0.73, "#72E972"),
    (0.82, "#4AE34A"),
    (0.91, "#22DD22"),
    (1.0, "#1CB51C"),
]
MATPLOTLIB_REVENUE_PALETTE = [
    "#D9FAD7",
    "#D0F8CC",
    "#C6F7C2",
    "#BCF5B8",
    "#B1F3AE",
    "#A7F2A4",
    "#9BF09A",
    "#90EE90",
    "#72E972",
    "#4AE34A",
    "#22DD22",
    "#1CB51C",
]
MATPLOTLIB_NO_DATA_COLOR = "#f3f4f6"
MATPLOTLIB_NEGATIVE_BAR_COLOR = "#b91c1c"
MATPLOTLIB_FIGURE_BG = "#F6FFF6"
MATPLOTLIB_PANEL_BG = "#FFFFFF"
MATPLOTLIB_TEXT_COLOR = "#1F2937"
MATPLOTLIB_MUTED_TEXT_COLOR = "#1f2937"
MATPLOTLIB_LABEL_STROKE = "#F8FFF8"
MATPLOTLIB_MAP_PADDING_RATIO = 0.035
MATPLOTLIB_LABEL_NUDGES = {
    "BGE": (-0.06, -0.03),
    "DPL": (0.12, -0.03),
    "JCPL": (-0.08, -0.02),
    "RECO": (0.14, 0.09),
    "METED": (-0.02, -0.03),
    "PECO": (0.04, -0.05),
    "PSEG": (0.05, -0.08),
    "AECO": (0.09, -0.03),
}
MATPLOTLIB_LABEL_SIZE_OVERRIDES = {
    "RECO": 6.1,
    "JCPL": 6.2,
    "PSEG": 6.2,
    "PECO": 6.3,
    "DPL": 6.3,
    "AECO": 6.3,
    "BGE": 6.4,
    "METED": 6.4,
}
_MATPLOTLIB_GEOMETRY_CACHE: dict[
    tuple[tuple[str, str, int, float, float, float, float], ...],
    tuple[list[MatplotlibZoneGeometry], tuple[float, float, float, float]],
] = {}


def build_node_map(dataframe: pd.DataFrame) -> ChartResult:
    """Build a geographic node map when usable coordinates are available."""

    if dataframe.empty:
        return ChartResult(None, "No rows are available for the map.")

    status = detect_coordinate_status(dataframe)
    if not status.has_coordinates:
        return ChartResult(None, status.message)

    required = {"Node_ID", "Latitude", "Longitude"}
    if not required.issubset(dataframe.columns):
        return ChartResult(None, status.message)

    map_data = dataframe.copy()
    map_data["Latitude"] = pd.to_numeric(map_data["Latitude"], errors="coerce")
    map_data["Longitude"] = pd.to_numeric(map_data["Longitude"], errors="coerce")
    map_data = map_data.dropna(subset=["Latitude", "Longitude"])
    if map_data.empty:
        return ChartResult(None, "No rows have complete coordinates for the map.")

    color_column = "Opportunity_Score" if "Opportunity_Score" in map_data.columns else None
    size_values = _positive_size_values(map_data, "Revenue_per_kW")
    hover_columns = [column for column in ("ISO_Region", "Annualized_Revenue", "Revenue_per_kW", "LMP_Volatility", "Risk_Label") if column in map_data.columns]

    figure = px.scatter_geo(
        map_data,
        lat="Latitude",
        lon="Longitude",
        color=color_column,
        size=size_values,
        hover_name="Node_ID",
        hover_data=hover_columns,
        scope="usa",
        projection="albers usa",
        color_continuous_scale="Viridis",
        title="Node Opportunity Map",
    )
    figure.update_layout(margin=dict(l=0, r=0, t=48, b=0), height=460)
    return ChartResult(figure, status.message)


def build_pjm_zone_choropleth(
    dataframe: pd.DataFrame,
    pjm_geojson: PjmZoneGeoJson | None,
    metric_column: str,
) -> tuple[ChartResult, object]:
    """Build a PJM zone polygon choropleth when polygons join to results."""

    join_result = prepare_pjm_zone_choropleth_data(dataframe, pjm_geojson, metric_column)
    diagnostics = join_result.diagnostics
    if not diagnostics.is_available or join_result.geojson is None or join_result.dataframe.empty:
        return ChartResult(None, diagnostics.message), diagnostics

    if metric_column not in join_result.dataframe.columns:
        return ChartResult(None, f"The selected metric '{metric_column}' is not available."), diagnostics

    chart_data = join_result.dataframe.copy()
    chart_data[metric_column] = pd.to_numeric(chart_data[metric_column], errors="coerce")
    chart_data = chart_data.dropna(subset=[metric_column])
    if chart_data.empty:
        return ChartResult(None, "No matched PJM zones have numeric values for the selected metric."), diagnostics

    metric_label = CHOROPLETH_METRIC_LABELS.get(metric_column, metric_column.replace("_", " "))
    chart_result, _ = create_pjm_matplotlib_figure(
        chart_data,
        pjm_geojson,
        metric=metric_column,
        metric_label=metric_label,
        time_selection="active dataset",
        category_label="All zones",
        time_context_label="Dataset",
        title=f"PJM Zone Choropleth: {metric_label}",
        compact=False,
    )
    return ChartResult(chart_result.figure, chart_result.message or diagnostics.message), diagnostics


def build_top_nodes_bar(dataframe: pd.DataFrame, top_n: int = 10) -> ChartResult:
    """Build a top-node bar chart using opportunity score when available."""

    if dataframe.empty:
        return ChartResult(None, "No rows are available for the top-node chart.")
    if "Node_ID" not in dataframe.columns:
        return ChartResult(None, "Node_ID is required for the top-node chart.")

    metric_column = "Opportunity_Score" if "Opportunity_Score" in dataframe.columns else "Revenue_per_kW"
    if metric_column not in dataframe.columns:
        return ChartResult(None, "Opportunity score or Revenue_per_kW is required for the top-node chart.")

    chart_data = dataframe.copy()
    chart_data[metric_column] = pd.to_numeric(chart_data[metric_column], errors="coerce")
    chart_data = chart_data.dropna(subset=[metric_column])
    if chart_data.empty:
        return ChartResult(None, f"No numeric values are available for {metric_column}.")

    chart_data = chart_data.sort_values(metric_column, ascending=False).head(top_n)
    color_column = "ISO_Region" if "ISO_Region" in chart_data.columns else None
    hover_columns = [column for column in ("Annualized_Revenue", "Revenue_per_kW", "LMP_Volatility", "Risk_Label") if column in chart_data.columns]

    figure = px.bar(
        chart_data,
        x=metric_column,
        y="Node_ID",
        color=color_column,
        orientation="h",
        hover_data=hover_columns,
        title=f"Top {min(top_n, len(chart_data))} Nodes by {metric_column.replace('_', ' ')}",
    )
    figure.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=48, b=0), height=420)
    return ChartResult(figure)


def build_volatility_revenue_scatter(dataframe: pd.DataFrame) -> ChartResult:
    """Build a volatility-vs-revenue scatter plot."""

    if dataframe.empty:
        return ChartResult(None, "No rows are available for the scatter plot.")

    required = {"LMP_Volatility", "Revenue_per_kW"}
    if not required.issubset(dataframe.columns):
        return ChartResult(None, "LMP_Volatility and Revenue_per_kW are required for the scatter plot.")

    chart_data = dataframe.copy()
    chart_data["LMP_Volatility"] = pd.to_numeric(chart_data["LMP_Volatility"], errors="coerce")
    chart_data["Revenue_per_kW"] = pd.to_numeric(chart_data["Revenue_per_kW"], errors="coerce")
    chart_data = chart_data.dropna(subset=["LMP_Volatility", "Revenue_per_kW"])
    if chart_data.empty:
        return ChartResult(None, "No rows have both volatility and revenue-per-kW values.")

    color_column = "Risk_Label" if "Risk_Label" in chart_data.columns else None
    size_column = "Annualized_Revenue" if "Annualized_Revenue" in chart_data.columns else None
    hover_name = "Node_ID" if "Node_ID" in chart_data.columns else None
    hover_columns = [column for column in ("ISO_Region", "Opportunity_Score", "Annualized_Revenue") if column in chart_data.columns]

    figure = px.scatter(
        chart_data,
        x="LMP_Volatility",
        y="Revenue_per_kW",
        color=color_column,
        size=size_column,
        hover_name=hover_name,
        hover_data=hover_columns,
        title="Volatility vs Revenue per kW",
    )
    figure.update_layout(margin=dict(l=0, r=0, t=48, b=0), height=430)
    return ChartResult(figure)


def build_monthly_revenue_chart(dataframe: pd.DataFrame, group_by: str = "Revenue_Category") -> ChartResult:
    """Build a monthly revenue line chart from long-format monthly revenue data."""

    if dataframe.empty:
        return ChartResult(None, "No monthly revenue rows are available.")

    required = {"Month", "Revenue"}
    if not required.issubset(dataframe.columns):
        return ChartResult(None, "Monthly revenue data requires Month and Revenue columns.")

    chart_data = dataframe.copy()
    chart_data["Month"] = pd.to_datetime(chart_data["Month"], errors="coerce")
    chart_data["Revenue"] = pd.to_numeric(chart_data["Revenue"], errors="coerce")
    chart_data = chart_data.dropna(subset=["Month", "Revenue"])
    if chart_data.empty:
        return ChartResult(None, "No rows have both Month and Revenue values.")

    if group_by not in chart_data.columns:
        group_by = "Revenue_Category" if "Revenue_Category" in chart_data.columns else ""

    group_columns = ["Month"] + ([group_by] if group_by else [])
    grouped = chart_data.groupby(group_columns, dropna=False, as_index=False)["Revenue"].sum()

    figure = px.line(
        grouped,
        x="Month",
        y="Revenue",
        color=group_by if group_by else None,
        markers=True,
        title=f"Monthly Revenue by {group_by.replace('_', ' ')}" if group_by else "Monthly Revenue",
    )
    figure.update_layout(margin=dict(l=0, r=0, t=48, b=0), height=430)
    return ChartResult(figure)


def build_monthly_revenue_bar(dataframe: pd.DataFrame, group_by: str = "Zone") -> ChartResult:
    """Build a total monthly revenue summary bar chart."""

    if dataframe.empty:
        return ChartResult(None, "No monthly revenue rows are available.")
    if "Revenue" not in dataframe.columns:
        return ChartResult(None, "Monthly revenue data requires a Revenue column.")

    chart_data = dataframe.copy()
    chart_data["Revenue"] = pd.to_numeric(chart_data["Revenue"], errors="coerce")
    chart_data = chart_data.dropna(subset=["Revenue"])
    if chart_data.empty:
        return ChartResult(None, "No numeric revenue values are available.")

    if group_by not in chart_data.columns:
        group_by = "Revenue_Category" if "Revenue_Category" in chart_data.columns else "Device"
    if group_by not in chart_data.columns:
        return ChartResult(None, "No grouping column is available for monthly revenue.")

    grouped = chart_data.groupby(group_by, dropna=False, as_index=False)["Revenue"].sum()
    grouped = grouped.sort_values("Revenue", ascending=False)

    figure = px.bar(
        grouped,
        x=group_by,
        y="Revenue",
        title=f"Total Monthly Revenue by {group_by.replace('_', ' ')}",
    )
    figure.update_layout(margin=dict(l=0, r=0, t=48, b=0), height=380)
    return ChartResult(figure)


def matplotlib_figure_to_png_bytes(figure: Figure) -> bytes:
    """Serialize a matplotlib figure to PNG bytes."""

    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=160,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    return buffer.getvalue()


def matplotlib_figures_to_zip_bytes(figures: list[Figure], names: list[str]) -> bytes:
    """Serialize matplotlib figures into a ZIP archive of PNG files."""

    if len(figures) != len(names):
        raise ValueError("The number of figures must match the number of file names.")

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for figure, name in zip(figures, names):
            safe_name = _safe_png_name(name)
            archive.writestr(safe_name, matplotlib_figure_to_png_bytes(figure))
    return buffer.getvalue()


def gif_bytes_to_html_img(gif_bytes: bytes, alt_text: str = "PJM animation preview") -> str:
    """Return a small HTML img tag for reliable in-browser GIF playback."""

    encoded = base64.b64encode(gif_bytes).decode("ascii")
    escaped_alt = alt_text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<img src="data:image/gif;base64,{encoded}" alt="{escaped_alt}" '
        'style="width:100%;height:auto;border-radius:8px;display:block;" />'
    )


def animation_frames_to_html_player(
    frame_png_bytes: list[bytes] | None,
    frame_labels: list[str] | None,
    alt_text: str = "PJM animation frame",
) -> str:
    """Return a self-contained HTML frame player with play/pause and scrubbing."""

    if not frame_png_bytes:
        return ""

    labels = list(frame_labels or [])
    if len(labels) < len(frame_png_bytes):
        labels.extend([""] * (len(frame_png_bytes) - len(labels)))
    labels = labels[: len(frame_png_bytes)]
    encoded_frames = [f"data:image/png;base64,{base64.b64encode(frame).decode('ascii')}" for frame in frame_png_bytes]
    player_id = hashlib.sha1(("".join(labels) + str(sum(len(frame) for frame in frame_png_bytes))).encode("utf-8")).hexdigest()[:10]
    escaped_alt = alt_text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    frames_json = json.dumps(encoded_frames)
    labels_json = json.dumps(labels)
    first_label = labels[0].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") if labels else ""

    return f"""
    <div id="flexworks-player-{player_id}" style="font-family: Arial, sans-serif; width: 100%; color: #111827;">
      <img
        id="flexworks-player-{player_id}-image"
        src="{encoded_frames[0]}"
        alt="{escaped_alt}"
        style="width: 100%; height: auto; border-radius: 8px; display: block; background: #f3f4f6;"
      />
      <div style="display: flex; align-items: center; gap: 0.55rem; margin-top: 0.6rem;">
        <button id="flexworks-player-{player_id}-play" style="background: #166534; color: white; border: 0; border-radius: 6px; padding: 0.42rem 0.72rem; cursor: pointer;">Play</button>
        <button id="flexworks-player-{player_id}-pause" style="background: #334155; color: white; border: 0; border-radius: 6px; padding: 0.42rem 0.72rem; cursor: pointer;">Pause</button>
        <input id="flexworks-player-{player_id}-slider" type="range" min="0" max="{len(encoded_frames) - 1}" value="0" step="1" style="flex: 1;" />
        <span id="flexworks-player-{player_id}-label" style="min-width: 9rem; text-align: right; font-size: 0.9rem; color: #1f2937;">{first_label}</span>
      </div>
    </div>
    <script>
      (() => {{
        const frames = {frames_json};
        const labels = {labels_json};
        const image = document.getElementById("flexworks-player-{player_id}-image");
        const slider = document.getElementById("flexworks-player-{player_id}-slider");
        const label = document.getElementById("flexworks-player-{player_id}-label");
        const playButton = document.getElementById("flexworks-player-{player_id}-play");
        const pauseButton = document.getElementById("flexworks-player-{player_id}-pause");
        let frameIndex = 0;
        let timer = null;

        function showFrame(index) {{
          frameIndex = Math.max(0, Math.min(frames.length - 1, Number(index)));
          image.src = frames[frameIndex];
          slider.value = String(frameIndex);
          label.textContent = labels[frameIndex] || "";
        }}

        function pause() {{
          if (timer !== null) {{
            window.clearInterval(timer);
            timer = null;
          }}
        }}

        function play() {{
          if (timer !== null || frames.length <= 1) {{
            return;
          }}
          timer = window.setInterval(() => {{
            showFrame((frameIndex + 1) % frames.length);
          }}, 90);
        }}

        slider.addEventListener("input", () => {{
          pause();
          showFrame(slider.value);
        }});
        playButton.addEventListener("click", play);
        pauseButton.addEventListener("click", pause);
        showFrame(0);
      }})();
    </script>
    """


def create_pjm_matplotlib_figure(
    dataframe: pd.DataFrame,
    pjm_geojson: PjmZoneGeoJson | None,
    metric: str = "Selected_Metric",
    time_selection: str | None = None,
    *,
    metric_label: str | None = None,
    category_label: str | None = None,
    time_context_label: str = "Selected time",
    title: str = "PJM Zone Performance",
    compact: bool = False,
    sort_order: str = "Top zones",
    metric_range: tuple[float, float] | None = None,
) -> tuple[ChartResult, dict[str, object]]:
    """Render a high-fidelity PJM polygon map with ranked bars using matplotlib.

    This renderer intentionally avoids Plotly choropleth geometry for the static
    PJM views. It builds real matplotlib paths from the GeoJSON coordinates and
    labels zones using a polylabel-style pole-of-inaccessibility calculation.
    """

    diagnostics = _cumulative_revenue_diagnostics(dataframe, pjm_geojson)
    if pjm_geojson is None:
        return ChartResult(None, "PJM zone map requires the PJM GeoJSON file."), diagnostics
    if dataframe.empty:
        return ChartResult(None, "No PJM zone rows are available for the selected view."), diagnostics
    if "Zone_Normalized" not in dataframe.columns:
        return ChartResult(None, "PJM map rendering requires normalized zone names."), diagnostics
    if metric not in dataframe.columns:
        return ChartResult(None, f"{metric} is not available for this PJM map."), diagnostics
    if not diagnostics["is_available"]:
        return ChartResult(None, "No PJM zones matched the GeoJSON."), diagnostics

    try:
        chart_data = dataframe.copy()
        chart_data[metric] = pd.to_numeric(chart_data[metric], errors="coerce")
        chart_data = chart_data.loc[chart_data["Zone_Normalized"].isin(pjm_geojson.zones)].dropna(subset=[metric])
        if chart_data.empty:
            return ChartResult(None, "Matched PJM zones do not have numeric values for the selected metric."), diagnostics

        chart_data = _collapse_matplotlib_zone_data(chart_data, metric)
        geometries, bbox = _matplotlib_zone_geometries(pjm_geojson)
        if not geometries:
            return ChartResult(None, "The PJM GeoJSON did not contain renderable polygon geometry."), diagnostics

        metric_label = metric_label or _first_display_value(
            chart_data,
            "Metric_Label",
            ISO_ZONE_SNAPSHOT_METRIC_LABELS.get(metric, CHOROPLETH_METRIC_LABELS.get(metric, metric.replace("_", " "))),
        )
        time_label = time_selection or _first_display_value(chart_data, "Time_Label", "selected time")
        category_label = category_label or _first_display_value(chart_data, "Revenue_Category_Filter", "All categories")
        subtitle = f"{time_context_label}: {time_label} | Metric: {metric_label} | Category: {category_label}"

        figure = _draw_pjm_matplotlib_map_bars(
            geometries=geometries,
            bbox=bbox,
            chart_data=chart_data,
            metric=metric,
            metric_label=metric_label,
            title=title,
            subtitle=subtitle,
            compact=compact,
            sort_order=sort_order,
            metric_range=metric_range,
        )
        return ChartResult(figure, None), diagnostics
    except Exception as exc:
        return ChartResult(None, f"PJM matplotlib rendering failed: {exc}"), diagnostics


def create_pjm_animation_gif_bytes(
    dataframe: pd.DataFrame | None,
    pjm_geojson: PjmZoneGeoJson | None,
    *,
    metric: str,
    start_time: object,
    end_time: object,
    frame_count: int,
    category: str,
    iso_region: str = "PJM",
    duration_ms: int = 300,
    transition_frames_between_keyframes: int = 5,
    transition_duration_ms: int = 60,
    max_rendered_frames: int = 300,
    progress_callback: Callable[[float], None] | None = None,
) -> GifAnimationResult:
    """Create a PJM map-and-bars GIF using the matplotlib static renderer."""

    empty_diagnostics = _cumulative_revenue_diagnostics(pd.DataFrame(), pjm_geojson)
    if pjm_geojson is None:
        return GifAnimationResult(None, "PJM zone map requires the PJM GeoJSON file for animation.", empty_diagnostics, [], [])
    if dataframe is None or dataframe.empty:
        return GifAnimationResult(None, "Animation requires time-series revenue data.", empty_diagnostics, [], [])
    if str(iso_region or "").upper() != "PJM":
        return GifAnimationResult(None, f"{iso_region or 'Selected ISO'} zone animation is not configured yet.", empty_diagnostics, [], [])

    try:
        from .analysis import aggregate_zone_metric
        from .temporal import filter_time_range, format_time_label, format_time_range_label, select_evenly_spaced_snapshots

        requested_frames = max(1, int(frame_count))
        range_dataframe = filter_time_range(dataframe, start_time, end_time)
        if range_dataframe.empty:
            return GifAnimationResult(None, "The selected range has no valid time points for animation.", empty_diagnostics, [], [])

        selected_times = select_evenly_spaced_snapshots(range_dataframe, start_time, end_time, requested_frames)
        if not selected_times:
            return GifAnimationResult(None, "The selected range has no valid time points for animation.", empty_diagnostics, [], [])

        frame_dataframes: list[pd.DataFrame] = []
        frame_labels: list[str] = []
        for selected_time in selected_times:
            snapshot = aggregate_zone_metric(
                range_dataframe,
                metric=metric,
                category=category,
                time_point=selected_time,
                iso_region=iso_region,
            )
            if snapshot.empty:
                continue
            frame_dataframes.append(snapshot)
            frame_labels.append(format_time_label(selected_time, str(snapshot["Time_Granularity"].dropna().iloc[0]) if "Time_Granularity" in snapshot.columns else "monthly"))

        if not frame_dataframes:
            return GifAnimationResult(None, "No animation frames could be built for the selected range.", empty_diagnostics, [], [])

        combined = pd.concat(frame_dataframes, ignore_index=True)
        diagnostics = _cumulative_revenue_diagnostics(combined, pjm_geojson)
        if not diagnostics["is_available"]:
            return GifAnimationResult(None, "No animation snapshot zones matched the PJM GeoJSON.", diagnostics, frame_dataframes, frame_labels)

        metric_range = _global_metric_range(frame_dataframes, "Selected_Metric")
        period_label = format_time_range_label(start_time, end_time, str(combined["Time_Granularity"].dropna().iloc[0]) if "Time_Granularity" in combined.columns else "monthly")
        render_frames = _build_interpolated_animation_frames(
            frame_dataframes,
            frame_labels,
            transition_frames_between_keyframes=transition_frames_between_keyframes,
            max_rendered_frames=max_rendered_frames,
        )
        total_frames = len(render_frames)
        images: list[Image.Image] = []
        frame_png_bytes: list[bytes] = []
        rendered_frame_labels: list[str] = []
        durations: list[int] = []
        for index, (frame_data, frame_label, is_key_frame) in enumerate(render_frames):
            chart_result, _ = create_pjm_matplotlib_figure(
                frame_data,
                pjm_geojson,
                metric="Selected_Metric",
                metric_label=metric,
                time_selection=f"{frame_label} ({period_label})",
                category_label=category,
                time_context_label="Frame",
                title="PJM Zone Performance",
                compact=False,
                metric_range=metric_range,
            )
            if chart_result.figure is None:
                return GifAnimationResult(None, chart_result.message or "Animation frame rendering failed.", diagnostics, frame_dataframes, frame_labels)
            frame_png = _matplotlib_figure_to_frame_png_bytes(chart_result.figure)
            frame_png_bytes.append(frame_png)
            rendered_frame_labels.append(frame_label)
            images.append(_png_bytes_to_palette_image(frame_png))
            durations.append(duration_ms if is_key_frame else transition_duration_ms)
            if progress_callback is not None:
                progress_callback((index + 1) / total_frames)

        output = BytesIO()
        images[0].save(
            output,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=durations if len(durations) > 1 else duration_ms,
            loop=0,
            disposal=2,
            optimize=False,
        )
        return GifAnimationResult(output.getvalue(), None, diagnostics, frame_dataframes, frame_labels, frame_png_bytes, rendered_frame_labels)
    except Exception as exc:
        return GifAnimationResult(None, f"PJM GIF animation rendering failed: {exc}", empty_diagnostics, [], [])


def _build_interpolated_animation_frames(
    frame_dataframes: list[pd.DataFrame],
    frame_labels: list[str],
    *,
    transition_frames_between_keyframes: int,
    max_rendered_frames: int,
) -> list[tuple[pd.DataFrame, str, bool]]:
    if not frame_dataframes:
        return []
    if len(frame_dataframes) == 1:
        return [(frame_dataframes[0], frame_labels[0], True)]

    transition_count = _resolved_transition_frame_count(
        key_frame_count=len(frame_dataframes),
        requested_transition_count=transition_frames_between_keyframes,
        max_rendered_frames=max_rendered_frames,
    )
    render_frames: list[tuple[pd.DataFrame, str, bool]] = []
    for index, (frame_data, frame_label) in enumerate(zip(frame_dataframes, frame_labels)):
        render_frames.append((frame_data, frame_label, True))
        if transition_count <= 0 or index >= len(frame_dataframes) - 1:
            continue

        next_frame = frame_dataframes[index + 1]
        for transition_index in range(1, transition_count + 1):
            fraction = transition_index / (transition_count + 1)
            interpolated = _interpolate_zone_metric_frame(frame_data, next_frame, fraction)
            render_frames.append((interpolated, frame_label, False))
    return render_frames


def _resolved_transition_frame_count(
    *,
    key_frame_count: int,
    requested_transition_count: int,
    max_rendered_frames: int,
) -> int:
    if key_frame_count <= 1 or requested_transition_count <= 0:
        return 0
    safe_max_frames = max(key_frame_count, int(max_rendered_frames))
    available_transition_slots = safe_max_frames - key_frame_count
    if available_transition_slots <= 0:
        return 0
    return min(int(requested_transition_count), available_transition_slots // (key_frame_count - 1))


def _interpolate_zone_metric_frame(start_frame: pd.DataFrame, end_frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if start_frame.empty or end_frame.empty or "Zone_Normalized" not in start_frame.columns or "Zone_Normalized" not in end_frame.columns:
        return start_frame.copy()

    start_by_zone = _frame_rows_by_zone(start_frame)
    end_by_zone = _frame_rows_by_zone(end_frame)
    all_zones = sorted(set(start_by_zone).union(end_by_zone))
    rows: list[dict[str, object]] = []
    for zone in all_zones:
        start_row = start_by_zone.get(zone, {})
        end_row = end_by_zone.get(zone, {})
        base_row = dict(start_row or end_row)
        for column in _interpolatable_numeric_columns(start_row, end_row):
            start_value = _to_float(start_row.get(column))
            end_value = _to_float(end_row.get(column))
            if start_value is None and end_value is None:
                continue
            if start_value is None:
                base_row[column] = end_value
            elif end_value is None:
                base_row[column] = start_value
            else:
                base_row[column] = start_value + ((end_value - start_value) * fraction)
        rows.append(base_row)
    return pd.DataFrame(rows)


def _frame_rows_by_zone(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    deduped = frame.dropna(subset=["Zone_Normalized"]).drop_duplicates(subset=["Zone_Normalized"], keep="first")
    return {str(row["Zone_Normalized"]): row.to_dict() for _, row in deduped.iterrows()}


def _interpolatable_numeric_columns(start_row: dict[str, object], end_row: dict[str, object]) -> list[str]:
    preferred_columns = [
        "Selected_Metric",
        "Monthly_Revenue",
        "Cumulative_Revenue",
        "Revenue_per_kW",
        "Annualized_Revenue",
        "Opportunity_Score",
        "Risk_Adjusted_Score",
        "LMP_Volatility",
    ]
    available = set(start_row).union(end_row)
    return [column for column in preferred_columns if column in available]


def _global_metric_range(frame_dataframes: list[pd.DataFrame], metric: str) -> tuple[float, float] | None:
    values: list[float] = []
    for frame in frame_dataframes:
        if metric not in frame.columns:
            continue
        numeric_values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        values.extend(float(value) for value in numeric_values.tolist())
    if not values:
        return None

    value_min = min(values)
    value_max = max(values)
    if math.isclose(value_min, value_max):
        value_min -= 1.0
        value_max += 1.0
    return value_min, value_max


def _revenue_green_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("flexworks_revenue_green", MATPLOTLIB_REVENUE_PALETTE)


def map_value_to_color(value: float, vmin: float, vmax: float) -> str:
    if math.isclose(float(vmin), float(vmax)):
        normalized = 0.5
    else:
        normalized = (float(value) - float(vmin)) / (float(vmax) - float(vmin))
    index = int(round(float(np.clip(normalized, 0.0, 1.0)) * (len(MATPLOTLIB_REVENUE_PALETTE) - 1)))
    return MATPLOTLIB_REVENUE_PALETTE[index]


def _metric_color(value: float, value_min: float, value_max: float, cmap: LinearSegmentedColormap) -> object:
    return map_value_to_color(value, value_min, value_max)


def _legacy_metric_color(value: float, value_min: float, value_max: float, cmap: LinearSegmentedColormap) -> object:
    if math.isclose(value_min, value_max):
        normalized = 0.5
    else:
        normalized = (float(value) - value_min) / (value_max - value_min)
    return cmap(float(np.clip(normalized, 0.0, 1.0)))


def _metric_color_hex(value: float, value_min: float, value_max: float) -> str:
    legacy_cmap = LinearSegmentedColormap.from_list("flexworks_legacy_revenue_green", ["#cfecc8", "#a1d99b", "#74c476", "#238b45", "#006d2c"])
    return to_hex(_legacy_metric_color(value, value_min, value_max, legacy_cmap))


def build_iso_zone_snapshot_map_bars(
    snapshot_data: pd.DataFrame,
    iso_region: str,
    pjm_geojson: PjmZoneGeoJson | None,
    metric_column: str = "Selected_Metric",
    metric_label: str | None = None,
    time_label: str | None = None,
    category_label: str | None = None,
    time_context_label: str = "Selected time",
    compact: bool = False,
) -> tuple[ChartResult, dict[str, object]]:
    """Build an ISO-focused zone snapshot map with ranked bars."""

    normalized_iso = str(iso_region or "").upper()
    if normalized_iso != "PJM":
        return (
            ChartResult(None, f"{normalized_iso or 'Selected ISO'} zone polygons are not configured yet. Falling back to point map."),
            _cumulative_revenue_diagnostics(snapshot_data, None),
        )

    diagnostics = _cumulative_revenue_diagnostics(snapshot_data, pjm_geojson)
    if pjm_geojson is None:
        return ChartResult(None, "PJM zone map requires the PJM GeoJSON file."), diagnostics
    if snapshot_data.empty:
        return ChartResult(None, "No time-series rows are available for the selected time."), diagnostics
    if not diagnostics["is_available"]:
        return ChartResult(None, "No selected-time zones matched the PJM GeoJSON."), diagnostics
    if metric_column not in snapshot_data.columns:
        return ChartResult(None, f"{metric_column} is not available for this snapshot."), diagnostics

    chart_data = snapshot_data.copy()
    chart_data[metric_column] = pd.to_numeric(chart_data[metric_column], errors="coerce")
    chart_data = chart_data.loc[chart_data["Zone_Normalized"].isin(pjm_geojson.zones)].dropna(subset=[metric_column])
    if chart_data.empty:
        return ChartResult(None, "Matched PJM zones do not have numeric values for the selected snapshot."), diagnostics

    metric_label = metric_label or _first_display_value(chart_data, "Metric_Label", ISO_ZONE_SNAPSHOT_METRIC_LABELS.get(metric_column, metric_column))
    time_label = time_label or _first_display_value(chart_data, "Time_Label", "selected time")
    category_label = category_label or _first_display_value(chart_data, "Revenue_Category_Filter", "All categories")

    chart_data["Selected_Metric_Display"] = chart_data[metric_column].apply(lambda value: _format_metric_value(value, metric_label))
    chart_data["Monthly_Revenue_Display"] = chart_data.get("Monthly_Revenue", pd.Series(index=chart_data.index)).apply(_format_currency)
    chart_data["Cumulative_Revenue_Display"] = chart_data.get("Cumulative_Revenue", pd.Series(index=chart_data.index)).apply(_format_currency)
    chart_data["Revenue_per_kW_Display"] = chart_data.get("Revenue_per_kW", pd.Series(index=chart_data.index)).apply(_format_dollars_per_kw)

    bar_data = chart_data.sort_values(metric_column, ascending=False).copy()
    colorscale, zmid = _revenue_colorscale(chart_data[metric_column])
    custom_columns = [
        "Zone",
        "ISO_Region",
        "Selected_Metric_Display",
        "Monthly_Revenue_Display",
        "Cumulative_Revenue_Display",
        "Revenue_per_kW_Display",
        "Revenue_Category_Filter",
    ]

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "choropleth"}, {"type": "xy"}]],
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.04,
        subplot_titles=("PJM zone map", f"Ranked by {metric_label}"),
    )
    figure.add_trace(
        go.Choropleth(
            geojson=pjm_geojson.geojson,
            locations=chart_data["Zone_Normalized"],
            featureidkey="properties.Zone_Normalized",
            z=chart_data[metric_column],
            text=chart_data["Zone"],
            customdata=_custom_data_matrix(chart_data, custom_columns),
            colorscale=colorscale,
            zmid=zmid,
            marker_line_color="#ffffff",
            marker_line_width=1.35,
            colorbar=_snapshot_colorbar(metric_label),
            hovertemplate=_snapshot_map_hover_template(metric_label),
            name=metric_label,
        ),
        row=1,
        col=1,
    )

    max_zone_labels = 4 if compact else 10
    label_points = _zone_label_points(
        pjm_geojson.geojson,
        chart_data["Zone_Normalized"].tolist(),
        chart_data["Zone"].tolist(),
        max_labels=max_zone_labels,
    )
    if label_points:
        figure.add_trace(
            go.Scattergeo(
                lon=[point["lon"] for point in label_points],
                lat=[point["lat"] for point in label_points],
                text=[point["zone"] for point in label_points],
                mode="text",
                textfont=dict(size=8 if compact else 9, color="#0f172a", family="Arial"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Bar(
            x=bar_data[metric_column],
            y=bar_data["Zone"],
            orientation="h",
            text=bar_data["Selected_Metric_Display"],
            textposition="outside",
            marker_color=["#b91c1c" if float(value) < 0 else "#2f7d32" for value in bar_data[metric_column]],
            cliponaxis=False,
            customdata=_custom_data_matrix(bar_data, custom_columns),
            hovertemplate=_snapshot_map_hover_template(metric_label),
            name=metric_label,
        ),
        row=1,
        col=2,
    )
    figure.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="#ffffff",
        row=1,
        col=1,
    )
    figure.update_yaxes(autorange="reversed", title_text=None, automargin=True, row=1, col=2)
    figure.update_xaxes(
        title_text=metric_label,
        tickformat=_metric_tickformat(metric_label),
        ticksuffix=_metric_ticksuffix(metric_label),
        zeroline=True,
        automargin=True,
        row=1,
        col=2,
    )
    figure.update_layout(
        title=dict(
            text=(
                "PJM Zone Performance"
                f"<br><sup>{time_context_label}: {time_label} | Metric: {metric_label} | Category: {category_label}</sup>"
            ),
            x=0.01,
            xanchor="left",
            font=dict(size=16 if compact else 21),
        ),
        showlegend=False,
        height=430 if compact else 640,
        margin=dict(l=0, r=24, t=86, b=58) if compact else dict(l=0, r=28, t=106, b=76),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        bargap=0.16,
    )
    return ChartResult(figure, None), diagnostics


def create_animated_zone_performance_figure(
    snapshots: list[pd.DataFrame],
    iso_region: str,
    pjm_geojson: PjmZoneGeoJson | None,
    metric_column: str = "Selected_Metric",
    metric_label: str | None = None,
    category_label: str | None = None,
    frame_labels: list[str] | None = None,
) -> tuple[ChartResult, dict[str, object]]:
    """Create a Plotly animation for ISO zone performance snapshots."""

    non_empty_snapshots = [snapshot for snapshot in snapshots if snapshot is not None and not snapshot.empty]
    combined = pd.concat(non_empty_snapshots, ignore_index=True) if non_empty_snapshots else pd.DataFrame()
    diagnostics = _cumulative_revenue_diagnostics(combined, pjm_geojson)

    if pjm_geojson is None:
        return ChartResult(None, "PJM zone map requires the PJM GeoJSON file for animation."), diagnostics
    if str(iso_region or "").upper() != "PJM":
        return ChartResult(None, f"{iso_region or 'Selected ISO'} zone animation is not configured yet."), diagnostics
    if not non_empty_snapshots:
        return ChartResult(None, "No snapshot data is available for animation."), diagnostics
    if not diagnostics["is_available"]:
        return ChartResult(None, "No animation snapshot zones matched the PJM GeoJSON."), diagnostics

    labels = frame_labels or [
        _first_display_value(snapshot, "Time_Label", f"Frame {index + 1}") for index, snapshot in enumerate(non_empty_snapshots)
    ]
    metric_label = metric_label or _first_display_value(non_empty_snapshots[0], "Metric_Label", ISO_ZONE_SNAPSHOT_METRIC_LABELS.get(metric_column, metric_column))
    category_label = category_label or _first_display_value(non_empty_snapshots[0], "Revenue_Category_Filter", "All categories")

    metric_values = combined[metric_column] if metric_column in combined.columns else pd.Series(dtype="float64")
    all_values = pd.to_numeric(metric_values, errors="coerce").dropna()
    colorscale, zmid = _revenue_colorscale(all_values)
    value_min, value_max = _animation_value_range(all_values)

    frame_figures: list[tuple[str, go.Figure]] = []
    for label, snapshot in zip(labels, non_empty_snapshots):
        frame_result, _ = build_iso_zone_snapshot_map_bars(
            snapshot,
            iso_region=iso_region,
            pjm_geojson=pjm_geojson,
            metric_column=metric_column,
            metric_label=metric_label,
            time_label=label,
            category_label=category_label,
            time_context_label="Selected time",
            compact=True,
        )
        if frame_result.figure is not None:
            frame_figures.append((label, frame_result.figure))

    if not frame_figures:
        return ChartResult(None, "No animation frames could be built for the selected range."), diagnostics

    first_label, figure = frame_figures[0]
    choropleth_index = _trace_index(figure, "choropleth")
    bar_index = _trace_index(figure, "bar")
    if choropleth_index is None or bar_index is None:
        return ChartResult(None, "Animation frames could not find both map and bar traces."), diagnostics

    _apply_animation_trace_scale(figure.data[choropleth_index], colorscale, zmid, value_min, value_max)
    figure.update_xaxes(range=[value_min, value_max], row=1, col=2)
    figure.update_layout(
        title=dict(
            text=(
                "PJM Zone Performance"
                f"<br><sup>Selected time: {first_label} | Metric: {metric_label} | Category: {category_label}</sup>"
            ),
            x=0.01,
            xanchor="left",
            font=dict(size=20),
        ),
        height=620,
        margin=dict(l=0, r=28, t=106, b=108),
    )

    frames: list[go.Frame] = []
    slider_steps: list[dict[str, object]] = []
    for label, frame_figure in frame_figures:
        frame_choropleth_index = _trace_index(frame_figure, "choropleth")
        frame_bar_index = _trace_index(frame_figure, "bar")
        if frame_choropleth_index is None or frame_bar_index is None:
            continue
        choropleth_trace = frame_figure.data[frame_choropleth_index]
        bar_trace = frame_figure.data[frame_bar_index]
        _apply_animation_trace_scale(choropleth_trace, colorscale, zmid, value_min, value_max)
        frame_name = str(label)
        frames.append(
            go.Frame(
                name=frame_name,
                data=[choropleth_trace, bar_trace],
                traces=[choropleth_index, bar_index],
                layout=go.Layout(
                    title=dict(
                        text=(
                            "PJM Zone Performance"
                            f"<br><sup>Selected time: {label} | Metric: {metric_label} | Category: {category_label}</sup>"
                        ),
                        x=0.01,
                        xanchor="left",
                        font=dict(size=20),
                    )
                ),
            )
        )
        slider_steps.append(
            {
                "label": frame_name,
                "method": "animate",
                "args": [
                    [frame_name],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    figure.frames = frames
    figure.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.01,
                "y": -0.08,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "mode": "immediate",
                                "frame": {"duration": 900, "redraw": True},
                                "transition": {"duration": 200},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.18,
                "y": -0.08,
                "len": 0.78,
                "xanchor": "left",
                "yanchor": "top",
                "currentvalue": {"prefix": "Time: ", "font": {"size": 12}},
                "steps": slider_steps,
            }
        ],
    )
    return ChartResult(figure, None), diagnostics


def build_pjm_cumulative_revenue_map_bars(
    selected_month_data: pd.DataFrame,
    pjm_geojson: PjmZoneGeoJson | None,
    metric_column: str = "Cumulative_Revenue",
    sort_order: str = "Top zones",
) -> tuple[ChartResult, dict[str, object]]:
    """Build a PJM zone map with ranked cumulative/monthly revenue bars."""

    diagnostics = _cumulative_revenue_diagnostics(selected_month_data, pjm_geojson)
    if pjm_geojson is None:
        return ChartResult(None, "PJM zone map requires the PJM GeoJSON file."), diagnostics
    if selected_month_data.empty:
        return ChartResult(None, "No monthly revenue rows are available for the selected month."), diagnostics
    if not diagnostics["is_available"]:
        return ChartResult(None, "No monthly revenue zones matched the PJM GeoJSON."), diagnostics
    if metric_column not in selected_month_data.columns:
        return ChartResult(None, f"{metric_column} is not available for this visualization."), diagnostics

    chart_data = selected_month_data.copy()
    chart_data[metric_column] = pd.to_numeric(chart_data[metric_column], errors="coerce")
    chart_data = chart_data.loc[chart_data["Zone_Normalized"].isin(pjm_geojson.zones)].dropna(subset=[metric_column])
    if chart_data.empty:
        return ChartResult(None, "Matched PJM zones do not have numeric revenue for the selected month."), diagnostics

    metric_label = "Monthly Revenue" if metric_column == "Monthly_Revenue" else "Cumulative Revenue"
    selected_month = pd.to_datetime(chart_data["Month"].iloc[0]).strftime("%B %Y")
    category_label = _first_display_value(chart_data, "Revenue_Category_Filter", "All categories")
    chart_data["Metric_Label"] = metric_label
    chart_data["Time_Label"] = selected_month
    chart_result, _ = create_pjm_matplotlib_figure(
        chart_data,
        pjm_geojson,
        metric=metric_column,
        metric_label=metric_label,
        time_selection=selected_month,
        category_label=category_label,
        time_context_label="Selected month",
        title=f"PJM Battery Revenue — {metric_label} by Zone",
        compact=False,
        sort_order=sort_order,
    )
    return ChartResult(chart_result.figure, chart_result.message), diagnostics


def _collapse_matplotlib_zone_data(dataframe: pd.DataFrame, metric: str) -> pd.DataFrame:
    aggregation: dict[str, object] = {
        "Zone": ("Zone", _first_non_empty_string),
        "ISO_Region": ("ISO_Region", _first_non_empty_string) if "ISO_Region" in dataframe.columns else ("Zone", lambda _: "PJM"),
        metric: (metric, "mean"),
    }
    for column in ("Metric_Label", "Time_Label", "Revenue_Category_Filter", "Risk_Label"):
        if column in dataframe.columns:
            aggregation[column] = (column, _first_non_empty_string)
    for column in ("Monthly_Revenue", "Cumulative_Revenue", "Revenue_per_kW", "Annualized_Revenue", "Opportunity_Score", "Risk_Adjusted_Score"):
        if column in dataframe.columns and column != metric:
            aggregation[column] = (column, "mean")

    return dataframe.groupby("Zone_Normalized", as_index=False, dropna=False).agg(**aggregation)


def _draw_pjm_matplotlib_map_bars(
    *,
    geometries: list[MatplotlibZoneGeometry],
    bbox: tuple[float, float, float, float],
    chart_data: pd.DataFrame,
    metric: str,
    metric_label: str,
    title: str,
    subtitle: str,
    compact: bool,
    sort_order: str,
    metric_range: tuple[float, float] | None = None,
) -> Figure:
    figure_width = 10.8 if compact else 14.2
    figure_height = 5.15 if compact else 7.45
    title_size = 13 if compact else 22
    subtitle_size = 8.5 if compact else 12
    zone_label_size = 5.2 if compact else 7.0
    bar_label_size = 6.2 if compact else 9.0
    y_label_size = 6.0 if compact else 8.4

    figure = Figure(figsize=(figure_width, figure_height), dpi=135, facecolor=MATPLOTLIB_FIGURE_BG)
    grid_spec = figure.add_gridspec(
        2,
        2,
        height_ratios=[0.23 if compact else 0.24, 0.77 if compact else 0.76],
        width_ratios=[1.62, 1.0],
        hspace=0.14 if compact else 0.16,
        wspace=0.08,
    )
    header_ax = figure.add_subplot(grid_spec[0, :])
    map_ax = figure.add_subplot(grid_spec[1, 0])
    bar_ax = figure.add_subplot(grid_spec[1, 1])
    figure.subplots_adjust(left=0.025, right=0.975, top=0.96, bottom=0.05)

    header_ax.axis("off")
    header_ax.set_facecolor(MATPLOTLIB_FIGURE_BG)
    header_ax.text(
        0.5,
        0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="normal",
        color=MATPLOTLIB_TEXT_COLOR,
        wrap=True,
    )
    header_ax.text(
        0.5,
        0.24,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        fontweight="semibold",
        color=MATPLOTLIB_MUTED_TEXT_COLOR,
        wrap=True,
    )

    values_by_zone = {
        str(row["Zone_Normalized"]): float(row[metric])
        for _, row in chart_data.iterrows()
        if _to_float(row.get(metric)) is not None
    }
    cmap = _revenue_green_cmap()
    numeric_values = pd.to_numeric(chart_data[metric], errors="coerce").dropna()
    if metric_range is not None:
        value_min = float(metric_range[0])
        value_max = float(metric_range[1])
    else:
        value_min = float(numeric_values.min())
        value_max = float(numeric_values.max())
    if math.isclose(value_min, value_max):
        value_min -= 1.0
        value_max += 1.0

    xmin, xmax, ymin, ymax = bbox
    xpad = max((xmax - xmin) * MATPLOTLIB_MAP_PADDING_RATIO, 0.01)
    ypad = max((ymax - ymin) * MATPLOTLIB_MAP_PADDING_RATIO, 0.01)
    map_ax.set_xlim(xmin - xpad, xmax + xpad)
    map_ax.set_ylim(ymin - ypad, ymax + ypad)
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.set_facecolor(MATPLOTLIB_PANEL_BG)
    map_ax.axis("off")

    for geometry in geometries:
        value = values_by_zone.get(geometry.normalized_zone)
        facecolor = MATPLOTLIB_NO_DATA_COLOR if value is None else _metric_color(value, value_min, value_max, cmap)
        patch = PathPatch(
            geometry.path,
            facecolor=facecolor,
            edgecolor=MATPLOTLIB_TEXT_COLOR,
            linewidth=0.72 if compact else 0.9,
            antialiased=True,
        )
        if hasattr(patch, "set_fillrule"):
            patch.set_fillrule("evenodd")
        map_ax.add_patch(patch)

    for geometry in geometries:
        label_x, label_y = _nudge_matplotlib_label(geometry.label_point, geometry.zone_name)
        label_text = geometry.zone_name
        label = map_ax.text(
            label_x,
            label_y,
            label_text,
            ha="center",
            va="center",
            fontsize=_matplotlib_zone_label_size(label_text, zone_label_size),
            fontweight="heavy",
            color=MATPLOTLIB_TEXT_COLOR,
            clip_on=False,
        )
        label.set_path_effects(
            [
                path_effects.Stroke(linewidth=1.35 if not compact else 1.05, foreground=MATPLOTLIB_LABEL_STROKE),
                path_effects.Normal(),
            ]
        )

    bar_data = chart_data.copy()
    if sort_order == "Top zones":
        bar_data = bar_data.sort_values(metric, ascending=True)
    else:
        bar_data = bar_data.sort_values(metric, ascending=False)

    bar_values = pd.to_numeric(bar_data[metric], errors="coerce").fillna(0.0).astype(float).tolist()
    bar_zones = bar_data["Zone"].astype(str).tolist()
    y_positions = np.arange(len(bar_data), dtype=float)
    bar_colors = [
        MATPLOTLIB_NEGATIVE_BAR_COLOR if value < 0 else _metric_color(value, value_min, value_max, cmap)
        for value in bar_values
    ]

    bar_ax.barh(y_positions, bar_values, height=0.62, color=bar_colors, edgecolor="none")
    x_min = min(0.0, value_min)
    x_max = max(0.0, value_max)
    span = x_max - x_min
    if span <= 0:
        span = max(abs(x_max), 1.0)
        x_min = -span * 0.05
        x_max = span
    left_pad = span * 0.22 if x_min < 0 else span * 0.04
    right_pad = span * 0.34
    bar_ax.set_xlim(x_min - left_pad, x_max + right_pad)
    bar_ax.axvline(0, color=MATPLOTLIB_TEXT_COLOR, linewidth=0.65, alpha=0.45)
    bar_ax.set_ylim(-0.6, len(bar_data) - 0.4)
    bar_ax.set_yticks(y_positions)
    bar_ax.set_yticklabels(bar_zones, fontsize=y_label_size, fontweight="semibold", color=MATPLOTLIB_TEXT_COLOR)
    bar_ax.set_xticks([])
    bar_ax.tick_params(axis="y", length=0, colors=MATPLOTLIB_TEXT_COLOR)
    bar_ax.tick_params(axis="x", length=0, labelbottom=False)
    bar_ax.set_title(f"Ranked by {metric_label}", fontsize=9 if compact else 12, color=MATPLOTLIB_TEXT_COLOR, pad=14 if not compact else 10)
    bar_ax.set_facecolor(MATPLOTLIB_PANEL_BG)
    bar_ax.grid(False)
    for spine in bar_ax.spines.values():
        spine.set_visible(False)

    label_offset = span * 0.018
    for y_position, value in zip(y_positions, bar_values):
        label_x = value + label_offset if value >= 0 else value - label_offset
        ha = "left" if value >= 0 else "right"
        bar_ax.text(
            label_x,
            y_position,
            _format_metric_value(value, metric_label),
            ha=ha,
            va="center",
            fontsize=bar_label_size,
            fontweight="semibold",
            color=MATPLOTLIB_TEXT_COLOR,
            clip_on=False,
        )

    return figure


def _matplotlib_figure_to_palette_image(figure: Figure) -> Image.Image:
    return _png_bytes_to_palette_image(_matplotlib_figure_to_frame_png_bytes(figure))


def _matplotlib_figure_to_frame_png_bytes(figure: Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=105,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    return buffer.getvalue()


def _png_bytes_to_palette_image(png_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(png_bytes)).convert("RGBA")
    adaptive_palette = getattr(getattr(Image, "Palette", object), "ADAPTIVE", None)
    if adaptive_palette is None:
        adaptive_palette = getattr(Image, "ADAPTIVE", 1)
    return image.convert("P", palette=adaptive_palette, colors=256)


def _safe_png_name(name: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in str(name).strip())
    cleaned = cleaned.strip("_") or "snapshot"
    return f"{cleaned}.png" if not cleaned.lower().endswith(".png") else cleaned


def _matplotlib_zone_geometries(pjm_geojson: PjmZoneGeoJson) -> tuple[list[MatplotlibZoneGeometry], tuple[float, float, float, float]]:
    cache_key = _matplotlib_geojson_cache_key(pjm_geojson)
    if cache_key in _MATPLOTLIB_GEOMETRY_CACHE:
        return _MATPLOTLIB_GEOMETRY_CACHE[cache_key]

    geometries: list[MatplotlibZoneGeometry] = []
    all_x: list[float] = []
    all_y: list[float] = []

    for feature in pjm_geojson.geojson.get("features", []):
        properties = feature.get("properties", {})
        zone_name = str(properties.get("Zone") or properties.get(pjm_geojson.zone_property) or "").strip()
        normalized_zone = str(properties.get("Zone_Normalized") or "").strip()
        if not zone_name or not normalized_zone:
            continue

        polygons = _extract_geojson_polygons(feature.get("geometry", {}))
        if not polygons:
            continue

        path = _build_matplotlib_polygon_path(polygons)
        label_polygon = max(polygons, key=_polygon_outer_area)
        label_point = _polylabel(label_polygon)

        for polygon in polygons:
            for ring in polygon:
                all_x.extend(ring[:, 0].tolist())
                all_y.extend(ring[:, 1].tolist())

        geometries.append(
            MatplotlibZoneGeometry(
                zone_name=zone_name,
                normalized_zone=normalized_zone,
                polygons=polygons,
                path=path,
                label_point=label_point,
            )
        )

    if not all_x or not all_y:
        return [], (0.0, 1.0, 0.0, 1.0)
    result = geometries, (min(all_x), max(all_x), min(all_y), max(all_y))
    _MATPLOTLIB_GEOMETRY_CACHE[cache_key] = result
    return result


def _matplotlib_geojson_cache_key(pjm_geojson: PjmZoneGeoJson) -> tuple[tuple[str, str, int, float, float, float, float], ...]:
    key_parts: list[tuple[str, str, int, float, float, float, float]] = []
    for feature in pjm_geojson.geojson.get("features", []):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        normalized_zone = str(properties.get("Zone_Normalized") or "").strip()
        geometry_type = str(geometry.get("type") or "")
        point_count, min_x, max_x, min_y, max_y = _coordinate_signature(geometry.get("coordinates"))
        key_parts.append((normalized_zone, geometry_type, point_count, min_x, max_x, min_y, max_y))
    return tuple(sorted(key_parts))


def _coordinate_signature(coordinates: object) -> tuple[int, float, float, float, float]:
    points: list[tuple[float, float]] = []
    _collect_coordinate_points(coordinates, points)
    if not points:
        return 0, 0.0, 0.0, 0.0, 0.0
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return (
        len(points),
        round(min(x_values), 6),
        round(max(x_values), 6),
        round(min(y_values), 6),
        round(max(y_values), 6),
    )


def _collect_coordinate_points(value: object, points: list[tuple[float, float]]) -> None:
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            points.append((float(value[0]), float(value[1])))
            return
        for item in value:
            _collect_coordinate_points(item, points)


def _extract_geojson_polygons(geometry: object) -> tuple[tuple[np.ndarray, ...], ...]:
    if not isinstance(geometry, dict):
        return tuple()

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        polygon_groups = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygon_groups = coordinates
    else:
        return tuple()

    polygons: list[tuple[np.ndarray, ...]] = []
    for polygon in polygon_groups or []:
        rings: list[np.ndarray] = []
        for ring in polygon or []:
            ring_array = np.asarray(ring, dtype=float)
            if ring_array.ndim != 2 or ring_array.shape[0] < 3 or ring_array.shape[1] < 2:
                continue
            ring_array = ring_array[:, :2]
            if not np.allclose(ring_array[0], ring_array[-1]):
                ring_array = np.vstack([ring_array, ring_array[0]])
            rings.append(ring_array)
        if rings:
            polygons.append(tuple(rings))
    return tuple(polygons)


def _build_matplotlib_polygon_path(polygons: tuple[tuple[np.ndarray, ...], ...]) -> MplPath:
    vertices: list[list[float]] = []
    codes: list[int] = []

    for polygon in polygons:
        for ring in polygon:
            for index, (x_value, y_value) in enumerate(ring):
                vertices.append([float(x_value), float(y_value)])
                if index == 0:
                    codes.append(MplPath.MOVETO)
                elif index == len(ring) - 1:
                    codes.append(MplPath.CLOSEPOLY)
                else:
                    codes.append(MplPath.LINETO)

    return MplPath(np.asarray(vertices, dtype=float), codes)


def _polygon_outer_area(polygon: tuple[np.ndarray, ...]) -> float:
    if not polygon:
        return 0.0
    return abs(_ring_signed_area(polygon[0]))


def _ring_signed_area(ring: np.ndarray) -> float:
    if len(ring) < 3:
        return 0.0
    x_values = ring[:, 0]
    y_values = ring[:, 1]
    return float(0.5 * np.sum((x_values[:-1] * y_values[1:]) - (x_values[1:] * y_values[:-1])))


def _point_in_ring(point: tuple[float, float], ring: np.ndarray) -> bool:
    px, py = point
    inside = False
    for index in range(len(ring) - 1):
        x1, y1 = ring[index]
        x2, y2 = ring[index + 1]
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / ((y2 - y1) or 1e-12) + x1):
            inside = not inside
    return inside


def _point_in_polygon(point: tuple[float, float], polygon: tuple[np.ndarray, ...]) -> bool:
    if not polygon or not _point_in_ring(point, polygon[0]):
        return False
    return not any(_point_in_ring(point, hole) for hole in polygon[1:])


def _point_segment_distance_sq(point: tuple[float, float], start: np.ndarray, end: np.ndarray) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return float((px - ax) ** 2 + (py - ay) ** 2)

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, float(t)))
    proj_x = ax + (t * dx)
    proj_y = ay + (t * dy)
    return float((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _signed_distance(point: tuple[float, float], polygon: tuple[np.ndarray, ...]) -> float:
    min_dist_sq = float("inf")
    for ring in polygon:
        for index in range(len(ring) - 1):
            min_dist_sq = min(min_dist_sq, _point_segment_distance_sq(point, ring[index], ring[index + 1]))

    distance = math.sqrt(min_dist_sq)
    return distance if _point_in_polygon(point, polygon) else -distance


def _polygon_centroid(ring: np.ndarray) -> tuple[float, float]:
    area_twice = 0.0
    centroid_x = 0.0
    centroid_y = 0.0

    for index in range(len(ring) - 1):
        x1, y1 = ring[index]
        x2, y2 = ring[index + 1]
        cross = x1 * y2 - x2 * y1
        area_twice += cross
        centroid_x += (x1 + x2) * cross
        centroid_y += (y1 + y2) * cross

    if abs(area_twice) < 1e-12:
        return float(np.mean(ring[:, 0])), float(np.mean(ring[:, 1]))

    factor = 1.0 / (3.0 * area_twice)
    return float(centroid_x * factor), float(centroid_y * factor)


def _make_polylabel_cell(x_value: float, y_value: float, half_size: float, polygon: tuple[np.ndarray, ...]) -> _PolylabelCell:
    distance = _signed_distance((x_value, y_value), polygon)
    max_distance = distance + half_size * math.sqrt(2)
    return _PolylabelCell(-max_distance, x_value, y_value, half_size, distance, max_distance)


def _polylabel(polygon: tuple[np.ndarray, ...], precision: float = 0.01) -> tuple[float, float]:
    outer = polygon[0]
    min_x = float(np.min(outer[:, 0]))
    min_y = float(np.min(outer[:, 1]))
    max_x = float(np.max(outer[:, 0]))
    max_y = float(np.max(outer[:, 1]))
    width = max_x - min_x
    height = max_y - min_y
    cell_size = min(width, height)
    if cell_size == 0:
        return float(outer[0, 0]), float(outer[0, 1])

    half_size = cell_size / 2.0
    queue: list[_PolylabelCell] = []
    x_value = min_x
    while x_value < max_x:
        y_value = min_y
        while y_value < max_y:
            heappush(queue, _make_polylabel_cell(x_value + half_size, y_value + half_size, half_size, polygon))
            y_value += cell_size
        x_value += cell_size

    centroid = _polygon_centroid(outer)
    best_cell = _make_polylabel_cell(centroid[0], centroid[1], 0.0, polygon)
    bbox_cell = _make_polylabel_cell((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, 0.0, polygon)
    if bbox_cell.distance > best_cell.distance:
        best_cell = bbox_cell

    while queue:
        cell = heappop(queue)
        if cell.distance > best_cell.distance:
            best_cell = cell
        if cell.max_distance - best_cell.distance <= precision:
            continue
        next_half_size = cell.h / 2.0
        heappush(queue, _make_polylabel_cell(cell.x - next_half_size, cell.y - next_half_size, next_half_size, polygon))
        heappush(queue, _make_polylabel_cell(cell.x + next_half_size, cell.y - next_half_size, next_half_size, polygon))
        heappush(queue, _make_polylabel_cell(cell.x - next_half_size, cell.y + next_half_size, next_half_size, polygon))
        heappush(queue, _make_polylabel_cell(cell.x + next_half_size, cell.y + next_half_size, next_half_size, polygon))

    return float(best_cell.x), float(best_cell.y)


def _nudge_matplotlib_label(label_point: tuple[float, float], zone_name: str) -> tuple[float, float]:
    offset_x, offset_y = MATPLOTLIB_LABEL_NUDGES.get(str(zone_name).upper(), (0.0, 0.0))
    return label_point[0] + offset_x, label_point[1] + offset_y


def _matplotlib_zone_label_size(zone_name: str, default_size: float) -> float:
    return MATPLOTLIB_LABEL_SIZE_OVERRIDES.get(str(zone_name).upper(), default_size)


def _first_non_empty_string(values: pd.Series) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def _positive_size_values(dataframe: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in dataframe.columns:
        return None

    values = pd.to_numeric(dataframe[column], errors="coerce")
    if values.dropna().empty:
        return None

    positive = values.clip(lower=0)
    if positive.max() == 0:
        return None
    return positive


def _add_choropleth_hover_fields(dataframe: pd.DataFrame) -> pd.DataFrame:
    chart_data = dataframe.copy()
    if "ISO_Region" not in chart_data.columns:
        chart_data["ISO_Region"] = "n/a"
    if "Risk_Label" not in chart_data.columns:
        chart_data["Risk_Label"] = "n/a"
    if "Annualized_Revenue" in chart_data.columns:
        chart_data["Annualized_Revenue_Display"] = chart_data["Annualized_Revenue"].apply(_format_currency)
    if "Revenue_per_kW" in chart_data.columns:
        chart_data["Revenue_per_kW_Display"] = chart_data["Revenue_per_kW"].apply(_format_dollars_per_kw)
    if "Opportunity_Score" in chart_data.columns:
        chart_data["Opportunity_Score_Display"] = chart_data["Opportunity_Score"].apply(_format_score)
    if "Risk_Adjusted_Score" in chart_data.columns:
        chart_data["Risk_Adjusted_Score_Display"] = chart_data["Risk_Adjusted_Score"].apply(_format_score)
    return chart_data


def _choropleth_hover_template(custom_data_columns: list[str]) -> str:
    labels = {
        "Zone": "Zone",
        "ISO_Region": "ISO",
        "Annualized_Revenue_Display": "Annualized Revenue",
        "Revenue_per_kW_Display": "Revenue per kW",
        "Opportunity_Score_Display": "Opportunity Score",
        "Risk_Adjusted_Score_Display": "Risk Adjusted Score",
        "Risk_Label": "Risk Label",
        "Node_Count": "Node Count",
    }
    lines = ["<b>%{customdata[0]}</b>"]
    for index, column in enumerate(custom_data_columns[1:], start=1):
        lines.append(f"{labels.get(column, column)}: %{{customdata[{index}]}}")
    return "<br>".join(lines) + "<extra></extra>"


def _colorbar_config(metric_column: str, metric_label: str) -> dict[str, object]:
    config: dict[str, object] = {"title": metric_label}
    if metric_column == "Revenue_per_kW":
        config.update({"tickprefix": "$", "ticksuffix": "/kW", "tickformat": ",.0f"})
    elif metric_column == "Annualized_Revenue":
        config.update({"tickprefix": "$", "tickformat": ",.0f"})
    else:
        config.update({"tickformat": ",.1f"})
    return config


def _format_currency(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.0f}"


def _format_dollars_per_kw(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.2f}/kW"


def _format_score(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number:,.2f}"


def _to_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _cumulative_revenue_diagnostics(
    selected_month_data: pd.DataFrame,
    pjm_geojson: PjmZoneGeoJson | None,
) -> dict[str, object]:
    geojson_zones = set(pjm_geojson.zones) if pjm_geojson is not None else set()
    data_zones = (
        set(selected_month_data["Zone_Normalized"].dropna().astype(str))
        if "Zone_Normalized" in selected_month_data.columns
        else set()
    )
    matched_zones = sorted(data_zones.intersection(geojson_zones))
    return {
        "geojson_zone_count": len(geojson_zones),
        "revenue_zone_count": len(data_zones),
        "matched_zone_count": len(matched_zones),
        "unmatched_revenue_zones": sorted(data_zones.difference(geojson_zones)),
        "unmatched_geojson_zones": sorted(geojson_zones.difference(data_zones)),
        "is_available": pjm_geojson is not None and bool(matched_zones),
    }


def _custom_data_matrix(dataframe: pd.DataFrame, columns: list[str]) -> list[list[object]]:
    prepared = dataframe.copy()
    for column in columns:
        if column not in prepared.columns:
            prepared[column] = "n/a"
    if "Monthly_Revenue" in columns:
        prepared["Monthly_Revenue"] = prepared["Monthly_Revenue"].apply(_format_currency)
    if "Cumulative_Revenue" in columns:
        prepared["Cumulative_Revenue"] = prepared["Cumulative_Revenue"].apply(_format_currency)
    return prepared[columns].to_numpy().tolist()


def _cumulative_map_hover_template() -> str:
    return (
        "<b>%{customdata[0]}</b><br>"
        "ISO: %{customdata[1]}<br>"
        "Monthly Revenue: %{customdata[2]}<br>"
        "Cumulative Revenue: %{customdata[3]}<br>"
        "Category Filter: %{customdata[4]}"
        "<extra></extra>"
    )


def _cumulative_bar_hover_template() -> str:
    return (
        "<b>%{customdata[0]}</b><br>"
        "ISO: %{customdata[1]}<br>"
        "Monthly Revenue: %{customdata[2]}<br>"
        "Cumulative Revenue: %{customdata[3]}<br>"
        "Category Filter: %{customdata[4]}"
        "<extra></extra>"
    )


def _snapshot_map_hover_template(metric_label: str) -> str:
    return (
        "<b>%{customdata[0]}</b><br>"
        "ISO: %{customdata[1]}<br>"
        f"{metric_label}: %{{customdata[2]}}<br>"
        "Monthly Revenue: %{customdata[3]}<br>"
        "Cumulative Revenue: %{customdata[4]}<br>"
        "Revenue per kW: %{customdata[5]}<br>"
        "Category Filter: %{customdata[6]}"
        "<extra></extra>"
    )


def _snapshot_colorbar(metric_label: str) -> dict[str, object]:
    return {
        "title": metric_label,
        "tickformat": _metric_tickformat(metric_label),
        "ticksuffix": _metric_ticksuffix(metric_label),
        "orientation": "h",
        "thickness": 10,
        "len": 0.44,
        "x": 0.29,
        "xanchor": "center",
        "y": -0.08,
        "yanchor": "top",
    }


def _cumulative_colorbar(metric_label: str) -> dict[str, object]:
    return {
        "title": metric_label,
        "tickformat": "$,.0f",
        "orientation": "h",
        "thickness": 10,
        "len": 0.44,
        "x": 0.29,
        "xanchor": "center",
        "y": -0.08,
        "yanchor": "top",
    }


def _revenue_colorscale(series: pd.Series) -> tuple[object, int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty or values.min() >= 0:
        return REVENUE_GREEN_COLORSCALE, None
    if values.max() <= 0:
        return [(0.0, "#7f1d1d"), (1.0, "#fecaca")], None
    return [(0.0, "#b91c1c"), (0.5, "#f8fafc"), (1.0, "#166534")], 0


def _animation_value_range(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0, 1.0

    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if minimum == maximum:
        padding = max(abs(maximum) * 0.1, 1.0)
        return minimum - padding, maximum + padding

    span = maximum - minimum
    padding = span * 0.12
    return min(0.0, minimum - padding), max(0.0, maximum + padding)


def _trace_index(figure: go.Figure, trace_type: str) -> int | None:
    for index, trace in enumerate(figure.data):
        if getattr(trace, "type", None) == trace_type:
            return index
    return None


def _apply_animation_trace_scale(
    trace: object,
    colorscale: object,
    zmid: int | None,
    value_min: float,
    value_max: float,
) -> None:
    if getattr(trace, "type", None) != "choropleth":
        return
    trace.colorscale = colorscale
    trace.zmin = value_min
    trace.zmax = value_max
    if zmid is not None:
        trace.zmid = zmid


def _format_metric_value(value: object, metric_label: str) -> str:
    if "kw" in metric_label.lower():
        return _format_dollars_per_kw(value)
    if "revenue" in metric_label.lower():
        return _format_currency(value)
    return _format_score(value)


def _metric_tickformat(metric_label: str) -> str:
    if "revenue" in metric_label.lower() or "kw" in metric_label.lower():
        return "$,.0f"
    return ",.1f"


def _metric_ticksuffix(metric_label: str) -> str:
    if "kw" in metric_label.lower():
        return "/kW"
    return ""


def _first_display_value(dataframe: pd.DataFrame, column: str, fallback: str) -> str:
    if column not in dataframe.columns:
        return fallback
    values = dataframe[column].dropna().astype(str)
    if values.empty:
        return fallback
    return values.iloc[0]


def _zone_label_points(
    geojson: dict[str, object],
    normalized_zones: list[str],
    display_zones: list[str],
    max_labels: int = 10,
) -> list[dict[str, object]]:
    display_by_zone = dict(zip(normalized_zones, display_zones))
    requested_zones = set(normalized_zones)
    label_points: list[dict[str, object]] = []
    for feature in geojson.get("features", []):  # type: ignore[union-attr]
        properties = feature.get("properties", {})  # type: ignore[union-attr]
        normalized_zone = properties.get("Zone_Normalized")
        if normalized_zone not in requested_zones:
            continue
        coordinates = _geometry_coordinates(feature.get("geometry", {}))  # type: ignore[union-attr]
        if not coordinates:
            continue
        lon = sum(point[0] for point in coordinates) / len(coordinates)
        lat = sum(point[1] for point in coordinates) / len(coordinates)
        lon_values = [point[0] for point in coordinates]
        lat_values = [point[1] for point in coordinates]
        area_hint = (max(lon_values) - min(lon_values)) * (max(lat_values) - min(lat_values))
        label_points.append(
            {
                "zone": display_by_zone.get(normalized_zone, normalized_zone),
                "lon": lon,
                "lat": lat,
                "area_hint": area_hint,
            }
        )
    return _thin_zone_label_points(label_points, max_labels=max_labels)


def _thin_zone_label_points(label_points: list[dict[str, object]], max_labels: int = 10) -> list[dict[str, object]]:
    """Keep direct map labels readable by dropping labels that would overlap nearby zones."""

    if max_labels <= 0:
        return []

    selected: list[dict[str, object]] = []
    ranked_points = sorted(
        label_points,
        key=lambda point: (-float(point.get("area_hint", 0.0)), str(point.get("zone", ""))),
    )
    for point in ranked_points:
        lon = float(point["lon"])
        lat = float(point["lat"])
        has_nearby_label = any(
            abs(lon - float(existing["lon"])) < 0.85 and abs(lat - float(existing["lat"])) < 0.42
            for existing in selected
        )
        if not has_nearby_label:
            selected.append(point)
        if len(selected) >= max_labels:
            break
    return sorted(selected, key=lambda point: str(point.get("zone", "")))


def _geometry_coordinates(geometry: dict[str, object]) -> list[tuple[float, float]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    points: list[tuple[float, float]] = []

    def collect(value: object) -> None:
        if (
            isinstance(value, list)
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            points.append((float(value[0]), float(value[1])))
            return
        if isinstance(value, list):
            for item in value:
                collect(item)

    if geometry_type in {"Polygon", "MultiPolygon"}:
        collect(coordinates)
    return points
