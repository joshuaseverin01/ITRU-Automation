"""Defensive Plotly visualization builders."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .geo import PjmZoneGeoJson, detect_coordinate_status, prepare_pjm_zone_choropleth_data


@dataclass(frozen=True)
class ChartResult:
    """Container for a Plotly figure or a user-facing skip message."""

    figure: go.Figure | None
    message: str | None = None


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
    (0.0, "#eaf7e6"),
    (0.25, "#cfeec9"),
    (0.5, "#a6d96a"),
    (0.75, "#31a354"),
    (1.0, "#006d2c"),
]


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

    chart_data = _add_choropleth_hover_fields(chart_data)
    custom_data_columns = [
        column
        for column in (
            "Zone",
            "ISO_Region",
            "Annualized_Revenue_Display",
            "Revenue_per_kW_Display",
            "Opportunity_Score_Display",
            "Risk_Adjusted_Score_Display",
            "Risk_Label",
            "Node_Count",
        )
        if column in chart_data.columns
    ]

    metric_label = CHOROPLETH_METRIC_LABELS.get(metric_column, metric_column.replace("_", " "))
    figure = px.choropleth(
        chart_data,
        geojson=join_result.geojson,
        locations="Zone_Normalized",
        featureidkey="properties.Zone_Normalized",
        color=metric_column,
        hover_name="Zone",
        custom_data=custom_data_columns,
        color_continuous_scale="Viridis",
        labels={metric_column: metric_label},
        scope="usa",
        projection="albers usa",
        title=f"PJM Zone Choropleth: {metric_label}",
    )
    figure.update_traces(
        marker_line_color="#f8fafc",
        marker_line_width=1.25,
        hovertemplate=_choropleth_hover_template(custom_data_columns),
    )
    figure.update_geos(
        fitbounds="locations",
        visible=True,
        showland=True,
        landcolor="#f8fafc",
        showsubunits=True,
        subunitcolor="#cbd5e1",
        countrycolor="#94a3b8",
        lakecolor="#ffffff",
        bgcolor="rgba(0,0,0,0)",
    )
    figure.update_layout(
        coloraxis_colorbar=_colorbar_config(metric_column, metric_label),
        margin=dict(l=0, r=0, t=58, b=0),
        height=540,
        title=dict(text=f"PJM Zone Choropleth: {metric_label}", x=0.01, xanchor="left"),
    )
    return ChartResult(figure, diagnostics.message), diagnostics


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
    bar_data = chart_data.sort_values(metric_column, ascending=sort_order == "Bottom zones").copy()
    bar_data["Revenue_Display"] = bar_data[metric_column].apply(_format_currency)
    chart_data["Revenue_Display"] = chart_data[metric_column].apply(_format_currency)
    colorscale, zmid = _revenue_colorscale(chart_data[metric_column])

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "choropleth"}, {"type": "xy"}]],
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.04,
        subplot_titles=("PJM zone map", f"{sort_order} by {metric_label}"),
    )
    figure.add_trace(
        go.Choropleth(
            geojson=pjm_geojson.geojson,
            locations=chart_data["Zone_Normalized"],
            featureidkey="properties.Zone_Normalized",
            z=chart_data[metric_column],
            text=chart_data["Zone"],
            customdata=_custom_data_matrix(
                chart_data,
                ["Zone", "ISO_Region", "Monthly_Revenue", "Cumulative_Revenue", "Revenue_Category_Filter"],
            ),
            colorscale=colorscale,
            zmid=zmid,
            marker_line_color="#f8fafc",
            marker_line_width=1.2,
            colorbar=_cumulative_colorbar(metric_label),
            hovertemplate=_cumulative_map_hover_template(),
            name=metric_label,
        ),
        row=1,
        col=1,
    )
    label_points = _zone_label_points(pjm_geojson.geojson, chart_data["Zone_Normalized"].tolist(), chart_data["Zone"].tolist())
    if label_points:
        figure.add_trace(
            go.Scattergeo(
                lon=[point["lon"] for point in label_points],
                lat=[point["lat"] for point in label_points],
                text=[point["zone"] for point in label_points],
                mode="text",
                textfont=dict(size=9, color="#0f172a"),
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
            text=bar_data["Revenue_Display"],
            textposition="outside",
            marker_color=["#b91c1c" if float(value) < 0 else "#386cb0" for value in bar_data[metric_column]],
            cliponaxis=False,
            customdata=_custom_data_matrix(
                bar_data,
                ["Zone", "ISO_Region", "Monthly_Revenue", "Cumulative_Revenue", "Revenue_Category_Filter"],
            ),
            hovertemplate=_cumulative_bar_hover_template(),
            name=metric_label,
        ),
        row=1,
        col=2,
    )
    figure.update_geos(
        fitbounds="locations",
        visible=True,
        showland=True,
        landcolor="#f8fafc",
        showsubunits=True,
        subunitcolor="#cbd5e1",
        countrycolor="#94a3b8",
        bgcolor="rgba(0,0,0,0)",
        row=1,
        col=1,
    )
    figure.update_yaxes(autorange="reversed", title_text=None, automargin=True, row=1, col=2)
    figure.update_xaxes(title_text=metric_label, tickformat="$,.0f", zeroline=True, automargin=True, row=1, col=2)
    figure.update_layout(
        title=dict(
            text=(
                f"PJM Battery Revenue — {metric_label} by Zone"
                f"<br><sup>Selected month: {selected_month} | Metric: {metric_label} | Category: {category_label}</sup>"
            ),
            x=0.01,
            xanchor="left",
            font=dict(size=20),
        ),
        showlegend=False,
        height=620,
        margin=dict(l=0, r=28, t=104, b=76),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        bargap=0.16,
    )
    return ChartResult(figure, None), diagnostics


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
