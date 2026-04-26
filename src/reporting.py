"""Deterministic Markdown reporting for FlexWorks arbitrage analysis."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from .analysis import DEFAULT_SCORE_WEIGHTS, SummaryMetrics, get_effective_score_weights, normalize_score_weights


@dataclass(frozen=True)
class ExecutiveSummary:
    """Deterministic zone-performance summary in export-ready formats."""

    markdown: str
    text: str
    top_zones: list[str]
    bottom_zones: list[str]
    spread: float | None


@dataclass(frozen=True)
class ZoneKpiOverview:
    """Top-line zone KPI values for the product overview."""

    zone_count: int
    metric_average: float | None
    top_zone: str | None
    spread: float | None


def generate_markdown_report(
    *,
    summary_metrics: SummaryMetrics,
    ranked_nodes: pd.DataFrame,
    iso_summary: pd.DataFrame,
    high_risk_high_reward_nodes: pd.DataFrame,
    cleaning_summary: object | None = None,
    coordinate_status: object | None = None,
    score_weights: Mapping[str, float] | None = None,
) -> str:
    """Generate a deterministic, client-ready Markdown report.

    The narrative is driven only by computed metrics and data quality summaries.
    It does not call an LLM or add unsupported interpretation.
    """

    weights = get_effective_score_weights(ranked_nodes, score_weights or DEFAULT_SCORE_WEIGHTS)
    sections = [
        "# FlexWorks Arbitrage Analyzer Report",
        _executive_summary(summary_metrics, ranked_nodes),
        _dataset_overview(summary_metrics, iso_summary),
        _opportunity_analysis(ranked_nodes, iso_summary),
        _risk_profile(summary_metrics, high_risk_high_reward_nodes),
        _recommendations(summary_metrics, ranked_nodes, high_risk_high_reward_nodes, coordinate_status),
        _methodology(weights),
        _data_quality_notes(cleaning_summary, coordinate_status),
    ]
    return "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"


def build_zone_kpi_overview(
    dataframe: pd.DataFrame,
    *,
    metric_column: str,
    zone_column: str = "Zone",
) -> ZoneKpiOverview:
    """Compute zone-level KPI cards for the dashboard overview."""

    if dataframe.empty or zone_column not in dataframe.columns:
        return ZoneKpiOverview(zone_count=0, metric_average=None, top_zone=None, spread=None)

    zone_data = dataframe[[zone_column]].copy()
    zone_data[zone_column] = zone_data[zone_column].astype("string").str.strip()
    zone_data = zone_data.dropna(subset=[zone_column])
    zone_data = zone_data.loc[zone_data[zone_column] != ""].copy()
    zone_count = int(zone_data[zone_column].nunique())

    if metric_column not in dataframe.columns:
        return ZoneKpiOverview(zone_count=zone_count, metric_average=None, top_zone=None, spread=None)

    working = dataframe[[zone_column, metric_column]].copy()
    working[zone_column] = working[zone_column].astype("string").str.strip()
    working[metric_column] = pd.to_numeric(working[metric_column], errors="coerce")
    working = working.dropna(subset=[zone_column, metric_column])
    working = working.loc[working[zone_column] != ""].copy()
    if working.empty:
        return ZoneKpiOverview(zone_count=zone_count, metric_average=None, top_zone=None, spread=None)

    grouped = working.groupby(zone_column, as_index=False, dropna=False)[metric_column].mean()
    metric_average = float(grouped[metric_column].mean())
    top_row = grouped.sort_values(metric_column, ascending=False).iloc[0]
    spread = float(grouped[metric_column].max() - grouped[metric_column].min())
    return ZoneKpiOverview(
        zone_count=zone_count,
        metric_average=metric_average,
        top_zone=str(top_row[zone_column]),
        spread=spread,
    )


def build_executive_summary(
    dataframe: pd.DataFrame,
    *,
    selected_iso: str,
    selected_metric: str,
    selected_period: str,
    zone_column: str = "Zone",
    metric_column: str = "Selected_Metric",
) -> ExecutiveSummary:
    """Build a deterministic executive summary for zone performance exports."""

    if dataframe.empty or zone_column not in dataframe.columns or metric_column not in dataframe.columns:
        message = (
            f"No zone performance data is available for {selected_iso or 'the selected ISO'} "
            f"over {selected_period or 'the selected period'}."
        )
        markdown = "\n".join(["# ISO Zone Performance Executive Summary", "", message, ""])
        return ExecutiveSummary(markdown=markdown, text=message, top_zones=[], bottom_zones=[], spread=None)

    summary_data = dataframe[[zone_column, metric_column]].copy()
    summary_data[metric_column] = pd.to_numeric(summary_data[metric_column], errors="coerce")
    summary_data = summary_data.dropna(subset=[zone_column, metric_column])
    if summary_data.empty:
        message = (
            f"No numeric {selected_metric} values are available for {selected_iso or 'the selected ISO'} "
            f"over {selected_period or 'the selected period'}."
        )
        markdown = "\n".join(["# ISO Zone Performance Executive Summary", "", message, ""])
        return ExecutiveSummary(markdown=markdown, text=message, top_zones=[], bottom_zones=[], spread=None)

    grouped = summary_data.groupby(zone_column, as_index=False, dropna=False)[metric_column].mean()
    ranked = grouped.sort_values(metric_column, ascending=False).reset_index(drop=True)
    top_rows = ranked.head(3)
    bottom_rows = ranked.tail(3).sort_values(metric_column, ascending=True)
    top_zones = top_rows[zone_column].astype(str).tolist()
    bottom_zones = bottom_rows[zone_column].astype(str).tolist()
    spread = float(ranked[metric_column].max() - ranked[metric_column].min())

    top_phrase = _join_items(top_zones)
    bottom_phrase = _join_items(bottom_zones)
    spread_text = _fmt_metric_value(spread, selected_metric)
    interpretation = (
        f"{selected_iso} shows meaningful locational variation in {selected_metric} over {selected_period}. "
        f"The top-performing zones were {top_phrase}, while the weakest zones were {bottom_phrase}. "
        f"The spread between the highest and lowest zone was {spread_text}, suggesting that location materially affects battery arbitrage value."
    )

    markdown_lines = [
        "# ISO Zone Performance Executive Summary",
        "",
        f"- ISO/RTO: {selected_iso}",
        f"- Metric: {selected_metric}",
        f"- Period: {selected_period}",
        f"- Top zones: {top_phrase}",
        f"- Bottom zones: {bottom_phrase}",
        f"- Spread: {spread_text}",
        "",
        interpretation,
        "",
    ]
    text_lines = [
        "ISO Zone Performance Executive Summary",
        f"ISO/RTO: {selected_iso}",
        f"Metric: {selected_metric}",
        f"Period: {selected_period}",
        f"Top zones: {top_phrase}",
        f"Bottom zones: {bottom_phrase}",
        f"Spread: {spread_text}",
        "",
        interpretation,
    ]
    return ExecutiveSummary(
        markdown="\n".join(markdown_lines),
        text="\n".join(text_lines),
        top_zones=top_zones,
        bottom_zones=bottom_zones,
        spread=spread,
    )


def export_dataframe_csv(dataframe: pd.DataFrame) -> bytes:
    """Serialize a dataframe to CSV bytes for Streamlit downloads."""

    return dataframe.to_csv(index=False).encode("utf-8")


def plotly_figure_to_html_bytes(figure: go.Figure) -> bytes:
    """Serialize a Plotly figure to standalone HTML bytes."""

    return figure.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")


def plotly_figures_to_html_bytes(figures: Sequence[go.Figure], title: str = "FlexWorks Plotly Export") -> bytes:
    """Serialize one or more Plotly figures into a single HTML document."""

    body_parts = [figure.to_html(full_html=False, include_plotlyjs="cdn" if index == 0 else False) for index, figure in enumerate(figures)]
    html = "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{title}</title>",
            "</head>",
            "<body>",
            *body_parts,
            "</body>",
            "</html>",
        ]
    )
    return html.encode("utf-8")


def safe_plotly_png_bytes(figure: go.Figure) -> tuple[bytes | None, str | None]:
    """Serialize a Plotly figure to PNG bytes when Kaleido is available."""

    try:
        return figure.to_image(format="png"), None
    except Exception:
        return None, "PNG export requires Kaleido. HTML export is still available."


def _executive_summary(summary: SummaryMetrics, ranked_nodes: pd.DataFrame) -> str:
    lines = ["## Executive Summary"]
    if summary.node_count == 0:
        lines.append("The filtered dataset contains no nodes, so no arbitrage opportunity conclusions can be drawn.")
        return "\n".join(lines)

    lines.append(
        f"The dataset includes {summary.node_count} node(s) across {summary.iso_count} ISO region(s). "
        f"Average revenue per kW is {_fmt_dollars_per_kw(summary.average_revenue_per_kw)} and median revenue per kW is {_fmt_dollars_per_kw(summary.median_revenue_per_kw)}."
    )

    if summary.top_opportunity_node:
        lines.append(
            f"The current scoring logic identifies `{summary.top_opportunity_node}` as the top opportunity node, "
            "which suggests location-specific economics are a material driver of value."
        )

    if summary.high_volatility_node_count:
        lines.append(
            f"{summary.high_volatility_node_count} node(s) are classified as high volatility, indicating that a portion of the apparent upside may warrant risk review."
        )

    if not ranked_nodes.empty and "Opportunity_Score" in ranked_nodes.columns:
        top_score = ranked_nodes.iloc[0].get("Opportunity_Score")
        lines.append(f"The highest computed opportunity score is {_fmt_number(top_score)}, based on normalized revenue and volatility inputs.")

    return "\n".join(lines)


def _dataset_overview(summary: SummaryMetrics, iso_summary: pd.DataFrame) -> str:
    lines = ["## Dataset Overview"]
    lines.append(f"- Nodes analyzed: {summary.node_count}")
    lines.append(f"- ISO regions represented: {', '.join(summary.iso_regions) if summary.iso_regions else 'none'}")
    lines.append(f"- Average annualized revenue: {_fmt_currency(summary.average_annualized_revenue)}")
    lines.append(f"- Median annualized revenue: {_fmt_currency(summary.median_annualized_revenue)}")
    lines.append(f"- LMP volatility range: {_fmt_number(summary.min_lmp_volatility)} to {_fmt_number(summary.max_lmp_volatility)}")

    if not iso_summary.empty:
        lines.append("\nISO comparison:")
        lines.append(
            _dataframe_to_markdown(
                iso_summary,
                [
                    "ISO_Region",
                    "Node_Count",
                    "Average_Annualized_Revenue",
                    "Average_Revenue_per_kW",
                    "Average_LMP_Volatility",
                    "Average_Opportunity_Score",
                ],
                max_rows=10,
            )
        )
    return "\n".join(lines)


def _opportunity_analysis(ranked_nodes: pd.DataFrame, iso_summary: pd.DataFrame) -> str:
    lines = ["## Opportunity Analysis"]
    if ranked_nodes.empty:
        lines.append("No ranked nodes are available for opportunity analysis.")
        return "\n".join(lines)

    top_node = ranked_nodes.iloc[0].get("Node_ID", "the top-ranked node")
    lines.append(
        f"`{top_node}` ranks highest under the selected score weights. This indicates the node may warrant priority review for arbitrage economics."
    )
    lines.append("\nTop ranked nodes:")
    lines.append(
        _dataframe_to_markdown(
            ranked_nodes,
            ["Rank", "Node_ID", "ISO_Region", "Opportunity_Score", "Annualized_Revenue", "Revenue_per_kW", "LMP_Volatility", "Risk_Label"],
            max_rows=10,
        )
    )

    if not iso_summary.empty:
        leading_iso = iso_summary.iloc[0].get("ISO_Region")
        lines.append(
            f"\nAt the ISO level, `{leading_iso}` has the strongest average opportunity score in the filtered dataset, which suggests zonal screening should precede detailed asset modeling."
        )
    return "\n".join(lines)


def _risk_profile(summary: SummaryMetrics, high_risk_high_reward_nodes: pd.DataFrame) -> str:
    lines = ["## Risk Profile"]
    if summary.node_count == 0:
        lines.append("Risk profile is unavailable because the filtered dataset contains no nodes.")
        return "\n".join(lines)

    lines.append(
        f"Average LMP volatility is {_fmt_number(summary.average_lmp_volatility)} with a median of {_fmt_number(summary.median_lmp_volatility)}. "
        "Risk labels are dataset-relative and should be interpreted as screening indicators."
    )

    if high_risk_high_reward_nodes.empty:
        lines.append("No nodes meet the current high-risk/high-reward screen.")
    else:
        lines.append(
            "The following nodes combine high volatility with top-quartile revenue per kW, which may warrant additional sensitivity analysis:"
        )
        lines.append(
            _dataframe_to_markdown(
                high_risk_high_reward_nodes,
                ["Rank", "Node_ID", "ISO_Region", "Revenue_per_kW", "LMP_Volatility", "Opportunity_Score"],
                max_rows=10,
            )
        )
    return "\n".join(lines)


def _recommendations(
    summary: SummaryMetrics,
    ranked_nodes: pd.DataFrame,
    high_risk_high_reward_nodes: pd.DataFrame,
    coordinate_status: object | None,
) -> str:
    lines = ["## Recommendations"]
    if summary.node_count == 0:
        lines.append("- Re-run the analysis with a non-empty dataset or broader ISO filter.")
        return "\n".join(lines)

    if not ranked_nodes.empty:
        top_node = ranked_nodes.iloc[0].get("Node_ID", "the top-ranked node")
        lines.append(f"- Prioritize diligence on `{top_node}` and adjacent nodes before expanding to lower-ranked opportunities.")

    if not high_risk_high_reward_nodes.empty:
        lines.append("- Run sensitivity checks for high-risk/high-reward nodes because volatility may improve upside while increasing forecast exposure.")
    else:
        lines.append("- Use the current ranking as an initial screen, then validate dispatch assumptions against hourly price behavior before investment decisions.")

    coordinate_message = _object_to_dict(coordinate_status).get("message") if coordinate_status is not None else None
    has_coordinates = _object_to_dict(coordinate_status).get("has_coordinates") if coordinate_status is not None else None
    if has_coordinates is False:
        lines.append("- Add node coordinates or a lookup table to evaluate geographic clustering and client-facing map outputs.")
    elif coordinate_message:
        lines.append(f"- Review map coverage: {coordinate_message}")

    return "\n".join(lines)


def _methodology(weights: Mapping[str, float]) -> str:
    return "\n".join(
        [
            "## Methodology",
            "The workflow loads a FlexWorks CSV export, validates required columns, trims whitespace, coerces numeric fields, removes blank node IDs, and aggregates duplicate Node_ID rows.",
            "Duplicate nodes are aggregated by averaging numeric fields and retaining the first non-empty categorical value.",
            (
            "Opportunity score is calculated as a weighted average of available min-max normalized metrics: "
                f"Annualized_Revenue {_fmt_percent(weights.get('Annualized_Revenue'))}, "
                f"Revenue_per_kW {_fmt_percent(weights.get('Revenue_per_kW'))}, "
                f"LMP_Volatility {_fmt_percent(weights.get('LMP_Volatility'))}."
            ),
            "Metrics with no valid values are excluded from scoring and the remaining weights are redistributed.",
            "Volatility risk labels use dataset-relative tertiles when LMP_Volatility is available: stable, moderate, and high volatility.",
            "The analysis is a screening model and does not assume perfect foresight or replace hourly dispatch validation.",
        ]
    )


def _data_quality_notes(cleaning_summary: object | None, coordinate_status: object | None) -> str:
    lines = ["## Data Quality Notes"]
    notes = _extract_notes(cleaning_summary)
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- No cleaning summary was provided.")

    coordinate_dict = _object_to_dict(coordinate_status)
    coordinate_message = coordinate_dict.get("message")
    if coordinate_message:
        lines.append(f"- {coordinate_message}")
    return "\n".join(lines)


def _dataframe_to_markdown(dataframe: pd.DataFrame, columns: Sequence[str], max_rows: int) -> str:
    selected_columns = [column for column in columns if column in dataframe.columns]
    if dataframe.empty or not selected_columns:
        return "_No rows available._"

    view = dataframe.loc[:, selected_columns].head(max_rows)
    header = "| " + " | ".join(selected_columns) + " |"
    separator = "| " + " | ".join("---" for _ in selected_columns) + " |"
    rows = []
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(_format_cell(row[column], column) for column in selected_columns) + " |")
    return "\n".join([header, separator, *rows])


def _extract_notes(cleaning_summary: object | None) -> list[str]:
    summary_dict = _object_to_dict(cleaning_summary)
    notes = summary_dict.get("notes")
    if isinstance(notes, list):
        return [str(note) for note in notes]
    return []


def _object_to_dict(value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _format_cell(value: object, column: str) -> str:
    if pd.isna(value):
        return ""
    if column in {"Revenue_per_kW", "Average_Revenue_per_kW"}:
        return _fmt_dollars_per_kw(value)
    if "Revenue" in column:
        return _fmt_currency(value)
    if column in {"LMP_Volatility", "Opportunity_Score", "Average_LMP_Volatility", "Average_Opportunity_Score"}:
        return _fmt_number(value)
    return str(value)


def _fmt_currency(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"${number:,.0f}"


def _fmt_number(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number:,.2f}"


def _fmt_dollars_per_kw(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"${number:,.2f}/kW"


def _fmt_percent(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.0f}%"


def _to_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _fmt_metric_value(value: object, metric_label: str) -> str:
    if "kw" in metric_label.lower():
        return _fmt_dollars_per_kw(value)
    if "revenue" in metric_label.lower():
        return _fmt_currency(value)
    return _fmt_number(value)


def _join_items(items: Sequence[str]) -> str:
    cleaned = [item for item in items if item]
    if not cleaned:
        return "n/a"
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"
