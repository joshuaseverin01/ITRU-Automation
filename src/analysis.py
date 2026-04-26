"""Analysis engine for node-level battery arbitrage opportunity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .geo import normalize_zone_name
from .temporal import (
    TIME_GRANULARITY_NONE,
    detect_time_granularity,
    filter_time_range,
    format_time_label,
    format_time_range_label,
    normalize_time_value,
    normalize_time_values,
    time_column_for_granularity,
)


DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "Annualized_Revenue": 0.35,
    "Revenue_per_kW": 0.45,
    "LMP_Volatility": 0.20,
}

RISK_LABEL_STABLE = "stable"
RISK_LABEL_MODERATE = "moderate"
RISK_LABEL_HIGH = "high volatility"
RISK_LABEL_UNKNOWN = "unknown"
ALL_REVENUE_CATEGORIES = "All categories"
SNAPSHOT_METRIC_MONTHLY_REVENUE = "Monthly Revenue"
SNAPSHOT_METRIC_CUMULATIVE_REVENUE = "Cumulative Revenue"
SNAPSHOT_METRIC_REVENUE_PER_KW = "Revenue per kW"

SNAPSHOT_METRIC_COLUMNS = {
    SNAPSHOT_METRIC_MONTHLY_REVENUE: "Monthly_Revenue",
    SNAPSHOT_METRIC_CUMULATIVE_REVENUE: "Cumulative_Revenue",
    SNAPSHOT_METRIC_REVENUE_PER_KW: "Revenue_per_kW",
}


@dataclass(frozen=True)
class SummaryMetrics:
    """Top-line metrics for the filtered analysis dataset."""

    node_count: int
    iso_count: int
    iso_regions: list[str]
    average_annualized_revenue: float | None
    median_annualized_revenue: float | None
    average_revenue_per_kw: float | None
    median_revenue_per_kw: float | None
    average_lmp_volatility: float | None
    median_lmp_volatility: float | None
    min_lmp_volatility: float | None
    max_lmp_volatility: float | None
    max_revenue_node: str | None
    max_revenue_per_kw_node: str | None
    top_opportunity_node: str | None
    high_volatility_node_count: int

    def to_dict(self) -> dict[str, object]:
        """Return a serializable representation for UI/reporting."""

        return {
            "node_count": self.node_count,
            "iso_count": self.iso_count,
            "iso_regions": self.iso_regions,
            "average_annualized_revenue": self.average_annualized_revenue,
            "median_annualized_revenue": self.median_annualized_revenue,
            "average_revenue_per_kw": self.average_revenue_per_kw,
            "median_revenue_per_kw": self.median_revenue_per_kw,
            "average_lmp_volatility": self.average_lmp_volatility,
            "median_lmp_volatility": self.median_lmp_volatility,
            "min_lmp_volatility": self.min_lmp_volatility,
            "max_lmp_volatility": self.max_lmp_volatility,
            "max_revenue_node": self.max_revenue_node,
            "max_revenue_per_kw_node": self.max_revenue_per_kw_node,
            "top_opportunity_node": self.top_opportunity_node,
            "high_volatility_node_count": self.high_volatility_node_count,
        }


def add_analysis_columns(
    dataframe: pd.DataFrame,
    score_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Add normalized metrics, opportunity score, and risk labels.

    The opportunity score is a transparent weighted average of normalized
    annualized revenue, revenue per kW, and LMP volatility. Volatility is
    treated as an arbitrage opportunity signal while risk labels separately
    identify exposure.
    """

    analyzed = dataframe.copy()
    weights = get_effective_score_weights(analyzed, score_weights)

    for column in DEFAULT_SCORE_WEIGHTS:
        analyzed[f"{column}_Normalized"] = _min_max_normalize(analyzed.get(column))

    score = pd.Series(0.0, index=analyzed.index)
    for column, weight in weights.items():
        normalized_column = f"{column}_Normalized"
        score = score + analyzed[normalized_column].fillna(0.0) * weight

    analyzed["Opportunity_Score"] = score.round(2)
    analyzed["Risk_Label"] = classify_volatility_risk(analyzed.get("LMP_Volatility"))
    return analyzed


def get_effective_score_weights(
    dataframe: pd.DataFrame,
    score_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return normalized weights for metrics with at least one valid value."""

    weights = normalize_score_weights(score_weights)
    available_weights: dict[str, float] = {}
    for column, weight in weights.items():
        if column in dataframe.columns and not pd.to_numeric(dataframe[column], errors="coerce").dropna().empty:
            available_weights[column] = weight
        else:
            available_weights[column] = 0.0

    total_available_weight = sum(available_weights.values())
    if total_available_weight <= 0:
        return weights
    return {column: weight / total_available_weight for column, weight in available_weights.items()}


def normalize_score_weights(score_weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """Normalize user-provided non-negative scoring weights to sum to 1."""

    raw_weights = dict(DEFAULT_SCORE_WEIGHTS)
    if score_weights:
        for column in DEFAULT_SCORE_WEIGHTS:
            raw_weights[column] = float(score_weights.get(column, raw_weights[column]))

    cleaned_weights = {column: max(weight, 0.0) for column, weight in raw_weights.items()}
    total_weight = sum(cleaned_weights.values())
    if total_weight <= 0:
        return dict(DEFAULT_SCORE_WEIGHTS)

    return {column: weight / total_weight for column, weight in cleaned_weights.items()}


def compute_summary_metrics(dataframe: pd.DataFrame) -> SummaryMetrics:
    """Compute top-line metrics for the current filtered dataset."""

    if dataframe.empty:
        return SummaryMetrics(
            node_count=0,
            iso_count=0,
            iso_regions=[],
            average_annualized_revenue=None,
            median_annualized_revenue=None,
            average_revenue_per_kw=None,
            median_revenue_per_kw=None,
            average_lmp_volatility=None,
            median_lmp_volatility=None,
            min_lmp_volatility=None,
            max_lmp_volatility=None,
            max_revenue_node=None,
            max_revenue_per_kw_node=None,
            top_opportunity_node=None,
            high_volatility_node_count=0,
        )

    iso_regions = sorted(dataframe["ISO_Region"].dropna().astype(str).unique().tolist()) if "ISO_Region" in dataframe else []

    return SummaryMetrics(
        node_count=int(dataframe["Node_ID"].nunique()) if "Node_ID" in dataframe else len(dataframe),
        iso_count=len(iso_regions),
        iso_regions=iso_regions,
        average_annualized_revenue=_safe_mean(dataframe.get("Annualized_Revenue")),
        median_annualized_revenue=_safe_median(dataframe.get("Annualized_Revenue")),
        average_revenue_per_kw=_safe_mean(dataframe.get("Revenue_per_kW")),
        median_revenue_per_kw=_safe_median(dataframe.get("Revenue_per_kW")),
        average_lmp_volatility=_safe_mean(dataframe.get("LMP_Volatility")),
        median_lmp_volatility=_safe_median(dataframe.get("LMP_Volatility")),
        min_lmp_volatility=_safe_min(dataframe.get("LMP_Volatility")),
        max_lmp_volatility=_safe_max(dataframe.get("LMP_Volatility")),
        max_revenue_node=_node_at_max(dataframe, "Annualized_Revenue"),
        max_revenue_per_kw_node=_node_at_max(dataframe, "Revenue_per_kW"),
        top_opportunity_node=_node_at_max(dataframe, "Opportunity_Score"),
        high_volatility_node_count=_count_high_volatility(dataframe),
    )


def rank_nodes(dataframe: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Return nodes ranked by opportunity score and revenue per kW."""

    if dataframe.empty:
        return dataframe.copy()

    ranked = dataframe.copy()
    sort_columns = [column for column in ("Opportunity_Score", "Revenue_per_kW", "Annualized_Revenue") if column in ranked]
    if sort_columns:
        ranked = ranked.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")

    ranked["Rank"] = np.arange(1, len(ranked) + 1)
    if top_n is not None:
        ranked = ranked.head(top_n)
    return ranked.reset_index(drop=True)


def identify_high_risk_high_reward(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Identify nodes with high volatility and top-quartile revenue per kW."""

    required_columns = {"Risk_Label", "Revenue_per_kW"}
    if dataframe.empty or not required_columns.issubset(dataframe.columns):
        return dataframe.head(0).copy()

    valid_revenue = pd.to_numeric(dataframe["Revenue_per_kW"], errors="coerce").dropna()
    if valid_revenue.empty:
        return dataframe.head(0).copy()

    reward_threshold = float(valid_revenue.quantile(0.75))
    mask = (dataframe["Risk_Label"] == RISK_LABEL_HIGH) & (pd.to_numeric(dataframe["Revenue_per_kW"], errors="coerce") >= reward_threshold)
    high_risk_high_reward = dataframe.loc[mask].copy()
    return rank_nodes(high_risk_high_reward)


def summarize_iso_regions(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate node-level results into ISO-level comparison metrics."""

    required_columns = {"ISO_Region"}
    if dataframe.empty or not required_columns.issubset(dataframe.columns):
        return pd.DataFrame()

    aggregations = {
        "Node_ID": "nunique",
        "Annualized_Revenue": "mean",
        "Revenue_per_kW": "mean",
        "LMP_Volatility": "mean",
    }
    if "Opportunity_Score" in dataframe.columns:
        aggregations["Opportunity_Score"] = "mean"

    summary = dataframe.groupby("ISO_Region", dropna=False).agg(aggregations).reset_index()
    summary = summary.rename(
        columns={
            "Node_ID": "Node_Count",
            "Annualized_Revenue": "Average_Annualized_Revenue",
            "Revenue_per_kW": "Average_Revenue_per_kW",
            "LMP_Volatility": "Average_LMP_Volatility",
            "Opportunity_Score": "Average_Opportunity_Score",
        }
    )

    sort_column = "Average_Opportunity_Score" if "Average_Opportunity_Score" in summary.columns else "Average_Revenue_per_kW"
    return summary.sort_values(sort_column, ascending=False, na_position="last").reset_index(drop=True)


def compute_zone_monthly_revenue(
    monthly_dataframe: pd.DataFrame | None,
    revenue_category: str = ALL_REVENUE_CATEGORIES,
    iso_region: str = "PJM",
) -> pd.DataFrame:
    """Aggregate monthly FlexWorks revenue by PJM zone and compute cumulative revenue."""

    if monthly_dataframe is None or monthly_dataframe.empty:
        return pd.DataFrame(
            columns=[
                "Zone",
                "Zone_Normalized",
                "ISO_Region",
                "Month",
                "Revenue_Category_Filter",
                "Monthly_Revenue",
                "Cumulative_Revenue",
            ]
        )

    required_columns = {"Zone", "Month", "Revenue"}
    if not required_columns.issubset(monthly_dataframe.columns):
        return pd.DataFrame(columns=["Zone", "Zone_Normalized", "ISO_Region", "Month", "Revenue_Category_Filter", "Monthly_Revenue", "Cumulative_Revenue"])

    working = monthly_dataframe.copy()
    if "ISO_Region" in working.columns:
        working = working.loc[working["ISO_Region"].astype("string").str.upper() == iso_region.upper()].copy()

    if revenue_category != ALL_REVENUE_CATEGORIES and "Revenue_Category" in working.columns:
        working = working.loc[working["Revenue_Category"].astype("string") == revenue_category].copy()

    working["Month"] = pd.to_datetime(working["Month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    working["Revenue"] = pd.to_numeric(working["Revenue"], errors="coerce")
    working["Zone"] = working["Zone"].astype("string").str.strip()
    working["Zone_Normalized"] = working["Zone"].apply(normalize_zone_name)
    working = working.dropna(subset=["Month", "Revenue", "Zone"])
    working = working.loc[working["Zone_Normalized"] != ""].copy()
    if working.empty:
        return pd.DataFrame(columns=["Zone", "Zone_Normalized", "ISO_Region", "Month", "Revenue_Category_Filter", "Monthly_Revenue", "Cumulative_Revenue"])

    grouped = (
        working.groupby(["Zone_Normalized", "Month"], as_index=False, dropna=False)
        .agg(
            Zone=("Zone", _first_string),
            ISO_Region=("ISO_Region", _first_string) if "ISO_Region" in working.columns else ("Zone", lambda _: iso_region),
            Monthly_Revenue=("Revenue", "sum"),
        )
        .sort_values(["Zone_Normalized", "Month"])
    )
    grouped["Revenue_Category_Filter"] = revenue_category
    grouped["Cumulative_Revenue"] = grouped.groupby("Zone_Normalized")["Monthly_Revenue"].cumsum()
    return grouped.reset_index(drop=True)


def filter_zone_revenue_to_month(zone_monthly_revenue: pd.DataFrame, selected_month: object) -> pd.DataFrame:
    """Return zone revenue rows for the selected month."""

    if zone_monthly_revenue.empty or "Month" not in zone_monthly_revenue.columns:
        return zone_monthly_revenue.head(0).copy()

    month = pd.to_datetime(selected_month, errors="coerce")
    if pd.isna(month):
        return zone_monthly_revenue.head(0).copy()

    normalized_month = month.to_period("M").to_timestamp()
    data = zone_monthly_revenue.copy()
    data["Month"] = pd.to_datetime(data["Month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return data.loc[data["Month"] == normalized_month].copy().reset_index(drop=True)


def compute_cumulative_by_zone(
    dataframe: pd.DataFrame | None,
    revenue_category: str = ALL_REVENUE_CATEGORIES,
    iso_region: str | None = "PJM",
) -> pd.DataFrame:
    """Aggregate time-series revenue by zone and compute cumulative revenue."""

    if dataframe is None or dataframe.empty:
        return _empty_zone_time_dataframe()

    granularity = detect_time_granularity(dataframe)
    time_column = time_column_for_granularity(granularity)
    required_columns = {"Zone", "Revenue"}
    if granularity == TIME_GRANULARITY_NONE or time_column is None or not required_columns.issubset(dataframe.columns):
        return _empty_zone_time_dataframe()

    working = dataframe.copy()
    if iso_region and "ISO_Region" in working.columns:
        working = working.loc[working["ISO_Region"].astype("string").str.upper() == iso_region.upper()].copy()

    if revenue_category != ALL_REVENUE_CATEGORIES and "Revenue_Category" in working.columns:
        working = working.loc[working["Revenue_Category"].astype("string") == revenue_category].copy()

    if working.empty:
        return _empty_zone_time_dataframe()

    working["_Time"] = normalize_time_values(working[time_column], granularity).to_numpy()
    working["Revenue"] = pd.to_numeric(working["Revenue"], errors="coerce")
    working["Zone"] = working["Zone"].astype("string").str.strip()
    working["Zone_Normalized"] = working["Zone"].apply(normalize_zone_name)
    working = working.dropna(subset=["_Time", "Revenue", "Zone"])
    working = working.loc[working["Zone_Normalized"] != ""].copy()
    if working.empty:
        return _empty_zone_time_dataframe()

    aggregation: dict[str, object] = {
        "Zone": ("Zone", _first_string),
        "ISO_Region": ("ISO_Region", _first_string) if "ISO_Region" in working.columns else ("Zone", lambda _: iso_region or ""),
        "Monthly_Revenue": ("Revenue", "sum"),
    }
    for column in ("Annualized_Revenue", "Revenue_per_kW", "Opportunity_Score", "Risk_Adjusted_Score"):
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
            aggregation[column] = (column, "mean")

    grouped = (
        working.groupby(["Zone_Normalized", "_Time"], as_index=False, dropna=False)
        .agg(**aggregation)
        .sort_values(["Zone_Normalized", "_Time"])
    )
    grouped["Revenue_Category_Filter"] = revenue_category
    grouped["Cumulative_Revenue"] = grouped.groupby("Zone_Normalized")["Monthly_Revenue"].cumsum()
    grouped["Time"] = pd.to_datetime(grouped["_Time"], errors="coerce")
    grouped["Time_Label"] = grouped["Time"].apply(lambda value: format_time_label(value, granularity))
    grouped["Time_Granularity"] = granularity
    if granularity == "monthly":
        grouped["Month"] = grouped["Time"]
    else:
        grouped["Timestamp"] = grouped["Time"]

    return grouped.drop(columns=["_Time"]).reset_index(drop=True)


def aggregate_zone_metric(
    dataframe: pd.DataFrame | None,
    metric: str,
    category: str = ALL_REVENUE_CATEGORIES,
    time_point: object | None = None,
    iso_region: str | None = "PJM",
) -> pd.DataFrame:
    """Return one zone-level snapshot for the selected metric and time point."""

    zone_time = compute_cumulative_by_zone(dataframe, revenue_category=category, iso_region=iso_region)
    if zone_time.empty:
        return zone_time

    metric_column = SNAPSHOT_METRIC_COLUMNS.get(metric, metric)
    if metric_column not in zone_time.columns:
        return zone_time.head(0).copy()

    granularity = str(zone_time["Time_Granularity"].dropna().iloc[0])
    if time_point is None:
        selected_time = zone_time["Time"].max()
    else:
        selected_time = normalize_time_value(time_point, granularity)
    if pd.isna(selected_time):
        return zone_time.head(0).copy()

    snapshot = zone_time.loc[zone_time["Time"] == selected_time].copy()
    if snapshot.empty:
        return snapshot

    snapshot[metric_column] = pd.to_numeric(snapshot[metric_column], errors="coerce")
    snapshot = snapshot.dropna(subset=[metric_column])
    if snapshot.empty:
        return snapshot

    snapshot["Selected_Metric"] = snapshot[metric_column]
    snapshot["Selected_Metric_Column"] = metric_column
    snapshot["Metric_Label"] = metric
    snapshot["Time_Label"] = snapshot["Time"].apply(lambda value: format_time_label(value, granularity))
    return snapshot.sort_values("Selected_Metric", ascending=False).reset_index(drop=True)


def aggregate_zone_metric_over_range(
    dataframe: pd.DataFrame | None,
    metric: str,
    category: str = ALL_REVENUE_CATEGORIES,
    start_time: object | None = None,
    end_time: object | None = None,
    iso_region: str | None = "PJM",
) -> pd.DataFrame:
    """Aggregate one zone-level metric over an inclusive selected time range."""

    if dataframe is None or dataframe.empty:
        return _empty_zone_time_dataframe()

    working = dataframe.copy()
    if iso_region and "ISO_Region" in working.columns:
        working = working.loc[working["ISO_Region"].astype("string").str.upper() == iso_region.upper()].copy()
    if category != ALL_REVENUE_CATEGORIES and "Revenue_Category" in working.columns:
        working = working.loc[working["Revenue_Category"].astype("string") == category].copy()
    if working.empty:
        return _empty_zone_time_dataframe()

    granularity = detect_time_granularity(working)
    time_column = time_column_for_granularity(granularity)
    required_columns = {"Zone", "Revenue"}
    if granularity == TIME_GRANULARITY_NONE or time_column is None or not required_columns.issubset(working.columns):
        return _empty_zone_time_dataframe()

    start = normalize_time_value(start_time if start_time is not None else working[time_column].min(), granularity)
    end = normalize_time_value(end_time if end_time is not None else working[time_column].max(), granularity)
    if pd.isna(start) or pd.isna(end) or start > end:
        return _empty_zone_time_dataframe()

    working = filter_time_range(working, start, end)
    if working.empty:
        return _empty_zone_time_dataframe()

    working["_Time"] = normalize_time_values(working[time_column], granularity).to_numpy()
    working["Revenue"] = pd.to_numeric(working["Revenue"], errors="coerce")
    working["Zone"] = working["Zone"].astype("string").str.strip()
    working["Zone_Normalized"] = working["Zone"].apply(normalize_zone_name)
    working = working.dropna(subset=["_Time", "Revenue", "Zone"])
    working = working.loc[working["Zone_Normalized"] != ""].copy()
    if working.empty:
        return _empty_zone_time_dataframe()

    aggregation: dict[str, object] = {
        "Zone": ("Zone", _first_string),
        "ISO_Region": ("ISO_Region", _first_string) if "ISO_Region" in working.columns else ("Zone", lambda _: iso_region or ""),
        "Monthly_Revenue": ("Revenue", "sum"),
        "Cumulative_Revenue": ("Revenue", "sum"),
    }
    for column in ("Annualized_Revenue", "Revenue_per_kW", "Opportunity_Score", "Risk_Adjusted_Score"):
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
            aggregation[column] = (column, "mean")

    grouped = working.groupby("Zone_Normalized", as_index=False, dropna=False).agg(**aggregation)
    metric_column = SNAPSHOT_METRIC_COLUMNS.get(metric, metric)
    if metric_column not in grouped.columns:
        return grouped.head(0).copy()

    grouped[metric_column] = pd.to_numeric(grouped[metric_column], errors="coerce")
    grouped = grouped.dropna(subset=[metric_column])
    if grouped.empty:
        return grouped

    grouped["Selected_Metric"] = grouped[metric_column]
    grouped["Selected_Metric_Column"] = metric_column
    grouped["Metric_Label"] = metric
    grouped["Revenue_Category_Filter"] = category
    grouped["Time_Granularity"] = granularity
    grouped["Time_Start"] = start
    grouped["Time_End"] = end
    grouped["Time_Label"] = format_time_range_label(start, end, granularity)
    return grouped.sort_values("Selected_Metric", ascending=False).reset_index(drop=True)


def classify_volatility_risk(volatility: pd.Series | None) -> pd.Series:
    """Classify volatility risk using dataset-relative tertiles."""

    if volatility is None:
        return pd.Series(dtype="string")

    numeric = pd.to_numeric(volatility, errors="coerce")
    labels = pd.Series(RISK_LABEL_UNKNOWN, index=numeric.index, dtype="string")
    valid = numeric.dropna()
    if valid.empty:
        return labels

    lower_threshold = float(valid.quantile(1 / 3))
    upper_threshold = float(valid.quantile(2 / 3))

    if lower_threshold == upper_threshold:
        labels.loc[numeric.notna()] = RISK_LABEL_MODERATE
        return labels

    labels.loc[numeric <= lower_threshold] = RISK_LABEL_STABLE
    labels.loc[(numeric > lower_threshold) & (numeric <= upper_threshold)] = RISK_LABEL_MODERATE
    labels.loc[numeric > upper_threshold] = RISK_LABEL_HIGH
    return labels


def _min_max_normalize(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")

    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    normalized = pd.Series(np.nan, index=numeric.index, dtype="float64")
    if valid.empty:
        return normalized

    minimum = float(valid.min())
    maximum = float(valid.max())
    if minimum == maximum:
        normalized.loc[numeric.notna()] = 50.0
        return normalized

    normalized.loc[numeric.notna()] = ((numeric.loc[numeric.notna()] - minimum) / (maximum - minimum)) * 100.0
    return normalized


def _safe_mean(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    value = pd.to_numeric(series, errors="coerce").mean()
    return None if pd.isna(value) else float(value)


def _safe_median(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    value = pd.to_numeric(series, errors="coerce").median()
    return None if pd.isna(value) else float(value)


def _safe_min(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    value = pd.to_numeric(series, errors="coerce").min()
    return None if pd.isna(value) else float(value)


def _safe_max(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    value = pd.to_numeric(series, errors="coerce").max()
    return None if pd.isna(value) else float(value)


def _node_at_max(dataframe: pd.DataFrame, metric_column: str) -> str | None:
    if dataframe.empty or metric_column not in dataframe or "Node_ID" not in dataframe:
        return None

    numeric = pd.to_numeric(dataframe[metric_column], errors="coerce")
    if numeric.dropna().empty:
        return None

    index = numeric.idxmax()
    node = dataframe.loc[index, "Node_ID"]
    return None if pd.isna(node) else str(node)


def _count_high_volatility(dataframe: pd.DataFrame) -> int:
    if "Risk_Label" not in dataframe:
        return 0
    return int((dataframe["Risk_Label"] == RISK_LABEL_HIGH).sum())


def _empty_zone_time_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Zone",
            "Zone_Normalized",
            "ISO_Region",
            "Time",
            "Time_Start",
            "Time_End",
            "Time_Label",
            "Time_Granularity",
            "Revenue_Category_Filter",
            "Monthly_Revenue",
            "Cumulative_Revenue",
            "Annualized_Revenue",
            "Revenue_per_kW",
            "Opportunity_Score",
            "Risk_Adjusted_Score",
            "Selected_Metric",
            "Selected_Metric_Column",
            "Metric_Label",
        ]
    )


def _first_string(series: pd.Series) -> str:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return ""
    return non_null.iloc[0]
