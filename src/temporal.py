"""Time handling helpers for FlexWorks revenue series."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


TIME_GRANULARITY_MONTHLY = "monthly"
TIME_GRANULARITY_TIMESTAMP = "timestamp"
TIME_GRANULARITY_NONE = "none"


def detect_time_granularity(dataframe: pd.DataFrame | None) -> str:
    """Detect the most specific usable time column in a dataframe."""

    if dataframe is None or dataframe.empty:
        return TIME_GRANULARITY_NONE

    if "Timestamp" in dataframe.columns and not _valid_datetimes(dataframe["Timestamp"]).empty:
        return TIME_GRANULARITY_TIMESTAMP
    if "Month" in dataframe.columns and not _valid_datetimes(dataframe["Month"]).empty:
        return TIME_GRANULARITY_MONTHLY
    return TIME_GRANULARITY_NONE


def time_column_for_granularity(granularity: str) -> str | None:
    """Return the source time column for a detected granularity."""

    if granularity == TIME_GRANULARITY_TIMESTAMP:
        return "Timestamp"
    if granularity == TIME_GRANULARITY_MONTHLY:
        return "Month"
    return None


def available_time_points(dataframe: pd.DataFrame | None) -> list[pd.Timestamp]:
    """Return sorted valid time points for the detected granularity."""

    granularity = detect_time_granularity(dataframe)
    time_column = time_column_for_granularity(granularity)
    if dataframe is None or time_column is None:
        return []

    values = normalize_time_values(dataframe[time_column], granularity).dropna().drop_duplicates()
    return sorted(pd.Timestamp(value) for value in values)


def normalize_time_value(value: object, granularity: str) -> pd.Timestamp | pd.NaT:
    """Normalize one time value according to the selected granularity."""

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    if granularity == TIME_GRANULARITY_MONTHLY:
        return pd.Timestamp(timestamp).to_period("M").to_timestamp()
    return pd.Timestamp(timestamp)


def normalize_time_values(values: Iterable[object] | pd.Series, granularity: str) -> pd.Series:
    """Normalize a vector of time values according to the selected granularity."""

    timestamps = pd.to_datetime(values, errors="coerce")
    if granularity == TIME_GRANULARITY_MONTHLY:
        return pd.Series(timestamps).dt.to_period("M").dt.to_timestamp()
    return pd.Series(timestamps)


def format_time_label(value: object, granularity: str) -> str:
    """Format a time value for UI labels and chart subtitles."""

    timestamp = normalize_time_value(value, granularity)
    if pd.isna(timestamp):
        return "n/a"
    if granularity == TIME_GRANULARITY_MONTHLY:
        return pd.Timestamp(timestamp).strftime("%B %Y")
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def format_time_range_label(start: object, end: object, granularity: str) -> str:
    """Format an inclusive time range for chart subtitles."""

    start_label = format_time_label(start, granularity)
    end_label = format_time_label(end, granularity)
    if start_label == end_label:
        return start_label
    return f"{start_label} to {end_label}"


def filter_time_range(dataframe: pd.DataFrame, start: object, end: object) -> pd.DataFrame:
    """Filter rows to an inclusive time range using detected time granularity."""

    if dataframe.empty:
        return dataframe.copy()

    granularity = detect_time_granularity(dataframe)
    time_column = time_column_for_granularity(granularity)
    if time_column is None:
        return dataframe.head(0).copy()

    start_time = normalize_time_value(start, granularity)
    end_time = normalize_time_value(end, granularity)
    if pd.isna(start_time) or pd.isna(end_time) or start_time > end_time:
        return dataframe.head(0).copy()

    filtered = dataframe.copy()
    filtered[time_column] = normalize_time_values(filtered[time_column], granularity).to_numpy()
    mask = filtered[time_column].between(start_time, end_time, inclusive="both")
    return filtered.loc[mask].copy().reset_index(drop=True)


def select_evenly_spaced_snapshots(
    dataframe: pd.DataFrame,
    start: object,
    end: object,
    n: int,
) -> list[pd.Timestamp]:
    """Select up to n evenly spaced available time points within a range."""

    if n <= 0:
        return []

    filtered = filter_time_range(dataframe, start, end)
    points = available_time_points(filtered)
    if len(points) <= n:
        return points

    indices = np.linspace(0, len(points) - 1, num=n)
    selected_indices = sorted({int(round(index)) for index in indices})
    return [points[index] for index in selected_indices]


def _valid_datetimes(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce").dropna()
    return values
