"""Cleaning pipeline for FlexWorks simulation exports."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .validation import COORDINATE_COLUMNS, REQUIRED_NUMERIC_COLUMNS, normalize_column_names


@dataclass(frozen=True)
class CleaningSummary:
    """Data quality summary produced by the cleaning pipeline."""

    original_rows: int
    cleaned_rows: int
    rows_removed_empty_node: int
    duplicate_rows_aggregated: int
    numeric_values_coerced_to_missing: dict[str, int] = field(default_factory=dict)
    missing_required_numeric: dict[str, int] = field(default_factory=dict)
    missing_coordinates: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable representation for UI/reporting."""

        return {
            "original_rows": self.original_rows,
            "cleaned_rows": self.cleaned_rows,
            "rows_removed_empty_node": self.rows_removed_empty_node,
            "duplicate_rows_aggregated": self.duplicate_rows_aggregated,
            "numeric_values_coerced_to_missing": self.numeric_values_coerced_to_missing,
            "missing_required_numeric": self.missing_required_numeric,
            "missing_coordinates": self.missing_coordinates,
            "notes": self.notes,
        }


def clean_flexworks_export(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, CleaningSummary]:
    """Clean a loaded FlexWorks export and aggregate duplicate nodes.

    Numeric metrics are averaged when duplicate ``Node_ID`` values are present.
    This avoids accidental revenue inflation from duplicated annualized rows
    while preserving the node-level comparison intent of the workflow.
    """

    original_rows = len(dataframe)
    cleaned = normalize_column_names(dataframe)
    cleaned = _strip_whitespace(cleaned)

    cleaned["Node_ID"] = cleaned["Node_ID"].astype("string").str.strip()
    empty_node_mask = cleaned["Node_ID"].isna() | (cleaned["Node_ID"] == "")
    rows_removed_empty_node = int(empty_node_mask.sum())
    cleaned = cleaned.loc[~empty_node_mask].copy()

    numeric_columns = _numeric_columns_to_clean(cleaned)
    numeric_values_coerced_to_missing: dict[str, int] = {}
    for column in numeric_columns:
        before_missing = int(cleaned[column].isna().sum())
        cleaned[column] = _coerce_numeric(cleaned[column])
        after_missing = int(cleaned[column].isna().sum())
        numeric_values_coerced_to_missing[column] = max(after_missing - before_missing, 0)

    duplicate_rows_aggregated = int(cleaned.duplicated(subset=["Node_ID"], keep=False).sum())
    if duplicate_rows_aggregated:
        cleaned = _aggregate_duplicate_nodes(cleaned, numeric_columns)

    missing_required_numeric = {
        column: int(cleaned[column].isna().sum())
        for column in REQUIRED_NUMERIC_COLUMNS
        if column in cleaned.columns
    }

    missing_coordinates = None
    if all(column in cleaned.columns for column in COORDINATE_COLUMNS):
        missing_coordinates = int(cleaned[list(COORDINATE_COLUMNS)].isna().any(axis=1).sum())

    notes = _build_cleaning_notes(
        rows_removed_empty_node=rows_removed_empty_node,
        duplicate_rows_aggregated=duplicate_rows_aggregated,
        numeric_values_coerced_to_missing=numeric_values_coerced_to_missing,
        missing_coordinates=missing_coordinates,
    )

    summary = CleaningSummary(
        original_rows=original_rows,
        cleaned_rows=len(cleaned),
        rows_removed_empty_node=rows_removed_empty_node,
        duplicate_rows_aggregated=duplicate_rows_aggregated,
        numeric_values_coerced_to_missing=numeric_values_coerced_to_missing,
        missing_required_numeric=missing_required_numeric,
        missing_coordinates=missing_coordinates,
        notes=notes,
    )

    return cleaned.reset_index(drop=True), summary


def _strip_whitespace(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    string_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
    return cleaned


def _numeric_columns_to_clean(dataframe: pd.DataFrame) -> list[str]:
    optional_numeric_columns = [column for column in COORDINATE_COLUMNS if column in dataframe.columns]
    return [column for column in (*REQUIRED_NUMERIC_COLUMNS, *optional_numeric_columns) if column in dataframe.columns]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    normalized = (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA})
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        .str.replace(r"[$,%]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(normalized, errors="coerce")


def _aggregate_duplicate_nodes(dataframe: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    aggregation: dict[str, str] = {}
    for column in dataframe.columns:
        if column == "Node_ID":
            continue
        if column in numeric_columns or pd.api.types.is_numeric_dtype(dataframe[column]):
            aggregation[column] = "mean"
        else:
            aggregation[column] = _first_non_null

    grouped = dataframe.groupby("Node_ID", as_index=False, dropna=False).agg(aggregation)
    return grouped


def _first_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def _build_cleaning_notes(
    *,
    rows_removed_empty_node: int,
    duplicate_rows_aggregated: int,
    numeric_values_coerced_to_missing: dict[str, int],
    missing_coordinates: int | None,
) -> list[str]:
    notes: list[str] = []

    if rows_removed_empty_node:
        notes.append(f"Removed {rows_removed_empty_node} row(s) with blank Node_ID values.")
    if duplicate_rows_aggregated:
        notes.append(
            f"Aggregated {duplicate_rows_aggregated} duplicate node row(s) by averaging numeric fields and keeping the first non-empty categorical value."
        )

    coerced_columns = [
        f"{column} ({count})" for column, count in numeric_values_coerced_to_missing.items() if count > 0
    ]
    if coerced_columns:
        notes.append("Coerced non-numeric values to missing in: " + ", ".join(coerced_columns) + ".")

    if missing_coordinates is None:
        notes.append("No coordinate columns were found; map visualizations will be skipped unless a lookup table is supplied.")
    elif missing_coordinates:
        notes.append(f"{missing_coordinates} node(s) are missing latitude or longitude values.")

    if not notes:
        notes.append("No material cleaning issues detected.")

    return notes
