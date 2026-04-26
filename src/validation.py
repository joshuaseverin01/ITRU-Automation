"""Schema validation helpers for FlexWorks exports."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS: tuple[str, ...] = (
    "Node_ID",
    "ISO_Region",
    "Annualized_Revenue",
    "Revenue_per_kW",
    "LMP_Volatility",
)

REQUIRED_NUMERIC_COLUMNS: tuple[str, ...] = (
    "Annualized_Revenue",
    "Revenue_per_kW",
    "LMP_Volatility",
)

COORDINATE_COLUMNS: tuple[str, str] = ("Latitude", "Longitude")


@dataclass(frozen=True)
class ValidationResult:
    """Result object returned by schema validation."""

    is_valid: bool
    missing_columns: list[str]
    available_columns: list[str]
    row_count: int
    warnings: list[str]


def normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with whitespace-trimmed string column names."""

    normalized = dataframe.copy()
    normalized.columns = [str(column).replace("\ufeff", "").strip() for column in normalized.columns]
    return normalized


def validate_required_columns(dataframe: pd.DataFrame) -> ValidationResult:
    """Validate the required FlexWorks export columns.

    Column-name whitespace is ignored, so a file with headers like
    ``" Node_ID "`` still validates and can be cleaned downstream.
    """

    available_columns = [str(column).strip() for column in dataframe.columns]
    available_set = set(available_columns)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in available_set]
    warnings: list[str] = []

    if dataframe.empty:
        warnings.append("The dataset has no rows after loading.")

    return ValidationResult(
        is_valid=not missing_columns and not dataframe.empty,
        missing_columns=missing_columns,
        available_columns=available_columns,
        row_count=len(dataframe),
        warnings=warnings,
    )


def validate_coordinate_lookup(dataframe: pd.DataFrame) -> ValidationResult:
    """Validate an optional coordinate lookup table."""

    available_columns = [str(column).strip() for column in dataframe.columns]
    available_set = set(available_columns)
    required = ("Node_ID", *COORDINATE_COLUMNS)
    missing_columns = [column for column in required if column not in available_set]
    warnings: list[str] = []

    if dataframe.empty:
        warnings.append("The coordinate lookup table has no rows.")

    return ValidationResult(
        is_valid=not missing_columns and not dataframe.empty,
        missing_columns=missing_columns,
        available_columns=available_columns,
        row_count=len(dataframe),
        warnings=warnings,
    )


def format_missing_columns_message(missing_columns: list[str]) -> str:
    """Build a concise user-facing missing-column message."""

    if not missing_columns:
        return ""
    missing = ", ".join(missing_columns)
    required = ", ".join(REQUIRED_COLUMNS)
    return f"Missing required column(s): {missing}. Required columns are: {required}."
