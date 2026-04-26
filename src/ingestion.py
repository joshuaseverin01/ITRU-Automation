"""Schema detection and parsing for FlexWorks CSV export variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re

import pandas as pd

from .validation import REQUIRED_COLUMNS, normalize_column_names


class ExportSchema(str, Enum):
    """Supported CSV export schemas."""

    MVP_NODE = "current MVP node schema"
    DEVICE_SUMMARY = "FlexWorks device summary schema"
    MONTHLY_WIDE = "FlexWorks monthly wide-format schema"
    UNKNOWN = "unknown schema"


@dataclass(frozen=True)
class ParsedExport:
    """Parsed representation of one uploaded CSV export."""

    schema: ExportSchema
    node_dataframe: pd.DataFrame | None = None
    monthly_dataframe: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)


MONTH_COLUMN_RE = re.compile(r"^\d{4}-\d{2}$")


def parse_flexworks_export(dataframe: pd.DataFrame) -> ParsedExport:
    """Detect and parse a supported FlexWorks export dataframe."""

    schema = detect_export_schema(dataframe)
    if schema == ExportSchema.MVP_NODE:
        return ParsedExport(
            schema=schema,
            node_dataframe=drop_blank_columns(normalize_column_names(dataframe)),
            notes=["Detected current MVP node schema."],
        )
    if schema == ExportSchema.DEVICE_SUMMARY:
        node_dataframe, notes = parse_device_summary_export(dataframe)
        return ParsedExport(schema=schema, node_dataframe=node_dataframe, notes=notes)
    if schema == ExportSchema.MONTHLY_WIDE:
        monthly_dataframe, notes = parse_monthly_wide_export(dataframe)
        return ParsedExport(schema=schema, monthly_dataframe=monthly_dataframe, notes=notes)

    return ParsedExport(
        schema=schema,
        notes=["Unsupported CSV schema. Upload a node summary, device summary, or monthly wide-format export."],
    )


def detect_export_schema(dataframe: pd.DataFrame) -> ExportSchema:
    """Detect whether a dataframe matches a supported export schema."""

    normalized = drop_blank_columns(normalize_column_names(dataframe))
    columns = list(normalized.columns)
    column_set = set(columns)

    if set(REQUIRED_COLUMNS).issubset(column_set):
        return ExportSchema.MVP_NODE
    if _is_device_summary(columns):
        return ExportSchema.DEVICE_SUMMARY
    if _is_monthly_wide(normalized):
        return ExportSchema.MONTHLY_WIDE
    return ExportSchema.UNKNOWN


def parse_device_summary_export(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize a FlexWorks device summary export to node-level app columns."""

    cleaned = drop_blank_columns(normalize_column_names(dataframe))
    annualized_column = _find_column(cleaned.columns, lambda value: "annualized" in value and "income" in value)
    revenue_per_kw_column = _find_column(cleaned.columns, lambda value: value == "revenue per kw")

    required_columns = ["Device", "Location", annualized_column, revenue_per_kw_column]
    missing = [str(column) for column in required_columns if not column or column not in cleaned.columns]
    if missing:
        raise ValueError(f"Device summary export is missing required column(s): {', '.join(missing)}.")

    parsed = pd.DataFrame()
    parsed["Device"] = cleaned["Device"].astype("string").str.strip()
    parsed["Device_ID"] = parsed["Device"]
    parsed["Node_ID"] = parsed["Device"]
    parsed["Type"] = cleaned["Type"].astype("string").str.strip() if "Type" in cleaned.columns else pd.NA
    parsed["Location"] = cleaned["Location"].astype("string").str.strip()

    location_parts = parsed["Location"].apply(_extract_location_parts)
    parsed["Zone"] = location_parts.apply(lambda value: value[0])
    parsed["ISO_Region"] = location_parts.apply(lambda value: value[1])
    parsed["Node_Name"] = parsed["Location"]

    if "Income" in cleaned.columns:
        parsed["Income_Total"] = coerce_numeric_series(cleaned["Income"])
    parsed["Annualized_Revenue"] = coerce_numeric_series(cleaned[annualized_column])
    parsed["Revenue_per_kW"] = coerce_numeric_series(cleaned[revenue_per_kw_column])
    parsed["LMP_Volatility"] = pd.NA

    parsed = parsed.dropna(subset=["Device"]).copy()
    notes = [
        "Detected FlexWorks device summary schema.",
        "Removed blank trailing columns from the device summary export.",
        "Mapped Device to Node_ID and Device_ID for compatibility with the node analysis workflow.",
        "Parsed Location into Zone and ISO_Region when values matched 'Zone (ISO)'.",
        "LMP_Volatility is not present in this export and is left missing.",
    ]
    return parsed.reset_index(drop=True), notes


def parse_monthly_wide_export(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Reshape a FlexWorks monthly wide-format revenue export into long format."""

    cleaned = drop_blank_columns(normalize_column_names(dataframe))
    if len(cleaned.columns) < 3:
        raise ValueError("Monthly export must include device, category, and month columns.")

    month_columns = get_month_columns(cleaned)
    if not month_columns:
        raise ValueError("Monthly export does not include YYYY-MM month columns.")

    device_column = cleaned.columns[0]
    category_column = cleaned.columns[1]
    working = cleaned.rename(columns={device_column: "Device", category_column: "Revenue_Category"}).copy()
    working["Device"] = working["Device"].astype("string").str.strip().replace("", pd.NA).ffill()
    working["Revenue_Category"] = working["Revenue_Category"].astype("string").str.strip().replace("", pd.NA)
    working = working.dropna(subset=["Device", "Revenue_Category"])

    long_data = working.melt(
        id_vars=["Device", "Revenue_Category"],
        value_vars=month_columns,
        var_name="Month",
        value_name="Revenue",
    )
    long_data["Month"] = pd.to_datetime(long_data["Month"], format="%Y-%m", errors="coerce")
    long_data["Revenue"] = coerce_numeric_series(long_data["Revenue"])
    long_data = long_data.dropna(subset=["Month", "Revenue_Category"]).reset_index(drop=True)

    notes = [
        "Detected FlexWorks monthly wide-format schema.",
        f"Reshaped {len(month_columns)} monthly columns into long format.",
        "Forward-filled Device values across grouped revenue category rows.",
    ]
    return long_data, notes


def join_monthly_to_device_summary(monthly_dataframe: pd.DataFrame | None, node_dataframe: pd.DataFrame | None) -> tuple[pd.DataFrame | None, list[str]]:
    """Join monthly revenue rows to device summary metadata using Device."""

    if monthly_dataframe is None or monthly_dataframe.empty:
        return monthly_dataframe, []
    if node_dataframe is None or node_dataframe.empty:
        return monthly_dataframe.copy(), ["Monthly revenue data was loaded without device summary metadata."]

    monthly = monthly_dataframe.copy()
    node_metadata = node_dataframe.copy()

    if "Device" not in monthly.columns:
        return monthly, ["Monthly revenue data does not include a Device column, so metadata was not joined."]
    if "Device" not in node_metadata.columns:
        if "Device_ID" in node_metadata.columns:
            node_metadata["Device"] = node_metadata["Device_ID"]
        elif "Node_ID" in node_metadata.columns:
            node_metadata["Device"] = node_metadata["Node_ID"]
        else:
            return monthly, ["Device summary metadata does not include Device, Device_ID, or Node_ID, so monthly rows were not joined."]

    monthly["Device"] = monthly["Device"].astype("string").str.strip()
    node_metadata["Device"] = node_metadata["Device"].astype("string").str.strip()
    metadata_columns = [
        "Device",
        "Node_ID",
        "Device_ID",
        "Type",
        "Location",
        "Node_Name",
        "Zone",
        "ISO_Region",
        "Annualized_Revenue",
        "Revenue_per_kW",
        "LMP_Volatility",
        "Latitude",
        "Longitude",
    ]
    metadata_columns = [column for column in metadata_columns if column in node_metadata.columns]
    metadata = node_metadata[metadata_columns].drop_duplicates(subset=["Device"], keep="first")
    joined = monthly.merge(metadata, on="Device", how="left")
    unmatched_devices = int(joined["Node_ID"].isna().groupby(joined["Device"]).max().sum()) if "Node_ID" in joined.columns else 0

    notes = ["Joined monthly revenue data to device summary metadata using Device."]
    if unmatched_devices:
        notes.append(f"{unmatched_devices} monthly device(s) did not match the device summary metadata.")
    return joined, notes


def drop_blank_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Drop empty trailing columns common in exported CSV files."""

    cleaned = dataframe.copy()
    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = cleaned.loc[:, [not _is_unnamed_blank_column(column, cleaned[column]) for column in cleaned.columns]]
    return cleaned


def get_month_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return columns whose names match YYYY-MM."""

    return [str(column).strip() for column in dataframe.columns if MONTH_COLUMN_RE.match(str(column).strip())]


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Coerce currency-like values such as '$1,496.08' into numeric values."""

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


def _is_device_summary(columns: list[str]) -> bool:
    lower_columns = {column.lower().strip() for column in columns}
    has_annualized_income = any("annualized" in column and "income" in column for column in lower_columns)
    return {"device", "location", "revenue per kw"}.issubset(lower_columns) and has_annualized_income


def _is_monthly_wide(dataframe: pd.DataFrame) -> bool:
    month_columns = get_month_columns(dataframe)
    if len(month_columns) < 2 or len(dataframe.columns) < 3:
        return False
    category_values = dataframe.iloc[:, 1].astype("string").str.strip().str.lower().dropna().unique().tolist()
    expected_categories = {"energy", "ancillary", "fcp"}
    return bool(expected_categories.intersection(category_values))


def _extract_location_parts(location: object) -> tuple[str | None, str | None]:
    if pd.isna(location):
        return None, None
    text = str(location).strip()
    match = re.match(r"^(.*?)\s*\((.*?)\)\s*$", text)
    if match:
        return match.group(1).strip() or None, match.group(2).strip() or None
    return text or None, None


def _find_column(columns: pd.Index, predicate: object) -> str | None:
    for column in columns:
        if callable(predicate) and predicate(str(column).lower().strip()):
            return str(column)
    return None


def _is_unnamed_blank_column(column: object, series: pd.Series) -> bool:
    return str(column).lower().startswith("unnamed:") and series.isna().all()
