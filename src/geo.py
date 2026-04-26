"""Geospatial helpers for coordinates and PJM zone polygons."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import IO, Any

import pandas as pd

from .validation import COORDINATE_COLUMNS, normalize_column_names


LATITUDE_ALIASES = ("Latitude", "latitude", "LATITUDE", "lat", "Lat")
LONGITUDE_ALIASES = ("Longitude", "longitude", "LONGITUDE", "lon", "Lon", "lng", "Lng", "long", "Long")
ZONE_COLUMN_CANDIDATES = (
    "Zone",
    "zone",
    "ZONE",
    "PJM_Zone",
    "PJM Zone",
    "zoneName",
    "PLANNING_ZONE_NAME",
    "Location",
    "Node_Name",
)
GEOJSON_ZONE_PROPERTY_PREFERENCE = (
    "zoneName",
    "PLANNING_ZONE_NAME",
    "Zone",
    "ZONE",
    "zone",
    "NAME",
    "Name",
    "name",
)

GeoJsonSource = str | Path | IO[str] | IO[bytes]
GeoJson = dict[str, Any]


@dataclass(frozen=True)
class CoordinateStatus:
    """Coordinate readiness summary for map rendering."""

    has_coordinates: bool
    rows_with_coordinates: int
    rows_missing_coordinates: int
    message: str

    def to_dict(self) -> dict[str, object]:
        """Return a serializable representation for UI/reporting."""

        return {
            "has_coordinates": self.has_coordinates,
            "rows_with_coordinates": self.rows_with_coordinates,
            "rows_missing_coordinates": self.rows_missing_coordinates,
            "message": self.message,
        }


@dataclass(frozen=True)
class PjmZoneGeoJson:
    """Loaded PJM zone GeoJSON with normalized join properties."""

    geojson: GeoJson
    zone_property: str
    zone_count: int
    zones: list[str]


@dataclass(frozen=True)
class ZoneJoinDiagnostics:
    """Diagnostics for joining FlexWorks zone results to PJM polygons."""

    geojson_zone_count: int
    flexworks_zone_count: int
    matched_zone_count: int
    unmatched_flexworks_zones: list[str]
    unmatched_geojson_zones: list[str]
    zone_property: str | None
    data_zone_column: str | None
    is_available: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        """Return a serializable representation for UI/reporting."""

        return {
            "geojson_zone_count": self.geojson_zone_count,
            "flexworks_zone_count": self.flexworks_zone_count,
            "matched_zone_count": self.matched_zone_count,
            "unmatched_flexworks_zones": self.unmatched_flexworks_zones,
            "unmatched_geojson_zones": self.unmatched_geojson_zones,
            "zone_property": self.zone_property,
            "data_zone_column": self.data_zone_column,
            "is_available": self.is_available,
            "message": self.message,
        }


@dataclass(frozen=True)
class ZoneJoinResult:
    """Aggregated zone-level results ready for choropleth rendering."""

    dataframe: pd.DataFrame
    geojson: GeoJson | None
    diagnostics: ZoneJoinDiagnostics


def standardize_coordinate_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with common coordinate aliases renamed to Latitude/Longitude."""

    standardized = normalize_column_names(dataframe)
    rename_map: dict[str, str] = {}

    latitude_column = _find_first_column(standardized, LATITUDE_ALIASES)
    longitude_column = _find_first_column(standardized, LONGITUDE_ALIASES)
    if latitude_column and latitude_column != "Latitude":
        rename_map[latitude_column] = "Latitude"
    if longitude_column and longitude_column != "Longitude":
        rename_map[longitude_column] = "Longitude"

    if rename_map:
        standardized = standardized.rename(columns=rename_map)
    return standardized


def merge_coordinate_lookup(
    dataframe: pd.DataFrame,
    coordinate_lookup: pd.DataFrame | None,
) -> tuple[pd.DataFrame, CoordinateStatus]:
    """Merge an optional Node_ID coordinate lookup into the analysis dataset."""

    merged = standardize_coordinate_columns(dataframe)

    if coordinate_lookup is not None and not coordinate_lookup.empty:
        lookup = standardize_coordinate_columns(coordinate_lookup)
        if {"Node_ID", *COORDINATE_COLUMNS}.issubset(lookup.columns):
            lookup = lookup[["Node_ID", *COORDINATE_COLUMNS]].copy()
            lookup["Node_ID"] = lookup["Node_ID"].astype("string").str.strip()
            lookup = lookup.dropna(subset=["Node_ID"]).drop_duplicates(subset=["Node_ID"], keep="first")
            lookup["Latitude"] = pd.to_numeric(lookup["Latitude"], errors="coerce")
            lookup["Longitude"] = pd.to_numeric(lookup["Longitude"], errors="coerce")

            if not all(column in merged.columns for column in COORDINATE_COLUMNS):
                merged = merged.merge(lookup, on="Node_ID", how="left")
            else:
                merged = merged.merge(lookup, on="Node_ID", how="left", suffixes=("", "_lookup"))
                for column in COORDINATE_COLUMNS:
                    lookup_column = f"{column}_lookup"
                    if lookup_column in merged.columns:
                        merged[column] = merged[column].fillna(merged[lookup_column])
                        merged = merged.drop(columns=[lookup_column])

    merged = _coerce_and_validate_coordinates(merged)
    return merged, detect_coordinate_status(merged)


def detect_coordinate_status(dataframe: pd.DataFrame) -> CoordinateStatus:
    """Inspect whether a dataframe has usable latitude and longitude values."""

    standardized = standardize_coordinate_columns(dataframe)
    if not all(column in standardized.columns for column in COORDINATE_COLUMNS):
        return CoordinateStatus(
            has_coordinates=False,
            rows_with_coordinates=0,
            rows_missing_coordinates=len(standardized),
            message="No Latitude/Longitude columns were found. Map visualization will be skipped.",
        )

    coordinate_values = standardized[list(COORDINATE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    valid_mask = coordinate_values.notna().all(axis=1)
    rows_with_coordinates = int(valid_mask.sum())
    rows_missing_coordinates = int((~valid_mask).sum())

    if rows_with_coordinates == 0:
        message = "Latitude/Longitude columns exist, but no rows have complete usable coordinates."
    elif rows_missing_coordinates:
        message = f"Map will include {rows_with_coordinates} node(s); {rows_missing_coordinates} node(s) are missing coordinates."
    else:
        message = f"Map is ready for {rows_with_coordinates} node(s)."

    return CoordinateStatus(
        has_coordinates=rows_with_coordinates > 0,
        rows_with_coordinates=rows_with_coordinates,
        rows_missing_coordinates=rows_missing_coordinates,
        message=message,
    )


def load_pjm_zone_geojson(source: GeoJsonSource) -> PjmZoneGeoJson:
    """Load PJM zone polygons and normalize a zone-name property for joining."""

    try:
        if isinstance(source, (str, Path)):
            with Path(source).open("r", encoding="utf-8-sig") as file:
                geojson = json.load(file)
        else:
            if hasattr(source, "seek"):
                source.seek(0)
            raw = source.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8-sig")
            geojson = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("The PJM zones GeoJSON could not be parsed.") from exc
    except OSError as exc:
        raise ValueError(f"The PJM zones GeoJSON could not be loaded: {exc}") from exc

    if geojson.get("type") != "FeatureCollection" or not isinstance(geojson.get("features"), list):
        raise ValueError("The PJM zones file must be a GeoJSON FeatureCollection.")

    features = geojson["features"]
    if not features:
        raise ValueError("The PJM zones GeoJSON does not contain any features.")

    zone_property = infer_geojson_zone_property(geojson)
    if zone_property is None:
        raise ValueError("No usable zone-name property was found in the PJM zones GeoJSON.")

    zones: list[str] = []
    for feature in features:
        properties = feature.setdefault("properties", {})
        raw_zone = properties.get(zone_property)
        normalized_zone = normalize_zone_name(raw_zone)
        properties["Zone"] = str(raw_zone).strip() if raw_zone is not None and str(raw_zone).strip() else normalized_zone
        properties["Zone_Normalized"] = normalized_zone
        if normalized_zone:
            zones.append(normalized_zone)

    unique_zones = sorted(set(zones))
    if not unique_zones:
        raise ValueError(f"The GeoJSON property '{zone_property}' did not contain usable zone names.")

    return PjmZoneGeoJson(
        geojson=geojson,
        zone_property=zone_property,
        zone_count=len(unique_zones),
        zones=unique_zones,
    )


def infer_geojson_zone_property(geojson: GeoJson, flexworks_zones: list[str] | None = None) -> str | None:
    """Infer the best GeoJSON property to use as the PJM zone name."""

    features = geojson.get("features", [])
    property_keys = sorted({key for feature in features for key in feature.get("properties", {}).keys()})
    if not property_keys:
        return None

    normalized_flexworks_zones = {normalize_zone_name(zone) for zone in flexworks_zones or [] if normalize_zone_name(zone)}
    candidates = _ordered_geojson_zone_candidates(property_keys)
    if not candidates:
        return None

    if normalized_flexworks_zones:
        scored: list[tuple[int, int, str]] = []
        for preference_index, candidate in enumerate(candidates):
            candidate_zones = {
                normalize_zone_name(feature.get("properties", {}).get(candidate))
                for feature in features
                if feature.get("properties", {}).get(candidate) is not None
            }
            matched_count = len(candidate_zones.intersection(normalized_flexworks_zones))
            scored.append((matched_count, -preference_index, candidate))
        best = max(scored)
        if best[0] > 0:
            return best[2]

    return candidates[0]


def prepare_pjm_zone_choropleth_data(
    dataframe: pd.DataFrame,
    pjm_geojson: PjmZoneGeoJson | None,
    metric_column: str,
) -> ZoneJoinResult:
    """Aggregate FlexWorks results by PJM zone and join them to polygon zones."""

    if pjm_geojson is None:
        diagnostics = ZoneJoinDiagnostics(
            geojson_zone_count=0,
            flexworks_zone_count=0,
            matched_zone_count=0,
            unmatched_flexworks_zones=[],
            unmatched_geojson_zones=[],
            zone_property=None,
            data_zone_column=None,
            is_available=False,
            message="No PJM zones GeoJSON is available. Falling back to point map.",
        )
        return ZoneJoinResult(pd.DataFrame(), None, diagnostics)

    if dataframe.empty:
        diagnostics = ZoneJoinDiagnostics(
            geojson_zone_count=pjm_geojson.zone_count,
            flexworks_zone_count=0,
            matched_zone_count=0,
            unmatched_flexworks_zones=[],
            unmatched_geojson_zones=pjm_geojson.zones,
            zone_property=pjm_geojson.zone_property,
            data_zone_column=None,
            is_available=False,
            message="No FlexWorks rows are available for a zone choropleth.",
        )
        return ZoneJoinResult(pd.DataFrame(), pjm_geojson.geojson, diagnostics)

    data_zone_column = find_best_zone_column(dataframe)
    if data_zone_column is None:
        diagnostics = ZoneJoinDiagnostics(
            geojson_zone_count=pjm_geojson.zone_count,
            flexworks_zone_count=0,
            matched_zone_count=0,
            unmatched_flexworks_zones=[],
            unmatched_geojson_zones=pjm_geojson.zones,
            zone_property=pjm_geojson.zone_property,
            data_zone_column=None,
            is_available=False,
            message="No usable zone column was found in the FlexWorks results. Falling back to point map.",
        )
        return ZoneJoinResult(pd.DataFrame(), pjm_geojson.geojson, diagnostics)

    zone_data = dataframe.copy()
    zone_data["Zone_Normalized"] = zone_data[data_zone_column].apply(normalize_zone_name)
    zone_data = zone_data.loc[zone_data["Zone_Normalized"].notna() & (zone_data["Zone_Normalized"] != "")].copy()
    if zone_data.empty:
        diagnostics = ZoneJoinDiagnostics(
            geojson_zone_count=pjm_geojson.zone_count,
            flexworks_zone_count=0,
            matched_zone_count=0,
            unmatched_flexworks_zones=[],
            unmatched_geojson_zones=pjm_geojson.zones,
            zone_property=pjm_geojson.zone_property,
            data_zone_column=data_zone_column,
            is_available=False,
            message=f"The FlexWorks column '{data_zone_column}' did not contain usable PJM zone names.",
        )
        return ZoneJoinResult(pd.DataFrame(), pjm_geojson.geojson, diagnostics)

    aggregated = _aggregate_zone_results(zone_data, data_zone_column)
    if metric_column not in aggregated.columns:
        diagnostics = _build_zone_join_diagnostics(
            aggregated=aggregated,
            pjm_geojson=pjm_geojson,
            data_zone_column=data_zone_column,
            is_available=False,
            message=f"The selected metric '{metric_column}' is not available for the PJM zone choropleth.",
        )
        return ZoneJoinResult(aggregated, pjm_geojson.geojson, diagnostics)

    aggregated[metric_column] = pd.to_numeric(aggregated[metric_column], errors="coerce")
    diagnostics = _build_zone_join_diagnostics(
        aggregated=aggregated,
        pjm_geojson=pjm_geojson,
        data_zone_column=data_zone_column,
        is_available=not aggregated.loc[aggregated["Zone_Normalized"].isin(pjm_geojson.zones), metric_column].dropna().empty,
        message="PJM zone choropleth is ready.",
    )
    if not diagnostics.is_available:
        diagnostics = ZoneJoinDiagnostics(
            geojson_zone_count=diagnostics.geojson_zone_count,
            flexworks_zone_count=diagnostics.flexworks_zone_count,
            matched_zone_count=diagnostics.matched_zone_count,
            unmatched_flexworks_zones=diagnostics.unmatched_flexworks_zones,
            unmatched_geojson_zones=diagnostics.unmatched_geojson_zones,
            zone_property=diagnostics.zone_property,
            data_zone_column=diagnostics.data_zone_column,
            is_available=False,
            message="No matched PJM zones have numeric values for the selected metric. Falling back to point map.",
        )

    return ZoneJoinResult(aggregated, pjm_geojson.geojson, diagnostics)


def find_best_zone_column(dataframe: pd.DataFrame) -> str | None:
    """Find the best available column containing zone names."""

    for candidate in ZONE_COLUMN_CANDIDATES:
        if candidate in dataframe.columns and _has_normalizable_zones(dataframe[candidate]):
            return candidate

    for column in dataframe.columns:
        column_text = str(column).lower()
        if "zone" in column_text and _has_normalizable_zones(dataframe[column]):
            return str(column)
    return None


def normalize_zone_name(value: object) -> str:
    """Normalize PJM zone names for resilient joins."""

    if value is None or pd.isna(value):
        return ""

    text = str(value).upper().strip()
    if not text:
        return ""

    parenthetical = re.match(r"^(.*?)\s*\((.*?)\)\s*$", text)
    if parenthetical:
        text = parenthetical.group(1).strip()

    text = re.sub(r"\b(PJM|ZONE|ZONES|AREA|PLANNING|TRANSMISSION|UTILITY|UTILITIES|COMPANY|CO|INC|LLC)\b", " ", text)
    text = text.replace("&", "AND")
    compact = re.sub(r"[^A-Z0-9]", "", text)

    aliases = {
        "BGANDE": "BGE",
        "BALTIMOREGASELECTRIC": "BGE",
        "DELMARVAPOWER": "DPL",
        "DUQUESNE": "DUQ",
        "DUQUESNELIGHT": "DUQ",
        "JERSEYCENTRALPOWERLIGHT": "JCPL",
        "JCPANDL": "JCPL",
        "ATLANTICCITYELECTRIC": "AECO",
        "PENNELEC": "PENELEC",
        "METED": "METED",
        "METROPOLITANEDISON": "METED",
        "PPL": "PPL",
        "PSEG": "PSEG",
        "PUBLICSERVICEELECTRICGAS": "PSEG",
        "PEPCO": "PEPCO",
        "POTOMACELECTRICPOWER": "PEPCO",
        "DOMINION": "DOM",
        "DOMINIONENERGY": "DOM",
        "COMED": "COMED",
        "COMMONWEALTHEDISON": "COMED",
        "DAYTON": "DAY",
        "DAYTONPOWERLIGHT": "DAY",
        "AEP": "AEP",
        "AMERICANELECTRICPOWER": "AEP",
        "APS": "APS",
        "ALLEGHENYPOWERSYSTEM": "APS",
    }
    return aliases.get(compact, compact)


def _coerce_and_validate_coordinates(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = standardize_coordinate_columns(dataframe)
    if not all(column in cleaned.columns for column in COORDINATE_COLUMNS):
        return cleaned

    cleaned["Latitude"] = pd.to_numeric(cleaned["Latitude"], errors="coerce")
    cleaned["Longitude"] = pd.to_numeric(cleaned["Longitude"], errors="coerce")

    invalid_latitude = ~cleaned["Latitude"].between(-90, 90) & cleaned["Latitude"].notna()
    invalid_longitude = ~cleaned["Longitude"].between(-180, 180) & cleaned["Longitude"].notna()
    cleaned.loc[invalid_latitude, "Latitude"] = pd.NA
    cleaned.loc[invalid_longitude, "Longitude"] = pd.NA
    return cleaned


def _find_first_column(dataframe: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in dataframe.columns:
            return alias
    return None


def _ordered_geojson_zone_candidates(property_keys: list[str]) -> list[str]:
    preferred = [candidate for candidate in GEOJSON_ZONE_PROPERTY_PREFERENCE if candidate in property_keys]
    remaining = [
        key
        for key in property_keys
        if key not in preferred and ("zone" in key.lower() or key.lower() in {"name", "zone_name"})
    ]
    return preferred + remaining


def _has_normalizable_zones(series: pd.Series) -> bool:
    normalized = series.dropna().apply(normalize_zone_name)
    return bool((normalized != "").any())


def _aggregate_zone_results(dataframe: pd.DataFrame, data_zone_column: str) -> pd.DataFrame:
    aggregation: dict[str, object] = {
        data_zone_column: _first_non_null,
    }
    for column in ("Zone", "ISO_Region", "Risk_Label"):
        if column in dataframe.columns and column != data_zone_column:
            aggregation[column] = _risk_rollup if column == "Risk_Label" else _first_non_null

    for column in (
        "Annualized_Revenue",
        "Revenue_per_kW",
        "Opportunity_Score",
        "Risk_Adjusted_Score",
        "LMP_Volatility",
    ):
        if column in dataframe.columns:
            aggregation[column] = "mean"

    if "Node_ID" in dataframe.columns:
        aggregation["Node_ID"] = "nunique"

    grouped = dataframe.groupby("Zone_Normalized", as_index=False, dropna=False).agg(aggregation)
    grouped = grouped.rename(columns={data_zone_column: "Zone_Display", "Node_ID": "Node_Count"})
    if "Zone" in grouped.columns and "Zone_Display" not in grouped.columns:
        grouped = grouped.rename(columns={"Zone": "Zone_Display"})
    grouped["Zone"] = grouped["Zone_Display"].fillna(grouped["Zone_Normalized"])
    return grouped


def _build_zone_join_diagnostics(
    *,
    aggregated: pd.DataFrame,
    pjm_geojson: PjmZoneGeoJson,
    data_zone_column: str,
    is_available: bool,
    message: str,
) -> ZoneJoinDiagnostics:
    flexworks_zones = sorted(set(aggregated["Zone_Normalized"].dropna().astype(str)))
    geojson_zones = set(pjm_geojson.zones)
    matched_zones = sorted(set(flexworks_zones).intersection(geojson_zones))
    unmatched_flexworks_zones = sorted(set(flexworks_zones).difference(geojson_zones))
    unmatched_geojson_zones = sorted(geojson_zones.difference(flexworks_zones))

    if not matched_zones:
        is_available = False
        message = "No FlexWorks zones matched the PJM GeoJSON. Falling back to point map."

    return ZoneJoinDiagnostics(
        geojson_zone_count=pjm_geojson.zone_count,
        flexworks_zone_count=len(flexworks_zones),
        matched_zone_count=len(matched_zones),
        unmatched_flexworks_zones=unmatched_flexworks_zones,
        unmatched_geojson_zones=unmatched_geojson_zones,
        zone_property=pjm_geojson.zone_property,
        data_zone_column=data_zone_column,
        is_available=is_available,
        message=message,
    )


def _first_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def _risk_rollup(series: pd.Series) -> str:
    risk_order = {"high volatility": 3, "moderate": 2, "stable": 1, "unknown": 0}
    values = [str(value) for value in series.dropna()]
    if not values:
        return "unknown"
    return max(values, key=lambda value: risk_order.get(value, 0))
