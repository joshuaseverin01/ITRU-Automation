#!/usr/bin/env python3
"""Python recreation of the PJM map + cumulative revenue bar GIF."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from heapq import heappop, heappush
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_pjm_graphic"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath


STEPS_BETWEEN = 8
DESIRED_SECONDS = 15

WIDTH = 1658
HEIGHT = 826
SCALE = 1
DPI = 100

BASE_DIR = Path(__file__).resolve().parent
PJM_GEOJSON_PATH = BASE_DIR / "pjm_zones.geojson"
NEW_DATA_DIR = BASE_DIR / "new data"
PORTFOLIO_2022_PATH = NEW_DATA_DIR / "portfolio_data (38).csv"
PORTFOLIO_2023_2025_PATH = NEW_DATA_DIR / "portfolio_data (37).csv"
PJM_DEVICES_MD_PATH = NEW_DATA_DIR / "pjm devices.md"
PJM_EXCEL_PATH = BASE_DIR / "PJM Run Dec 16.xlsx"
GIF_FILE = BASE_DIR / "pjm_map_plus_bars_python.gif"

MAIN_TITLE = "PJM Battery Revenue — 10 kW Battery"
PALETTE = ["#EAF7E6", "#CFEEC9", "#A6D96A", "#31A354", "#006D2C"]
RELATIVE_BRIGHTNESS_GAMMA = 1.0
ABSOLUTE_CONTRAST_GAMMA = 0.8
ABSOLUTE_CONTRAST_FLOOR = 0.62
LOW_CONTRAST_BASE = "#ECF6E7"

SIZE_BASE = 16
SIZE_ZONE_SF_MM = 4.4
SIZE_VALUE_LABEL_MM = 5.2
SIZE_ZONE_UNDER = 8.5
TITLE_SIZE = 28
SUBTITLE_SIZE = 15
HEADER_HEIGHT_PT = TITLE_SIZE + SUBTITLE_SIZE + 18
MAP_LINEWIDTH_MM = 0.4
BAR_HEIGHT = 0.65
BAR_LABEL_OFFSET_RATIO = 0.006
MAP_PADDING_RATIO = 0.03

PANEL_WIDTH_RATIOS = [0.05, 5, 2, 0.05]

LABEL_NUDGES = {
    "BGE": (-0.06, -0.03),
    "DPL": (0.12, -0.03),
    "JCPL": (-0.08, -0.02),
    "RECO": (0.14, 0.09),
    "METED": (-0.02, -0.03),
    "PECO": (0.04, -0.05),
    "PSEG": (0.05, -0.08),
    "AECO": (0.09, -0.03),
}

ZONE_LABEL_SIZE_OVERRIDES_MM = {
    "RECO": 3.6,
    "JCPL": 3.7,
    "PSEG": 3.7,
    "PECO": 3.8,
    "DPL": 3.8,
    "AECO": 3.8,
    "BGE": 3.9,
    "METED": 3.9,
}

LEGACY_ZONE_ORDER = [
    "BGE",
    "DPL",
    "DUQ",
    "JCPL",
    "RECO",
    "COMED",
    "DAY",
    "PENELEC",
    "METED",
    "PPL",
    "AECO",
    "PECO",
    "PSEG",
    "PEPCO",
    "DOM",
    "APS",
    "ATSI",
    "DEOK",
    "AEP",
    "EKPC",
    "OVEC",
]


@dataclass(frozen=True)
class ZoneGeometry:
    zone_name: str
    rings: tuple[np.ndarray, ...]
    path: MplPath
    label_point: tuple[float, float]


@dataclass(order=True)
class QueueCell:
    sort_index: float
    x: float
    y: float
    h: float
    distance: float
    max_distance: float


@dataclass
class ArtistState:
    canvas: FigureCanvasAgg
    subtitle_text: Any
    map_patches: dict[str, PathPatch]
    bar_patches: dict[str, Rectangle]
    value_texts: dict[str, Any]


def mm_to_pt(value_mm: float) -> float:
    return value_mm * 72.0 / 25.4


def choose_fonts() -> tuple[str, str, str]:
    families = {font.name for font in font_manager.fontManager.ttflist}

    base_font = "Helvetica" if "Helvetica" in families else "Arial" if "Arial" in families else "sans-serif"
    title_font = base_font
    zone_font = "Arial Black" if "Arial Black" in families else base_font

    return base_font, title_font, zone_font


def format_dollar(value: float) -> str:
    rounded = int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return f"${rounded:,.0f}"


def point_in_ring(point: tuple[float, float], ring: np.ndarray) -> bool:
    px, py = point
    inside = False

    for idx in range(len(ring) - 1):
        x1, y1 = ring[idx]
        x2, y2 = ring[idx + 1]

        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / ((y2 - y1) or 1e-12) + x1):
            inside = not inside

    return inside


def point_in_polygon(point: tuple[float, float], rings: tuple[np.ndarray, ...]) -> bool:
    if not point_in_ring(point, rings[0]):
        return False

    for hole in rings[1:]:
        if point_in_ring(point, hole):
            return False

    return True


def point_segment_distance_sq(point: tuple[float, float], a: np.ndarray, b: np.ndarray) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay

    if dx == 0 and dy == 0:
        return (px - ax) ** 2 + (py - ay) ** 2

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))

    proj_x = ax + t * dx
    proj_y = ay + t * dy

    return (px - proj_x) ** 2 + (py - proj_y) ** 2


def signed_distance(point: tuple[float, float], rings: tuple[np.ndarray, ...]) -> float:
    min_dist_sq = float("inf")

    for ring in rings:
        for idx in range(len(ring) - 1):
            min_dist_sq = min(
                min_dist_sq,
                point_segment_distance_sq(point, ring[idx], ring[idx + 1]),
            )

    distance = math.sqrt(min_dist_sq)
    return distance if point_in_polygon(point, rings) else -distance


def polygon_centroid(ring: np.ndarray) -> tuple[float, float]:
    area_twice = 0.0
    cx = 0.0
    cy = 0.0

    for idx in range(len(ring) - 1):
        x1, y1 = ring[idx]
        x2, y2 = ring[idx + 1]
        cross = x1 * y2 - x2 * y1
        area_twice += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross

    if abs(area_twice) < 1e-12:
        return float(np.mean(ring[:, 0])), float(np.mean(ring[:, 1]))

    factor = 1.0 / (3.0 * area_twice)
    return cx * factor, cy * factor


def make_cell(x: float, y: float, h: float, rings: tuple[np.ndarray, ...]) -> QueueCell:
    distance = signed_distance((x, y), rings)
    max_distance = distance + h * math.sqrt(2)
    return QueueCell(-max_distance, x, y, h, distance, max_distance)


def polylabel(rings: tuple[np.ndarray, ...], precision: float = 1e-4) -> tuple[float, float]:
    outer = rings[0]
    min_x = float(np.min(outer[:, 0]))
    min_y = float(np.min(outer[:, 1]))
    max_x = float(np.max(outer[:, 0]))
    max_y = float(np.max(outer[:, 1]))

    width = max_x - min_x
    height = max_y - min_y
    cell_size = min(width, height)

    if cell_size == 0:
        return float(outer[0, 0]), float(outer[0, 1])

    h = cell_size / 2.0
    queue: list[QueueCell] = []

    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            heappush(queue, make_cell(x + h, y + h, h, rings))
            y += cell_size
        x += cell_size

    centroid = polygon_centroid(outer)
    best_cell = make_cell(centroid[0], centroid[1], 0.0, rings)
    bbox_cell = make_cell((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, 0.0, rings)

    if bbox_cell.distance > best_cell.distance:
        best_cell = bbox_cell

    while queue:
        cell = heappop(queue)

        if cell.distance > best_cell.distance:
            best_cell = cell

        if cell.max_distance - best_cell.distance <= precision:
            continue

        next_h = cell.h / 2.0

        heappush(queue, make_cell(cell.x - next_h, cell.y - next_h, next_h, rings))
        heappush(queue, make_cell(cell.x + next_h, cell.y - next_h, next_h, rings))
        heappush(queue, make_cell(cell.x - next_h, cell.y + next_h, next_h, rings))
        heappush(queue, make_cell(cell.x + next_h, cell.y + next_h, next_h, rings))

    return best_cell.x, best_cell.y


def build_polygon_path(rings: tuple[np.ndarray, ...]) -> MplPath:
    vertices: list[list[float]] = []
    codes: list[int] = []

    for ring in rings:
        coords = ring
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        for idx, (x, y) in enumerate(coords):
            vertices.append([x, y])
            if idx == 0:
                codes.append(MplPath.MOVETO)
            elif idx == len(coords) - 1:
                codes.append(MplPath.CLOSEPOLY)
            else:
                codes.append(MplPath.LINETO)

    return MplPath(np.asarray(vertices, dtype=float), codes)


def load_zone_geometries(path: Path) -> tuple[list[ZoneGeometry], tuple[float, float, float, float]]:
    geojson = json.loads(path.read_text())

    zones: list[ZoneGeometry] = []
    all_x: list[float] = []
    all_y: list[float] = []

    for feature in geojson["features"]:
        zone_name = str(feature["properties"]["zoneName"])
        rings = tuple(np.asarray(ring, dtype=float) for ring in feature["geometry"]["coordinates"])
        path_obj = build_polygon_path(rings)
        label_point = polylabel(rings)

        for ring in rings:
            all_x.extend(ring[:, 0].tolist())
            all_y.extend(ring[:, 1].tolist())

        zones.append(
            ZoneGeometry(
                zone_name=zone_name,
                rings=rings,
                path=path_obj,
                label_point=label_point,
            )
        )

    bbox = (min(all_x), max(all_x), min(all_y), max(all_y))
    return zones, bbox


def extract_zone_order_from_markdown(markdown_path: Path) -> list[str]:
    if not markdown_path.exists():
        return []

    text = markdown_path.read_text()
    return re.findall(r"Location:\s*([A-Z]+)\s+\(PJM\)", text)


def resolve_zone_order() -> list[str]:
    zones_from_markdown = extract_zone_order_from_markdown(PJM_DEVICES_MD_PATH)
    expected_set = sorted(LEGACY_ZONE_ORDER)

    if len(zones_from_markdown) == len(LEGACY_ZONE_ORDER):
        if sorted(set(zones_from_markdown)) == expected_set:
            return zones_from_markdown

        missing = [zone for zone in LEGACY_ZONE_ORDER if zone not in zones_from_markdown]
        seen: set[str] = set()
        repaired: list[str] = []

        for zone_name in zones_from_markdown:
            if zone_name in seen and missing:
                repaired.append(missing.pop(0))
            else:
                repaired.append(zone_name)
                seen.add(zone_name)

        if sorted(set(repaired)) == expected_set:
            return repaired

    return LEGACY_ZONE_ORDER.copy()


def load_portfolio_csv(csv_path: Path, zone_order: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    block_size = 4
    expected_rows = len(zone_order) * block_size

    if len(raw) < expected_rows:
        raise ValueError(f"Expected at least {expected_rows} rows in {csv_path.name}, found {len(raw)}.")

    month_headers = raw.columns[2:].tolist()
    months = pd.to_datetime([f"{label}-01" for label in month_headers], format="%Y-%m-%d")

    records: list[dict[str, Any]] = []

    for block_idx, zone_name in enumerate(zone_order):
        block = raw.iloc[block_idx * block_size : (block_idx + 1) * block_size].reset_index(drop=True)
        revenue_components: dict[str, np.ndarray] = {}

        for _, row in block.iterrows():
            revenue_type = str(row.iloc[1]).strip()
            values = pd.to_numeric(row.iloc[2:], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            revenue_components[revenue_type] = values

        if "Energy" not in revenue_components:
            raise ValueError(f"Missing Energy row for zone block {block_idx + 1} in {csv_path.name}.")

        total_values = revenue_components["Energy"] + revenue_components.get(
            "Capacity",
            np.zeros(len(months), dtype=float),
        )

        for month, revenue in zip(months, total_values):
            records.append(
                {
                    "zoneName": zone_name,
                    "Month": month,
                    "Revenue": float(revenue),
                }
            )

    return pd.DataFrame.from_records(records)


def load_revenue_data_from_new_files() -> pd.DataFrame:
    zone_order = resolve_zone_order()
    frames = [
        load_portfolio_csv(PORTFOLIO_2022_PATH, zone_order),
        load_portfolio_csv(PORTFOLIO_2023_2025_PATH, zone_order),
    ]
    tidy = pd.concat(frames, ignore_index=True)
    tidy = (
        tidy.groupby(["zoneName", "Month"], as_index=False)["Revenue"]
        .sum()
        .sort_values(["zoneName", "Month"])
        .reset_index(drop=True)
    )
    return tidy


def load_revenue_data_from_legacy_excel(excel_path: Path) -> pd.DataFrame:
    devices = pd.read_excel(excel_path, sheet_name="devices")
    devices = devices.loc[:, ["Device", "Location"]].copy()
    devices["zoneName"] = (
        devices["Location"]
        .astype(str)
        .str.replace(" (PJM)", "", regex=False)
        .str.strip()
    )

    raw = pd.read_excel(excel_path, sheet_name="portfolio_data (29)", header=None)
    month_headers = raw.iloc[0, 2:].tolist()
    months = pd.to_datetime([f"{label}-01" for label in month_headers], format="%Y-%m-%d")

    records: list[dict[str, Any]] = []

    for row_idx in range(1, len(raw), 4):
        device = raw.iat[row_idx, 0]
        revenue_type = raw.iat[row_idx, 1]

        if pd.isna(device) or revenue_type != "Energy":
            continue

        values = pd.to_numeric(raw.iloc[row_idx, 2:], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        for month, revenue in zip(months, values):
            records.append(
                {
                    "Device": str(device),
                    "Month": month,
                    "Revenue": float(revenue),
                }
            )

    tidy = pd.DataFrame.from_records(records)
    tidy = tidy.merge(devices[["Device", "zoneName"]], on="Device", how="left")

    if tidy["zoneName"].isna().any():
        missing_devices = tidy.loc[tidy["zoneName"].isna(), "Device"].unique().tolist()
        raise ValueError(f"Device-to-zone mapping failed for: {missing_devices}")

    return tidy.loc[:, ["zoneName", "Month", "Revenue"]].sort_values(["zoneName", "Month"]).reset_index(drop=True)


def load_revenue_data() -> pd.DataFrame:
    if PORTFOLIO_2022_PATH.exists() and PORTFOLIO_2023_2025_PATH.exists():
        return load_revenue_data_from_new_files()

    return load_revenue_data_from_legacy_excel(PJM_EXCEL_PATH)


def build_color_scale() -> tuple[np.ndarray, np.ndarray]:
    positions = np.linspace(0.0, 1.0, num=len(PALETTE))
    colors = np.asarray([to_rgb(color) for color in PALETTE], dtype=float)
    return positions, colors


def nudge_label(label_point: tuple[float, float], zone_name: str) -> tuple[float, float]:
    offset_x, offset_y = LABEL_NUDGES.get(zone_name, (0.0, 0.0))
    return label_point[0] + offset_x, label_point[1] + offset_y


def zone_label_size_mm(zone_name: str) -> float:
    return ZONE_LABEL_SIZE_OVERRIDES_MM.get(zone_name, SIZE_ZONE_SF_MM)


def frame_relative_positions(values_by_zone: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values_by_zone.items(), key=lambda item: item[1])
    if not ordered:
        return {}

    if len(ordered) == 1:
        zone_name, _ = ordered[0]
        return {zone_name: 0.5}

    relative_positions: dict[str, float] = {}
    denominator = len(ordered) - 1

    for idx, (zone_name, _) in enumerate(ordered):
        relative_positions[zone_name] = idx / denominator

    return relative_positions


def color_for_value(
    value: float,
    relative_normalized: float,
    y_max_global: float,
    positions: np.ndarray,
    colors: np.ndarray,
) -> tuple[float, float, float]:
    if y_max_global <= 0:
        return tuple(colors[0])

    relative_normalized = relative_normalized ** RELATIVE_BRIGHTNESS_GAMMA

    base_red = float(np.interp(relative_normalized, positions, colors[:, 0]))
    base_green = float(np.interp(relative_normalized, positions, colors[:, 1]))
    base_blue = float(np.interp(relative_normalized, positions, colors[:, 2]))

    absolute_normalized = float(np.clip(value / y_max_global, 0.0, 1.0))
    absolute_normalized = absolute_normalized ** ABSOLUTE_CONTRAST_GAMMA
    contrast = ABSOLUTE_CONTRAST_FLOOR + ((1.0 - ABSOLUTE_CONTRAST_FLOOR) * absolute_normalized)

    low_contrast_base = np.asarray(to_rgb(LOW_CONTRAST_BASE), dtype=float)
    red = float((low_contrast_base[0] * (1.0 - contrast)) + (base_red * contrast))
    green = float((low_contrast_base[1] * (1.0 - contrast)) + (base_green * contrast))
    blue = float((low_contrast_base[2] * (1.0 - contrast)) + (base_blue * contrast))
    return red, green, blue


def build_smoothed_series(cum_long: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, np.ndarray], list[pd.Timestamp]]:
    months = pd.DatetimeIndex(sorted(cum_long["Month"].unique()))
    if len(months) < 2:
        raise ValueError("Not enough months to animate.")

    frame_count = (len(months) - 1) * STEPS_BETWEEN + 1
    fine_idx = np.linspace(1.0, float(len(months)), num=frame_count)
    month_idx = np.arange(1.0, len(months) + 1.0, dtype=float)

    by_zone: dict[str, np.ndarray] = {}
    for zone_name, group in cum_long.groupby("zoneName", sort=False):
        values = group.sort_values("Month")["CumRevenue"].to_numpy(dtype=float)
        by_zone[zone_name] = np.interp(fine_idx, month_idx, values)

    month_labels: list[pd.Timestamp] = []
    for value in fine_idx:
        month_idx_label = max(1, min(len(months), int(math.floor(value))))
        month_labels.append(months[month_idx_label - 1])

    return months, fine_idx, by_zone, month_labels


def build_frame_durations(frame_count: int) -> list[int]:
    # GIF delays are centisecond-based, so distribute 5/6 cs frames to hit the target total duration.
    total_centiseconds = int(round(DESIRED_SECONDS * 100.0))
    base_centiseconds = total_centiseconds // frame_count
    extra_centiseconds = total_centiseconds - (base_centiseconds * frame_count)

    durations_ms: list[int] = []
    carried = 0

    for _ in range(frame_count):
        centiseconds = base_centiseconds
        carried += extra_centiseconds
        if carried >= frame_count:
            centiseconds += 1
            carried -= frame_count
        durations_ms.append(centiseconds * 10)

    return durations_ms


def create_artist_state(
    zones: list[ZoneGeometry],
    bbox: tuple[float, float, float, float],
    final_order: list[str],
    initial_values: dict[str, float],
    title_font: str,
    subtitle_font: str,
    base_font: str,
    zone_font: str,
    y_max_global: float,
    positions: np.ndarray,
    colors: np.ndarray,
    subtitle: str,
) -> ArtistState:
    figure_width = int(WIDTH * SCALE)
    figure_height = int(HEIGHT * SCALE)
    figure = Figure(
        figsize=(
            np.nextafter(figure_width / DPI, math.inf),
            np.nextafter(figure_height / DPI, math.inf),
        ),
        dpi=DPI,
        facecolor="white",
    )
    canvas = FigureCanvasAgg(figure)

    total_height_pt = figure_height / DPI * 72.0
    body_height_pt = max(total_height_pt - HEADER_HEIGHT_PT, 1.0)

    grid_spec = figure.add_gridspec(
        2,
        4,
        height_ratios=[HEADER_HEIGHT_PT, body_height_pt],
        width_ratios=PANEL_WIDTH_RATIOS,
        hspace=0.0,
        wspace=0.12,
    )

    header_ax = figure.add_subplot(grid_spec[0, :])
    map_ax = figure.add_subplot(grid_spec[1, 1])
    bar_ax = figure.add_subplot(grid_spec[1, 2])

    figure.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)

    header_ax.axis("off")
    header_ax.text(
        0.5,
        0.72,
        MAIN_TITLE,
        ha="center",
        va="center",
        fontsize=TITLE_SIZE,
        fontfamily=title_font,
        fontweight="normal",
        color="black",
    )
    subtitle_text = header_ax.text(
        0.5,
        0.18,
        subtitle,
        ha="center",
        va="center",
        fontsize=SUBTITLE_SIZE,
        fontfamily=subtitle_font,
        fontweight="semibold",
        color="black",
    )

    xmin, xmax, ymin, ymax = bbox
    xpad = (xmax - xmin) * MAP_PADDING_RATIO
    ypad = (ymax - ymin) * MAP_PADDING_RATIO
    map_ax.set_xlim(xmin - xpad, xmax + xpad)
    map_ax.set_ylim(ymin - ypad, ymax + ypad)
    map_ax.set_aspect("equal")
    map_ax.axis("off")

    linewidth_pt = mm_to_pt(MAP_LINEWIDTH_MM)
    map_patches: dict[str, PathPatch] = {}
    relative_positions_by_zone = frame_relative_positions(initial_values)

    for zone in zones:
        facecolor = color_for_value(
            initial_values[zone.zone_name],
            relative_positions_by_zone[zone.zone_name],
            y_max_global,
            positions,
            colors,
        )
        patch = PathPatch(
            zone.path,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=linewidth_pt,
            antialiased=True,
        )
        if hasattr(patch, "set_fillrule"):
            patch.set_fillrule("evenodd")
        map_ax.add_patch(patch)
        map_patches[zone.zone_name] = patch

    for zone in zones:
        label_x, label_y = nudge_label(zone.label_point, zone.zone_name)
        label = map_ax.text(
            label_x,
            label_y,
            zone.zone_name,
            ha="center",
            va="center",
            fontsize=mm_to_pt(zone_label_size_mm(zone.zone_name)),
            fontfamily=zone_font,
            fontweight="heavy",
            color="black",
            clip_on=False,
        )
        label.set_path_effects(
            [
                path_effects.Stroke(linewidth=1.2, foreground="white"),
                path_effects.Normal(),
            ]
        )

    y_positions = np.arange(len(final_order), dtype=float)
    bar_values = [initial_values[zone_name] for zone_name in final_order]
    bar_colors = [
        color_for_value(
            value,
            relative_positions_by_zone[zone_name],
            y_max_global,
            positions,
            colors,
        )
        for zone_name, value in zip(final_order, bar_values)
    ]

    bars = bar_ax.barh(y_positions, bar_values, height=BAR_HEIGHT, color=bar_colors, edgecolor="none")
    bar_ax.set_xlim(0.0, 1.08 * y_max_global)
    bar_ax.set_ylim(-0.5, len(final_order) - 0.5)
    bar_ax.set_yticks(y_positions)
    bar_ax.set_yticklabels(final_order, fontfamily=base_font, fontsize=SIZE_ZONE_UNDER, fontweight="semibold", color="black")
    bar_ax.set_xticks([])
    bar_ax.tick_params(axis="y", length=0, colors="black")
    bar_ax.tick_params(axis="x", length=0, labelbottom=False)
    bar_ax.set_facecolor("white")
    bar_ax.grid(False)

    for spine in bar_ax.spines.values():
        spine.set_visible(False)

    value_texts: dict[str, Any] = {}
    label_offset = BAR_LABEL_OFFSET_RATIO * y_max_global

    for zone_name, bar in zip(final_order, bars):
        value = initial_values[zone_name]
        text = bar_ax.text(
            value + label_offset,
            bar.get_y() + (bar.get_height() / 2.0),
            format_dollar(value),
            ha="left",
            va="center",
            fontsize=mm_to_pt(SIZE_VALUE_LABEL_MM),
            fontfamily=base_font,
            fontweight="semibold",
            color="black",
            clip_on=False,
        )
        value_texts[zone_name] = text

    bar_patches = {zone_name: bar for zone_name, bar in zip(final_order, bars)}
    return ArtistState(canvas=canvas, subtitle_text=subtitle_text, map_patches=map_patches, bar_patches=bar_patches, value_texts=value_texts)


def update_artists(
    state: ArtistState,
    zones: list[ZoneGeometry],
    final_order: list[str],
    values_by_zone: dict[str, float],
    subtitle: str,
    y_max_global: float,
    positions: np.ndarray,
    colors: np.ndarray,
) -> None:
    state.subtitle_text.set_text(subtitle)
    relative_positions_by_zone = frame_relative_positions(values_by_zone)

    for zone in zones:
        value = values_by_zone[zone.zone_name]
        state.map_patches[zone.zone_name].set_facecolor(
            color_for_value(value, relative_positions_by_zone[zone.zone_name], y_max_global, positions, colors)
        )

    label_offset = BAR_LABEL_OFFSET_RATIO * y_max_global

    for zone_name in final_order:
        value = values_by_zone[zone_name]
        bar = state.bar_patches[zone_name]
        bar.set_width(value)
        bar.set_facecolor(color_for_value(value, relative_positions_by_zone[zone_name], y_max_global, positions, colors))

        text = state.value_texts[zone_name]
        text.set_x(value + label_offset)
        text.set_text(format_dollar(value))


def render_gif() -> None:
    base_font, title_font, zone_font = choose_fonts()
    zones, bbox = load_zone_geometries(PJM_GEOJSON_PATH)
    tidy = load_revenue_data()

    zones_geo = sorted(zone.zone_name for zone in zones)
    zones_xls = sorted(tidy["zoneName"].unique().tolist())
    if zones_geo != zones_xls:
        raise ValueError("Zone mapping failed: revenue data zone names do not match GeoJSON zone names.")

    cum_long = tidy.copy()
    cum_long["CumRevenue"] = cum_long.groupby("zoneName")["Revenue"].cumsum()

    y_max_global = float(cum_long["CumRevenue"].max())
    final_order = (
        cum_long.groupby("zoneName", as_index=False)["CumRevenue"]
        .max()
        .sort_values("CumRevenue")["zoneName"]
        .tolist()
    )

    positions, colors = build_color_scale()

    _, _, series_by_zone, month_labels = build_smoothed_series(cum_long)
    frame_count = len(month_labels)
    durations_ms = build_frame_durations(frame_count)

    initial_values = {zone_name: float(series_by_zone[zone_name][0]) for zone_name in final_order}
    initial_values.update({zone.zone_name: float(series_by_zone[zone.zone_name][0]) for zone in zones})

    state = create_artist_state(
        zones=zones,
        bbox=bbox,
        final_order=final_order,
        initial_values=initial_values,
        title_font=title_font,
        subtitle_font=title_font,
        base_font=base_font,
        zone_font=zone_font,
        y_max_global=y_max_global,
        positions=positions,
        colors=colors,
        subtitle=month_labels[0].strftime("%b %Y"),
    )

    frames: list[Image.Image] = []

    for frame_idx in range(frame_count):
        values = {zone_name: float(series[frame_idx]) for zone_name, series in series_by_zone.items()}
        subtitle = month_labels[frame_idx].strftime("%b %Y")

        update_artists(
            state=state,
            zones=zones,
            final_order=final_order,
            values_by_zone=values,
            subtitle=subtitle,
            y_max_global=y_max_global,
            positions=positions,
            colors=colors,
        )

        state.canvas.draw()
        frame = np.asarray(state.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        frames.append(Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE, colors=256))

    frames[0].save(
        GIF_FILE,
        save_all=True,
        append_images=frames[1:],
        duration=durations_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )

    total_duration = sum(durations_ms) / 1000.0
    print(f"Saved GIF: {GIF_FILE.resolve()}")
    print(f"Frames: {frame_count}")
    print(f"Target GIF duration: {total_duration:.2f} seconds")


if __name__ == "__main__":
    render_gif()
