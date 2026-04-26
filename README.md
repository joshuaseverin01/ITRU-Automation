# Flexworks Arbitrage Intelligence Dashboard

Turn battery arbitrage simulations into zone-level market strategy.

Flexworks Arbitrage Intelligence Dashboard is a local-first Streamlit application for cleaning Flexworks simulation exports, analyzing battery arbitrage performance, comparing zones across ISOs, visualizing PJM zone performance over time, and exporting strategy-ready data, visuals, and summaries.

## Portfolio Pitch

I automated a manual internship workflow that converted Flexworks battery arbitrage simulations into zone-level market intelligence, interactive maps, animations, and executive-ready exports.

The project compresses a multi-day consulting workflow into a local dashboard: upload raw Flexworks CSVs, normalize messy revenue fields, join device metadata to monthly revenue, map PJM zones with GeoJSON polygons, inspect static and time-based performance views, and export CSV, Plotly HTML, and deterministic executive summaries.

## What the App Does

- Loads raw Flexworks-style CSV exports and validates the detected schema.
- Cleans whitespace, blank export columns, duplicated device or node rows, and currency-formatted numeric fields.
- Converts Flexworks monthly wide-format revenue exports into long-form time-series data.
- Joins device summary metadata to monthly revenue using device identifiers.
- Computes opportunity scores, risk labels, summary metrics, rankings, and zone-level KPI cards.
- Displays point maps, PJM zone choropleths, cumulative revenue map-and-bar views, and ISO-focused time-series map-and-bar views.
- Supports snapshot, time-range, multi-snapshot, and animation modes for PJM zonal performance.
- Generates deterministic Markdown and text executive summaries without external APIs.
- Exports processed CSVs and interactive Plotly HTML visuals.

## Key Features

- **Robust ingestion:** Detects current MVP node uploads, Flexworks device summary exports, and Flexworks monthly wide revenue exports.
- **PJM zone mapping:** Uses GeoJSON zone polygons for PJM choropleths and ISO-focused performance views.
- **Strategic KPI overview:** Shows number of zones, selected metric average, top zone, and top-to-bottom spread.
- **Zonal Market Performance:** Lets users explore PJM performance by exact month/timestamp, selected range, multiple snapshots, or animation.
- **Interactive visual exports:** Saves current Plotly figures as standalone HTML files.
- **Strategy Export Center:** Downloads processed data and deterministic executive summaries as Markdown or plain text.
- **Demo-ready UI:** Includes a polished landing state, usage guidance, bundled sample dataset option, and client-facing section names.

## Technical Stack

- Python
- Streamlit
- Pandas
- Plotly
- GeoJSON / geospatial mapping
- unittest

## Supported Data Inputs

### 1. Current MVP Node Schema

Node-level CSVs must include:

- `Node_ID`
- `ISO_Region`
- `Annualized_Revenue`
- `Revenue_per_kW`
- `LMP_Volatility`

Optional map columns:

- `Latitude`
- `Longitude`

### 2. Flexworks Device Summary Schema

Device summary CSVs must include:

- `Device`
- `Location`
- `Annualized Income`, or a column containing both `Annualized` and `Income`
- `Revenue per kW`

Blank trailing export columns are ignored. Currency fields such as `$1,496.08` are coerced to numeric values. `Location` values such as `BGE (PJM)` are parsed into `Zone = BGE` and `ISO_Region = PJM`.

Device summary fields are normalized as:

- `Device` -> `Node_ID`, `Device_ID`
- `Location` -> `Node_Name`, `Zone`
- `Annualized Income` -> `Annualized_Revenue`
- `Revenue per kW` -> `Revenue_per_kW`

`LMP_Volatility` is not present in this export type and is left missing.

### 3. Flexworks Monthly Wide Revenue Schema

Monthly wide-format CSVs are detected when month columns follow `YYYY-MM`, such as `2022-01` through `2024-12`. The first two columns are treated as device and revenue category, then reshaped into:

- `Device`
- `Revenue_Category`
- `Month`
- `Revenue`

Rows grouped by device are handled by forward-filling the device value across category rows such as `Energy`, `Ancillary`, and `FCP`.

When a device summary and monthly revenue export are uploaded together, monthly revenue rows are joined to device metadata using `Device`, enabling monthly revenue charts by device, zone, and category.

### 4. Optional Coordinate Lookup

Coordinate lookup CSVs can be uploaded with:

- `Node_ID`
- `Latitude`
- `Longitude`

When polygon mapping is unavailable, valid coordinates power the point-map fallback.

### 5. Optional PJM GeoJSON

The app can load PJM zone polygons from a GeoJSON FeatureCollection. It detects zone-name properties such as:

- `zoneName`
- `PLANNING_ZONE_NAME`
- `Zone`
- other property names containing `zone`

Zone joins use normalized names: uppercase, trimmed whitespace, common symbols removed, and common PJM naming variants mapped to short zone labels. If a polygon join is unavailable or fails, the app shows diagnostics and falls back gracefully.

## How to Run Locally

From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app runs locally and does not require a database, authentication, deployment service, or external API.

## How to Run Tests

```bash
python -m unittest discover
```

For a quick syntax validation:

```bash
python -m py_compile app.py src/*.py
```

## Demo Mode

If no file is uploaded, the sidebar can load the bundled sample dataset from:

```text
sample_data/sample_flexworks_export.csv
```

This demo mode is intended for interface walkthroughs and portfolio presentations. It does not replace analysis of real Flexworks simulation exports.

## Export Outputs

- Cleaned CSV
- Ranked CSV
- Processed zone performance CSV
- Interactive Plotly HTML visuals
- Markdown executive summary
- Plain text executive summary
- Optional PNG export when Plotly image export dependencies, such as Kaleido, are installed in the environment

Kaleido is intentionally not required for the core app. If it is unavailable, HTML export remains supported and the app shows a clear PNG-export warning.

## Methodology

- Column names and string values are whitespace-trimmed.
- Required numeric fields are coerced to numeric values.
- Blank `Node_ID` rows are removed.
- Duplicate `Node_ID` rows are aggregated by averaging numeric fields and keeping the first non-empty categorical value.
- Opportunity score is a weighted average of available normalized `Annualized_Revenue`, `Revenue_per_kW`, and `LMP_Volatility`.
- Metrics with no valid values are excluded from scoring and the remaining weights are redistributed.
- Volatility risk labels are dataset-relative tertiles when `LMP_Volatility` is available: stable, moderate, and high volatility.
- Monthly revenue aggregation sums revenue by zone and time period.
- `Revenue per kW` is averaged across selected time ranges where period aggregation is needed.
- Reports are deterministic Markdown/text generated only from computed metrics.

## Known Limitations

- PJM is the only ISO with polygon choropleth support in the current implementation.
- ERCOT, CAISO, and other ISOs currently use generic node or coordinate fallbacks unless polygon GeoJSON support is added.
- The app screens arbitrage value from simulation outputs; it does not run a full battery dispatch optimizer.
- Hourly and daily time-series support is architected, but the bundled sample data is monthly.
- PNG export depends on optional Plotly image export support.
- Executive summaries are deterministic and intentionally not LLM-generated.

## Future Roadmap

- ERCOT/CAISO polygon support
- Full LLM-generated narrative summaries
- GIF/video animation export
- Scenario comparison mode
- Cloud deployment

