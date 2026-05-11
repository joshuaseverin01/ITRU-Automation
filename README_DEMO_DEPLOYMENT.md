# FlexWorks Public Demo Deployment

## Branch To Deploy

Deploy the `demo` branch.

This branch is intentionally demo-first. It defaults to bundled sample data and does not expose production file upload controls unless `APP_MODE=production` is explicitly set for local testing.

## Streamlit Community Cloud Settings

- Repository: `joshuaseverin01/ITRU-Automation`
- Branch: `demo`
- Main file path: `flexworks_arbitrage_analyzer/app.py`
- Python version: Python 3.11+ recommended. The app has been tested locally with Python 3.13.
- Secrets required: none for the standard public demo.

Do not set `APP_MODE=production` for the public demo deployment. With no app mode secret configured, this branch opens in public demo mode.

## Demo Data Included

The demo uses bundled non-sensitive PJM sample files:

```text
demo_data/
  flexworks_export.csv
  device_to_zone_mapping.csv
  zones.geojson
```

These files are sample workflow assets for portfolio/demo use. They are not investment advice and should not be treated as live market results.

## How The Demo Differs From The Full Internal App

The public demo branch:

- Uses bundled sample data only.
- Hides FlexWorks CSV upload controls.
- Hides coordinate lookup upload controls.
- Hides zones GeoJSON upload controls.
- Lets visitors load bundled demo files and run sample analysis.
- Preserves representative dashboard outputs, maps, charts, exports, Blog Post Creator, and Presentation Draft Creator.

The full internal app remains on `main` and supports uploading real FlexWorks exports, optional coordinate lookup/device-to-zone CSVs, and zones GeoJSON files.

## AI And Secrets

No API key is required for the demo. The Presentation Draft Creator degrades gracefully to deterministic local deck generation when `OPENAI_API_KEY` is not configured.

If an OpenAI key is configured for private testing, the app can request structured slide JSON for presentation drafts, but public demo deployment does not require this.

## Local Demo Run

From the repository root:

```bash
cd flexworks_arbitrage_analyzer
pip install -r requirements.txt
streamlit run app.py
```

The app should open in public demo mode by default on the `demo` branch.

## Local Production Preview From This Branch

For development only, you can preview the full upload workflow from the demo branch:

```bash
APP_MODE=production streamlit run app.py
```

Do not use this setting for the public Streamlit Community Cloud demo app.

## Streamlit Hibernation Note

Streamlit Community Cloud apps can hibernate after inactivity. If the demo has not had traffic recently, visitors may briefly see a wake-up page before the app becomes available.
