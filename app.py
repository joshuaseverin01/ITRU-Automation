"""Streamlit entrypoint for FlexWorks Arbitrage Analyzer."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.analysis import (
    ALL_REVENUE_CATEGORIES,
    DEFAULT_SCORE_WEIGHTS,
    SNAPSHOT_METRIC_CUMULATIVE_REVENUE,
    SNAPSHOT_METRIC_MONTHLY_REVENUE,
    SNAPSHOT_METRIC_REVENUE_PER_KW,
    add_analysis_columns,
    aggregate_zone_metric,
    aggregate_zone_metric_over_range,
    compute_summary_metrics,
    compute_zone_monthly_revenue,
    filter_zone_revenue_to_month,
    identify_high_risk_high_reward,
    rank_nodes,
    summarize_iso_regions,
)
from src.cleaning import CleaningSummary, clean_flexworks_export
from src.data_loader import DataLoadError, load_csv
from src.geo import (
    CoordinateStatus,
    PjmZoneGeoJson,
    ZoneJoinDiagnostics,
    detect_coordinate_status,
    load_pjm_zone_geojson,
    merge_coordinate_lookup,
    standardize_coordinate_columns,
)
from src.ingestion import ExportSchema, ParsedExport, join_monthly_to_device_summary, parse_flexworks_export
from src.reporting import (
    build_executive_summary,
    build_zone_kpi_overview,
    export_dataframe_csv,
    generate_markdown_report,
    plotly_figure_to_html_bytes,
    plotly_figures_to_html_bytes,
    safe_plotly_png_bytes,
)
from src.temporal import (
    TIME_GRANULARITY_MONTHLY,
    TIME_GRANULARITY_NONE,
    available_time_points,
    default_frame_count_for_range,
    detect_time_granularity,
    format_time_label,
    format_time_range_label,
    select_evenly_spaced_snapshots,
)
from src.validation import format_missing_columns_message, validate_coordinate_lookup, validate_required_columns
from src.visualization import (
    animation_frames_to_html_player,
    build_iso_zone_snapshot_map_bars,
    build_monthly_revenue_bar,
    build_monthly_revenue_chart,
    build_node_map,
    build_pjm_cumulative_revenue_map_bars,
    build_pjm_zone_choropleth,
    build_top_nodes_bar,
    build_volatility_revenue_scatter,
    create_pjm_animation_gif_bytes,
    create_pjm_matplotlib_figure,
    gif_bytes_to_html_img,
    matplotlib_figure_to_png_bytes,
    matplotlib_figures_to_zip_bytes,
)


PROJECT_ROOT = Path(__file__).parent
DEMO_DATA_DIR = PROJECT_ROOT / "demo_data"
DEMO_FLEXWORKS_EXPORT_PATH = DEMO_DATA_DIR / "flexworks_export.csv"
DEMO_DEVICE_ZONE_MAPPING_PATH = DEMO_DATA_DIR / "device_to_zone_mapping.csv"
DEMO_ZONES_GEOJSON_PATH = DEMO_DATA_DIR / "zones.geojson"
DEMO_FILE_PATHS = (
    DEMO_FLEXWORKS_EXPORT_PATH,
    DEMO_DEVICE_ZONE_MAPPING_PATH,
    DEMO_ZONES_GEOJSON_PATH,
)
DEFAULT_PJM_GEOJSON_PATH = DEMO_ZONES_GEOJSON_PATH
MAX_MULTI_SNAPSHOTS = 12
MAX_ANIMATION_FRAMES = 60
ANALYSIS_STATE_KEY = "flexworks_analysis_state"
WALKTHROUGH_STATE_KEY = "show_walkthrough"


def main() -> None:
    st.set_page_config(page_title="Flexworks Arbitrage Intelligence Dashboard", layout="wide")
    _render_header()

    try:
        _render_app()
    except Exception as exc:  # pragma: no cover - final UI safety net
        st.error("The analysis could not be completed. Check the input files and try again.")
        st.caption(str(exc))
    _render_footer()


def _render_header() -> None:
    st.title("Flexworks Arbitrage Intelligence Dashboard")
    st.caption("Turn battery arbitrage simulations into zone-level market strategy.")
    with st.expander("How to use this app", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- Upload Flexworks output.",
                    "- Select ISO and metric.",
                    "- Explore static, range, multi-snapshot, or animation views.",
                    "- Export CSV, HTML visuals, and executive summary.",
                ]
            )
        )


def _render_empty_state(has_demo_data: bool) -> None:
    st.info(
        "Upload a Flexworks export and click Run Analysis to generate market intelligence outputs. "
        "Upload a Flexworks simulation export and, for time-series views, a device-to-zone mapping file. "
        "The dashboard will clean the files, join device metadata to revenue, map PJM zones, rank market performance, "
        "and generate strategy-ready exports."
    )
    st.markdown(
        "\n".join(
            [
                "Expected uploads:",
                "- Device summary CSV with Device, Location, Annualized Income, and Revenue per kW.",
                "- Monthly wide-format revenue CSV with device/category rows and YYYY-MM columns.",
                "- Optional PJM zones GeoJSON for polygon maps.",
            ]
        )
    )
    if has_demo_data:
        st.caption("Demo files are available from the sidebar. Load them, then click Run Analysis to explore the PJM workflow.")


def _render_footer() -> None:
    st.divider()
    st.caption("Built to convert Flexworks simulation outputs into investment-ready market intelligence.")


def _ensure_walkthrough_state(session_state: MutableMapping[str, object]) -> bool:
    """Initialize and return the first-time walkthrough visibility flag."""

    if WALKTHROUGH_STATE_KEY not in session_state:
        session_state[WALKTHROUGH_STATE_KEY] = True
    return bool(session_state[WALKTHROUGH_STATE_KEY])


def _dismiss_walkthrough(session_state: MutableMapping[str, object]) -> None:
    """Hide the walkthrough for the current Streamlit session."""

    session_state[WALKTHROUGH_STATE_KEY] = False


def _close_walkthrough_and_rerun(session_state: MutableMapping[str, object], rerun: Callable[[], None]) -> None:
    """Hide the walkthrough and immediately refresh the Streamlit UI."""

    _dismiss_walkthrough(session_state)
    rerun()


def _reopen_walkthrough(session_state: MutableMapping[str, object]) -> None:
    """Show the walkthrough again in the current Streamlit session."""

    session_state[WALKTHROUGH_STATE_KEY] = True


def _render_walkthrough() -> None:
    if not _ensure_walkthrough_state(st.session_state):
        return

    if callable(getattr(st, "dialog", None)):
        _render_walkthrough_dialog()
        return

    _render_walkthrough_card()


def _render_walkthrough_dialog() -> None:
    dialog = getattr(st, "dialog")

    @dialog("Welcome to the Flexworks Arbitrage Intelligence Dashboard")
    def walkthrough_dialog() -> None:
        st.write("This tool converts Flexworks battery arbitrage simulations into zone-level market intelligence.")
        st.markdown(
            "\n".join(
                [
                    "1. Load demo files or upload your own Flexworks export CSV, mapping CSV, and zones GeoJSON.",
                    "2. Click **Run Analysis** to process the data.",
                    "3. Explore Snapshot, Time range, Multi-snapshot, and Animation modes.",
                    "4. Download PNGs, GIFs, CSVs, and summaries from the Strategy Export Center.",
                ]
            )
        )
        if st.button(
            "Got it",
            type="primary",
            key="dismiss_walkthrough_modal",
        ):
            _close_walkthrough_and_rerun(st.session_state, st.rerun)

    walkthrough_dialog()


def _render_walkthrough_card() -> None:
    st.markdown(
        """
        <div style="
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            background: #f8fafc;
            padding: 1.1rem 1.25rem;
            margin: 0.25rem 0 1.25rem 0;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
            color: #111827;
        ">
            <h3 style="margin: 0 0 0.5rem 0; color: #111827;">
                Welcome to the Flexworks Arbitrage Intelligence Dashboard
            </h3>
            <p style="margin: 0 0 0.7rem 0; color: #1f2937;">
                This tool converts Flexworks battery arbitrage simulations into zone-level market intelligence.
            </p>
            <ol style="margin: 0 0 0 1.25rem; padding: 0; color: #1f2937;">
                <li>Load demo files or upload your own Flexworks export CSV, mapping CSV, and zones GeoJSON.</li>
                <li>Click <strong>Run Analysis</strong> to process the data.</li>
                <li>Explore Snapshot, Time range, Multi-snapshot, and Animation modes.</li>
                <li>Download PNGs, GIFs, CSVs, and summaries from the Strategy Export Center.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(
        "Got it",
        type="primary",
        key="dismiss_walkthrough_card",
    ):
        _close_walkthrough_and_rerun(st.session_state, st.rerun)


def _render_app() -> None:
    state = _analysis_state()
    if st.sidebar.button("Show walkthrough", use_container_width=True, key="show_walkthrough_button"):
        _reopen_walkthrough(st.session_state)
    _render_walkthrough()

    uploaded_exports = st.sidebar.file_uploader(
        "FlexWorks Export CSVs",
        type=["csv"],
        accept_multiple_files=True,
        key="staged_flexworks_exports",
        help="Upload one or more Flexworks simulation export files containing revenue or market performance data.",
    )
    st.sidebar.caption("Upload one or more Flexworks simulation export files containing revenue or market performance data.")
    uploaded_lookup = st.sidebar.file_uploader(
        "Device-to-Zone Mapping CSV",
        type=["csv"],
        key="staged_coordinate_lookup",
        help="Optional mapping file that connects devices/nodes from the simulation export to zone names used in the map.",
    )
    st.sidebar.caption("Optional mapping file that connects devices/nodes from the simulation export to zone names used in the map.")
    uploaded_pjm_geojson = st.sidebar.file_uploader(
        "Zones GeoJSON",
        type=["geojson", "json"],
        key="staged_pjm_geojson",
        help="Upload zone polygon boundaries. The GeoJSON must include a zone name field matching the processed data.",
    )
    st.sidebar.caption("Upload zone polygon boundaries. The GeoJSON must include a zone name field matching the processed data.")
    demo_files_available = _demo_files_available()
    with st.sidebar.expander("Demo files", expanded=False):
        st.caption("Use these bundled PJM sample files to test the dashboard without your own Flexworks export.")
        demo_clicked = st.button(
            "Load Demo Files",
            disabled=not demo_files_available,
            use_container_width=True,
            key="load_demo_files",
        )
        if not demo_files_available:
            st.warning("Bundled demo files are missing from demo_data/.")
        elif st.session_state.get("demo_files_loaded"):
            st.success("Demo files loaded. Click Run Analysis to generate sample PJM market intelligence outputs.")
    if demo_clicked:
        st.session_state["demo_files_loaded"] = True
        st.session_state["demo_files_notice"] = True
    if st.session_state.pop("demo_files_notice", False):
        st.sidebar.success("Demo files loaded. Click Run Analysis to generate sample PJM market intelligence outputs.")
    if uploaded_exports:
        st.session_state["demo_files_loaded"] = False
    use_demo_files = bool(st.session_state.get("demo_files_loaded")) and demo_files_available
    if use_demo_files:
        st.sidebar.caption("Demo inputs staged: Flexworks monthly export, device-to-zone mapping, and zones GeoJSON.")

    use_local_pjm_geojson = st.sidebar.checkbox(
        "Use bundled zones GeoJSON",
        value=DEFAULT_PJM_GEOJSON_PATH.exists() and uploaded_pjm_geojson is None and not use_demo_files,
        disabled=uploaded_pjm_geojson is not None or use_demo_files or not DEFAULT_PJM_GEOJSON_PATH.exists(),
        key="staged_use_local_pjm_geojson",
    )
    _stage_uploaded_inputs(uploaded_exports, uploaded_lookup, uploaded_pjm_geojson, use_demo_files, use_local_pjm_geojson)

    staged_signature = _staged_input_signature(uploaded_exports, uploaded_lookup, uploaded_pjm_geojson, use_demo_files, use_local_pjm_geojson)
    has_staged_exports = _has_staged_flexworks_input(uploaded_exports, use_demo_files)
    run_clicked = st.sidebar.button(
        "Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=not has_staged_exports,
    )
    if run_clicked:
        _run_analysis_workflow(
            uploaded_exports=uploaded_exports,
            use_demo_files=use_demo_files,
            uploaded_lookup=uploaded_lookup,
            uploaded_pjm_geojson=uploaded_pjm_geojson,
            use_local_pjm_geojson=use_local_pjm_geojson,
            input_signature=staged_signature,
        )
        state = _analysis_state()

    if not has_staged_exports:
        st.sidebar.caption("Upload a Flexworks export or load demo files before running analysis.")

    st.sidebar.divider()
    st.sidebar.subheader("Scoring Weights")
    score_weights = _render_weight_controls()

    if state.get("analysis_error"):
        st.error(str(state["analysis_error"]))

    if not state.get("analysis_has_run"):
        _render_empty_state(demo_files_available)
        return

    if staged_signature != state.get("input_signature"):
        st.info("New file uploaded. Click Run Analysis to refresh results.")
        st.sidebar.warning("New file uploaded. Click Run Analysis to refresh results.")

    parsed_exports = state.get("parsed_exports") or []
    _render_schema_status(parsed_exports)
    node_data = state.get("node_data")
    monthly_data = state.get("monthly_data")

    if node_data is None:
        st.warning("Monthly revenue data was loaded, but node-level analysis requires a current node schema or device summary export.")
        st.info(
            "PJM Cumulative Revenue Map + Bars requires both a FlexWorks device summary export and a monthly revenue export so monthly rows can be joined to PJM zones."
        )
        if monthly_data is not None:
            _render_monthly_revenue_section(monthly_data)
        return

    cleaned_with_coordinates = state["cleaned_with_coordinates"]
    cleaning_summary = state["cleaning_summary"]
    pjm_geojson = state.get("pjm_geojson")
    monthly_revenue = state.get("monthly_revenue")
    monthly_notes = state.get("monthly_notes") or []
    if pjm_geojson is not None:
        st.sidebar.caption(
            f"Active PJM GeoJSON: {pjm_geojson.zone_count} zones using `{pjm_geojson.zone_property}`."
        )

    iso_options = _available_iso_regions(cleaned_with_coordinates)
    selected_isos = st.sidebar.multiselect("ISO filter", iso_options, default=iso_options)
    top_n = st.sidebar.slider("Top nodes", min_value=5, max_value=50, value=10, step=5)

    filtered_data = _filter_by_iso(cleaned_with_coordinates, selected_isos)
    if filtered_data.empty:
        st.warning("The selected filters returned zero rows. Adjust the ISO filter to continue.")
        _render_cleaning_summary(cleaning_summary)
        return

    st.session_state["selected_iso_filters"] = selected_isos
    st.session_state["active_dataset"] = filtered_data
    analyzed_data = add_analysis_columns(filtered_data, score_weights)
    st.session_state["processed_dataframe"] = analyzed_data
    ranked_nodes = rank_nodes(analyzed_data)
    st.session_state["ranked_dataframe"] = ranked_nodes
    top_ranked_nodes = ranked_nodes.head(top_n)
    iso_summary = summarize_iso_regions(analyzed_data)
    high_risk_high_reward = identify_high_risk_high_reward(analyzed_data)
    summary_metrics = compute_summary_metrics(analyzed_data)
    coordinate_status = detect_coordinate_status(analyzed_data)
    report = generate_markdown_report(
        summary_metrics=summary_metrics,
        ranked_nodes=ranked_nodes,
        iso_summary=iso_summary,
        high_risk_high_reward_nodes=high_risk_high_reward,
        cleaning_summary=cleaning_summary,
        coordinate_status=coordinate_status,
        score_weights=score_weights,
    )

    _render_kpi_overview(analyzed_data)
    _render_summary_cards(summary_metrics)
    _render_exports(cleaned_with_coordinates, ranked_nodes, report, monthly_revenue)
    _render_visualizations(analyzed_data, top_n, coordinate_status, pjm_geojson, selected_isos, monthly_revenue)
    _render_monthly_revenue_section(monthly_revenue, monthly_notes)
    _render_tables(top_ranked_nodes, iso_summary, high_risk_high_reward)
    _render_report(report)
    _render_cleaning_summary(cleaning_summary)


def _default_analysis_state() -> dict[str, object]:
    return {
        "analysis_has_run": False,
        "analysis_error": None,
        "input_signature": None,
        "parsed_exports": [],
        "node_data": None,
        "monthly_data": None,
        "cleaned_with_coordinates": None,
        "cleaning_summary": None,
        "pjm_geojson": None,
        "monthly_revenue": None,
        "monthly_notes": [],
        "active_dataset_name": None,
    }


def _analysis_state() -> dict[str, object]:
    if ANALYSIS_STATE_KEY not in st.session_state:
        st.session_state[ANALYSIS_STATE_KEY] = _default_analysis_state()
    return st.session_state[ANALYSIS_STATE_KEY]


def _stage_uploaded_inputs(
    uploaded_exports: list[object] | None,
    uploaded_lookup: object | None,
    uploaded_pjm_geojson: object | None,
    use_demo_files: bool,
    use_local_pjm_geojson: bool,
) -> None:
    staged_exports = list(uploaded_exports or [])
    st.session_state["staged_uploaded_csv_objects"] = staged_exports
    st.session_state["staged_uploaded_csv_names"] = [getattr(uploaded_file, "name", "uploaded CSV") for uploaded_file in staged_exports]
    st.session_state["staged_coordinate_lookup_object"] = uploaded_lookup
    st.session_state["staged_coordinate_lookup_name"] = getattr(uploaded_lookup, "name", None)
    st.session_state["staged_uploaded_geojson_object"] = uploaded_pjm_geojson
    st.session_state["staged_uploaded_geojson_name"] = getattr(uploaded_pjm_geojson, "name", None)
    st.session_state["staged_use_demo_files_flag"] = use_demo_files
    st.session_state["staged_use_local_pjm_geojson_flag"] = use_local_pjm_geojson


def _demo_files_available() -> bool:
    return all(path.exists() for path in DEMO_FILE_PATHS)


def _has_staged_flexworks_input(uploaded_exports: list[object] | None, use_demo_files: bool) -> bool:
    return bool(uploaded_exports) or bool(use_demo_files and _demo_files_available())


def _staged_input_signature(
    uploaded_exports: list[object] | None,
    uploaded_lookup: object | None,
    uploaded_pjm_geojson: object | None,
    use_demo_files: bool,
    use_local_pjm_geojson: bool,
) -> tuple[object, ...]:
    if use_demo_files:
        export_signature: object = (
            "demo",
            DEMO_FLEXWORKS_EXPORT_PATH.name,
            _path_mtime_ns(DEMO_FLEXWORKS_EXPORT_PATH),
            DEMO_DEVICE_ZONE_MAPPING_PATH.name,
            _path_mtime_ns(DEMO_DEVICE_ZONE_MAPPING_PATH),
        )
    else:
        export_signature = tuple(_uploaded_file_signature(uploaded_file) for uploaded_file in uploaded_exports or [])

    geojson_signature: object
    if uploaded_pjm_geojson is not None:
        geojson_signature = ("uploaded", _uploaded_file_signature(uploaded_pjm_geojson))
    elif use_demo_files:
        geojson_signature = ("demo", DEMO_ZONES_GEOJSON_PATH.name, _path_mtime_ns(DEMO_ZONES_GEOJSON_PATH))
    elif use_local_pjm_geojson:
        geojson_signature = ("local", str(DEFAULT_PJM_GEOJSON_PATH), _path_mtime_ns(DEFAULT_PJM_GEOJSON_PATH))
    else:
        geojson_signature = None

    return (
        ("exports", export_signature),
        ("coordinate_lookup", _uploaded_file_signature(uploaded_lookup) if uploaded_lookup is not None else None),
        ("pjm_geojson", geojson_signature),
    )


def _uploaded_file_signature(uploaded_file: object) -> tuple[str, int | None]:
    file_name = getattr(uploaded_file, "name", "uploaded file")
    file_size = getattr(uploaded_file, "size", None)
    if file_size is None and hasattr(uploaded_file, "getbuffer"):
        try:
            file_size = len(uploaded_file.getbuffer())
        except Exception:
            file_size = None
    return str(file_name), int(file_size) if file_size is not None else None


def _path_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _run_analysis_workflow(
    *,
    uploaded_exports: list[object] | None,
    use_demo_files: bool,
    uploaded_lookup: object | None,
    uploaded_pjm_geojson: object | None,
    use_local_pjm_geojson: bool,
    input_signature: tuple[object, ...],
) -> None:
    st.session_state[ANALYSIS_STATE_KEY] = _default_analysis_state()
    progress = st.progress(0)

    try:
        with st.status("Running analysis...", expanded=True) as status:
            st.write("Reading uploaded file...")
            parsed_exports = _load_flexworks_exports(
                uploaded_exports,
                use_demo_files,
                uploaded_mapping_csv=uploaded_lookup,
                stop_on_error=True,
            )
            if not parsed_exports:
                raise ValueError("No Flexworks exports were loaded. Upload a CSV or enable demo mode, then run analysis.")
            unsupported_files = [file_name for file_name, parsed in parsed_exports if parsed.schema == ExportSchema.UNKNOWN]
            if unsupported_files:
                raise ValueError(
                    "Unsupported Flexworks export schema for: "
                    + ", ".join(unsupported_files)
                    + ". Upload a node summary, device summary, or monthly wide-format export."
                )
            progress.progress(25)

            st.write("Cleaning Flexworks export...")
            node_data = _select_node_dataframe(parsed_exports)
            monthly_data = _select_monthly_dataframe(parsed_exports)
            cleaned_with_coordinates = None
            cleaning_summary = None
            monthly_revenue = monthly_data
            monthly_notes: list[str] = []

            if node_data is not None:
                validation_result = validate_required_columns(node_data)
                if not validation_result.is_valid:
                    raise ValueError(format_missing_columns_message(validation_result.missing_columns))
                cleaned_data, cleaning_summary = clean_flexworks_export(node_data)
                coordinate_lookup = (
                    None
                    if _uploaded_mapping_used_as_flexworks_export(parsed_exports, uploaded_lookup)
                    else _load_coordinate_lookup(uploaded_lookup)
                )
                cleaned_with_coordinates, _ = merge_coordinate_lookup(cleaned_data, coordinate_lookup)
            progress.progress(50)

            st.write("Matching zones...")
            pjm_geojson = _load_pjm_geojson(uploaded_pjm_geojson, use_local_pjm_geojson, use_demo_files=use_demo_files)
            progress.progress(70)

            st.write("Building market intelligence outputs...")
            if node_data is not None:
                monthly_revenue, monthly_notes = join_monthly_to_device_summary(monthly_data, cleaned_with_coordinates)
            progress.progress(90)

            st.session_state[ANALYSIS_STATE_KEY] = {
                "analysis_has_run": True,
                "analysis_error": None,
                "input_signature": input_signature,
                "parsed_exports": parsed_exports,
                "node_data": node_data,
                "monthly_data": monthly_data,
                "cleaned_with_coordinates": cleaned_with_coordinates,
                "cleaning_summary": cleaning_summary,
                "pjm_geojson": pjm_geojson,
                "monthly_revenue": monthly_revenue,
                "monthly_notes": monthly_notes,
                "active_dataset_name": _active_dataset_name(parsed_exports, use_demo_files),
            }
            st.session_state["processed_dataframe"] = cleaned_with_coordinates
            st.session_state["active_dataset"] = cleaned_with_coordinates
            progress.progress(100)
            st.write("Analysis complete.")
            status.update(label="Analysis complete.", state="complete", expanded=False)
    except Exception as exc:
        st.session_state[ANALYSIS_STATE_KEY] = {
            **_default_analysis_state(),
            "analysis_error": str(exc),
            "input_signature": input_signature,
        }
    finally:
        progress.empty()


def _active_dataset_name(parsed_exports: list[tuple[str, ParsedExport]], use_demo_files: bool) -> str:
    if use_demo_files:
        return "Bundled PJM demo files"
    return ", ".join(file_name for file_name, _ in parsed_exports)


def _uploaded_mapping_used_as_flexworks_export(
    parsed_exports: list[tuple[str, ParsedExport]],
    uploaded_mapping_csv: object | None,
) -> bool:
    if uploaded_mapping_csv is None:
        return False
    mapping_name = getattr(uploaded_mapping_csv, "name", None)
    return any(file_name == mapping_name and parsed.schema != ExportSchema.UNKNOWN for file_name, parsed in parsed_exports)


def _load_flexworks_exports(
    uploaded_exports: list[object] | None,
    use_demo_files: bool,
    *,
    uploaded_mapping_csv: object | None = None,
    stop_on_error: bool = False,
) -> list[tuple[str, ParsedExport]]:
    sources: list[tuple[str, object]] = []
    if use_demo_files:
        if uploaded_mapping_csv is None:
            sources.append((DEMO_DEVICE_ZONE_MAPPING_PATH.name, DEMO_DEVICE_ZONE_MAPPING_PATH))
        else:
            sources.append((getattr(uploaded_mapping_csv, "name", "Device-to-Zone Mapping CSV"), uploaded_mapping_csv))
        sources.append((DEMO_FLEXWORKS_EXPORT_PATH.name, DEMO_FLEXWORKS_EXPORT_PATH))
    else:
        sources.extend((getattr(uploaded_file, "name", "uploaded CSV"), uploaded_file) for uploaded_file in uploaded_exports or [])
        if uploaded_mapping_csv is not None:
            sources.append((getattr(uploaded_mapping_csv, "name", "Device-to-Zone Mapping CSV"), uploaded_mapping_csv))

    parsed_exports: list[tuple[str, ParsedExport]] = []
    for file_name, source in sources:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            raw_data = load_csv(source)
            parsed = parse_flexworks_export(raw_data)
            if source is uploaded_mapping_csv and parsed.schema == ExportSchema.UNKNOWN:
                continue
            parsed_exports.append((file_name, parsed))
        except (DataLoadError, ValueError) as exc:
            if stop_on_error:
                raise ValueError(f"{file_name}: {exc}") from exc
            st.error(f"{file_name}: {exc}")

    if use_demo_files and uploaded_mapping_csv is not None and not any(parsed.node_dataframe is not None for _, parsed in parsed_exports):
        try:
            demo_mapping = parse_flexworks_export(load_csv(DEMO_DEVICE_ZONE_MAPPING_PATH))
            parsed_exports.insert(0, (DEMO_DEVICE_ZONE_MAPPING_PATH.name, demo_mapping))
        except (DataLoadError, ValueError) as exc:
            if stop_on_error:
                raise ValueError(f"{DEMO_DEVICE_ZONE_MAPPING_PATH.name}: {exc}") from exc
            st.error(f"{DEMO_DEVICE_ZONE_MAPPING_PATH.name}: {exc}")
    return parsed_exports


def _select_node_dataframe(parsed_exports: list[tuple[str, ParsedExport]]) -> pd.DataFrame | None:
    node_frames = [parsed.node_dataframe for _, parsed in parsed_exports if parsed.node_dataframe is not None]
    if not node_frames:
        return None
    if len(node_frames) > 1:
        st.warning("Multiple node/device summary exports were uploaded. Using the first one for node-level analysis.")
    return node_frames[0]


def _select_monthly_dataframe(parsed_exports: list[tuple[str, ParsedExport]]) -> pd.DataFrame | None:
    monthly_frames = [parsed.monthly_dataframe for _, parsed in parsed_exports if parsed.monthly_dataframe is not None]
    if not monthly_frames:
        return None
    return pd.concat(monthly_frames, ignore_index=True)


def _render_schema_status(parsed_exports: list[tuple[str, ParsedExport]]) -> None:
    with st.sidebar.expander("Detected schemas", expanded=False):
        for file_name, parsed in parsed_exports:
            st.write(f"{file_name}: {parsed.schema.value}")
            for note in parsed.notes:
                st.caption(note)


def _load_coordinate_lookup(uploaded_lookup: object | None) -> pd.DataFrame | None:
    if uploaded_lookup is None:
        return None

    try:
        lookup = standardize_coordinate_columns(load_csv(uploaded_lookup))
    except DataLoadError as exc:
        st.sidebar.warning(str(exc))
        return None

    validation_result = validate_coordinate_lookup(lookup)
    if not validation_result.is_valid:
        missing = ", ".join(validation_result.missing_columns)
        st.sidebar.warning(
            f"Coordinate lookup ignored. Missing required lookup column(s): {missing}. "
            "Required lookup columns are: Node_ID, Latitude, Longitude."
        )
        return None

    return lookup


def _load_pjm_geojson(
    uploaded_geojson: object | None,
    use_local_pjm_geojson: bool,
    *,
    use_demo_files: bool = False,
) -> PjmZoneGeoJson | None:
    if uploaded_geojson is None and not use_local_pjm_geojson and not use_demo_files:
        return None

    if uploaded_geojson is not None:
        source = uploaded_geojson
    elif use_demo_files:
        source = DEMO_ZONES_GEOJSON_PATH
    else:
        source = DEFAULT_PJM_GEOJSON_PATH
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        loaded_geojson = load_pjm_zone_geojson(source)
    except ValueError as exc:
        st.sidebar.warning(str(exc))
        return None

    st.sidebar.caption(
        f"PJM GeoJSON loaded: {loaded_geojson.zone_count} zones using `{loaded_geojson.zone_property}`."
    )
    return loaded_geojson


def _render_weight_controls() -> dict[str, float]:
    annualized_revenue = st.sidebar.slider(
        "Annualized revenue",
        min_value=0,
        max_value=100,
        value=int(DEFAULT_SCORE_WEIGHTS["Annualized_Revenue"] * 100),
    )
    revenue_per_kw = st.sidebar.slider(
        "Revenue per kW",
        min_value=0,
        max_value=100,
        value=int(DEFAULT_SCORE_WEIGHTS["Revenue_per_kW"] * 100),
    )
    lmp_volatility = st.sidebar.slider(
        "LMP volatility",
        min_value=0,
        max_value=100,
        value=int(DEFAULT_SCORE_WEIGHTS["LMP_Volatility"] * 100),
    )
    return {
        "Annualized_Revenue": float(annualized_revenue),
        "Revenue_per_kW": float(revenue_per_kw),
        "LMP_Volatility": float(lmp_volatility),
    }


def _available_iso_regions(dataframe: pd.DataFrame) -> list[str]:
    if "ISO_Region" not in dataframe.columns:
        return []
    return sorted(dataframe["ISO_Region"].dropna().astype(str).unique().tolist())


def _filter_by_iso(dataframe: pd.DataFrame, selected_isos: list[str]) -> pd.DataFrame:
    if "ISO_Region" not in dataframe.columns:
        return dataframe.copy()
    if not selected_isos:
        return dataframe.head(0).copy()
    return dataframe.loc[dataframe["ISO_Region"].astype(str).isin(selected_isos)].copy()


def _render_summary_cards(summary_metrics: object) -> None:
    st.subheader("Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", f"{summary_metrics.node_count:,}")
    col2.metric("ISO Regions", f"{summary_metrics.iso_count:,}")
    col3.metric("Avg Revenue/kW", _fmt_dollars_per_kw(summary_metrics.average_revenue_per_kw))
    col4.metric("Avg Volatility", _fmt_number(summary_metrics.average_lmp_volatility))

    col5, col6, col7 = st.columns(3)
    col5.metric("Top Opportunity Node", summary_metrics.top_opportunity_node or "n/a")
    col6.metric("Max Revenue Node", summary_metrics.max_revenue_node or "n/a")
    col7.metric("High Volatility Nodes", f"{summary_metrics.high_volatility_node_count:,}")


def _render_kpi_overview(analyzed_data: pd.DataFrame) -> None:
    st.subheader("Market Intelligence Overview")
    metric_options = [
        column
        for column in ("Revenue_per_kW", "Annualized_Revenue", "Opportunity_Score", "Risk_Adjusted_Score")
        if column in analyzed_data.columns and not pd.to_numeric(analyzed_data[column], errors="coerce").dropna().empty
    ]
    if not metric_options:
        st.warning("No numeric market metric is available for KPI overview.")
        return

    labels = {
        "Revenue_per_kW": "Revenue per kW",
        "Annualized_Revenue": "Annualized Revenue",
        "Opportunity_Score": "Opportunity Score",
        "Risk_Adjusted_Score": "Risk-adjusted Score",
    }
    selected_label = st.selectbox("KPI metric", [labels[column] for column in metric_options], key="overview_kpi_metric")
    metric_column = {labels[column]: column for column in metric_options}[selected_label]
    kpis = build_zone_kpi_overview(analyzed_data, metric_column=metric_column)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Zones", f"{kpis.zone_count:,}")
    col2.metric(f"Average {selected_label}", _fmt_metric_value(kpis.metric_average, metric_column))
    col3.metric("Top Zone", kpis.top_zone or "n/a")
    col4.metric("Top-to-Bottom Spread", _fmt_metric_value(kpis.spread, metric_column))
    st.caption(
        "These KPIs translate cleaned Flexworks results into a market-screening view: breadth of zonal coverage, average value, leader, and locational spread."
    )


def _render_exports(
    cleaned_data: pd.DataFrame,
    ranked_nodes: pd.DataFrame,
    report: str,
    monthly_revenue: pd.DataFrame | None = None,
) -> None:
    st.subheader("Core Data Exports")
    st.caption("Download cleaned datasets and the full deterministic methodology report for auditability and downstream analysis.")
    col1, col2, col3, col4 = st.columns(4)
    col1.download_button(
        "Cleaned CSV",
        data=cleaned_data.to_csv(index=False),
        file_name="flexworks_cleaned.csv",
        mime="text/csv",
    )
    col2.download_button(
        "Ranked CSV",
        data=ranked_nodes.to_csv(index=False),
        file_name="flexworks_ranked.csv",
        mime="text/csv",
    )
    col3.download_button(
        "Markdown Report",
        data=report,
        file_name="flexworks_arbitrage_report.md",
        mime="text/markdown",
    )
    if monthly_revenue is not None and not monthly_revenue.empty:
        col4.download_button(
            "Monthly CSV",
            data=monthly_revenue.to_csv(index=False),
            file_name="flexworks_monthly_revenue_long.csv",
            mime="text/csv",
        )


def _render_visualizations(
    analyzed_data: pd.DataFrame,
    top_n: int,
    coordinate_status: CoordinateStatus,
    pjm_geojson: PjmZoneGeoJson | None,
    active_iso_filters: list[str],
    monthly_revenue: pd.DataFrame | None,
) -> None:
    st.subheader("Market Intelligence Views")
    metric_column = _render_zone_metric_selector(analyzed_data, pjm_geojson)
    zone_result = None
    zone_diagnostics = None
    if pjm_geojson is not None and metric_column is not None:
        zone_result, zone_diagnostics = build_pjm_zone_choropleth(analyzed_data, pjm_geojson, metric_column)

    map_options = ["Point map"]
    if pjm_geojson is not None and metric_column is not None:
        map_options.append("Zone choropleth")
    default_map_index = 1 if zone_diagnostics is not None and zone_diagnostics.is_available else 0
    map_mode = st.radio("Map mode", map_options, index=default_map_index, horizontal=True)

    if zone_diagnostics is not None:
        if zone_diagnostics.is_available:
            st.caption(
                "Zone choropleth highlights where market value concentrates across PJM polygons using zone-level averages from the active dataset."
            )
        _render_zone_diagnostics(zone_diagnostics)

    if map_mode == "Zone choropleth" and zone_result is not None and zone_result.figure is not None:
        st.caption("Polygon map: compares PJM zones to reveal locational value differences before deeper asset diligence.")
        _render_chart(st, zone_result.figure)
    elif map_mode == "Zone choropleth" and zone_diagnostics is not None and not zone_diagnostics.is_available:
        st.warning(_build_choropleth_unavailable_message(zone_diagnostics, active_iso_filters))
        st.caption("Point map fallback: plots individual node/device coordinates when polygon joins are unavailable.")
        _render_point_map(analyzed_data, coordinate_status)
    else:
        st.caption("Point map: shows where uploaded devices or nodes sit geographically when zone polygons are not available.")
        _render_point_map(analyzed_data, coordinate_status)

    _render_iso_zone_performance_snapshot(analyzed_data, monthly_revenue, pjm_geojson, active_iso_filters)

    col1, col2 = st.columns(2)
    bar_result = build_top_nodes_bar(analyzed_data, top_n=top_n)
    if bar_result.figure is not None:
        col1.caption("Top-node ranking surfaces priority locations for follow-up diligence.")
        col1.plotly_chart(bar_result.figure, use_container_width=True)
    else:
        col1.warning(bar_result.message)

    scatter_result = build_volatility_revenue_scatter(analyzed_data)
    if scatter_result.figure is not None:
        col2.caption("Revenue-versus-volatility view separates attractive upside from market-risk exposure.")
        col2.plotly_chart(scatter_result.figure, use_container_width=True)
    else:
        col2.warning(scatter_result.message)

    _render_pjm_cumulative_revenue_map_bars(monthly_revenue, pjm_geojson, active_iso_filters)


def _render_chart(container: object, figure: object) -> None:
    """Render either a Plotly figure or a matplotlib figure in Streamlit."""

    if isinstance(figure, go.Figure):
        container.plotly_chart(figure, use_container_width=True)
    elif hasattr(figure, "savefig"):
        container.pyplot(figure, clear_figure=False)
    else:
        container.warning("The chart could not be rendered because the figure type is unsupported.")


def _render_zone_metric_selector(analyzed_data: pd.DataFrame, pjm_geojson: PjmZoneGeoJson | None) -> str | None:
    if pjm_geojson is None:
        return None

    metric_options = [
        column
        for column in ("Revenue_per_kW", "Annualized_Revenue", "Opportunity_Score", "Risk_Adjusted_Score")
        if column in analyzed_data.columns and not pd.to_numeric(analyzed_data[column], errors="coerce").dropna().empty
    ]
    if not metric_options:
        st.warning("No numeric metric is available for a PJM zone choropleth.")
        return None

    labels = {
        "Revenue_per_kW": "Revenue per kW ($/kW)",
        "Annualized_Revenue": "Annualized Revenue",
        "Opportunity_Score": "Opportunity Score",
        "Risk_Adjusted_Score": "Risk Adjusted Score",
    }
    selected_label = st.selectbox("Zone choropleth metric", [labels[column] for column in metric_options])
    return {labels[column]: column for column in metric_options}[selected_label]


def _render_point_map(analyzed_data: pd.DataFrame, coordinate_status: CoordinateStatus) -> None:
    if coordinate_status.has_coordinates:
        st.caption(coordinate_status.message)
    else:
        st.warning(coordinate_status.message)
    map_result = build_node_map(analyzed_data)
    if map_result.figure is not None:
        st.plotly_chart(map_result.figure, use_container_width=True)
    elif map_result.message:
        st.warning(map_result.message)


def _build_choropleth_unavailable_message(
    diagnostics: ZoneJoinDiagnostics,
    active_iso_filters: list[str],
) -> str:
    iso_filters = ", ".join(active_iso_filters) if active_iso_filters else "none"
    return (
        "Zone choropleth is available only when the current filtered dataset contains PJM zone-level data. "
        "Your GeoJSON loaded correctly, but the active dataset does not contain matching PJM zones.\n\n"
        f"- Active ISO filters: {iso_filters}\n"
        f"- PJM zones found in current data: {diagnostics.matched_zone_count}\n"
        f"- GeoJSON zones loaded: {diagnostics.geojson_zone_count}"
    )


def _render_iso_zone_performance_snapshot(
    analyzed_data: pd.DataFrame,
    monthly_revenue: pd.DataFrame | None,
    pjm_geojson: PjmZoneGeoJson | None,
    active_iso_filters: list[str],
) -> None:
    st.subheader("Zonal Market Performance")
    st.caption(
        "Compare how battery arbitrage value moves across zones, categories, and time so location strategy is visible at a glance."
    )

    if monthly_revenue is None or monthly_revenue.empty:
        st.info("ISO zone performance snapshots require time-series revenue data. Upload a FlexWorks monthly revenue export.")
        return

    filtered_monthly = monthly_revenue.copy()
    if active_iso_filters and "ISO_Region" in filtered_monthly.columns:
        filtered_monthly = filtered_monthly.loc[filtered_monthly["ISO_Region"].astype(str).isin(active_iso_filters)].copy()

    iso_options = _available_iso_regions(filtered_monthly)
    if not iso_options:
        st.warning("No ISO/RTO values are available in the active time-series dataset.")
        st.caption(f"Active ISO filters: {', '.join(active_iso_filters) if active_iso_filters else 'none'}")
        return

    default_iso_index = iso_options.index("PJM") if "PJM" in iso_options else 0
    col1, col2, col3, col4 = st.columns(4)
    selected_iso = col1.selectbox("ISO/RTO", iso_options, index=default_iso_index, key="iso_snapshot_iso")

    iso_monthly = filtered_monthly.loc[filtered_monthly["ISO_Region"].astype(str) == selected_iso].copy() if "ISO_Region" in filtered_monthly.columns else filtered_monthly.copy()
    if iso_monthly.empty:
        st.warning("No time-series rows are available for the selected ISO/RTO and active filters.")
        return

    metric_options = [SNAPSHOT_METRIC_MONTHLY_REVENUE, SNAPSHOT_METRIC_CUMULATIVE_REVENUE]
    if "Revenue_per_kW" in iso_monthly.columns and not pd.to_numeric(iso_monthly["Revenue_per_kW"], errors="coerce").dropna().empty:
        metric_options.append(SNAPSHOT_METRIC_REVENUE_PER_KW)
    view_mode = col2.radio("Mode", ["Snapshot", "Time range", "Multi-snapshot", "Animation"], horizontal=True, key="iso_time_view_mode")
    selected_metric = col3.selectbox("Metric", metric_options, key="iso_snapshot_metric")
    revenue_category = col4.selectbox("Revenue category", _monthly_category_options(iso_monthly), key="iso_snapshot_category")

    category_filtered = iso_monthly.copy()
    if revenue_category != ALL_REVENUE_CATEGORIES and "Revenue_Category" in category_filtered.columns:
        category_filtered = category_filtered.loc[category_filtered["Revenue_Category"].astype(str) == revenue_category].copy()

    granularity = detect_time_granularity(category_filtered)
    if granularity == TIME_GRANULARITY_NONE:
        st.warning("Snapshot mode requires a Month or Timestamp column. This dataset does not contain usable time data.")
        return

    time_points = available_time_points(category_filtered)
    if not time_points:
        st.warning("No valid time points are available for the selected ISO/RTO and category.")
        return

    time_labels = [format_time_label(time_point, granularity) for time_point in time_points]
    time_by_label = dict(zip(time_labels, time_points))
    time_control_label = "Month" if granularity == TIME_GRANULARITY_MONTHLY else "Timestamp"

    if view_mode == "Snapshot":
        selected_time_label = st.select_slider(time_control_label, options=time_labels, value=time_labels[-1], key="iso_snapshot_time")
        selected_time = time_by_label[selected_time_label]
        zone_values = aggregate_zone_metric(
            iso_monthly,
            metric=selected_metric,
            category=revenue_category,
            time_point=selected_time,
            iso_region=selected_iso,
        )
        time_label = selected_time_label
        time_context_label = "Selected time"
        empty_message = "The selected time has no zone-level data for the selected metric and category. Pick another valid time point or broaden the category filter."
        compact = False
    elif view_mode == "Time range":
        range_col1, range_col2 = st.columns(2)
        start_label = range_col1.selectbox(f"Start {time_control_label}", time_labels, index=0, key="iso_range_start")
        start_index = time_labels.index(start_label)
        valid_end_labels = time_labels[start_index:]
        end_label = range_col2.selectbox(
            f"End {time_control_label}",
            valid_end_labels,
            index=len(valid_end_labels) - 1,
            key="iso_range_end",
        )
        start_time = time_by_label[start_label]
        end_time = time_by_label[end_label]
        if start_time > end_time:
            st.warning("Start time must be before or equal to end time. Choose a later end time to continue.")
            return
        zone_values = aggregate_zone_metric_over_range(
            iso_monthly,
            metric=selected_metric,
            category=revenue_category,
            start_time=start_time,
            end_time=end_time,
            iso_region=selected_iso,
        )
        time_label = format_time_range_label(start_time, end_time, granularity)
        time_context_label = "Selected range"
        empty_message = "The selected time range has no zone-level data for the selected metric and category. Widen the range or choose All categories."
        compact = False
    elif view_mode == "Multi-snapshot":
        multi_col1, multi_col2, multi_col3 = st.columns(3)
        start_label = multi_col1.selectbox(f"Start {time_control_label}", time_labels, index=0, key="iso_multi_start")
        start_index = time_labels.index(start_label)
        valid_end_labels = time_labels[start_index:]
        end_label = multi_col2.selectbox(
            f"End {time_control_label}",
            valid_end_labels,
            index=len(valid_end_labels) - 1,
            key="iso_multi_end",
        )
        max_snapshots = min(MAX_MULTI_SNAPSHOTS, len(valid_end_labels))
        requested_snapshots = multi_col3.slider(
            "Snapshots",
            min_value=1,
            max_value=max_snapshots,
            value=min(4, max_snapshots),
            step=1,
            key="iso_multi_snapshot_count",
        )
        start_time = time_by_label[start_label]
        end_time = time_by_label[end_label]
        if start_time > end_time:
            st.warning("Start time must be before or equal to end time. Choose a later end time to continue.")
            return
        selected_times = select_evenly_spaced_snapshots(category_filtered, start_time, end_time, requested_snapshots)
        if not selected_times:
            st.warning("The selected range has no valid time points for multi-snapshot mode.")
            return
        if selected_iso != "PJM":
            st.warning(f"{selected_iso} zone polygons are not configured yet. Multi-snapshot mode requires zone polygons.")
            return
        if pjm_geojson is None:
            st.warning("PJM zone map requires the PJM GeoJSON file for multi-snapshot mode.")
            return

        st.caption(
            f"Showing {len(selected_times)} evenly spaced {time_control_label.lower()} snapshot(s) from "
            f"{format_time_range_label(start_time, end_time, granularity)} to show whether leadership is persistent or episodic."
        )
        export_frames: list[pd.DataFrame] = []
        snapshot_figures: list[object] = []
        snapshot_names: list[str] = []
        for selected_time in selected_times:
            selected_time_label = format_time_label(selected_time, granularity)
            zone_values = aggregate_zone_metric(
                iso_monthly,
                metric=selected_metric,
                category=revenue_category,
                time_point=selected_time,
                iso_region=selected_iso,
            )
            if zone_values.empty:
                st.warning(f"No zone-level data is available for {selected_time_label}.")
                continue
            export_frame = zone_values.copy()
            export_frame["Export_Frame"] = selected_time_label
            export_frames.append(export_frame)
            chart_result, diagnostics = create_pjm_matplotlib_figure(
                zone_values,
                pjm_geojson=pjm_geojson,
                metric="Selected_Metric",
                metric_label=selected_metric,
                time_selection=selected_time_label,
                category_label=revenue_category,
                time_context_label="Selected time",
                compact=True,
            )
            if chart_result.figure is not None:
                _render_chart(st, chart_result.figure)
                snapshot_figures.append(chart_result.figure)
                snapshot_names.append(f"{_export_file_stem(selected_iso, selected_metric, selected_time_label, 'multi_snapshot')}.png")
            else:
                st.warning(chart_result.message)
                _render_snapshot_join_diagnostics(diagnostics)
        if snapshot_figures:
            st.download_button(
                "Download multi-snapshot PNGs",
                data=matplotlib_figures_to_zip_bytes(snapshot_figures, snapshot_names),
                file_name=f"{_export_file_stem(selected_iso, selected_metric, format_time_range_label(start_time, end_time, granularity), 'multi_snapshot')}.zip",
                mime="application/zip",
            )
        if export_frames:
            _render_iso_export_report(
                zone_data=pd.concat(export_frames, ignore_index=True),
                figures=[],
                selected_iso=selected_iso,
                selected_metric=selected_metric,
                selected_period=format_time_range_label(start_time, end_time, granularity),
                mode_label="multi_snapshot",
            )
        return
    else:
        animation_col1, animation_col2, animation_col3 = st.columns(3)
        start_label = animation_col1.selectbox(f"Start {time_control_label}", time_labels, index=0, key="iso_animation_start")
        start_index = time_labels.index(start_label)
        valid_end_labels = time_labels[start_index:]
        end_label = animation_col2.selectbox(
            f"End {time_control_label}",
            valid_end_labels,
            index=len(valid_end_labels) - 1,
            key="iso_animation_end",
        )
        start_time = time_by_label[start_label]
        end_time = time_by_label[end_label]
        if start_time > end_time:
            st.warning("Start time must be before or equal to end time. Choose a later end time to continue.")
            return
        default_key_frames = default_frame_count_for_range(category_filtered, start_time, end_time, MAX_ANIMATION_FRAMES)
        if default_key_frames <= 0:
            st.warning("The selected range has no valid time points for animation.")
            return
        requested_frames = animation_col3.slider(
            "Monthly/key frames",
            min_value=1,
            max_value=default_key_frames,
            value=default_key_frames,
            step=1,
            key="iso_animation_frame_count",
            help=(
                f"Uses every available {time_control_label.lower()} in the selected range when it fits under "
                f"the {MAX_ANIMATION_FRAMES}-frame cap; otherwise it samples evenly."
            ),
        )
        selected_times = select_evenly_spaced_snapshots(category_filtered, start_time, end_time, requested_frames)
        if not selected_times:
            st.warning("The selected range has no valid time points for animation.")
            return
        if selected_iso != "PJM":
            st.warning(f"{selected_iso} zone polygons are not configured yet. Animation mode requires zone polygons.")
            return
        if pjm_geojson is None:
            st.warning("PJM zone map requires the PJM GeoJSON file for animation mode.")
            return

        frame_labels = [format_time_label(selected_time, granularity) for selected_time in selected_times]
        if len(frame_labels) == 1:
            st.caption("Only one valid time point is selected, so playback contains a single frame.")
        st.caption("Animation renders with matplotlib map frames for reliable zone styling; GIF download remains available.")

        cache_key = _animation_gif_cache_key(
            iso_monthly,
            selected_iso=selected_iso,
            selected_metric=selected_metric,
            revenue_category=revenue_category,
            start_label=start_label,
            end_label=end_label,
            requested_frames=requested_frames,
        )
        animation_cache = st.session_state.setdefault("pjm_animation_gif_cache", {})
        gif_result = animation_cache.get(cache_key)
        if gif_result is None:
            progress = st.progress(0)
            with st.status("Rendering PJM animation GIF...", expanded=True) as status:
                st.write("Building matplotlib map frames...")
                gif_result = create_pjm_animation_gif_bytes(
                    category_filtered,
                    pjm_geojson,
                    metric=selected_metric,
                    category=revenue_category,
                    start_time=start_time,
                    end_time=end_time,
                    frame_count=requested_frames,
                    iso_region=selected_iso,
                    progress_callback=lambda value: progress.progress(min(max(float(value), 0.0), 1.0)),
                )
                if gif_result.gif_bytes is not None:
                    animation_cache[cache_key] = gif_result
                    status.update(label="Animation GIF ready.", state="complete", expanded=False)
                else:
                    status.update(label="Animation GIF could not be rendered.", state="error", expanded=True)
            progress.empty()

        diagnostics = gif_result.diagnostics
        if gif_result.gif_bytes is not None:
            st.caption(
                f"Animation uses {len(gif_result.frame_labels)} {time_control_label.lower()} key frame(s) from "
                f"{format_time_range_label(start_time, end_time, granularity)}, with interpolated transition frames for smoother playback."
            )
            player_rendered = False
            try:
                player_html = animation_frames_to_html_player(
                    gif_result.frame_png_bytes,
                    gif_result.rendered_frame_labels,
                    "PJM zone performance animation",
                )
                if player_html:
                    components.html(player_html, height=850, scrolling=False)
                    st.caption("Use the playback controls to pause or scrub through the selected period.")
                    player_rendered = True
            except Exception as exc:
                st.warning(f"Interactive animation controls could not be rendered. Showing GIF fallback instead. {exc}")
            if not player_rendered:
                st.markdown(
                    gif_bytes_to_html_img(gif_result.gif_bytes, "PJM zone performance animation"),
                    unsafe_allow_html=True,
                )
                st.caption("GIF preview loops in-browser using the same matplotlib map styling as the static views.")
            st.download_button(
                "Download animated GIF",
                data=gif_result.gif_bytes,
                file_name=f"{_export_file_stem(selected_iso, selected_metric, format_time_range_label(start_time, end_time, granularity), 'animation')}.gif",
                mime="image/gif",
            )
            export_frames = []
            for frame_label, frame_data in zip(gif_result.frame_labels, gif_result.frame_dataframes):
                if frame_data.empty:
                    continue
                export_frame = frame_data.copy()
                export_frame["Export_Frame"] = frame_label
                export_frames.append(export_frame)
            if export_frames:
                _render_iso_export_report(
                    zone_data=pd.concat(export_frames, ignore_index=True),
                    figures=[],
                    selected_iso=selected_iso,
                    selected_metric=selected_metric,
                    selected_period=format_time_range_label(start_time, end_time, granularity),
                    mode_label="animation",
                )
        else:
            st.warning(gif_result.message)
            _render_snapshot_join_diagnostics(diagnostics)
        return

    if zone_values.empty:
        st.warning(empty_message)
        return

    if selected_iso != "PJM":
        st.warning(f"{selected_iso} zone polygons are not configured yet. Falling back to the point map when coordinates are available.")
        selected_iso_nodes = _filter_by_iso(analyzed_data, [selected_iso])
        _render_point_map(selected_iso_nodes, detect_coordinate_status(selected_iso_nodes))
        return
    if pjm_geojson is None:
        st.warning("PJM zone map requires the PJM GeoJSON file. Falling back to the point map when coordinates are available.")
        selected_iso_nodes = _filter_by_iso(analyzed_data, [selected_iso])
        _render_point_map(selected_iso_nodes, detect_coordinate_status(selected_iso_nodes))
        return

    chart_result, diagnostics = create_pjm_matplotlib_figure(
        zone_values,
        pjm_geojson=pjm_geojson,
        metric="Selected_Metric",
        metric_label=selected_metric,
        time_selection=time_label,
        category_label=revenue_category,
        time_context_label=time_context_label,
        compact=compact,
    )
    if chart_result.figure is not None:
        _render_chart(st, chart_result.figure)
        download_label = "Download snapshot PNG" if view_mode == "Snapshot" else "Download time range PNG"
        st.download_button(
            download_label,
            data=matplotlib_figure_to_png_bytes(chart_result.figure),
            file_name=f"{_export_file_stem(selected_iso, selected_metric, time_label, view_mode.lower().replace(' ', '_'))}.png",
            mime="image/png",
        )
        _render_iso_export_report(
            zone_data=zone_values,
            figures=[],
            selected_iso=selected_iso,
            selected_metric=selected_metric,
            selected_period=time_label,
            mode_label=view_mode.lower().replace(" ", "_"),
        )
    else:
        st.warning(chart_result.message)
        selected_iso_nodes = _filter_by_iso(analyzed_data, [selected_iso])
        _render_point_map(selected_iso_nodes, detect_coordinate_status(selected_iso_nodes))

    _render_snapshot_join_diagnostics(diagnostics)


def _render_iso_export_report(
    *,
    zone_data: pd.DataFrame,
    figures: list[go.Figure],
    selected_iso: str,
    selected_metric: str,
    selected_period: str,
    mode_label: str,
) -> None:
    st.subheader("Strategy Export Center")
    if zone_data.empty:
        st.info("No processed zone performance data is available to export.")
        return

    summary = build_executive_summary(
        zone_data,
        selected_iso=selected_iso,
        selected_metric=selected_metric,
        selected_period=selected_period,
    )
    with st.expander("Executive summary preview", expanded=True):
        st.markdown(summary.markdown)

    file_stem = _export_file_stem(selected_iso, selected_metric, selected_period, mode_label)
    data_col, html_col, png_col, md_col, txt_col = st.columns(5)
    data_col.download_button(
        "Zone CSV",
        data=export_dataframe_csv(zone_data),
        file_name=f"{file_stem}_zone_data.csv",
        mime="text/csv",
        key=f"{file_stem}_zone_csv",
    )

    if figures:
        html_bytes = (
            plotly_figure_to_html_bytes(figures[0])
            if len(figures) == 1
            else plotly_figures_to_html_bytes(figures, title="FlexWorks ISO Zone Performance")
        )
        html_col.download_button(
            "Plotly HTML",
            data=html_bytes,
            file_name=f"{file_stem}_figure.html",
            mime="text/html",
            key=f"{file_stem}_plotly_html",
        )

        if len(figures) == 1:
            png_bytes, png_message = safe_plotly_png_bytes(figures[0])
            if png_bytes is not None:
                png_col.download_button(
                    "PNG",
                    data=png_bytes,
                    file_name=f"{file_stem}_figure.png",
                    mime="image/png",
                    key=f"{file_stem}_plotly_png",
                )
            elif png_message:
                png_col.warning(png_message)
        else:
            png_col.warning("PNG export is available for single-figure modes. HTML export is still available.")
    else:
        html_col.info("Interactive HTML visual export is available for Plotly animation. Static PJM maps render with matplotlib.")

    md_col.download_button(
        "Summary MD",
        data=summary.markdown.encode("utf-8"),
        file_name=f"{file_stem}_summary.md",
        mime="text/markdown",
        key=f"{file_stem}_summary_md",
    )
    txt_col.download_button(
        "Summary TXT",
        data=summary.text.encode("utf-8"),
        file_name=f"{file_stem}_summary.txt",
        mime="text/plain",
        key=f"{file_stem}_summary_txt",
    )


def _render_pjm_cumulative_revenue_map_bars(
    monthly_revenue: pd.DataFrame | None,
    pjm_geojson: PjmZoneGeoJson | None,
    active_iso_filters: list[str],
) -> None:
    st.subheader("PJM Cumulative Revenue Map + Bars")

    if monthly_revenue is None or monthly_revenue.empty:
        st.info("Cumulative revenue visualization requires a FlexWorks monthly revenue export.")
        return
    if pjm_geojson is None:
        st.warning("PJM zone map requires the PJM GeoJSON file.")
        return

    filtered_monthly = monthly_revenue.copy()
    if active_iso_filters and "ISO_Region" in filtered_monthly.columns:
        filtered_monthly = filtered_monthly.loc[filtered_monthly["ISO_Region"].astype(str).isin(active_iso_filters)].copy()

    category_options = _monthly_category_options(filtered_monthly)
    col1, col2, col3, col4 = st.columns(4)
    revenue_category = col1.selectbox("Revenue category", category_options, key="pjm_cumulative_category")
    metric_label = col2.radio(
        "Metric",
        ["Cumulative Revenue", "Monthly Revenue"],
        horizontal=True,
        key="pjm_cumulative_metric",
    )
    sort_order = col3.selectbox("Bars", ["Top zones", "Bottom zones"], key="pjm_cumulative_sort")

    zone_monthly = compute_zone_monthly_revenue(filtered_monthly, revenue_category=revenue_category)
    if zone_monthly.empty:
        st.warning(
            "No PJM monthly revenue rows are available for the current filters and category selection."
        )
        st.caption(f"Active ISO filters: {', '.join(active_iso_filters) if active_iso_filters else 'none'}")
        st.caption(f"GeoJSON zones loaded: {pjm_geojson.zone_count}")
        return

    months = sorted(pd.to_datetime(zone_monthly["Month"]).dropna().unique())
    month_options = [pd.Timestamp(month).strftime("%Y-%m") for month in months]
    selected_month_label = col4.select_slider("Month", options=month_options, value=month_options[-1])
    selected_month_data = filter_zone_revenue_to_month(zone_monthly, selected_month_label)
    metric_column = "Monthly_Revenue" if metric_label == "Monthly Revenue" else "Cumulative_Revenue"
    chart_result, diagnostics = build_pjm_cumulative_revenue_map_bars(
        selected_month_data,
        pjm_geojson,
        metric_column=metric_column,
        sort_order=sort_order,
    )

    if chart_result.figure is not None:
        st.caption(
            "Revenue is aggregated by PJM zone and month. Cumulative revenue sums monthly revenue from the first available month through the selected month."
        )
        _render_chart(st, chart_result.figure)
    else:
        st.warning(chart_result.message)

    _render_cumulative_revenue_diagnostics(diagnostics)


def _monthly_category_options(monthly_revenue: pd.DataFrame) -> list[str]:
    preferred_order = ["Energy", "Ancillary", "FCP"]
    if "Revenue_Category" not in monthly_revenue.columns:
        return [ALL_REVENUE_CATEGORIES]

    available = monthly_revenue["Revenue_Category"].dropna().astype(str).unique().tolist()
    ordered = [category for category in preferred_order if category in available]
    ordered.extend(sorted(category for category in available if category not in ordered))
    return [ALL_REVENUE_CATEGORIES, *ordered]


def _animation_gif_cache_key(
    dataframe: pd.DataFrame,
    *,
    selected_iso: str,
    selected_metric: str,
    revenue_category: str,
    start_label: str,
    end_label: str,
    requested_frames: int,
) -> tuple[object, ...]:
    revenue_total = None
    if "Revenue" in dataframe.columns:
        revenue_total = round(float(pd.to_numeric(dataframe["Revenue"], errors="coerce").fillna(0.0).sum()), 4)
    return (
        selected_iso,
        selected_metric,
        revenue_category,
        start_label,
        end_label,
        int(requested_frames),
        len(dataframe),
        tuple(dataframe.columns),
        revenue_total,
    )


def _render_cumulative_revenue_diagnostics(diagnostics: dict[str, object]) -> None:
    unmatched_revenue_zones = diagnostics.get("unmatched_revenue_zones") or []
    unmatched_geojson_zones = diagnostics.get("unmatched_geojson_zones") or []
    expanded = bool(unmatched_revenue_zones)
    with st.expander("PJM cumulative revenue join diagnostics", expanded=expanded):
        col1, col2, col3 = st.columns(3)
        col1.metric("GeoJSON zones", diagnostics.get("geojson_zone_count", 0))
        col2.metric("Revenue zones", diagnostics.get("revenue_zone_count", 0))
        col3.metric("Matched zones", diagnostics.get("matched_zone_count", 0))
        if unmatched_revenue_zones:
            st.write("Unmatched revenue zones:")
            st.write(", ".join(str(zone) for zone in unmatched_revenue_zones))
        if unmatched_geojson_zones:
            st.write("GeoJSON zones without revenue in selected month:")
            st.write(", ".join(str(zone) for zone in unmatched_geojson_zones))


def _render_snapshot_join_diagnostics(diagnostics: dict[str, object]) -> None:
    unmatched_revenue_zones = diagnostics.get("unmatched_revenue_zones") or []
    unmatched_geojson_zones = diagnostics.get("unmatched_geojson_zones") or []
    expanded = bool(unmatched_revenue_zones)
    with st.expander("PJM snapshot join diagnostics", expanded=expanded):
        col1, col2, col3 = st.columns(3)
        col1.metric("GeoJSON zones", diagnostics.get("geojson_zone_count", 0))
        col2.metric("Snapshot zones", diagnostics.get("revenue_zone_count", 0))
        col3.metric("Matched zones", diagnostics.get("matched_zone_count", 0))
        if unmatched_revenue_zones:
            st.write("Unmatched snapshot zones:")
            st.write(", ".join(str(zone) for zone in unmatched_revenue_zones))
        if unmatched_geojson_zones:
            st.write("GeoJSON zones without data at selected time:")
            st.write(", ".join(str(zone) for zone in unmatched_geojson_zones))


def _render_zone_diagnostics(diagnostics: ZoneJoinDiagnostics) -> None:
    with st.expander("PJM zone join diagnostics", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("GeoJSON zones", diagnostics.geojson_zone_count)
        col2.metric("FlexWorks zones", diagnostics.flexworks_zone_count)
        col3.metric("Matched zones", diagnostics.matched_zone_count)
        st.write(f"GeoJSON zone property: `{diagnostics.zone_property or 'n/a'}`")
        st.write(f"FlexWorks zone column: `{diagnostics.data_zone_column or 'n/a'}`")
        if diagnostics.unmatched_flexworks_zones:
            st.write("Unmatched FlexWorks zones:")
            st.write(", ".join(diagnostics.unmatched_flexworks_zones))
        if diagnostics.unmatched_geojson_zones:
            st.write("Unmatched GeoJSON zones:")
            st.write(", ".join(diagnostics.unmatched_geojson_zones))


def _render_tables(ranked_nodes: pd.DataFrame, iso_summary: pd.DataFrame, high_risk_high_reward: pd.DataFrame) -> None:
    st.subheader("Tables")
    tab1, tab2, tab3 = st.tabs(["Top Nodes", "ISO Summary", "High Risk / High Reward"])
    tab1.dataframe(_display_columns(ranked_nodes), use_container_width=True, hide_index=True)

    if iso_summary.empty:
        tab2.info("No ISO summary is available.")
    else:
        tab2.dataframe(iso_summary, use_container_width=True, hide_index=True)

    if high_risk_high_reward.empty:
        tab3.info("No nodes meet the high-risk/high-reward screen.")
    else:
        tab3.dataframe(_display_columns(high_risk_high_reward), use_container_width=True, hide_index=True)


def _render_monthly_revenue_section(monthly_revenue: pd.DataFrame | None, notes: list[str] | None = None) -> None:
    if monthly_revenue is None or monthly_revenue.empty:
        return

    st.subheader("Monthly Revenue")
    st.caption("Use monthly revenue views to trace whether value comes from persistent zonal advantage or isolated market events.")
    for note in notes or []:
        st.caption(note)

    filtered = monthly_revenue.copy()
    devices = _sorted_unique(filtered, "Device")
    categories = _sorted_unique(filtered, "Revenue_Category")
    zones = _sorted_unique(filtered, "Zone")

    col1, col2, col3, col4 = st.columns(4)
    selected_devices = col1.multiselect("Devices", devices, default=devices, key="monthly_devices") if devices else []
    selected_categories = (
        col2.multiselect("Categories", categories, default=categories, key="monthly_categories") if categories else []
    )
    selected_zones = col3.multiselect("Zones", zones, default=zones, key="monthly_zones") if zones else []
    group_options = ["Revenue_Category", "Device"] + (["Zone"] if zones else [])
    group_by = col4.selectbox("Group monthly chart by", group_options, key="monthly_group_by")

    if selected_devices:
        filtered = filtered.loc[filtered["Device"].astype(str).isin(selected_devices)]
    if selected_categories:
        filtered = filtered.loc[filtered["Revenue_Category"].astype(str).isin(selected_categories)]
    if selected_zones and "Zone" in filtered.columns:
        filtered = filtered.loc[filtered["Zone"].astype(str).isin(selected_zones)]

    if filtered.empty:
        st.warning("The monthly revenue filters returned zero rows.")
        return

    line_result = build_monthly_revenue_chart(filtered, group_by=group_by)
    bar_result = build_monthly_revenue_bar(filtered, group_by=group_by)
    chart_col1, chart_col2 = st.columns(2)
    if line_result.figure is not None:
        chart_col1.plotly_chart(line_result.figure, use_container_width=True)
    else:
        chart_col1.warning(line_result.message)
    if bar_result.figure is not None:
        chart_col2.plotly_chart(bar_result.figure, use_container_width=True)
    else:
        chart_col2.warning(bar_result.message)

    with st.expander("Monthly revenue long-format data"):
        st.dataframe(_display_monthly_columns(filtered).head(500), use_container_width=True, hide_index=True)


def _render_report(report: str) -> None:
    st.subheader("Report Preview")
    st.markdown(report)


def _render_cleaning_summary(cleaning_summary: CleaningSummary) -> None:
    with st.expander("Data Quality Notes"):
        for note in cleaning_summary.notes:
            st.write(f"- {note}")
        st.json(cleaning_summary.to_dict())


def _display_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Rank",
        "Node_ID",
        "Device_ID",
        "Device",
        "ISO_Region",
        "Zone",
        "Node_Name",
        "Opportunity_Score",
        "Annualized_Revenue",
        "Revenue_per_kW",
        "LMP_Volatility",
        "Risk_Label",
        "Latitude",
        "Longitude",
    ]
    return dataframe[[column for column in columns if column in dataframe.columns]].copy()


def _display_monthly_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Device",
        "Node_ID",
        "Zone",
        "ISO_Region",
        "Revenue_Category",
        "Month",
        "Revenue",
        "Annualized_Revenue",
        "Revenue_per_kW",
    ]
    return dataframe[[column for column in columns if column in dataframe.columns]].copy()


def _sorted_unique(dataframe: pd.DataFrame, column: str) -> list[str]:
    if column not in dataframe.columns:
        return []
    values = dataframe[column].dropna().astype(str).unique().tolist()
    return sorted(values)


def _export_file_stem(selected_iso: str, selected_metric: str, selected_period: str, mode_label: str) -> str:
    raw = f"flexworks_{selected_iso}_{mode_label}_{selected_metric}_{selected_period}".lower()
    cleaned = "".join(character if character.isalnum() else "_" for character in raw)
    return "_".join(part for part in cleaned.split("_") if part)


def _fmt_number(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    return f"{number:,.2f}"


def _fmt_dollars_per_kw(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    return f"${number:,.2f}/kW"


def _fmt_metric_value(value: object, metric_column: str) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    if metric_column == "Revenue_per_kW":
        return f"${number:,.2f}/kW"
    if "Revenue" in metric_column:
        return f"${number:,.0f}"
    return f"{number:,.2f}"


if __name__ == "__main__":
    main()
