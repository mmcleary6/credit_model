from __future__ import annotations

import os
import sys
from pathlib import Path
import random
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from asset_modeling.credit import loan_portfolio
from data.sofr import get_sofr_data

REQUIRED_COLUMNS = [
    "investment_name",
    "investment_date",
    "maturity_date",
    "par_value",
    "spread",
    "base_rate",
    "sofr_assumption",
    "pik_interest",
    "amortization",
    "oid",
    "exit_fee",
    "prepayment_date",
]

NUMERIC_COLUMNS = [
    "par_value",
    "spread",
    "pik_interest",
    "amortization",
    "oid",
    "exit_fee",
]

LINE_BLUE = "#2563eb"
LIGHT_BLUE = "#7dd3fc"
FRED_API_KEY = os.getenv("FRED_API_KEY")
SOFR_RATES = pd.DataFrame()


def _default_row(index: int) -> dict:
    return {
        "investment_name": f"Loan {index}",
        "investment_date": "2020-12-31",
        "maturity_date": "2030-12-31",
        "par_value": 15000000,
        "spread": 0.06,
        "base_rate": "SOFR",
        "sofr_assumption": 0.04,
        "pik_interest": 0.02,
        "amortization": 0.01,
        "oid": 0.02,
        "exit_fee": 0.02,
        "prepayment_date": "",
    }


def _random_quarter_end(
    random_generator: random.Random,
    start_date: str,
    end_date: str,
) -> pd.Timestamp:
    quarter_ends = pd.date_range(start=start_date, end=end_date, freq="QE")
    return random_generator.choice(list(quarter_ends))


def default_schedule() -> pd.DataFrame:
    random_generator = random.Random()
    rows = []

    for index in range(1, 26):
        row = _default_row(index)
        investment_date = _random_quarter_end(
            random_generator,
            "2020-12-31",
            "2024-12-31",
        )
        maturity_date = _random_quarter_end(
            random_generator,
            "2030-12-31",
            "2035-12-31",
        )
        row["investment_date"] = investment_date.strftime("%Y-%m-%d")
        row["maturity_date"] = maturity_date.strftime("%Y-%m-%d")
        row["par_value"] = int(
            round(random_generator.uniform(15000000, 100000000) / 1_000_000)
            * 1_000_000
        )
        row["oid"] = round(random_generator.uniform(0.02, 0.05), 4)
        row["sofr_assumption"] = round(random_generator.uniform(0.035, 0.04), 4)
        row["spread"] = round(random_generator.uniform(0.08, 0.11), 4)
        rows.append(row)

    return pd.DataFrame(rows)


def _sort_schedule_by_investment_date(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = (
        df.assign(_investment_date_sort=pd.to_datetime(df["investment_date"]))
        .sort_values("_investment_date_sort", kind="stable")
        .drop(columns="_investment_date_sort")
        .reset_index(drop=True)
    )
    sorted_df["investment_name"] = [f"Loan {i + 1}" for i in range(len(sorted_df))]
    return sorted_df



def _normalize_numeric_value(value: object):
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "")
        if cleaned == "":
            return None
        if cleaned.endswith("%"):
            percentage_text = cleaned[:-1].strip()
            try:
                return float(percentage_text) / 100.0
            except ValueError:
                return value
        return cleaned
    return value


def _coerce_sofr_assumption(value: object) -> str | float:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            raise ValueError(
                "Column 'sofr_assumption' must be numeric or the literal 'actual'."
            )
        if cleaned.lower() == "actual":
            return "actual"
        normalized_value = _normalize_numeric_value(cleaned)
    else:
        normalized_value = value

    try:
        return float(normalized_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Column 'sofr_assumption' must be numeric or the literal 'actual'."
        ) from exc


def _schedule_uses_actual_sofr(df: pd.DataFrame) -> bool:
    return df["sofr_assumption"].apply(
        lambda value: isinstance(value, str) and value.lower() == "actual"
    ).any()


def _first_assumed_sofr_date(sofr_df: pd.DataFrame) -> str | None:
    if sofr_df.empty or "date" not in sofr_df.columns or "rate_status" not in sofr_df.columns:
        return None

    working_df = sofr_df.copy()
    working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce")
    assumed_dates = working_df.loc[
        working_df["rate_status"].astype(str).str.lower() == "assumed",
        "date",
    ].dropna()

    if assumed_dates.empty:
        return None

    return assumed_dates.min().strftime("%Y-%m-%d")


def _apply_rate_shock_to_assumed_sofr(
    sofr_df: pd.DataFrame,
    rate_shock_bps: int | float,
    shock_start_date: object | None,
) -> pd.DataFrame:
    if sofr_df.empty or "sofr" not in sofr_df.columns or "rate_status" not in sofr_df.columns:
        return sofr_df

    if not rate_shock_bps:
        return sofr_df

    shocked_df = sofr_df.copy()
    shocked_df["date"] = pd.to_datetime(shocked_df["date"], errors="coerce")
    shocked_df["sofr"] = pd.to_numeric(shocked_df["sofr"], errors="coerce")
    assumed_mask = shocked_df["rate_status"].astype(str).str.lower() == "assumed"

    if shock_start_date is not None:
        shock_start_timestamp = pd.Timestamp(shock_start_date)
        assumed_mask = assumed_mask & (shocked_df["date"] >= shock_start_timestamp)

    shocked_df.loc[assumed_mask, "sofr"] = (
        shocked_df.loc[assumed_mask, "sofr"] + (float(rate_shock_bps) / 10_000.0)
    )
    return shocked_df


def _apply_rate_shock_to_schedule(
    schedule_df: pd.DataFrame,
    rate_shock_bps: int | float,
    shock_start_date: object | None,
) -> pd.DataFrame:
    if schedule_df.empty or "sofr_assumption" not in schedule_df.columns:
        return schedule_df

    if not rate_shock_bps:
        return schedule_df

    shocked_schedule = schedule_df.copy()
    shock_decimal = float(rate_shock_bps) / 10_000.0
    numeric_mask = shocked_schedule["sofr_assumption"].apply(
        lambda value: not (isinstance(value, str) and value.lower() == "actual")
    )

    if shock_start_date is not None and "maturity_date" in shocked_schedule.columns:
        shock_start_timestamp = pd.Timestamp(shock_start_date)
        numeric_mask = numeric_mask & (
            pd.to_datetime(shocked_schedule["maturity_date"], errors="coerce")
            >= shock_start_timestamp
        )

    shocked_schedule.loc[numeric_mask, "sofr_assumption"] = (
        pd.to_numeric(shocked_schedule.loc[numeric_mask, "sofr_assumption"], errors="coerce")
        + shock_decimal
    )
    return shocked_schedule


def _build_linear_rate_change_factors(
    date_series: pd.Series,
    change_start_date: object | None,
) -> pd.Series:
    working_dates = pd.to_datetime(date_series, errors="coerce")
    factors = pd.Series(0.0, index=date_series.index, dtype=float)

    valid_dates = working_dates.dropna()
    if valid_dates.empty:
        return factors

    if change_start_date is None:
        start_timestamp = valid_dates.min()
    else:
        start_timestamp = pd.Timestamp(change_start_date)

    eligible_mask = working_dates >= start_timestamp
    eligible_dates = working_dates.loc[eligible_mask].dropna()
    if eligible_dates.empty:
        return factors

    end_timestamp = eligible_dates.max()
    total_days = (end_timestamp - start_timestamp).days

    if total_days <= 0:
        factors.loc[eligible_mask] = 1.0
        return factors

    factors.loc[eligible_mask] = (
        (working_dates.loc[eligible_mask] - start_timestamp).dt.days / total_days
    ).clip(lower=0.0, upper=1.0)
    return factors


def _apply_linear_rate_change_to_assumed_sofr(
    sofr_df: pd.DataFrame,
    rate_change_bps: int | float,
    change_start_date: object | None,
) -> pd.DataFrame:
    if sofr_df.empty or "sofr" not in sofr_df.columns or "rate_status" not in sofr_df.columns:
        return sofr_df

    if not rate_change_bps:
        return sofr_df

    changed_df = sofr_df.copy()
    changed_df["date"] = pd.to_datetime(changed_df["date"], errors="coerce")
    changed_df["sofr"] = pd.to_numeric(changed_df["sofr"], errors="coerce")
    assumed_mask = changed_df["rate_status"].astype(str).str.lower() == "assumed"
    assumed_dates = changed_df.loc[assumed_mask, "date"]
    factors = _build_linear_rate_change_factors(assumed_dates, change_start_date)
    delta = (float(rate_change_bps) / 10_000.0) * factors
    changed_df.loc[assumed_mask, "sofr"] = (
        changed_df.loc[assumed_mask, "sofr"] + delta
    )
    return changed_df


def _apply_linear_rate_change_to_schedule(
    schedule_df: pd.DataFrame,
    rate_change_bps: int | float,
    change_start_date: object | None,
) -> pd.DataFrame:
    if schedule_df.empty or "sofr_assumption" not in schedule_df.columns:
        return schedule_df

    if not rate_change_bps:
        return schedule_df

    changed_schedule = schedule_df.copy()
    numeric_mask = changed_schedule["sofr_assumption"].apply(
        lambda value: not (isinstance(value, str) and value.lower() == "actual")
    )
    maturity_dates = pd.to_datetime(changed_schedule.loc[numeric_mask, "maturity_date"], errors="coerce")
    factors = _build_linear_rate_change_factors(maturity_dates, change_start_date)
    delta = (float(rate_change_bps) / 10_000.0) * factors
    changed_schedule.loc[numeric_mask, "sofr_assumption"] = (
        pd.to_numeric(changed_schedule.loc[numeric_mask, "sofr_assumption"], errors="coerce")
        + delta
    )
    return changed_schedule


@lru_cache(maxsize=1)
def _get_daily_sofr_history(api_key: str, end_date: str) -> pd.DataFrame:
    return get_sofr_data(api_key=api_key, frequency="D", end_date=end_date)


def _coerce_schedule(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in working_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    working_df = working_df[REQUIRED_COLUMNS]

    if working_df.empty:
        raise ValueError("Schedule must contain at least one row.")

    for col in ["investment_name", "base_rate"]:
        working_df[col] = working_df[col].astype(str).str.strip()
        if (working_df[col] == "").any():
            raise ValueError(f"Column '{col}' cannot contain blank values.")

    working_df["base_rate"] = working_df["base_rate"].str.upper()

    for col in ["investment_date", "maturity_date"]:
        working_df[col] = pd.to_datetime(working_df[col], errors="coerce")
        if working_df[col].isna().any():
            raise ValueError(f"Column '{col}' has invalid dates.")

    prepayment = pd.to_datetime(working_df["prepayment_date"], errors="coerce")
    working_df["prepayment_date"] = prepayment.astype("object").where(~prepayment.isna(), None)

    try:
        working_df["sofr_assumption"] = working_df["sofr_assumption"].apply(
            _coerce_sofr_assumption
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    for col in NUMERIC_COLUMNS:
        normalized_numeric = working_df[col].apply(_normalize_numeric_value)
        working_df[col] = pd.to_numeric(normalized_numeric, errors="coerce").astype(float)
        if working_df[col].isna().any():
            raise ValueError(
                f"Column '{col}' must be numeric (examples: 1000000, 1,000,000, 8%)."
            )

    if (working_df["par_value"] <= 0).any():
        raise ValueError("'par_value' must be greater than 0.")

    if (working_df["maturity_date"] <= working_df["investment_date"]).any():
        raise ValueError("Each maturity_date must be after investment_date.")

    return working_df


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#1f2937", plot_bgcolor="#1f2937", margin={"l": 20, "r": 20, "t": 30, "b": 20})
    return fig


def _clean_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.apply(lambda x: not (isinstance(x, float) and math.isnan(x))))


def _format_column_label(column_name: str) -> str:
    if str(column_name).lower() == "sofr_assumption":
        return "SOFR"

    replacements = {
        "irr": "IRR",
        "tvpi": "TVPI",
        "sofr": "SOFR",
        "pik": "PIK",
        "oid": "OID",
    }
    parts = str(column_name).split("_")
    formatted_parts = [replacements.get(part.lower(), part.title()) for part in parts]
    return " ".join(formatted_parts)


def _format_table_headers(df: pd.DataFrame) -> pd.DataFrame:
    renamed_df = df.copy()
    renamed_df.columns = [_format_column_label(column) for column in renamed_df.columns]
    return renamed_df


app_ui = ui.page_navbar(
    ui.head_content(
        ui.tags.link(rel="icon", type="image/svg+xml", href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect fill='%232563eb' width='100' height='100'/><text x='50' y='70' font-size='50' font-weight='bold' fill='white' text-anchor='middle' font-family='monospace'>mmc</text></svg>"),
        ui.tags.style(
            ui.HTML("""
            .navbar {
                background-color: #1f2937 !important;
                position: sticky !important;
                top: 0;
                z-index: 1030;
            }
            body {
                padding-top: 0;
                background-color: #111827 !important;
                color: #f9fafb !important;
            }
            .navbar-brand {
                color: #ffffff !important;
            }
            .navbar-dark .navbar-brand {
                color: #ffffff !important;
            }
            .navbar-dark .navbar-text {
                color: #ffffff !important;
            }
            .navbar-dark .navbar-nav .nav-link {
                color: #ffffff !important;
            }
            .navbar-dark .navbar-nav .nav-link.active {
                color: #ffffff !important;
            }
            .navbar-dark .navbar-nav .nav-link:hover {
                color: #e0e0e0 !important;
            }
            .nav-link {
                color: #ffffff !important;
            }
            .navbar-title {
                color: #ffffff !important;
            }
            #schedule_df table thead th,
            #schedule_df [role="columnheader"] {
                white-space: normal !important;
                width: max-content !important;
                min-width: max-content !important;
            }
            #schedule_df table tbody td,
            #schedule_df [role="gridcell"] {
                white-space: nowrap !important;
                width: max-content !important;
                min-width: max-content !important;
            }
            #schedule_df table thead th,
            #schedule_df [role="columnheader"] {
                background-color: #343a40 !important;
                color: #ffffff !important;
                font-size: 0.8rem !important;
            }
            #schedule_df table tbody td,
            #schedule_df [role="gridcell"] {
                font-size: 0.8rem !important;
                color: #d1d5db !important;
            }
            #schedule_df table thead th:nth-child(2),
            #schedule_df table thead th:nth-child(3),
            #schedule_df table thead th:nth-child(6),
            #schedule_df table tbody td:nth-child(2),
            #schedule_df table tbody td:nth-child(3),
            #schedule_df table tbody td:nth-child(6),
            #schedule_df [role="columnheader"][aria-colindex="2"],
            #schedule_df [role="columnheader"][aria-colindex="3"],
            #schedule_df [role="columnheader"][aria-colindex="6"],
            #schedule_df [role="gridcell"][aria-colindex="2"],
            #schedule_df [role="gridcell"][aria-colindex="3"],
            #schedule_df [role="gridcell"][aria-colindex="6"] {
                text-align: center !important;
                justify-content: center !important;
            }
            #portfolio_outputs_table table thead th:nth-child(1),
            #portfolio_outputs_table table thead th:nth-child(6),
            #portfolio_outputs_table table thead th:nth-child(7),
            #portfolio_outputs_table [role="columnheader"][aria-colindex="1"],
            #portfolio_outputs_table [role="columnheader"][aria-colindex="6"],
            #portfolio_outputs_table [role="columnheader"][aria-colindex="7"] {
                text-align: center !important;
                justify-content: center !important;
            }
            #portfolio_outputs_table table tbody td:nth-child(1),
            #portfolio_outputs_table table tbody td:nth-child(6),
            #portfolio_outputs_table table tbody td:nth-child(7),
            #portfolio_outputs_table [role="gridcell"][aria-colindex="1"],
            #portfolio_outputs_table [role="gridcell"][aria-colindex="6"],
            #portfolio_outputs_table [role="gridcell"][aria-colindex="7"] {
                text-align: center !important;
            }
            #funds_summary_table table thead th,
            #funds_summary_table [role="columnheader"] {
                white-space: normal !important;
                width: max-content !important;
                min-width: max-content !important;
            }
            #funds_summary_table table tbody td,
            #funds_summary_table [role="gridcell"] {
                white-space: nowrap !important;
                width: max-content !important;
                min-width: max-content !important;
            }
            #funds_summary_table table thead th,
            #funds_summary_table [role="columnheader"] {
                background-color: #343a40 !important;
                color: #ffffff !important;
                font-size: 0.8rem !important;
            }
            #funds_summary_table table tbody td,
            #funds_summary_table [role="gridcell"] {
                font-size: 0.8rem !important;
                color: #d1d5db !important;
            }
            #portfolio_outputs_table table thead th,
            #portfolio_outputs_table [role="columnheader"] {
                background-color: #343a40 !important;
                color: #ffffff !important;
                font-size: 0.8rem !important;
                white-space: normal !important;
                width: max-content !important;
                min-width: max-content !important;
            }
            #portfolio_outputs_table table tbody td,
            #portfolio_outputs_table [role="gridcell"] {
                font-size: 0.8rem !important;
                color: #d1d5db !important;
                white-space: nowrap !important;
                width: max-content !important;
                min-width: max-content !important;
            }
            #schedule_df table thead th:nth-child(2),
            #schedule_df [role="columnheader"][aria-colindex="2"],
            #schedule_df table thead th:nth-child(3),
            #schedule_df [role="columnheader"][aria-colindex="3"],
            #schedule_df table thead th:nth-child(7),
            #schedule_df [role="columnheader"][aria-colindex="7"],
            #funds_summary_table table thead th:nth-child(2),
            #funds_summary_table [role="columnheader"][aria-colindex="2"],
            #funds_summary_table table thead th:nth-child(3),
            #funds_summary_table [role="columnheader"][aria-colindex="3"],
            #funds_summary_table table thead th:nth-child(7),
            #funds_summary_table [role="columnheader"][aria-colindex="7"] {
                max-width: 3rem !important;
            }
            #schedule_df table thead th,
            #schedule_df [role="columnheader"],
            #portfolio_outputs_table table thead th,
            #portfolio_outputs_table [role="columnheader"],
            #funds_summary_table table thead th,
            #funds_summary_table [role="columnheader"] {
                text-align: center !important;
                justify-content: center !important;
                align-items: center !important;
                min-height: 62px !important;
                line-height: 1.2 !important;
                padding-top: 10px !important;
                padding-bottom: 10px !important;
            }
            #schedule_df table thead th,
            #schedule_df table tbody td,
            #schedule_df [role="columnheader"],
            #schedule_df [role="gridcell"],
            #portfolio_outputs_table table thead th,
            #portfolio_outputs_table table tbody td,
            #portfolio_outputs_table [role="columnheader"],
            #portfolio_outputs_table [role="gridcell"],
            #funds_summary_table table thead th,
            #funds_summary_table table tbody td,
            #funds_summary_table [role="columnheader"],
            #funds_summary_table [role="gridcell"] {
                padding-left: 10px !important;
                padding-right: 10px !important;
            }
            #schedule_df {
                width: fit-content;
                margin-left: auto;
                margin-right: auto;
                max-width: 100%;
            }
            #schedule_df .shiny-data-grid-summary,
            #schedule_df .data-grid-summary,
            #schedule_df [role="status"],
            #schedule_df [aria-live="polite"],
            #portfolio_outputs_table .shiny-data-grid-summary,
            #portfolio_outputs_table .data-grid-summary,
            #portfolio_outputs_table [role="status"],
            #portfolio_outputs_table [aria-live="polite"],
            #funds_summary_table .shiny-data-grid-summary,
            #funds_summary_table .data-grid-summary,
            #funds_summary_table [role="status"],
            #funds_summary_table [aria-live="polite"] {
                font-size: 0.72rem !important;
                color: #9ca3af !important;
            }

            /* === DARK MODE === */
            .bslib-page-navbar, .tab-content, .tab-pane {
                background-color: #111827 !important;
            }
            .bslib-sidebar-layout > .sidebar, .sidebar {
                background-color: #1f2937 !important;
                border-color: #374151 !important;
            }
            .bslib-sidebar-layout > .main {
                background-color: #111827 !important;
            }
            .card, .bslib-card, .card-body {
                background-color: #1f2937 !important;
                border-color: #374151 !important;
                color: #f9fafb !important;
            }
            label, .control-label, p, h1, h2, h3, h4, h5, h6 {
                color: #f9fafb !important;
            }
            .form-control, .form-select, select,
            input[type="text"], input[type="number"], input[type="date"] {
                background-color: #374151 !important;
                color: #f9fafb !important;
                border-color: #4b5563 !important;
            }
            .form-control:focus, .form-select:focus {
                background-color: #374151 !important;
                color: #f9fafb !important;
                border-color: #6b7280 !important;
                box-shadow: 0 0 0 0.2rem rgba(107,114,128,0.25) !important;
            }
            .btn {
                background-color: #374151 !important;
                border-color: #4b5563 !important;
                color: #f9fafb !important;
            }
            .btn:hover {
                background-color: #4b5563 !important;
                border-color: #6b7280 !important;
                color: #ffffff !important;
            }
            .irs--shiny .irs-line {
                background-color: #374151 !important;
                border-color: #374151 !important;
            }
            .irs--shiny .irs-bar {
                background-color: #4b5563 !important;
            }
            .irs--shiny .irs-handle {
                background: #9ca3af !important;
                border-color: #6b7280 !important;
            }
            .irs--shiny .irs-min, .irs--shiny .irs-max,
            .irs--shiny .irs-from, .irs--shiny .irs-to,
            .irs--shiny .irs-single {
                background-color: #4b5563 !important;
                color: #f9fafb !important;
            }
            .selectize-input, .selectize-dropdown {
                background-color: #374151 !important;
                color: #f9fafb !important;
                border-color: #4b5563 !important;
            }
            .selectize-dropdown .option:hover,
            .selectize-dropdown .option.active {
                background-color: #4b5563 !important;
                color: #ffffff !important;
            }
            #schedule_df table tbody tr,
            #schedule_df [role="row"],
            #portfolio_outputs_table table tbody tr,
            #portfolio_outputs_table [role="row"],
            #funds_summary_table table tbody tr,
            #funds_summary_table [role="row"] {
                background-color: #1f2937 !important;
            }
            #schedule_df table tbody tr:nth-child(even),
            #portfolio_outputs_table table tbody tr:nth-child(even),
            #funds_summary_table table tbody tr:nth-child(even) {
                background-color: #253347 !important;
            }
            #schedule_df table, #portfolio_outputs_table table, #funds_summary_table table {
                border-color: #374151 !important;
            }
            #schedule_df table td, #schedule_df table th,
            #portfolio_outputs_table table td, #portfolio_outputs_table table th,
            #funds_summary_table table td, #funds_summary_table table th {
                border-color: #374151 !important;
            }
            """)
        )
    ),
    ui.nav_panel(
        "Loan Schedule",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_action_button("add_row", "Add investment"),
                ui.input_action_button("remove_selected_rows", "Remove investment"),
                ui.input_action_button("reset_schedule", "Reset defaults"),
            ),
            ui.p(
                "Schedule of Investments. Click 'Add investment' to add rows to the schedule. Select rows and click 'Remove investment' to delete them. Click 'Reset defaults' to restore the original sample schedule.",
            ),
            ui.div(
                {"style": "width: fit-content; margin-left: auto; margin-right: auto;"},
                ui.output_data_frame("schedule_df"),
                ui.div(
                    {"style": "text-align: right; margin-top: 15px;"},
                    ui.download_button("download_schedule_csv", "Download Loan Schedule CSV"),
                ),
            ),
            ui.output_ui("schedule_error"),
        ),
    ),
    ui.nav_panel(
        "Portfolio Outputs",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Scenario Analysis"),
                ui.div(
                    {
                        "style": "border: 1px solid #374151; border-radius: 8px; padding: 12px; margin-bottom: 12px; font-size: 0.85rem;",
                    },
                    ui.h5(
                        "Rate Shock",
                        style="margin-top: 0; margin-bottom: 12px; font-size: 0.95rem; font-weight: 700; text-align: center;",
                    ),
                    ui.div(
                        {"style": "font-size: 0.75rem;"},
                        ui.input_date(
                            "rate_shock_start_date",
                            "Start Date",
                            value=None,
                        ),
                    ),
                    ui.input_slider(
                        "scenario_shift",
                        "bps",
                        min=-300,
                        max=300,
                        value=0,
                        step=25,
                    ),
                    ui.div(
                        {"style": "font-size: 0.75rem;"},
                        ui.input_action_button("run_rate_shock", "Run Rate Shock"),
                    ),
                ),
                ui.div(
                    {
                        "style": "border: 1px solid #374151; border-radius: 8px; padding: 12px; margin-bottom: 12px; font-size: 0.85rem;",
                    },
                    ui.h5(
                        "Rate Change",
                        style="margin-top: 0; margin-bottom: 12px; font-size: 0.95rem; font-weight: 700; text-align: center;",
                    ),
                    ui.div(
                        {"style": "font-size: 0.75rem;"},
                        ui.input_date(
                            "rate_change_start_date",
                            "Start Date",
                            value=None,
                        ),
                    ),
                    ui.input_slider(
                        "rate_change_bps",
                        "bps",
                        min=-300,
                        max=300,
                        value=0,
                        step=25,
                    ),
                    ui.div(
                        {"style": "font-size: 0.75rem;"},
                        ui.input_action_button("run_rate_change", "Run Rate Change"),
                    ),
                ),
            ),
            ui.output_ui("portfolio_error"),
            ui.layout_columns(
                output_widget("cashflow_combined_chart"),
                output_widget("sofr_rate_chart"),
                col_widths=(6, 6),
            ),
            ui.layout_columns(
                output_widget("tvpi_chart"),
                output_widget("portfolio_irr_chart"),
                col_widths=(6, 6),
            ),
            ui.div(
                {"style": "display: flex; flex-direction: row; justify-content: center; align-items: flex-start; width: 100%;"},
                ui.div(
                    {"style": "flex: 0 0 auto;"},
                    ui.h4("Portfolio Results", style="font-size: 0.95rem;"),
                    ui.output_data_frame("portfolio_outputs_table"),
                    ui.div(
                        {"style": "text-align: right; margin-top: 15px;"},
                        ui.download_button("download_portfolio_outputs_csv", "Download Portfolio Results CSV"),
                    ),
                ),
                ui.div(
                    {"style": "flex: 0 0 auto; margin-left: 20px; overflow-x: auto; max-width: 100%;"},
                    ui.h4("Funds Summary", style="font-size: 0.95rem;"),
                    ui.output_data_frame("funds_summary_table"),
                    ui.div(
                        {"style": "text-align: right; margin-top: 15px;"},
                        ui.download_button("download_funds_summary_csv", "Download Funds Summary CSV"),
                    ),
                ),
            ),
        ),
    ),
    title=ui.tags.img(src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 350 100'><defs><linearGradient id='textGrad' x1='0%' y1='0%' x2='0%' y2='100%'><stop offset='0%' style='stop-color:%23d1d5db;stop-opacity:1' /><stop offset='100%' style='stop-color:%23ffffff;stop-opacity:1' /></linearGradient></defs><rect fill='%232563eb' width='350' height='100'/><polygon points='0,80 40,40 80,60 120,30 160,50 200,20 240,45 280,35 320,55 350,40 350,100 0,100' fill='%231e3a8a' opacity='0.7'/><polyline points='0,80 40,40 80,60 120,30 160,50 200,20 240,45 280,35 320,55 350,40 350,100 0,100' stroke='black' stroke-width='1.5' fill='none'/><polyline points='20,90 60,50 100,75 140,45 180,65 220,35 260,60 300,50 330,70 350,60 350,100 20,100' stroke='black' stroke-width='1' fill='none' opacity='0.6'/><text x='175' y='70' font-size='48' font-weight='700' font-stretch='condensed' fill='url(%23textGrad)' stroke='black' stroke-width='1' text-anchor='middle' font-family='Arial'>MMC Capital</text></svg>", style="height: 40px; width: auto;"),
    window_title="Portfolio Analytics"
)


def server(input, output, session):
    schedule_state = reactive.value(_sort_schedule_by_investment_date(default_schedule()))
    rate_shock_start_date_default = reactive.value(None)
    rate_change_start_date_default = reactive.value(None)
    applied_rate_shock_bps = reactive.value(0)
    applied_rate_shock_start_date = reactive.value(None)
    applied_rate_change_bps = reactive.value(0)
    applied_rate_change_start_date = reactive.value(None)

    @reactive.effect
    @reactive.event(input.add_row)
    def _add_row():
        df = schedule_state().copy()
        df.loc[len(df)] = _default_row(len(df) + 1)
        schedule_state.set(_sort_schedule_by_investment_date(df))

    @reactive.effect
    @reactive.event(input.reset_schedule)
    def _reset_schedule():
        schedule_state.set(_sort_schedule_by_investment_date(default_schedule()))

    PCT_DISPLAY_COLUMNS = [
        "spread",
        "pik_interest",
        "amortization",
        "oid",
        "exit_fee",
    ]

    @render.data_frame
    def schedule_df():
        display_df = schedule_state().copy()
        for col in PCT_DISPLAY_COLUMNS:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda v: f"{float(v) * 100:.2f}%" if v != "" and v is not None else v
                )
        if "sofr_assumption" in display_df.columns:
            display_df["sofr_assumption"] = display_df["sofr_assumption"].apply(
                lambda value: "actual"
                if isinstance(value, str) and value.lower() == "actual"
                else (f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
            )
        if "par_value" in display_df.columns:
            display_df["par_value"] = display_df["par_value"].apply(
                lambda v: f"{float(v):,.0f}" if v != "" and v is not None else v
            )
        display_df = _format_table_headers(display_df)

        centered_columns = [
            "Investment Date",
            "Maturity Date",
            "Par Value",
            "Spread",
            "Base Rate",
            "SOFR",
            "PIK Interest",
            "Amortization",
            "OID",
            "Exit Fee",
        ]
        center_styles = []
        for column_name in centered_columns:
            if column_name in display_df.columns:
                center_styles.append(
                    {
                        "cols": [display_df.columns.get_loc(column_name)],
                        "style": {"textAlign": "center"},
                    }
                )

        return render.DataGrid(
            display_df,
            editable=True,
            selection_mode="rows",
            width="fit-content",
            styles=center_styles,
        )

    @render.download(filename="loan_schedule.csv")
    def download_schedule_csv():
        display_df = schedule_state().copy()
        for col in PCT_DISPLAY_COLUMNS:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda v: f"{float(v) * 100:.2f}%" if v != "" and v is not None else v
                )
        if "sofr_assumption" in display_df.columns:
            display_df["sofr_assumption"] = display_df["sofr_assumption"].apply(
                lambda value: "actual"
                if isinstance(value, str) and value.lower() == "actual"
                else (f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
            )
        if "par_value" in display_df.columns:
            display_df["par_value"] = display_df["par_value"].apply(
                lambda v: f"{float(v):,.0f}" if v != "" and v is not None else v
            )
        display_df = _format_table_headers(display_df)
        yield display_df.to_csv(index=False)

    @schedule_df.set_patches_fn
    def _patch_schedule(*, patches: list[render.CellPatch]) -> list[render.CellPatch]:
        df = schedule_state().copy()
        for patch in patches:
            row_index = patch["row_index"]
            column_index = patch["column_index"]
            new_value = patch["value"]
            column_name = df.columns[column_index]
            if column_name in NUMERIC_COLUMNS:
                normalized_value = _normalize_numeric_value(new_value)
                if normalized_value is None:
                    df.iat[row_index, column_index] = ""
                else:
                    try:
                        coerced = float(normalized_value)
                        df.iat[row_index, column_index] = coerced
                        patch["value"] = coerced
                    except (TypeError, ValueError):
                        df.iat[row_index, column_index] = new_value
            else:
                df.iat[row_index, column_index] = new_value
        schedule_state.set(df)
        return patches

    @reactive.effect
    @reactive.event(input.remove_selected_rows)
    def _remove_selected_rows():
        df = schedule_state().copy()
        selected_rows = schedule_df.cell_selection()["rows"]
        if selected_rows:
            df = df.drop(index=list(selected_rows)).reset_index(drop=True)
            if len(df) > 0:
                schedule_state.set(df)

    @reactive.calc
    def portfolio_results():
        global SOFR_RATES

        normalized_schedule = _coerce_schedule(schedule_state())
        max_maturity_date = normalized_schedule["maturity_date"].max()
        rate_shock_bps = applied_rate_shock_bps()
        shock_start_date = applied_rate_shock_start_date()
        rate_change_bps = applied_rate_change_bps()
        change_start_date = applied_rate_change_start_date()

        if FRED_API_KEY:
            SOFR_RATES = _get_daily_sofr_history(
                FRED_API_KEY,
                pd.Timestamp(max_maturity_date).strftime("%Y-%m-%d"),
            ).copy()
            SOFR_RATES = _apply_rate_shock_to_assumed_sofr(
                SOFR_RATES,
                rate_shock_bps,
                shock_start_date,
            )
            SOFR_RATES = _apply_linear_rate_change_to_assumed_sofr(
                SOFR_RATES,
                rate_change_bps,
                change_start_date,
            )
        else:
            SOFR_RATES = pd.DataFrame()

        shocked_schedule = _apply_rate_shock_to_schedule(
            normalized_schedule,
            rate_shock_bps,
            shock_start_date,
        )
        shocked_schedule = _apply_linear_rate_change_to_schedule(
            shocked_schedule,
            rate_change_bps,
            change_start_date,
        )

        sofr_rates = None
        if _schedule_uses_actual_sofr(shocked_schedule):
            if not FRED_API_KEY:
                raise ValueError(
                    "FRED_API_KEY is required when any loan uses sofr_assumption='actual'."
                )
            sofr_rates = SOFR_RATES.copy()

        result = loan_portfolio(shocked_schedule, sofr_rates=sofr_rates)

        if isinstance(result, tuple):
            if len(result) >= 3:
                portfolio_df, funds_df, funds_summary_df = result[0], result[1], result[2]
            elif len(result) >= 2:
                portfolio_df, funds_df = result[0], result[1]
                funds_summary_df = pd.DataFrame()
            else:
                portfolio_df, funds_df = pd.DataFrame(), pd.DataFrame()
                funds_summary_df = pd.DataFrame()
        else:
            portfolio_df, funds_df = pd.DataFrame(), pd.DataFrame()
            funds_summary_df = pd.DataFrame()

        if portfolio_df.empty:
            raise ValueError("Portfolio output is empty. Add valid schedule rows.")

        return portfolio_df, funds_df, funds_summary_df

    @reactive.effect
    def _initialize_portfolio_outputs():
        schedule_state()
        try:
            portfolio_results()
        except Exception:
            return

    @reactive.effect
    def _set_default_rate_shock_start_date():
        try:
            normalized_schedule = _coerce_schedule(schedule_state())
            max_maturity_date = normalized_schedule["maturity_date"].max()

            if not FRED_API_KEY:
                return

            raw_sofr_rates = _get_daily_sofr_history(
                FRED_API_KEY,
                pd.Timestamp(max_maturity_date).strftime("%Y-%m-%d"),
            )
        except Exception:
            return

        default_start_date = _first_assumed_sofr_date(raw_sofr_rates)
        current_value = input.rate_shock_start_date()
        previous_default = rate_shock_start_date_default()
        current_value_text = (
            current_value.isoformat() if hasattr(current_value, "isoformat") else None
        )

        if default_start_date is None:
            return

        should_update = False
        if current_value_text is None:
            should_update = True
        elif previous_default is not None and current_value_text == previous_default:
            should_update = current_value_text != default_start_date

        if should_update:
            ui.update_date(
                "rate_shock_start_date",
                value=default_start_date,
                session=session,
            )

        if previous_default != default_start_date:
            rate_shock_start_date_default.set(default_start_date)

        rate_change_current_value = input.rate_change_start_date()
        rate_change_previous_default = rate_change_start_date_default()
        rate_change_current_value_text = (
            rate_change_current_value.isoformat()
            if hasattr(rate_change_current_value, "isoformat")
            else None
        )

        rate_change_should_update = False
        if rate_change_current_value_text is None:
            rate_change_should_update = True
        elif (
            rate_change_previous_default is not None
            and rate_change_current_value_text == rate_change_previous_default
        ):
            rate_change_should_update = rate_change_current_value_text != default_start_date

        if rate_change_should_update:
            ui.update_date(
                "rate_change_start_date",
                value=default_start_date,
                session=session,
            )

        if rate_change_previous_default != default_start_date:
            rate_change_start_date_default.set(default_start_date)

    @reactive.effect
    @reactive.event(input.run_rate_shock)
    def _run_rate_shock():
        applied_rate_shock_bps.set(input.scenario_shift())
        applied_rate_shock_start_date.set(input.rate_shock_start_date())

    @reactive.effect
    @reactive.event(input.run_rate_change)
    def _run_rate_change():
        applied_rate_change_bps.set(input.rate_change_bps())
        applied_rate_change_start_date.set(input.rate_change_start_date())

    @render.ui
    def schedule_error():
        try:
            _coerce_schedule(schedule_state())
            return ui.div()
        except Exception as exc:
            return ui.div(str(exc), style="color: #b91c1c;")

    @render.ui
    def portfolio_error():
        try:
            portfolio_results()
            return ui.div()
        except Exception as exc:
            return ui.div(str(exc), style="color: #b91c1c;")

    @render_widget
    def cashflow_combined_chart():
        try:
            portfolio_df, _, _ = portfolio_results()
        except Exception as exc:
            return _empty_figure(str(exc))

        cumulative = portfolio_df.copy()
        cumulative["cumulative_cashflow"] = cumulative["total_payment"].cumsum()
        bar_colors = [
            "#16a34a" if payment >= 0 else "#dc2626"
            for payment in portfolio_df["total_payment"]
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=portfolio_df["quarter_end"],
                y=portfolio_df["total_payment"],
                name="Cash Flow",
                opacity=0.6,
                marker_color=bar_colors,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cumulative["quarter_end"],
                y=cumulative["cumulative_cashflow"],
                mode="lines+markers",
                name="Cumulative Cash Flow",
                line={"color": LINE_BLUE, "width": 3},
                marker={"color": LINE_BLUE, "size": 7},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=portfolio_df["quarter_end"],
                y=portfolio_df["ending_balance"],
                mode="lines",
                name="Remaining Balance",
                line={"color": LIGHT_BLUE, "width": 2},
                opacity=0.2,
            )
        )
        fig.update_layout(
            title=dict(text="Cash Flow and Remaining Balance", y=0.95, yanchor="top"),
            template="plotly_dark",
            paper_bgcolor="#1f2937",
            plot_bgcolor="#1f2937",
            legend={"orientation": "h", "y": 1.05, "x": 0},
            margin={"l": 20, "r": 20, "t": 80, "b": 20},
        )
        fig.update_xaxes(title="Quarter End")
        fig.update_yaxes(title_text="Value")
        return fig

    @render_widget
    def sofr_rate_chart():
        try:
            portfolio_results()
        except Exception as exc:
            return _empty_figure(str(exc))

        if SOFR_RATES.empty or "sofr" not in SOFR_RATES.columns:
            return _empty_figure("SOFR rate data is unavailable.")

        sofr_df = SOFR_RATES.copy()
        sofr_df["date"] = pd.to_datetime(sofr_df["date"], errors="coerce")
        sofr_df["sofr"] = pd.to_numeric(sofr_df["sofr"], errors="coerce")
        if "rate_status" in sofr_df.columns:
            sofr_df["rate_status"] = sofr_df["rate_status"].fillna("assumed")
        else:
            sofr_df["rate_status"] = "assumed"
        sofr_df = sofr_df.dropna(subset=["date", "sofr"])

        if sofr_df.empty:
            return _empty_figure("SOFR rate data is unavailable.")

        sofr_by_quarter = (
            sofr_df.loc[
                sofr_df["date"].dt.is_quarter_end,
                ["date", "sofr", "rate_status"],
            ]
            .sort_values("date", kind="stable")
            .reset_index(drop=True)
        )

        if sofr_by_quarter.empty:
            sofr_by_quarter = sofr_df[["date", "sofr", "rate_status"]].sort_values(
                "date", kind="stable"
            ).reset_index(drop=True)

        fig = go.Figure()
        status_colors = {
            "actual": LINE_BLUE,
            "assumed": LIGHT_BLUE,
        }

        for rate_status, status_group in sofr_by_quarter.groupby("rate_status", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=status_group["date"],
                    y=status_group["sofr"],
                    mode="lines+markers",
                    name=f"SOFR Rate ({str(rate_status).title()})",
                    line={"color": status_colors.get(str(rate_status).lower(), LIGHT_BLUE), "width": 3},
                    marker={"color": status_colors.get(str(rate_status).lower(), LIGHT_BLUE), "size": 7},
                )
            )
        fig.update_layout(
            title=dict(text="SOFR Rate", y=0.95, yanchor="top"),
            template="plotly_dark",
            paper_bgcolor="#1f2937",
            plot_bgcolor="#1f2937",
            margin={"l": 20, "r": 20, "t": 80, "b": 50},
            annotations=[
                {
                    "text": "source: Federal Reserve Bank of ST. LOUIS",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": -0.22,
                    "showarrow": False,
                    "xanchor": "left",
                    "font": {"size": 10, "color": "#9ca3af"},
                }
            ],
        )
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title="SOFR Rate", tickformat=".2%")
        return fig

    @render_widget
    def tvpi_chart():
        try:
            portfolio_df, _, _ = portfolio_results()
        except Exception as exc:
            return _empty_figure(str(exc))

        tvpi_series = _clean_series(portfolio_df["tvpi"])
        tvpi_df = portfolio_df[["quarter_end"]].copy()
        tvpi_df["tvpi"] = tvpi_series
        tvpi_df = tvpi_df.dropna(subset=["tvpi"])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=tvpi_df["quarter_end"],
                y=tvpi_df["tvpi"],
                mode="lines+markers",
                name="TVPI",
                line={"color": LINE_BLUE, "width": 3},
                marker={"color": LINE_BLUE, "size": 7},
            )
        )
        fig.update_layout(title=dict(text="Portfolio Gross TVPI", y=0.95, yanchor="top"), template="plotly_dark", paper_bgcolor="#1f2937", plot_bgcolor="#1f2937", margin={"l": 20, "r": 20, "t": 80, "b": 20})
        fig.update_xaxes(title="Quarter End")
        fig.update_yaxes(title="TVPI", tickformat=".2f")
        return fig

    @render_widget
    def portfolio_irr_chart():
        try:
            portfolio_df, _, _ = portfolio_results()
        except Exception as exc:
            return _empty_figure(str(exc))

        irr_series = _clean_series(portfolio_df["irr"])
        irr_df = portfolio_df[["quarter_end"]].copy()
        irr_df["irr"] = irr_series
        irr_df = irr_df.dropna(subset=["irr"])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=irr_df["quarter_end"],
                y=irr_df["irr"],
                mode="lines+markers",
                name="Portfolio IRR",
                line={"color": LINE_BLUE, "width": 3},
                marker={"color": LINE_BLUE, "size": 7},
            )
        )
        fig.update_layout(title=dict(text="Portfolio Gross IRR", y=0.95, yanchor="top"), template="plotly_dark", paper_bgcolor="#1f2937", plot_bgcolor="#1f2937", margin={"l": 20, "r": 20, "t": 80, "b": 20})
        fig.update_xaxes(title="Quarter End")
        fig.update_yaxes(title="IRR", tickformat=".2%")
        return fig

    @render.data_frame
    def portfolio_outputs_table():
        try:
            portfolio_df, _, _ = portfolio_results()
        except Exception:
            return render.DataGrid(pd.DataFrame(), editable=False, width="fit-content")

        display_df = portfolio_df.copy()
        if "remaining_balance_payment" in display_df.columns:
            display_df = display_df.drop(columns=["remaining_balance_payment"])
        if "beginning_balance" in display_df.columns and "ending_balance" in display_df.columns:
            columns = list(display_df.columns)
            columns.remove("beginning_balance")
            ending_balance_index = columns.index("ending_balance")
            columns.insert(ending_balance_index, "beginning_balance")
            display_df = display_df[columns]
        if "quarter_end" in display_df.columns:
            display_df["quarter_end"] = pd.to_datetime(display_df["quarter_end"]).dt.strftime("%Y-%m-%d")
        for col in ["invested_amount", "total_payment", "beginning_balance", "ending_balance"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda value: f"{float(value):,.2f}" if pd.notna(value) else ""
                )
        if "irr" in display_df.columns:
            display_df["irr"] = display_df["irr"].apply(
                lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else ""
            )
        if "tvpi" in display_df.columns:
            display_df["tvpi"] = display_df["tvpi"].apply(
                lambda value: f"{float(value):.2f}x" if pd.notna(value) else ""
            )
        display_df = _format_table_headers(display_df)

        centered_columns = ["Quarter End", "IRR", "TVPI"]
        center_styles = []
        for column_name in centered_columns:
            if column_name in display_df.columns:
                center_styles.append(
                    {
                        "cols": [display_df.columns.get_loc(column_name)],
                        "style": {"textAlign": "center"},
                    }
                )

        return render.DataGrid(
            display_df,
            editable=False,
            width="fit-content",
            styles=center_styles,
        )

    @render.download(filename="portfolio_results.csv")
    def download_portfolio_outputs_csv():
        try:
            portfolio_df, _, _ = portfolio_results()
        except Exception:
            yield pd.DataFrame().to_csv(index=False)
            return

        display_df = portfolio_df.copy()
        if "remaining_balance_payment" in display_df.columns:
            display_df = display_df.drop(columns=["remaining_balance_payment"])
        if "beginning_balance" in display_df.columns and "ending_balance" in display_df.columns:
            columns = list(display_df.columns)
            columns.remove("beginning_balance")
            ending_balance_index = columns.index("ending_balance")
            columns.insert(ending_balance_index, "beginning_balance")
            display_df = display_df[columns]
        if "quarter_end" in display_df.columns:
            display_df["quarter_end"] = pd.to_datetime(display_df["quarter_end"]).dt.strftime("%Y-%m-%d")
        for col in ["invested_amount", "total_payment", "beginning_balance", "ending_balance"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda value: f"{float(value):,.2f}" if pd.notna(value) else ""
                )
        if "irr" in display_df.columns:
            display_df["irr"] = display_df["irr"].apply(
                lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else ""
            )
        if "tvpi" in display_df.columns:
            display_df["tvpi"] = display_df["tvpi"].apply(
                lambda value: f"{float(value):.2f}x" if pd.notna(value) else ""
            )
        display_df = _format_table_headers(display_df)
        yield display_df.to_csv(index=False)

    @render.data_frame
    def funds_summary_table():
        try:
            _, _, funds_summary_df = portfolio_results()
        except Exception:
            return render.DataGrid(pd.DataFrame(), editable=False, width="fit-content")

        display_df = funds_summary_df.copy()
        if "investment_date" in display_df.columns:
            display_df["investment_date"] = pd.to_datetime(
                display_df["investment_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        if "maturity_date" in display_df.columns:
            display_df["maturity_date"] = pd.to_datetime(
                display_df["maturity_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        if "prepayment_date" in display_df.columns:
            display_df["prepayment_date"] = pd.to_datetime(
                display_df["prepayment_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            display_df["prepayment_date"] = display_df["prepayment_date"].fillna("")

        for col in ["par_value", "total_payment"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda value: f"{float(value):,.2f}" if pd.notna(value) else ""
                )

        if "sofr_assumption" in display_df.columns:
            display_df["sofr_assumption"] = display_df["sofr_assumption"].apply(
                lambda value: "actual"
                if isinstance(value, str) and value.lower() == "actual"
                else (f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
            )

        for col in ["spread", "pik_interest", "amortization", "oid", "exit_fee", "irr"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else ""
                )

        display_df = _format_table_headers(display_df)

        centered_columns = [
            "Investment Date",
            "Maturity Date",
            "Prepayment Date",
            "Spread",
            "SOFR",
            "PIK Interest",
            "Amortization",
            "OID",
            "Exit Fee",
            "IRR",
        ]
        center_styles = []
        for column_name in centered_columns:
            if column_name in display_df.columns:
                center_styles.append(
                    {
                        "cols": [display_df.columns.get_loc(column_name)],
                        "style": {"textAlign": "center"},
                    }
                )

        return render.DataGrid(
            display_df,
            editable=False,
            width="fit-content",
            styles=center_styles,
        )

    @render.download(filename="funds_summary.csv")
    def download_funds_summary_csv():
        try:
            _, _, funds_summary_df = portfolio_results()
        except Exception:
            yield pd.DataFrame().to_csv(index=False)
            return

        display_df = funds_summary_df.copy()
        if "investment_date" in display_df.columns:
            display_df["investment_date"] = pd.to_datetime(
                display_df["investment_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        if "maturity_date" in display_df.columns:
            display_df["maturity_date"] = pd.to_datetime(
                display_df["maturity_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        if "prepayment_date" in display_df.columns:
            display_df["prepayment_date"] = pd.to_datetime(
                display_df["prepayment_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            display_df["prepayment_date"] = display_df["prepayment_date"].fillna("")

        for col in ["par_value", "total_payment"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda value: f"{float(value):,.2f}" if pd.notna(value) else ""
                )

        if "sofr_assumption" in display_df.columns:
            display_df["sofr_assumption"] = display_df["sofr_assumption"].apply(
                lambda value: "actual"
                if isinstance(value, str) and value.lower() == "actual"
                else (f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
            )

        for col in ["spread", "pik_interest", "amortization", "oid", "exit_fee", "irr"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else ""
                )

        display_df = _format_table_headers(display_df)
        yield display_df.to_csv(index=False)


app = App(app_ui, server)
