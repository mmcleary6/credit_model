
from __future__ import annotations

import os
import sys
from pathlib import Path
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from asset_modeling.credit import loan_portfolio
from data.ratings import get_effective_yield
from data.sofr import get_sofr_data

REQUIRED_COLUMNS = [
    "company_uid",
    "investment_uid",
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
    "pik_interest",
    "amortization",
    "oid",
    "exit_fee",
]

LINE_BLUE = "#2563eb"
LIGHT_BLUE = "#7dd3fc"
CHART_LINE_WIDTH = 2
CHART_MARKER_SIZE = 7
DEFAULT_SCHEDULE_PATH = Path(__file__).resolve().parent.parent / "app/loan_schedule.csv"
FRED_API_KEY = os.getenv("FRED_API_KEY")
SOFR_RATES = pd.DataFrame()
DEFAULT_SCHEDULE_INVESTMENT_END_DATE = "2024-12-31"


def _default_row(index: int) -> dict:
    return {
        "company_uid": f"company-{index}",
        "investment_uid": f"investment-{index}",
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


@lru_cache(maxsize=1)
def _get_daily_b_yield_history(api_key: str, end_date: str) -> pd.DataFrame:
    return get_effective_yield(
        api_key=api_key,
        rating="B",
        frequency="D",
        end_date=end_date,
    )


def _lookup_rate_value_on_or_before_date(
    rate_df: pd.DataFrame,
    lookup_date: object,
    value_column: str,
) -> float | None:
    if rate_df.empty or value_column not in rate_df.columns or "date" not in rate_df.columns:
        return None

    working_df = rate_df[["date", value_column]].copy()
    working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce").dt.normalize()
    working_df[value_column] = pd.to_numeric(working_df[value_column], errors="coerce")
    working_df = working_df.dropna(subset=["date", value_column]).sort_values("date")

    if working_df.empty:
        return None

    normalized_lookup_date = pd.Timestamp(lookup_date).normalize()
    matching_rows = working_df.loc[working_df["date"] == normalized_lookup_date, value_column]
    if not matching_rows.empty:
        return float(matching_rows.iloc[-1])

    prior_rows = working_df.loc[working_df["date"] <= normalized_lookup_date, value_column]
    if not prior_rows.empty:
        return float(prior_rows.iloc[-1])

    return float(working_df.iloc[0][value_column])


def default_schedule() -> pd.DataFrame:
    if not DEFAULT_SCHEDULE_PATH.exists():
        raise FileNotFoundError(
            f"Default loan schedule file not found: {DEFAULT_SCHEDULE_PATH}"
        )

    default_df = pd.read_csv(DEFAULT_SCHEDULE_PATH, keep_default_na=False)
    missing_columns = [
        column_name for column_name in REQUIRED_COLUMNS if column_name not in default_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Default loan schedule is missing required columns: {missing_columns}"
        )

    return default_df[REQUIRED_COLUMNS].copy()


def _sort_schedule_by_investment_date(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = (
        df.assign(_investment_date_sort=pd.to_datetime(df["investment_date"]))
        .sort_values("_investment_date_sort", kind="stable")
        .drop(columns="_investment_date_sort")
        .reset_index(drop=True)
    )
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


def _schedule_uses_actual_spread(df: pd.DataFrame) -> bool:
    return df["spread"].apply(
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


def _add_status_colored_line_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    trace_name: str,
) -> None:
    if df.empty or x_column not in df.columns or y_column not in df.columns:
        return

    working_df = df[[x_column, y_column]].copy()
    if "status" in df.columns:
        working_df["status"] = df["status"].fillna("projected").astype(str).str.lower()
    else:
        working_df["status"] = "actual"

    working_df = working_df.dropna(subset=[x_column, y_column]).reset_index(drop=True)
    if working_df.empty:
        return

    status_colors = {
        "actual": LINE_BLUE,
        "projected": LIGHT_BLUE,
    }

    segment_ids = working_df["status"].ne(working_df["status"].shift()).cumsum()
    legend_shown_for_status: set[str] = set()

    for _, segment in working_df.groupby(segment_ids, sort=False):
        segment_status = str(segment["status"].iloc[0]).lower()
        color = status_colors.get(segment_status, LIGHT_BLUE)

        x_values = segment[x_column].tolist()
        y_values = segment[y_column].tolist()
        marker_sizes = [CHART_MARKER_SIZE] * len(segment)

        segment_start_index = int(segment.index[0])
        if segment_start_index > 0:
            previous_row = working_df.iloc[segment_start_index - 1]
            x_values.insert(0, previous_row[x_column])
            y_values.insert(0, previous_row[y_column])
            marker_sizes.insert(0, 0)

        status_label = "Actual" if segment_status == "actual" else "Projected"
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=f"{trace_name} ({status_label})",
                line={"color": color, "width": CHART_LINE_WIDTH},
                marker={"color": color, "size": marker_sizes},
                showlegend=segment_status not in legend_shown_for_status,
            )
        )
        legend_shown_for_status.add(segment_status)


def _apply_standard_chart_layout(
    fig: go.Figure,
    title_text: str,
    *,
    left_margin: int = 20,
    right_margin: int = 20,
    bottom_margin: int = 20,
) -> None:
    fig.update_layout(
        title=dict(text=title_text, y=0.95, yanchor="top"),
        template="plotly_dark",
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": left_margin, "r": right_margin, "t": 90, "b": bottom_margin},
    )


def _format_column_label(column_name: str) -> str:
    if str(column_name).lower() == "sofr_assumption":
        return "SOFR"

    replacements = {
        "irr": "IRR",
        "tvpi": "TVPI",
        "nav": "NAV",
        "ncf": "NCF",
        "sofr": "SOFR",
        "pik": "PIK",
        "oid": "OID",
        "uid": "UID",
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
                font-size: 0.9rem !important;
            }
            #schedule_df table tbody td,
            #schedule_df [role="gridcell"] {
                font-size: 0.9rem !important;
                color: #d1d5db !important;
            }
            #schedule_df table thead th:nth-child(6),
            #schedule_df table tbody td:nth-child(6),
            #schedule_df [role="columnheader"][aria-colindex="6"],
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
            #funds_detail_table table thead th,
            #funds_detail_table [role="columnheader"] {
                white-space: normal !important;
                width: max-content !important;
                min-width: max-content !important;
            }
            #funds_detail_table table tbody td,
            #funds_detail_table [role="gridcell"] {
                white-space: nowrap !important;
                width: max-content !important;
                min-width: max-content !important;
            }
            #funds_detail_table table thead th,
            #funds_detail_table [role="columnheader"] {
                background-color: #343a40 !important;
                color: #ffffff !important;
                font-size: 0.8rem !important;
            }
            #funds_detail_table table tbody td,
            #funds_detail_table [role="gridcell"] {
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
            #schedule_df table thead th:nth-child(1),
            #schedule_df table thead th:nth-child(2),
            #schedule_df table tbody td:nth-child(1),
            #schedule_df table tbody td:nth-child(2),
            #schedule_df [role="columnheader"][aria-colindex="1"],
            #schedule_df [role="columnheader"][aria-colindex="2"],
            #schedule_df [role="gridcell"][aria-colindex="1"],
            #schedule_df [role="gridcell"][aria-colindex="2"] {
                width: 10rem !important;
                min-width: 10rem !important;
                max-width: 10rem !important;
            }
            #schedule_df table thead th:nth-child(4),
            #schedule_df table thead th:nth-child(5),
            #schedule_df table thead th:nth-child(14),
            #schedule_df table tbody td:nth-child(4),
            #schedule_df table tbody td:nth-child(5),
            #schedule_df table tbody td:nth-child(14),
            #schedule_df [role="columnheader"][aria-colindex="4"],
            #schedule_df [role="columnheader"][aria-colindex="5"],
            #schedule_df [role="columnheader"][aria-colindex="14"],
            #schedule_df [role="gridcell"][aria-colindex="4"],
            #schedule_df [role="gridcell"][aria-colindex="5"],
            #schedule_df [role="gridcell"][aria-colindex="14"] {
                width: 9rem !important;
                min-width: 9rem !important;
                max-width: 9rem !important;
            }
            #schedule_df table thead th:nth-child(7),
            #schedule_df table thead th:nth-child(8),
            #schedule_df table thead th:nth-child(9),
            #schedule_df table tbody td:nth-child(7),
            #schedule_df table tbody td:nth-child(8),
            #schedule_df table tbody td:nth-child(9),
            #schedule_df [role="columnheader"][aria-colindex="7"],
            #schedule_df [role="columnheader"][aria-colindex="8"],
            #schedule_df [role="columnheader"][aria-colindex="9"],
            #schedule_df [role="gridcell"][aria-colindex="7"],
            #schedule_df [role="gridcell"][aria-colindex="8"],
            #schedule_df [role="gridcell"][aria-colindex="9"] {
                width: 7.5rem !important;
                min-width: 7.5rem !important;
                max-width: 7.5rem !important;
            }
            #schedule_df table thead th:nth-child(10),
            #schedule_df table thead th:nth-child(11),
            #schedule_df table thead th:nth-child(12),
            #schedule_df table thead th:nth-child(13),
            #schedule_df table tbody td:nth-child(10),
            #schedule_df table tbody td:nth-child(11),
            #schedule_df table tbody td:nth-child(12),
            #schedule_df table tbody td:nth-child(13),
            #schedule_df [role="columnheader"][aria-colindex="10"],
            #schedule_df [role="columnheader"][aria-colindex="11"],
            #schedule_df [role="columnheader"][aria-colindex="12"],
            #schedule_df [role="columnheader"][aria-colindex="13"],
            #schedule_df [role="gridcell"][aria-colindex="10"],
            #schedule_df [role="gridcell"][aria-colindex="11"],
            #schedule_df [role="gridcell"][aria-colindex="12"],
            #schedule_df [role="gridcell"][aria-colindex="13"] {
                width: 7.25rem !important;
                min-width: 7.25rem !important;
                max-width: 7.25rem !important;
            }
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
            #funds_detail_table table thead th,
            #funds_detail_table [role="columnheader"],
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
            #funds_detail_table table thead th,
            #funds_detail_table table tbody td,
            #funds_detail_table [role="columnheader"],
            #funds_detail_table [role="gridcell"],
            #funds_summary_table table thead th,
            #funds_summary_table table tbody td,
            #funds_summary_table [role="columnheader"],
            #funds_summary_table [role="gridcell"] {
                padding-left: 10px !important;
                padding-right: 10px !important;
            }
            #schedule_df table thead th:nth-child(2),
            #schedule_df table thead th:nth-child(3),
            #schedule_df table tbody td:nth-child(2),
            #schedule_df table tbody td:nth-child(3),
            #schedule_df [role="columnheader"][aria-colindex="2"],
            #schedule_df [role="columnheader"][aria-colindex="3"],
            #schedule_df [role="gridcell"][aria-colindex="2"],
            #schedule_df [role="gridcell"][aria-colindex="3"] {
                text-align: left !important;
                justify-content: flex-start !important;
            }
            #schedule_df {
                width: 100%;
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
            #funds_detail_table .shiny-data-grid-summary,
            #funds_detail_table .data-grid-summary,
            #funds_detail_table [role="status"],
            #funds_detail_table [aria-live="polite"],
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
            #funds_detail_table table tbody tr,
            #funds_detail_table [role="row"],
            #funds_summary_table table tbody tr,
            #funds_summary_table [role="row"] {
                background-color: #1f2937 !important;
            }
            #schedule_df table tbody tr:nth-child(even),
            #portfolio_outputs_table table tbody tr:nth-child(even),
            #funds_detail_table table tbody tr:nth-child(even),
            #funds_summary_table table tbody tr:nth-child(even) {
                background-color: #253347 !important;
            }
            #schedule_df table, #portfolio_outputs_table table, #funds_detail_table table, #funds_summary_table table {
                border-color: #374151 !important;
            }
            #schedule_df table td, #schedule_df table th,
            #portfolio_outputs_table table td, #portfolio_outputs_table table th,
            #funds_detail_table table td, #funds_detail_table table th,
            #funds_summary_table table td, #funds_summary_table table th {
                border-color: #374151 !important;
            }
            #cashflow_combined_chart,
            #sofr_rate_chart,
            #tvpi_chart,
            #portfolio_irr_chart {
                width: 100%;
                min-width: 0;
                overflow: hidden;
            }
            #cashflow_combined_chart .js-plotly-plot,
            #cashflow_combined_chart .plot-container,
            #cashflow_combined_chart .plotly,
            #sofr_rate_chart .js-plotly-plot,
            #sofr_rate_chart .plot-container,
            #sofr_rate_chart .plotly,
            #tvpi_chart .js-plotly-plot,
            #tvpi_chart .plot-container,
            #tvpi_chart .plotly,
            #portfolio_irr_chart .js-plotly-plot,
            #portfolio_irr_chart .plot-container,
            #portfolio_irr_chart .plotly {
                width: 100% !important;
                max-width: 100% !important;
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
                ui.hr(),
                ui.input_action_button("run_schedule", "Run Schedule", class_="btn-primary"),
                ui.output_ui("run_schedule_status"),
            ),
            ui.p(
                "Schedule of Investments. Click 'Add investment' to add rows to the schedule. Select rows and click 'Remove investment' to delete them. Click 'Reset defaults' to restore the original sample schedule.",
            ),
            ui.div(
                {
                    "style": "display: flex; flex-direction: column; gap: 12px; width: 100%;"
                },
                ui.div(
                    {
                        "style": "display: flex; flex-direction: row; justify-content: center; align-items: flex-start; gap: 20px; width: 100%;"
                    },
                    ui.div(
                        {"style": "flex: 1 1 0; min-width: 0; max-width: 100%;"},
                        ui.h4("Schedule Of Investments", style="font-size: 0.95rem; margin-top: 0; margin-bottom: 0.5rem;"),
                        ui.div(
                            {"style": "overflow: auto; max-width: 100%; height: calc(100vh - 265px); max-height: calc(100vh - 265px);"},
                            ui.output_data_frame("schedule_df"),
                        ),
                        ui.div(
                            {"style": "text-align: right; margin-top: 15px;"},
                            ui.download_button("download_schedule_csv", "Download Loan Schedule CSV"),
                        ),
                    ),
                ),
            ),
            ui.output_ui("schedule_error"),
        ),
    ),
    ui.nav_panel(
        "Investments",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "selected_investment_name",
                    "Investment",
                    choices=[],
                ),
            ),
            ui.div(
                {
                    "style": "display: flex; flex-direction: column; gap: 12px; width: 100%;"
                },
                ui.div(
                    {"style": "flex: 1 1 0; min-width: 0; max-width: 100%;"},
                    ui.h4("Investment Cash Flows", style="font-size: 0.95rem; margin-top: 0; margin-bottom: 0.5rem;"),
                    ui.div(
                        {"style": "overflow: auto; max-width: 100%; max-height: 70vh;"},
                        ui.output_data_frame("funds_detail_table"),
                    ),
                ),
            ),
        ),
    ),
    ui.nav_panel(
        "Gross Cash Flows",
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
            ui.div(
                {
                    "style": "display: flex; flex-direction: row; align-items: stretch; gap: 20px; width: 100%; flex-wrap: nowrap;"
                },
                ui.div(
                    {"style": "flex: 1 1 0; min-width: 0; overflow: hidden;"},
                    output_widget("cashflow_combined_chart"),
                ),
                ui.div(
                    {"style": "flex: 1 1 0; min-width: 0; overflow: hidden;"},
                    output_widget("sofr_rate_chart"),
                ),
            ),
            ui.div(
                {
                    "style": "display: flex; flex-direction: row; align-items: stretch; gap: 20px; width: 100%; flex-wrap: nowrap;"
                },
                ui.div(
                    {"style": "flex: 1 1 0; min-width: 0; overflow: hidden;"},
                    output_widget("tvpi_chart"),
                ),
                ui.div(
                    {"style": "flex: 1 1 0; min-width: 0; overflow: hidden;"},
                    output_widget("portfolio_irr_chart"),
                ),
            ),
            ui.div(
                {"style": "display: flex; flex-direction: row; justify-content: center; align-items: flex-start; gap: 20px; width: 100%; min-width: 0;"},
                ui.div(
                    {"style": "flex: 1 1 0; min-width: 0; max-width: 100%;"},
                    ui.h4("Portfolio Results", style="font-size: 0.95rem;"),
                    ui.div(
                        {"style": "overflow: auto; max-width: 100%; max-height: 70vh;"},
                        ui.output_data_frame("portfolio_outputs_table"),
                    ),
                    ui.div(
                        {"style": "text-align: right; margin-top: 15px;"},
                        ui.download_button("download_portfolio_outputs_csv", "Download Portfolio Results CSV"),
                    ),
                ),
                ui.div(
                    {"style": "flex: 1 1 0; min-width: 0; max-width: 100%;"},
                    ui.h4("Funds Summary", style="font-size: 0.95rem;"),
                    ui.div(
                        {"style": "overflow: auto; max-width: 100%; max-height: 70vh;"},
                        ui.output_data_frame("funds_summary_table"),
                    ),
                    ui.div(
                        {"style": "text-align: right; margin-top: 15px;"},
                        ui.download_button("download_funds_summary_csv", "Download Funds Summary CSV"),
                    ),
                ),
            ),
        ),
    ),
    title=ui.tags.img(src="logo4.PNG", style="height: 40px; width: auto;"),
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

    # Persisted base run results (populated by "Run Schedule" button)
    base_portfolio_df = reactive.value(pd.DataFrame())
    base_funds_df = reactive.value(pd.DataFrame())
    base_funds_summary_df = reactive.value(pd.DataFrame())
    base_schedule_snapshot = reactive.value(pd.DataFrame())
    base_sofr_rates = reactive.value(pd.DataFrame())
    base_spread_rates = reactive.value(pd.DataFrame())
    schedule_has_been_run = reactive.value(False)

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

    PCT_DISPLAY_COLUMNS = ["pik_interest", "amortization", "oid", "exit_fee"]

    @render.data_frame
    def schedule_df():
        display_df = schedule_state().copy()
        for col in ["pik_interest", "amortization", "oid", "exit_fee"]:
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
            width="100%",
            height="100%",
            styles=center_styles,
        )

    @render.download(filename="loan_schedule.csv")
    def download_schedule_csv():
        display_df = schedule_state().copy()
        for col in ["pik_interest", "amortization", "oid", "exit_fee"]:
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

    def _execute_run_schedule():
        global SOFR_RATES

        normalized_schedule = _coerce_schedule(schedule_state())

        max_maturity_date = normalized_schedule["maturity_date"].max()
        end_date_text = pd.Timestamp(max_maturity_date).strftime("%Y-%m-%d")

        needs_actual_sofr = _schedule_uses_actual_sofr(normalized_schedule)
        needs_actual_spread = _schedule_uses_actual_spread(normalized_schedule)

        if (needs_actual_sofr or needs_actual_spread) and not FRED_API_KEY:
            raise ValueError(
                "FRED_API_KEY is required to run the schedule with actual SOFR or spread values."
            )

        if needs_actual_sofr:
            SOFR_RATES = _get_daily_sofr_history(FRED_API_KEY, end_date_text).copy()
            sofr_rates = SOFR_RATES.copy()
        else:
            SOFR_RATES = pd.DataFrame()
            sofr_rates = None

        if needs_actual_spread:
            spread_rates = _get_daily_b_yield_history(FRED_API_KEY, end_date_text).copy()
        else:
            spread_rates = None

        result = loan_portfolio(
            normalized_schedule,
            sofr_rates=sofr_rates,
            spreads=spread_rates,
        )

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

        base_portfolio_df.set(portfolio_df)
        base_funds_df.set(funds_df)
        base_funds_summary_df.set(funds_summary_df)
        base_schedule_snapshot.set(normalized_schedule)
        base_sofr_rates.set(SOFR_RATES.copy())
        base_spread_rates.set(spread_rates.copy())
        schedule_has_been_run.set(True)

    @reactive.effect
    @reactive.event(input.run_schedule)
    def _run_schedule():
        _execute_run_schedule()

    @reactive.effect
    def _run_schedule_on_load():
        if not schedule_has_been_run():
            _execute_run_schedule()

    @render.ui
    def run_schedule_status():
        if schedule_has_been_run():
            return ui.div(
                "Schedule has been run.",
                style="color: #10b981; font-size: 0.8rem; margin-top: 8px;",
            )
        return ui.div(
            "Click 'Run Schedule' to compute results.",
            style="color: #9ca3af; font-size: 0.8rem; margin-top: 8px;",
        )

    @reactive.calc
    def portfolio_results():
        global SOFR_RATES

        if not schedule_has_been_run():
            raise ValueError("Run the schedule on the Loan Schedule tab first.")

        snapshot_schedule = base_schedule_snapshot()
        if snapshot_schedule.empty:
            raise ValueError("Run the schedule on the Loan Schedule tab first.")

        rate_shock_bps = applied_rate_shock_bps()
        shock_start_date = applied_rate_shock_start_date()
        rate_change_bps = applied_rate_change_bps()
        change_start_date = applied_rate_change_start_date()

        has_scenario = bool(rate_shock_bps) or bool(rate_change_bps)

        if not has_scenario:
            SOFR_RATES = base_sofr_rates()
            return base_portfolio_df(), base_funds_df(), base_funds_summary_df()

        # Re-run with scenario adjustments applied to the persisted snapshot
        SOFR_RATES = base_sofr_rates().copy()
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

        shocked_schedule = _apply_rate_shock_to_schedule(
            snapshot_schedule.copy(),
            rate_shock_bps,
            shock_start_date,
        )
        shocked_schedule = _apply_linear_rate_change_to_schedule(
            shocked_schedule,
            rate_change_bps,
            change_start_date,
        )

        sofr_rates = SOFR_RATES.copy()
        spread_rates = base_spread_rates().copy()

        result = loan_portfolio(
            shocked_schedule,
            sofr_rates=sofr_rates,
            spreads=spread_rates,
        )

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
            raise ValueError("Portfolio output is empty.")

        return portfolio_df, funds_df, funds_summary_df

    @reactive.effect
    def _set_default_rate_shock_start_date():
        if not schedule_has_been_run():
            return

        raw_sofr_rates = base_sofr_rates()
        if raw_sofr_rates.empty:
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

    @reactive.calc
    def investment_name_choices() -> list[str]:
        try:
            _, funds_df, _ = portfolio_results()
        except Exception:
            return []

        if funds_df.empty or "investment_name" not in funds_df.columns:
            return []

        return (
            funds_df["investment_name"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

    @reactive.effect
    def _update_selected_investment_name():
        choices = investment_name_choices()
        current_value = input.selected_investment_name()
        selected_value = current_value if current_value in choices else None

        if selected_value is None and choices:
            selected_value = choices[0]

        ui.update_select(
            "selected_investment_name",
            choices=choices,
            selected=selected_value,
            session=session,
        )

    @render_widget
    def cashflow_combined_chart():
        try:
            portfolio_df, _, _ = portfolio_results()
        except Exception as exc:
            return _empty_figure(str(exc))

        cumulative = portfolio_df.copy()
        cumulative["cumulative_cashflow"] = cumulative["total_payment"].cumsum()
        cumulative["status"] = portfolio_df.get("status", "actual")
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
        _add_status_colored_line_traces(
            fig,
            cumulative,
            "quarter_end",
            "cumulative_cashflow",
            "Cumulative Cash Flow",
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
        _apply_standard_chart_layout(fig, "Cash Flow and Remaining Balance")
        fig.update_xaxes(title="Quarter End")
        fig.update_yaxes(title_text="Value")
        return fig

    @render_widget
    def sofr_rate_chart():
        try:
            portfolio_df, _, _ = portfolio_results()
        except Exception as exc:
            return _empty_figure(str(exc))

        portfolio_start_date = pd.to_datetime(
            portfolio_df.get("quarter_end"),
            errors="coerce",
        ).min()

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

        if pd.notna(portfolio_start_date):
            sofr_by_quarter = sofr_by_quarter.loc[
                sofr_by_quarter["date"] >= portfolio_start_date
            ].reset_index(drop=True)

        if sofr_by_quarter.empty:
            return _empty_figure("SOFR rate data is unavailable for the portfolio period.")

        sofr_by_quarter = sofr_by_quarter.rename(columns={"date": "quarter_end"})
        sofr_by_quarter["status"] = (
            sofr_by_quarter["rate_status"]
            .fillna("assumed")
            .astype(str)
            .str.lower()
            .replace({"assumed": "projected"})
        )

        fig = go.Figure()
        _add_status_colored_line_traces(
            fig,
            sofr_by_quarter,
            "quarter_end",
            "sofr",
            "SOFR Rate",
        )
        _apply_standard_chart_layout(
            fig,
            "SOFR Rate",
            bottom_margin=75,
        )
        fig.update_layout(
            annotations=[
                {
                    "text": "source: Federal Reserve Bank of ST. LOUIS",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": -0.2,
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
        tvpi_df["status"] = portfolio_df.get("status", "actual")
        tvpi_df["tvpi"] = tvpi_series
        tvpi_df = tvpi_df.dropna(subset=["tvpi"])

        fig = go.Figure()
        _add_status_colored_line_traces(
            fig,
            tvpi_df,
            "quarter_end",
            "tvpi",
            "TVPI",
        )
        _apply_standard_chart_layout(fig, "Portfolio Gross TVPI")
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
        irr_df["status"] = portfolio_df.get("status", "actual")
        irr_df["irr"] = irr_series
        irr_df = irr_df.dropna(subset=["irr"])

        fig = go.Figure()
        _add_status_colored_line_traces(
            fig,
            irr_df,
            "quarter_end",
            "irr",
            "Portfolio IRR",
        )
        _apply_standard_chart_layout(fig, "Portfolio Gross IRR")
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
        for col in [
            "invested_amount",
            "amortization",
            "cash_interest",
            "fees",
            "total_payment",
            "beginning_balance",
            "ending_balance",
            "nav",
            "contributions",
            "distributions",
            "ncf",
            "cumulative_contributions",
            "cumulative_distributions",
            "cumulative_ncf",
        ]:
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
        for col in [
            "invested_amount",
            "amortization",
            "cash_interest",
            "fees",
            "total_payment",
            "beginning_balance",
            "ending_balance",
            "nav",
            "contributions",
            "distributions",
            "ncf",
            "cumulative_contributions",
            "cumulative_distributions",
            "cumulative_ncf",
        ]:
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
    def funds_detail_table():
        try:
            _, funds_df, _ = portfolio_results()
        except Exception:
            return render.DataGrid(pd.DataFrame(), editable=False, width="fit-content")

        selected_investment_name = input.selected_investment_name()
        if (
            funds_df.empty
            or "investment_name" not in funds_df.columns
            or not selected_investment_name
        ):
            return render.DataGrid(pd.DataFrame(), editable=False, width="fit-content")

        display_df = funds_df.loc[
            funds_df["investment_name"] == selected_investment_name
        ].copy()

        if display_df.empty:
            return render.DataGrid(pd.DataFrame(), editable=False, width="fit-content")

        display_df = display_df.drop(
            columns=["original_investment", "base_rate"],
            errors="ignore",
        )

        for column_name in ["quarter_end", "prepayment_date"]:
            if column_name in display_df.columns:
                display_df[column_name] = pd.to_datetime(
                    display_df[column_name], errors="coerce"
                ).dt.strftime("%Y-%m-%d")
                display_df[column_name] = display_df[column_name].fillna("")

        currency_columns = [
            "par_value",
            "original_investment",
            "invested_amount",
            "beginning_balance",
            "amortization",
            "pik_interest",
            "ending_balance",
            "cash_interest",
            "fees",
            "remaining_balance_payment",
            "total_payment",
            "oid_value",
            "nav_oid",
            "nav",
            "contributions",
            "distributions",
            "ncf",
            "cumulative_contributions",
            "cumulative_distributions",
            "cumulative_ncf",
        ]
        for column_name in currency_columns:
            if column_name in display_df.columns:
                display_df[column_name] = display_df[column_name].apply(
                    lambda value: f"{float(value):,.2f}" if pd.notna(value) else ""
                )

        rate_columns = [
            "sofr_rate",
            "spread",
            "effective_yield",
            "effective_yield_change",
            "pik_rate",
            "cash_interest_rate",
            "oid",
            "exit_fee",
            "irr",
        ]
        for column_name in rate_columns:
            if column_name in display_df.columns:
                display_df[column_name] = display_df[column_name].apply(
                    lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else ""
                )

        if "tvpi" in display_df.columns:
            display_df["tvpi"] = display_df["tvpi"].apply(
                lambda value: f"{float(value):.2f}x" if pd.notna(value) else ""
            )

        display_df = _format_table_headers(display_df)

        centered_columns = [
            "Quarter End",
            "Prepayment Date",
            "Base Rate",
            "Rate Status",
            "SOFR Rate",
            "Spread",
            "PIK Rate",
            "Cash Interest Rate",
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

        for col in ["pik_interest", "amortization", "oid", "exit_fee", "irr"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else ""
                )
        if "spread" in display_df.columns:
            display_df["spread"] = display_df["spread"].apply(
                lambda value: "actual"
                if isinstance(value, str) and value.lower() == "actual"
                else (f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
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

        for col in ["pik_interest", "amortization", "oid", "exit_fee", "irr"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda value: f"{float(value) * 100:.2f}%" if pd.notna(value) else ""
                )
        if "spread" in display_df.columns:
            display_df["spread"] = display_df["spread"].apply(
                lambda value: "actual"
                if isinstance(value, str) and value.lower() == "actual"
                else (f"{float(value) * 100:.2f}%" if pd.notna(value) else "")
            )

        display_df = _format_table_headers(display_df)
        yield display_df.to_csv(index=False)


app = App(app_ui, server, static_assets=Path(__file__).resolve().parent)
