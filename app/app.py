from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from asset_modeling.credit import loan_portfolio

REQUIRED_COLUMNS = [
    "investment_name",
    "investment_date",
    "maturity_date",
    "loan_size",
    "spread",
    "base_rate",
    "sofr_assumption",
    "cash_interest_rate",
    "pik_interest",
    "amortization",
    "oid",
    "exit_fee",
    "prepayment_date",
]

NUMERIC_COLUMNS = [
    "loan_size",
    "spread",
    "sofr_assumption",
    "cash_interest_rate",
    "pik_interest",
    "amortization",
    "oid",
    "exit_fee",
]


def _default_row(index: int) -> dict:
    return {
        "investment_name": f"Loan {index}",
        "investment_date": "2024-03-31",
        "maturity_date": "2029-03-31",
        "loan_size": 100000000,
        "spread": 0.06,
        "base_rate": "SOFR",
        "sofr_assumption": 0.04,
        "cash_interest_rate": 0.08,
        "pik_interest": 0.02,
        "amortization": 0.01,
        "oid": 0.02,
        "exit_fee": 0.02,
        "prepayment_date": "",
    }


def default_schedule() -> pd.DataFrame:
    return pd.DataFrame([_default_row(1), _default_row(2)])



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

    for col in ["investment_date", "maturity_date"]:
        working_df[col] = pd.to_datetime(working_df[col], errors="coerce")
        if working_df[col].isna().any():
            raise ValueError(f"Column '{col}' has invalid dates.")

    prepayment = pd.to_datetime(working_df["prepayment_date"], errors="coerce")
    working_df["prepayment_date"] = prepayment.astype("object").where(~prepayment.isna(), None)

    for col in NUMERIC_COLUMNS:
        normalized_numeric = working_df[col].apply(_normalize_numeric_value)
        working_df[col] = pd.to_numeric(normalized_numeric, errors="coerce").astype(float)
        if working_df[col].isna().any():
            raise ValueError(
                f"Column '{col}' must be numeric (examples: 1000000, 1,000,000, 8%)."
            )

    if (working_df["loan_size"] <= 0).any():
        raise ValueError("'loan_size' must be greater than 0.")

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
    fig.update_layout(template="plotly_white", margin={"l": 20, "r": 20, "t": 30, "b": 20})
    return fig


def _clean_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.apply(lambda x: not (isinstance(x, float) and math.isnan(x))))


app_ui = ui.page_navbar(
    ui.nav_panel(
        "Loan Schedule",
        ui.p(
            "Edit all rows/columns directly. Changes feed the portfolio outputs page.",
        ),
        ui.layout_columns(
            ui.input_action_button("add_row", "Add row"),
            ui.input_action_button("remove_last_row", "Remove last row"),
            ui.input_action_button("reset_schedule", "Reset defaults"),
        ),
        ui.output_data_frame("schedule_df"),
        ui.output_ui("schedule_error"),
    ),
    ui.nav_panel(
        "Portfolio Outputs",
        ui.output_ui("portfolio_error"),
        ui.layout_columns(
            output_widget("cashflow_chart"),
            output_widget("cumulative_chart"),
            col_widths=(6, 6),
        ),
        ui.layout_columns(
            output_widget("balance_chart"),
            output_widget("interest_principal_chart"),
            col_widths=(6, 6),
        ),
        ui.h4("IRR Summary"),
        ui.output_table("irr_summary"),
    ),
    title="Private Credit Portfolio App",
)


def server(input, output, session):
    schedule_state = reactive.value(default_schedule())

    @reactive.effect
    @reactive.event(input.add_row)
    def _add_row():
        df = schedule_state().copy()
        df.loc[len(df)] = _default_row(len(df) + 1)
        schedule_state.set(df)

    @reactive.effect
    @reactive.event(input.remove_last_row)
    def _remove_last_row():
        df = schedule_state().copy()
        if len(df) > 1:
            schedule_state.set(df.iloc[:-1].reset_index(drop=True))

    @reactive.effect
    @reactive.event(input.reset_schedule)
    def _reset_schedule():
        schedule_state.set(default_schedule())

    @render.data_frame
    def schedule_df():
        return render.DataGrid(schedule_state(), editable=True)

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

    @reactive.calc
    def portfolio_results():
        normalized_schedule = _coerce_schedule(schedule_state())
        result = loan_portfolio(normalized_schedule)

        if isinstance(result, tuple):
            portfolio_df, funds_df = result
        else:
            portfolio_df, funds_df = pd.DataFrame(), pd.DataFrame()

        if portfolio_df.empty:
            raise ValueError("Portfolio output is empty. Add valid schedule rows.")

        return portfolio_df, funds_df

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
    def cashflow_chart():
        try:
            portfolio_df, _ = portfolio_results()
        except Exception as exc:
            return _empty_figure(str(exc))

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=portfolio_df["quarter_end"],
                y=portfolio_df["total_payment"],
                name="Total Payment",
            )
        )
        fig.update_layout(title="Cash Flow Over Time", template="plotly_white")
        fig.update_xaxes(title="Quarter End")
        fig.update_yaxes(title="Cash Flow")
        return fig

    @render_widget
    def cumulative_chart():
        try:
            portfolio_df, _ = portfolio_results()
        except Exception as exc:
            return _empty_figure(str(exc))

        cumulative = portfolio_df.copy()
        cumulative["cumulative_cashflow"] = cumulative["total_payment"].cumsum()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cumulative["quarter_end"],
                y=cumulative["cumulative_cashflow"],
                mode="lines+markers",
                name="Cumulative Cash Flow",
            )
        )
        fig.update_layout(title="Cumulative Cash Flow", template="plotly_white")
        fig.update_xaxes(title="Quarter End")
        fig.update_yaxes(title="Cumulative Cash Flow")
        return fig

    @render_widget
    def balance_chart():
        try:
            portfolio_df, _ = portfolio_results()
        except Exception as exc:
            return _empty_figure(str(exc))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=portfolio_df["quarter_end"],
                y=portfolio_df["ending_balance"],
                mode="lines+markers",
                name="Outstanding Balance",
            )
        )
        fig.update_layout(title="Outstanding Balance", template="plotly_white")
        fig.update_xaxes(title="Quarter End")
        fig.update_yaxes(title="Balance")
        return fig

    @render_widget
    def interest_principal_chart():
        try:
            _, funds_df = portfolio_results()
        except Exception as exc:
            return _empty_figure(str(exc))

        grouped = (
            funds_df.groupby("quarter_end", as_index=False)
            .agg(
                {
                    "cash_interest": "sum",
                    "amortization": "sum",
                    "remaining_balance_payment": "sum",
                }
            )
            .sort_values("quarter_end")
        )
        grouped["principal_component"] = (
            grouped["amortization"] + grouped["remaining_balance_payment"]
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=grouped["quarter_end"],
                y=grouped["cash_interest"],
                name="Interest",
            )
        )
        fig.add_trace(
            go.Bar(
                x=grouped["quarter_end"],
                y=grouped["principal_component"],
                name="Principal",
            )
        )
        fig.update_layout(
            barmode="stack",
            title="Interest vs Principal",
            template="plotly_white",
        )
        fig.update_xaxes(title="Quarter End")
        fig.update_yaxes(title="Cash Amount")
        return fig

    @render.table
    def irr_summary():
        try:
            portfolio_df, _ = portfolio_results()
        except Exception:
            return pd.DataFrame([{"latest_irr": None, "max_irr": None, "min_irr": None}])

        irr_values = _clean_series(portfolio_df["irr"])
        irr_values = irr_values.dropna()

        if irr_values.empty:
            return pd.DataFrame([{"latest_irr": None, "max_irr": None, "min_irr": None}])

        return pd.DataFrame(
            [
                {
                    "latest_irr": float(irr_values.iloc[-1]),
                    "max_irr": float(irr_values.max()),
                    "min_irr": float(irr_values.min()),
                }
            ]
        )


app = App(app_ui, server)
