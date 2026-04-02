# %%

"""Private credit cash flow and portfolio modeling utilities.

This module contains the public entry points used to build single-loan cash
flow schedules and aggregate them into a portfolio view. The implementation is
quarterly by construction and supports either observed SOFR data or fixed rate
assumptions.

Notes
-----
The private credit model assumes that amortization is applied to original par,
PIK interest accretes to balance each quarter, and exit fees are charged at the
end of the loan term or prepayment event.

investment_date: (t0) the date the investment is made, and the initial cash outflow occurs (no payment occurs at this time, but the cash outflow is recorded as a negative payment at this date)
amortization: applied to the original loan amount, not the remaining balance
pik_interest: applied to the remaining balance at the end of each quarter, added to the balance (Fixed PIK interest, not Floating PIK interest)
cash_interest: applied to the remaining balance at the end of each quarter, paid in cash
exit_fee: applied to the remaining balance at the end of loan, paid in cash
oid_income: Not amortized, recognized at the time of investment, reduces the initial cash outflow
prepayment: if prepayment occurs, the loan is paid off at the prepayment date, and no further payments occur after the prepayment date

Total_Rate = SOFR (4) + Spread (6) = Cash Interest Rate (8) + PIK Interest Rate (2)
"""

import pandas as pd
from .utilities import _irr_quarterly, _format_irr, _get_periodicity_config, _build_payment_dates, _prepare_sofr_table, _resolve_sofr_rate

# import pandas as pd
# import sys
# import os
# from pathlib import Path
# from utilities import _irr_quarterly, _format_irr, _get_periodicity_config, _build_payment_dates, _prepare_sofr_table, _resolve_sofr_rate
# sys.path.append(str(Path(__file__).resolve().parent.parent))
# from data.sofr import get_sofr_data
# fred_api_key = os.getenv("FRED_API_KEY")

def private_credit_loan_model(
    investment_name,
    investment_date,
    maturity_date,
    par_value,
    spread,
    base_rate,
    sofr_rates,
    # sofr_floor,
    pik_interest,
    amortization,
    oid,
    exit_fee,
    prepayment_date
):
    """Build a quarterly cash flow schedule for a single private credit loan.

    Parameters
    ----------
    investment_name : str
        Display name used to identify the loan in returned tables.
    investment_date : str or pandas.Timestamp
        Quarter-end date on which the investment is funded and the initial
        negative cash flow is recorded.
    maturity_date : str or pandas.Timestamp
        Contractual maturity date for the loan. Must be after
        ``investment_date``.
    par_value : float
        Original principal balance of the loan.
    spread : float
        Annualized credit spread expressed as a decimal.
    base_rate : str
        Name of the base reference rate. When set to ``"SOFR"``, the model
        attempts to resolve quarterly SOFR values from ``sofr_rates``.
    sofr_rates : pandas.DataFrame or float or None
        SOFR source used by the model. Provide a DataFrame with ``date`` and
        ``sofr`` columns for observed rates, a float for a fixed annualized
        assumption, or ``None`` to use the default assumption of ``0.04``.
    pik_interest : float
        Annualized PIK rate expressed as a decimal.
    amortization : float
        Annual amortization rate applied to original par value.
    oid : float
        Original issue discount expressed as a decimal reduction to funded
        proceeds.
    exit_fee : float
        Exit fee charged at maturity or prepayment, expressed as a decimal of
        original par value.
    prepayment_date : str or pandas.Timestamp or None
        Optional loan payoff date. When provided, cash flows stop on this date
        instead of the contractual maturity date.

    Returns
    -------
    pandas.DataFrame
        Quarterly loan cash flows including beginning and ending balances,
        component payments, resolved SOFR information, and a running IRR.

    Raises
    ------
    ValueError
        Raised when ``maturity_date`` is not after ``investment_date`` or when
        ``par_value`` is not positive.

    Notes
    -----
    The modeled quarterly rate is defined as
    ``cash_interest_rate = SOFR + spread - pik_interest``. Amortization is
    calculated on original par value rather than current balance.
    """

    # investment_name="Example Corp Term Loan"
    # investment_date=pd.Timestamp("2024-12-31")
    # maturity_date=pd.Timestamp("2029-12-31")
    # par_value=1_000_000
    # spread=0.06
    # base_rate='SOFR'
    # # sofr_assumption=0.04
    # # sofr_floor = 0.01
    # pik_interest=0.02
    # amortization=0.01
    # oid=0.02
    # exit_fee=0.02
    # prepayment_date=None
    # sofr_rates=get_sofr_data(api_key=fred_api_key, frequency='D')

    if prepayment_date is None or pd.isna(prepayment_date):
        end_date = maturity_date
    else:
        end_date = prepayment_date

    investment_date = pd.Timestamp(investment_date).normalize()
    maturity_date = pd.Timestamp(maturity_date).normalize()

    end_date = pd.Timestamp(end_date).normalize()

    if maturity_date <= investment_date:
        raise ValueError("maturity_date must be after investment_date")
    if par_value <= 0:
        raise ValueError("par_value must be positive")

    base_rate_name = str(base_rate).strip().upper()

    if sofr_rates is not None:
        if isinstance(sofr_rates, pd.DataFrame):
            prepared_sofr_rates = _prepare_sofr_table(sofr_rates)
            prepared_sofr_rates = prepared_sofr_rates.sort_values(by='date', ascending=False).reset_index(drop=True)
            sofr_assumption = prepared_sofr_rates['sofr'].iloc[0] if base_rate_name == "SOFR" else .04
        elif isinstance(sofr_rates, (int, float)):
            prepared_sofr_rates = None
            sofr_assumption = float(sofr_rates)
    else:
        prepared_sofr_rates = None
        sofr_assumption = 0.04   

    quarter_ends = pd.date_range(
        start=investment_date,
        end=end_date,
        freq="QE",
    )

    if len(quarter_ends) == 0:
        quarter_ends = pd.DatetimeIndex([end_date])

    quarterly_amortization = par_value * (amortization / 4.0)
    # quarterly_pik_interest = par_value * (pik_interest / 4.0)

    rows = []
    beginning_balance = float(par_value)

    for quarter_end in quarter_ends:

        if base_rate_name == "SOFR":
            if isinstance(sofr_rates, pd.DataFrame):
                resolved_sofr, rate_status = _resolve_sofr_rate(quarter_end, prepared_sofr_rates)
                if resolved_sofr is None:
                    resolved_sofr = sofr_assumption
                    rate_status = "assumed"
            else:
                resolved_sofr = sofr_assumption
                rate_status = "assumed"
        else:
            resolved_sofr = 0.04
            rate_status = "assumed"

        if quarter_end == investment_date:
            invested_amount = par_value * (1 - oid)
            sofr_rate = 0
            spread_rate = 0
            pik_rate = 0
            cash_interest_rate = 0
            amortization_amount = 0
            pik_interest_amount = 0
            cash_interest_amount = 0
            ending_balance = par_value
            fees = 0.0
            remaining_balance_payment = 0
            total_payment = -(par_value * (1 - oid))
        else:
            invested_amount = par_value * (1 - oid)
            sofr_rate = resolved_sofr
            spread_rate = spread
            pik_rate = pik_interest
            cash_interest_rate = sofr_rate + spread_rate - pik_rate
            amortization_amount = quarterly_amortization
            pik_interest_amount = beginning_balance * (pik_interest / 4.0)
            cash_interest_amount = beginning_balance * (cash_interest_rate / 4.0)
            ending_balance = beginning_balance - amortization_amount + pik_interest_amount
            
            if quarter_end == end_date:
                fees = par_value * (exit_fee)
                remaining_balance_payment = ending_balance
            else:
                fees = 0.0
                remaining_balance_payment = 0

            total_payment = cash_interest_amount + amortization_amount + fees + remaining_balance_payment

        rows.append(
            {
                "investment_name": investment_name,
                "quarter_end": quarter_end,
                "par_value": par_value,
                "invested_amount": invested_amount,
                "base_rate": base_rate,
                "sofr_rate": resolved_sofr,
                "rate_status": rate_status,
                "spread": spread_rate,
                "pik_rate": pik_rate,
                "cash_interest_rate": cash_interest_rate,
                "oid": oid,
                "exit_fee": exit_fee,
                "prepayment_date": prepayment_date,
                "beginning_balance": beginning_balance,
                "amortization": amortization_amount,
                "pik_interest": pik_interest_amount,
                "ending_balance": ending_balance,
                "cash_interest": cash_interest_amount,
                "fees": fees,
                "remaining_balance_payment": remaining_balance_payment,
                "total_payment": total_payment
            }
        )

        beginning_balance = ending_balance

    loan_df = pd.DataFrame(rows)

    irr_values = []
    total_payments = loan_df["total_payment"].tolist()
    remaining_balance_payments = loan_df["remaining_balance_payment"].tolist()
    adjusted_payments = [
        total_payment - remaining_balance_payment
        for total_payment, remaining_balance_payment in zip(
            total_payments,
            remaining_balance_payments,
        )
    ]
    ending_balances = loan_df["ending_balance"].tolist()

    for idx, _ in enumerate(adjusted_payments):
        cumulative_cashflows = adjusted_payments[: idx + 1]
        cumulative_cashflows.append(ending_balances[idx])
        quarterly_irr = _irr_quarterly(cumulative_cashflows)
        irr_values.append(_format_irr(quarterly_irr, idx + 1))

    loan_df["irr"] = irr_values
    loan_df.at[0, "irr"] = None
    return loan_df

def loan_portfolio(schedule_of_investments, sofr_rates=None):
    """Aggregate multiple loans into portfolio-level output tables.

    Parameters
    ----------
    schedule_of_investments : pandas.DataFrame or mapping
        Tabular loan schedule with one row per investment. Required columns are
        ``investment_name``, ``investment_date``, ``maturity_date``,
        ``par_value``, ``spread``, ``base_rate``, ``sofr_assumption``,
        ``pik_interest``, ``amortization``, ``oid``, ``exit_fee``, and
        ``prepayment_date``.
    sofr_rates : pandas.DataFrame or None, default=None
        Optional SOFR history used for loans whose ``sofr_assumption`` is set to
        ``"actual"``.

    Returns
    -------
    tuple of pandas.DataFrame
        Three DataFrames containing:

        - a portfolio-level quarterly summary,
        - loan-level quarterly cash flows for all investments,
        - a one-row-per-loan summary with total payment and terminal IRR.

    Raises
    ------
    ValueError
        Raised when the investment schedule does not include the required
        columns.
    """

    ####
    # schedule_of_investments = pd.DataFrame(
    #     {
    #         "investment_name": ["A Corp Term Loan", "B Corp Term Loan",  "C Corp Term Loan",  "D Corp Term Loan",  "E Corp Term Loan"],
    #         "investment_date": [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30"), pd.Timestamp("2020-09-30"), pd.Timestamp("2020-12-31"), pd.Timestamp("2021-03-31")],
    #         "maturity_date": [pd.Timestamp("2025-06-30"), pd.Timestamp("2027-06-30"), pd.Timestamp("2026-12-31"), pd.Timestamp("2028-03-31"), pd.Timestamp("2029-06-30")],
    #         "par_value": [1_000_000, 1_250_000, 1_000_000, 1_050_000, 1_600_000],
    #         "spread": [0.07, 0.06, 0.06, 0.08, 0.07],
    #         "base_rate": ['SOFR', 'SOFR', 'SOFR', 'SOFR', 'SOFR'],
    #         "sofr_assumption": ['actual', 'actual', 'actual', 'actual', 'actual'],
    #         "pik_interest": [0.02, 0.02, 0.02, 0.02, 0.02],
    #         "amortization": [0.01, 0.01, 0.01, 0.01, 0.01],
    #         "oid": [0.03, 0.02, 0.03, 0.04, 0.02],
    #         "exit_fee": [0.02, 0.02, 0.02, 0.02, 0.02],
    #         "prepayment_date": [None, None, None, None, None],
    #     }
    # )
    # sofr_rates=get_sofr_data(api_key=fred_api_key, frequency='D')
    ####

    if not isinstance(schedule_of_investments, pd.DataFrame):
        schedule_of_investments = pd.DataFrame(schedule_of_investments)

    required_columns = [
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
        "prepayment_date"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in schedule_of_investments.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    loan_cashflow_frames = []
    for _, investment in schedule_of_investments.iterrows():
        if investment["sofr_assumption"] == "actual":
            sofr_rates_for_loan = sofr_rates
        elif isinstance(investment["sofr_assumption"], (int, float)):
            sofr_rates_for_loan = float(investment["sofr_assumption"])
        else:
            sofr_rates_for_loan = 0.04

        loan_cashflow_frames.append(
            private_credit_loan_model(
                investment_name=investment["investment_name"],
                investment_date=investment["investment_date"],
                maturity_date=investment["maturity_date"],
                par_value=investment["par_value"],
                spread=investment["spread"],
                base_rate=investment["base_rate"],
                sofr_rates=sofr_rates_for_loan,
                # cash_interest_rate=investment["cash_interest_rate"],
                pik_interest=investment["pik_interest"],
                amortization=investment["amortization"],
                oid=investment["oid"],
                exit_fee=investment["exit_fee"],
                prepayment_date=investment["prepayment_date"],
            )
        )

    if not loan_cashflow_frames:
        return pd.DataFrame([])

    funds_df = pd.concat(loan_cashflow_frames, ignore_index=True).reset_index(drop=True)

    portfolio_df = funds_df.groupby("quarter_end").agg(
        {
            "invested_amount": "sum",
            "total_payment": "sum",
            "remaining_balance_payment": "sum",
            "ending_balance": "sum",
            "beginning_balance": "sum"
        }
    ).reset_index()

    portfolio_df['irr'] = None
    paid_in_capital = (-portfolio_df["total_payment"].clip(upper=0)).cumsum()
    cumulative_distributions = portfolio_df["total_payment"].clip(lower=0).cumsum()
    portfolio_df["tvpi"] = (
        (cumulative_distributions + portfolio_df["ending_balance"])
        / paid_in_capital.where(paid_in_capital != 0)
    )

    for i in range(1, len(portfolio_df)):
        # i = 5
        quarter_date = portfolio_df.at[i, "quarter_end"]
        cfs = portfolio_df[portfolio_df["quarter_end"] <= quarter_date].copy()
        cfs['irr_cfs'] = cfs['total_payment']
        cfs.at[i, 'irr_cfs'] = cfs.at[i, 'irr_cfs'] - cfs.at[i, 'remaining_balance_payment'] + cfs.at[i, 'ending_balance']
        quarterly_irr = _irr_quarterly(cfs['irr_cfs'].tolist())
        portfolio_df.at[i, 'irr'] = _format_irr(quarterly_irr, i)

    funds_summary_df = (
        funds_df
        .sort_values(["investment_name", "quarter_end"], kind="stable")
        .groupby("investment_name", as_index=False)
        .agg(
            total_payment=("total_payment", "sum"),
            irr=("irr", "last"),
        )
    )

    schedule_summary_columns = [
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
    schedule_summary_df = (
        schedule_of_investments[schedule_summary_columns]
        .drop_duplicates(subset=["investment_name"])
        .copy()
    )
    funds_summary_df = schedule_summary_df.merge(
        funds_summary_df,
        on="investment_name",
        how="left",
    )

    return portfolio_df, funds_df, funds_summary_df