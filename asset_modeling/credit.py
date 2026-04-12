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
from .utilities import _irr_quarterly, _format_irr, _get_periodicity_config, _build_payment_dates, _prepare_sofr_table, _resolve_sofr_rate, npv

# import pandas as pd
# import sys
# import os
# from pathlib import Path
# from utilities import _irr_quarterly, _format_irr, _get_periodicity_config, _build_payment_dates, _prepare_sofr_table, _resolve_sofr_rate, npv
# sys.path.append(str(Path(__file__).resolve().parent.parent))
# from data.sofr import get_sofr_data
# from data.ratings import get_effective_yield
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
    prepayment_date,
    company_uid=None,
    investment_uid=None,
    as_of_date=None,
):
    """Build a quarterly cash flow schedule for a single private credit loan.

    Parameters
    ----------
    investment_name : str
        Display name used to identify the loan in returned tables.
    company_uid : str or None, default=None
        Optional company identifier carried through to the returned cash flow
        schedule.
    investment_uid : str or None, default=None
        Optional investment identifier carried through to the returned cash
        flow schedule.
    investment_date : str or pandas.Timestamp
        Quarter-end date on which the investment is funded and the initial
        negative cash flow is recorded.
    maturity_date : str or pandas.Timestamp
        Contractual maturity date for the loan. Must be after
        ``investment_date``.
    par_value : float
        Original principal balance of the loan.
    spread : pandas.DataFrame or float or None
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
    as_of_date : str or pandas.Timestamp or None, default=None
        Reference date used to label each quarter-end row as either
        ``"actual"`` or ``"projected"``. When omitted, the current date is
        used.

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

    NAV is calculated as the initial investment + accrued OID value + accrued PIK interest adjusted for change in spread.
        - NAV goes to zero on maturity or prepayment date.
    """

    # investment_name="Example Corp Term Loan"
    # investment_date=pd.Timestamp("2024-12-31")
    # maturity_date=pd.Timestamp("2029-12-31")
    # par_value=1_000_000
    # # spread=0.06
    # base_rate='SOFR'
    # # sofr_assumption=0.04
    # # sofr_floor = 0.01
    # pik_interest=0.02
    # amortization=0.01
    # oid=0.02
    # exit_fee=0.02
    # prepayment_date=None
    # sofr_rates=get_sofr_data(api_key=fred_api_key, frequency='D', end_date=maturity_date)
    # spread=get_effective_yield(rating="B", api_key=fred_api_key, frequency='D', end_date=maturity_date)
    # as_of_date=pd.Timestamp("2026-03-31")

    if prepayment_date is None or (isinstance(prepayment_date, str) and prepayment_date.strip() == "") or pd.isna(prepayment_date):
        end_date = maturity_date
    else:
        end_date = prepayment_date

    investment_date = pd.Timestamp(investment_date).normalize()
    if as_of_date is None or pd.isna(as_of_date):
        as_of_date = pd.Timestamp.today().normalize()
    else:
        as_of_date = pd.Timestamp(as_of_date).normalize()
    maturity_date = pd.Timestamp(maturity_date).normalize()

    end_date = pd.Timestamp(end_date)
    if pd.isna(end_date):
        end_date = pd.Timestamp(maturity_date)
    end_date = end_date.normalize()

    if maturity_date <= investment_date:
        raise ValueError("maturity_date must be after investment_date")
    if par_value <= 0:
        raise ValueError("par_value must be positive")

    spread_table = None
    if isinstance(spread, pd.DataFrame):
        spread_table = spread.copy()
        spread_table["date"] = pd.to_datetime(spread_table["date"]).dt.normalize()
        if "yield_status" not in spread_table.columns:
            if "rate_status" in spread_table.columns:
                spread_table = spread_table.rename(columns={"rate_status": "yield_status"})
            else:
                spread_table["yield_status"] = "actual"
        spread_table = spread_table.sort_values("date", ascending=False).reset_index(drop=True)
        match = spread_table.loc[spread_table["date"] == investment_date, "effective_yield"]
        if not match.empty:
            spread = float(match.iloc[0])
        else:
            prior = spread_table.loc[spread_table["date"] <= investment_date]
            if not prior.empty:
                spread = float(prior.iloc[0]["effective_yield"])
            else:
                spread = float(spread_table.iloc[-1]["effective_yield"])
    else:
        spread = float(spread)

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
    original_investment = par_value * (1 - oid)
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
            invested_amount = original_investment
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
            total_payment = -original_investment
        else:
            invested_amount = 0.0
            sofr_rate = resolved_sofr
            spread_rate = spread
            pik_rate = pik_interest
            cash_interest_rate = sofr_rate + spread_rate - pik_rate
            # cash_interest_rate = sofr_rate + spread_rate
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
                "company_uid": company_uid,
                "investment_uid": investment_uid,
                "quarter_end": quarter_end,
                "par_value": par_value,
                "original_investment": original_investment,
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
    loan_df["status"] = "projected"
    loan_df.loc[loan_df["quarter_end"] <= as_of_date, "status"] = "actual"

    if spread_table is not None:
        loan_df = loan_df.merge(
            spread_table[["date", "effective_yield", "yield_status"]],
            left_on="quarter_end",
            right_on="date",
            how="left",
        ).drop(columns="date")
    else:
        loan_df["effective_yield"] = float(spread)
        loan_df["yield_status"] = "fixed"

    oid_total_value = par_value - original_investment
    num_periods = len(loan_df) - 1
    loan_df["oid_value"] = 0.0
    if num_periods > 0:
        loan_df.loc[1:, "oid_value"] = oid_total_value / num_periods

    loan_df["nav_oid"] = 0.0
    loan_df.at[0, "nav_oid"] = original_investment
    for i in range(1, len(loan_df)):
        loan_df.at[i, "nav_oid"] = loan_df.at[i - 1, "nav_oid"] + loan_df.at[i, "oid_value"]

    # loan_df["nav_oid_pik"] = loan_df["nav_oid"] + loan_df["pik_interest"]

    loan_df["effective_yield_change"] = loan_df["effective_yield"] - spread
    loan_df["nav"] = loan_df["nav_oid"] * (1 - loan_df["effective_yield_change"]) #######
    loan_df.loc[len(loan_df) - 1, "nav"] = 0
    loan_df["contributions"] = loan_df["invested_amount"] * -1.0
    loan_df["distributions"] = loan_df["cash_interest"] + loan_df["amortization"] + loan_df["fees"] + loan_df["remaining_balance_payment"]
    loan_df["ncf"] = loan_df["contributions"] + loan_df["distributions"]
    loan_df["cumulative_contributions"] = loan_df["contributions"].cumsum()
    loan_df["cumulative_distributions"] = loan_df["distributions"].cumsum()
    loan_df["cumulative_ncf"] = loan_df["ncf"].cumsum()
    loan_df["tvpi"] = (loan_df["cumulative_distributions"] + loan_df["nav"]) / -loan_df["cumulative_contributions"]

    # Effective Duration

    # npv
    loan_df["effective_duration"] = None
    npv_0 = npv(spread/4, loan_df["distributions"].tolist())

    for i in range(1, len(loan_df)):
        # i = 1
        # up scenario
        loan_df_duration = loan_df.copy()
        date_duration = loan_df_duration.loc[i, "quarter_end"]
        spread_up = spread + 0.01
        loan_df_duration.loc[i:(len(loan_df_duration) - 1), "spread"] = spread_up
        loan_df_duration["cash_interest_rate"] = loan_df_duration["sofr_rate"] + loan_df_duration["spread"] - loan_df_duration["pik_rate"]
        loan_df_duration["cash_interest"] = loan_df_duration["beginning_balance"] * (loan_df_duration["cash_interest_rate"] / 4.0)
        loan_df_duration.loc[0, "cash_interest"] = 0.0
        loan_df_duration["distributions"] = loan_df_duration["cash_interest"] + loan_df_duration["amortization"] + loan_df_duration["fees"] + loan_df_duration["remaining_balance_payment"]
        npv_up = npv(spread/4, loan_df_duration["distributions"].tolist())

        # down scenario
        loan_df_duration = loan_df.copy()
        date_duration = loan_df_duration.loc[i, "quarter_end"]
        spread_down = spread - 0.01
        loan_df_duration.loc[i:(len(loan_df_duration) - 1), "spread"] = spread_down
        loan_df_duration["cash_interest_rate"] = loan_df_duration["sofr_rate"] + loan_df_duration["spread"] - loan_df_duration["pik_rate"]
        loan_df_duration["cash_interest"] = loan_df_duration["beginning_balance"] * (loan_df_duration["cash_interest_rate"] / 4.0)
        loan_df_duration.loc[0, "cash_interest"] = 0.0
        loan_df_duration["distributions"] = loan_df_duration["cash_interest"] + loan_df_duration["amortization"] + loan_df_duration["fees"] + loan_df_duration["remaining_balance_payment"]
        npv_down = npv(spread/4, loan_df_duration["distributions"].tolist())

        loan_df.loc[i, "effective_duration"] = (npv_up - npv_down) / (2 * npv_0 * 0.01)

    # IRR calculation: for each quarter, calculate the IRR of all cash flows up to that quarter, treating the NAV at that quarter as a final positive cash flow
    
    loan_df["nav"] = loan_df["nav_oid"] * (1 - (loan_df["effective_yield_change"] * loan_df["effective_duration"])/100) #######
    loan_df.loc[len(loan_df) - 1, "nav"] = 0
    loan_df.loc[0, "nav"] = loan_df.loc[0, "original_investment"]
    
    irr_values = []
    ncf = loan_df["ncf"].tolist()
    nav = loan_df["nav"].tolist()

    for idx, _ in enumerate(ncf):
        cumulative_cashflows = ncf[: idx + 1]
        cumulative_cashflows.append(nav[idx])
        quarterly_irr = _irr_quarterly(cumulative_cashflows)
        irr_values.append(_format_irr(quarterly_irr, idx + 1))

    loan_df["irr"] = irr_values
    loan_df.at[0, "irr"] = None
    return loan_df

def loan_portfolio(schedule_of_investments, sofr_rates=None, spreads=None, as_of_date=None):
    """Aggregate multiple loans into portfolio-level output tables.

    Parameters
    ----------
    schedule_of_investments : pandas.DataFrame or mapping
        Tabular loan schedule with one row per investment. Required columns are
        ``investment_name``, ``company_uid``, ``investment_uid``,
        ``investment_date``, ``maturity_date``, ``par_value``, ``spread``,
        ``base_rate``, ``sofr_assumption``, ``pik_interest``,
        ``amortization``, ``oid``, ``exit_fee``, and ``prepayment_date``.
    sofr_rates : pandas.DataFrame or None, default=None
        Optional SOFR history used for loans whose ``sofr_assumption`` is set to
        ``"actual"``.
    spreads : pandas.DataFrame or None, default=None
        Optional spread history used for loans whose ``spread`` is set to
        ``"actual"``. Expected to be the output of
        ``data.ratings.get_effective_yield()``.
    as_of_date : str or pandas.Timestamp or None, default=None
        Reference date forwarded to ``private_credit_loan_model()`` to label
        each cash flow row as ``"actual"`` or ``"projected"``.

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
    #         "spread": ['actual', 'actual', 'actual', 'actual', 'actual'],
    #         "base_rate": ['SOFR', 'SOFR', 'SOFR', 'SOFR', 'SOFR'],
    #         "sofr_assumption": ['actual', 'actual', 'actual', 'actual', 'actual'],
    #         "pik_interest": [0.02, 0.02, 0.02, 0.02, 0.02],
    #         "amortization": [0.01, 0.01, 0.01, 0.01, 0.01],
    #         "oid": [0.03, 0.02, 0.03, 0.04, 0.02],
    #         "exit_fee": [0.02, 0.02, 0.02, 0.02, 0.02],
    #         "prepayment_date": [None, None, None, None, None],
    #     }
    # )
    # sofr_rates=get_sofr_data(api_key=fred_api_key, frequency='D', end_date=schedule_of_investments['maturity_date'].max())
    # spreads=get_effective_yield(rating="B", api_key=fred_api_key, frequency='D', end_date=schedule_of_investments['maturity_date'].max())

    ####

    if not isinstance(schedule_of_investments, pd.DataFrame):
        schedule_of_investments = pd.DataFrame(schedule_of_investments)

    if as_of_date is None or pd.isna(as_of_date):
        as_of_date = pd.Timestamp.today().normalize()
    else:
        as_of_date = pd.Timestamp(as_of_date).normalize()

    required_columns = [
        "investment_name",
        "company_uid",
        "investment_uid",
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

        spread_value = investment["spread"]
        if isinstance(spread_value, str) and spread_value.strip().lower() == "actual":
            if spreads is None:
                raise ValueError(
                    "spreads is required when schedule_of_investments uses 'actual' spread values."
                )
            spread_for_loan = spreads
        else:
            try:
                spread_for_loan = float(spread_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "spread must be numeric or the literal 'actual'."
                ) from exc

        loan_cashflow_frames.append(
            private_credit_loan_model(
                investment_name=investment["investment_name"],
                company_uid=investment["company_uid"],
                investment_uid=investment["investment_uid"],
                investment_date=investment["investment_date"],
                as_of_date=as_of_date,
                maturity_date=investment["maturity_date"],
                par_value=investment["par_value"],
                spread=spread_for_loan,
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
            "amortization": "sum",
            "cash_interest": "sum",
            "fees": "sum",
            "remaining_balance_payment": "sum",
            "total_payment": "sum",
            "remaining_balance_payment": "sum",
            "ending_balance": "sum",
            "beginning_balance": "sum",
            "nav": "sum",
            "contributions": "sum",
            "distributions": "sum",
            "ncf": "sum",
            "cumulative_contributions": "sum",
            "cumulative_distributions": "sum",
            "cumulative_ncf": "sum",
        }
    ).reset_index()
    portfolio_df["status"] = "projected"
    portfolio_df.loc[portfolio_df["quarter_end"] <= as_of_date, "status"] = "actual"

    portfolio_df['irr'] = None
    portfolio_df["tvpi"] = (portfolio_df["cumulative_distributions"] + portfolio_df["nav"]) / -portfolio_df["cumulative_contributions"]

    for i in range(1, len(portfolio_df)):
        # i = 5
        quarter_date = portfolio_df.at[i, "quarter_end"]
        cfs = portfolio_df[portfolio_df["quarter_end"] <= quarter_date].copy()
        cfs['irr_cfs'] = cfs['ncf']
        cfs.at[i, 'irr_cfs'] = cfs.at[i, 'irr_cfs'] + cfs.at[i, 'nav']
        quarterly_irr = _irr_quarterly(cfs['irr_cfs'].tolist())
        portfolio_df.at[i, 'irr'] = _format_irr(quarterly_irr, i)

    funds_summary_df = (
        funds_df
        .sort_values(["company_uid", "investment_uid", "investment_name", "quarter_end"], kind="stable")
        .groupby(["company_uid", "investment_uid", "investment_name"], as_index=False)
        .agg(
            total_payment=("total_payment", "sum"),
            irr=("irr", "last"),
        )
    )

    schedule_summary_columns = [
        "investment_name",
        "company_uid",
        "investment_uid",
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
        .drop_duplicates(subset=["company_uid", "investment_uid", "investment_name"])
        .copy()
    )
    funds_summary_df = schedule_summary_df.merge(
        funds_summary_df,
        on=["company_uid", "investment_uid", "investment_name"],
        how="left",
    )

    return portfolio_df, funds_df, funds_summary_df