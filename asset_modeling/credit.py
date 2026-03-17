# %%

# Model a fixed income investment with a par value of $1000, a coupon rate of 5%, and a maturity of 10 years. Calculate the annual coupon payment and the total amount received at maturity.
import pandas as pd


def _irr_quarterly(cashflows, tolerance=1e-10, max_iterations=200):
    has_positive = any(cf > 0 for cf in cashflows)
    has_negative = any(cf < 0 for cf in cashflows)
    if not (has_positive and has_negative):
        return None

    def npv(rate):
        return sum(cf / ((1 + rate) ** idx) for idx, cf in enumerate(cashflows))

    low = -0.9999
    high = 10.0
    npv_low = npv(low)
    npv_high = npv(high)

    if npv_low == 0:
        return low
    if npv_high == 0:
        return high
    if npv_low * npv_high > 0:
        return None

    for _ in range(max_iterations):
        mid = (low + high) / 2
        npv_mid = npv(mid)

        if abs(npv_mid) <= tolerance:
            return mid

        if npv_low * npv_mid < 0:
            high = mid
        else:
            low = mid
            npv_low = npv_mid

    return (low + high) / 2

def _get_periodicity_config(periodicity):
    period_map = {
        "annual": (1, 12),
        "semi-annually": (2, 6),
        "semi-annual": (2, 6),
        "semiannual": (2, 6),
        "quarterly": (4, 3),
    }

    if periodicity not in period_map:
        raise ValueError(
            "periodicity must be one of: annual, semi-annually, quarterly"
        )

    return period_map[periodicity]


def _build_payment_dates(start_date, total_periods, months_per_period):
    try:
        start_timestamp = pd.Timestamp(start_date).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("start_date must be a valid date") from exc

    return [
        start_timestamp + pd.DateOffset(months=months_per_period * payment_number)
        for payment_number in range(1, total_periods + 1)
    ]


def standard_loan(
    par_value,
    coupon_rate,
    maturity_years,
    periodicity,
    start_date,
):
    payments_per_year, months_per_period = _get_periodicity_config(periodicity)
    total_periods = int(maturity_years * payments_per_year)

    if total_periods <= 0:
        raise ValueError("maturity_years must produce at least one payment period")

    coupon_payment = par_value * coupon_rate / payments_per_year
    payment_dates = _build_payment_dates(
        start_date,
        total_periods,
        months_per_period,
    )

    rows = []
    for payment_number, payment_date in enumerate(payment_dates, start=1):
        principal_payment = par_value if payment_number == total_periods else 0.0
        total_payment = coupon_payment + principal_payment

        rows.append(
            {
                "payment_number": payment_number,
                "periodicity": periodicity,
                "payment_date": payment_date,
                "years_elapsed": payment_number / payments_per_year,
                "coupon_payment": coupon_payment,
                "principal_payment": principal_payment,
                "total_payment": total_payment,
            }
        )

    cash_flows = pd.DataFrame(rows)
    return cash_flows

def amortizing_loan(
    principal,
    annual_rate,
    maturity_years,
    periodicity,
    start_date,
):
    payments_per_year, months_per_period = _get_periodicity_config(periodicity)
    total_periods = int(maturity_years * payments_per_year)

    if total_periods <= 0:
        raise ValueError("maturity_years must produce at least one payment period")

    period_rate = annual_rate / payments_per_year

    if period_rate == 0:
        scheduled_payment = principal / total_periods
    else:
        scheduled_payment = (
            principal
            * period_rate
            / (1 - (1 + period_rate) ** (-total_periods))
        )

    balance = float(principal)
    payment_dates = _build_payment_dates(
        start_date,
        total_periods,
        months_per_period,
    )
    rows = []

    for payment_number, payment_date in enumerate(payment_dates, start=1):
        beginning_balance = balance
        interest_payment = beginning_balance * period_rate

        if payment_number == total_periods:
            principal_payment = beginning_balance
            scheduled_payment_actual = interest_payment + principal_payment
        else:
            principal_payment = scheduled_payment - interest_payment
            scheduled_payment_actual = scheduled_payment

        ending_balance = beginning_balance - principal_payment

        rows.append(
            {
                "payment_number": payment_number,
                "periodicity": periodicity,
                "payment_date": payment_date,
                "years_elapsed": payment_number / payments_per_year,
                "beginning_balance": round(beginning_balance, 2),
                "total_payment": round(scheduled_payment_actual, 2),
                "interest_payment": round(interest_payment, 2),
                "principal_payment": round(principal_payment, 2),
                "ending_balance": round(max(ending_balance, 0.0), 2),
            }
        )

        balance = ending_balance

    return pd.DataFrame(rows)

def bullet_loan(
    principal,
    annual_rate,
    maturity_years,
    periodicity,
    start_date,
):
    payments_per_year, months_per_period = _get_periodicity_config(periodicity)
    total_periods = int(maturity_years * payments_per_year)

    if total_periods <= 0:
        raise ValueError("maturity_years must produce at least one payment period")

    period_rate = annual_rate / payments_per_year
    balance = float(principal)
    payment_dates = _build_payment_dates(
        start_date,
        total_periods,
        months_per_period,
    )
    rows = []

    for payment_number, payment_date in enumerate(payment_dates, start=1):
        beginning_balance = balance
        interest_payment = beginning_balance * period_rate
        principal_payment = principal if payment_number == total_periods else 0.0
        total_payment = interest_payment + principal_payment
        ending_balance = beginning_balance - principal_payment

        rows.append(
            {
                "payment_number": payment_number,
                "periodicity": periodicity,
                "payment_date": payment_date,
                "years_elapsed": payment_number / payments_per_year,
                "beginning_balance": round(beginning_balance, 2),
                "interest_payment": round(interest_payment, 2),
                "principal_payment": round(principal_payment, 2),
                "total_payment": round(total_payment, 2),
                "ending_balance": round(max(ending_balance, 0.0), 2),
            }
        )

        balance = ending_balance

    return pd.DataFrame(rows)
# Assumptions
"""
investment_date: (t0) the date the investment is made, and the initial cash outflow occurs (no payment occurs at this time, but the cash outflow is recorded as a negative payment at this date)
amortization: applied to the original loan amount, not the remaining balance
pik_interest: applied to the remaining balance at the end of each quarter, added to the balance
cash_interest: applied to the remaining balance at the end of each quarter, paid in cash
exit_fee: applied to the remaining balance at the end of loan, paid in cash
oid_income: Not amortized, recognized at the time of investment, reduces the initial cash outflow
prepayment: if prepayment occurs, the loan is paid off at the prepayment date, and no further payments occur after the prepayment date
"""
def private_credit_loan_model(
    investment_name,
    investment_date,
    maturity_date,
    loan_size,
    spread,
    base_rate,
    sofr_assumption,
    cash_interest_rate,
    pik_interest,
    amortization,
    oid,
    exit_fee,
    prepayment_date,
):
    
    if prepayment_date is None:
        end_date = maturity_date
    else:
        end_date = prepayment_date

    investment_date = pd.Timestamp(investment_date).normalize()
    maturity_date = pd.Timestamp(maturity_date).normalize()
    


    end_date = pd.Timestamp(end_date).normalize()

    if maturity_date <= investment_date:
        raise ValueError("maturity_date must be after investment_date")
    if loan_size <= 0:
        raise ValueError("loan_size must be positive")

    quarter_ends = pd.date_range(
        start=investment_date,
        end=end_date,
        freq="QE",
    )

    if len(quarter_ends) == 0:
        quarter_ends = pd.DatetimeIndex([end_date])

    quarterly_amortization = loan_size * (amortization / 4.0)
    # quarterly_pik_interest = loan_size * (pik_interest / 4.0)

    rows = []
    beginning_balance = float(loan_size)


    for quarter_end in quarter_ends:

        if quarter_end == investment_date:
            invested_amount = loan_size * (1 - oid)
            amortization_amount = 0
            pik_interest_amount = 0
            cash_interest_amount = 0
            ending_balance = loan_size
            fees = 0.0
            remaining_balance_payment = 0
            total_payment = -(loan_size * (1 - oid))
        else:
            invested_amount = loan_size * (1 - oid)
            amortization_amount = quarterly_amortization
            pik_interest_amount = beginning_balance * (pik_interest / 4.0)
            cash_interest_amount = beginning_balance * (cash_interest_rate / 4.0)
            ending_balance = beginning_balance - amortization_amount + pik_interest_amount
            
            if quarter_end == end_date:
                fees = beginning_balance * (exit_fee)
                remaining_balance_payment = beginning_balance
            else:
                fees = 0.0
                remaining_balance_payment = 0

            total_payment = cash_interest_amount + amortization_amount + fees + remaining_balance_payment

        rows.append(
            {
                "investment_name": investment_name,
                "quarter_end": quarter_end,
                "invested_amount": invested_amount,
                "spread": spread,
                "base_rate": base_rate,
                "sofr_assumption": sofr_assumption,
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
        if quarterly_irr is None:
            irr_values.append(None)
        else:
            irr_values.append(((1 + quarterly_irr) ** 4) - 1)

    loan_df["irr"] = irr_values
    loan_df.at[0, "irr"] = None
    return loan_df

# Example usage:
loan_model = private_credit_loan_model(
    investment_name="ABC Corp Term Loan",
    investment_date="2024-03-31",
    maturity_date="2030-06-30",       
    loan_size=1_000_000_000,
    spread=0.06,
    base_rate = 'SOFR',
    sofr_assumption=0.04,
    cash_interest_rate=0.08,
    pik_interest=0.02,
    amortization=0.01,
    oid=0.02,
    exit_fee=0.02,
    prepayment_date=None
)

def loan_portfolio(schedule_of_investments):
    if not isinstance(schedule_of_investments, pd.DataFrame):
        schedule_of_investments = pd.DataFrame(schedule_of_investments)

    required_columns = [
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

    missing_columns = [
        column for column in required_columns
        if column not in schedule_of_investments.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    loan_cashflow_frames = []
    for _, investment in schedule_of_investments.iterrows():
        loan_cashflow_frames.append(
            private_credit_loan_model(
                investment_name=investment["investment_name"],
                investment_date=investment["investment_date"],
                maturity_date=investment["maturity_date"],
                loan_size=investment["loan_size"],
                spread=investment["spread"],
                base_rate=investment["base_rate"],
                sofr_assumption=investment["sofr_assumption"],
                cash_interest_rate=investment["cash_interest_rate"],
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

    for i in range(1, len(portfolio_df)):
        # i = 5
        quarter_date = portfolio_df.at[i, "quarter_end"]
        cfs = portfolio_df[portfolio_df["quarter_end"] <= quarter_date]
        cfs['irr_cfs'] = cfs['total_payment']
        cfs.at[i, 'irr_cfs'] = cfs.at[i, 'irr_cfs'] - cfs.at[i, 'remaining_balance_payment'] + cfs.at[i, 'ending_balance']
        portfolio_df.at[i, 'irr'] = (1 + _irr_quarterly(cfs['irr_cfs'].tolist()))**4 - 1

    return portfolio_df, funds_df