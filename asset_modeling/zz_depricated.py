import pandas as pd
from utilities import _irr_quarterly, _format_irr, _get_periodicity_config, _build_payment_dates

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