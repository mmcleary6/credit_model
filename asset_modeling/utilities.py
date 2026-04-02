"""Shared helper functions for cash flow and rate preprocessing.

The functions in this module support the higher-level modeling APIs in
``asset_modeling.credit``. They are intentionally small and focused so the
main modeling code can stay oriented around business logic.
"""

import pandas as pd

def _irr_quarterly(cashflows, tolerance=1e-10, max_iterations=200):
    """Estimate a quarterly IRR using a bisection search.

    Parameters
    ----------
    cashflows : sequence of float
        Ordered cash flow series.
    tolerance : float, default=1e-10
        Absolute NPV tolerance used to stop the iteration.
    max_iterations : int, default=200
        Maximum number of bisection steps.

    Returns
    -------
    float or None
        Quarterly internal rate of return if a sign-changing root can be found;
        otherwise ``None``.
    """

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


def _format_irr(quarterly_irr, cumulative_quarters):
    """Annualize a quarterly IRR once at least one year of data is available.

    Parameters
    ----------
    quarterly_irr : float or None
        Quarterly IRR estimate.
    cumulative_quarters : int
        Number of quarters represented in the IRR window.

    Returns
    -------
    float or None
        The raw quarterly IRR for windows shorter than four quarters, the
        annualized IRR afterward, or ``None`` when the input IRR is missing.
    """

    if quarterly_irr is None:
        return None
    if cumulative_quarters < 4:
        return quarterly_irr
    return ((1 + quarterly_irr) ** 4) - 1

def _get_periodicity_config(periodicity):
    """Map a payment frequency label to schedule settings.

    Parameters
    ----------
    periodicity : str
        Payment frequency label.

    Returns
    -------
    tuple of int
        A pair containing payments per year and months per payment period.

    Raises
    ------
    ValueError
        Raised when ``periodicity`` is not a supported label.
    """

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
    """Create forward payment dates from a normalized start date.

    Parameters
    ----------
    start_date : str or pandas.Timestamp
        First accrual date for the schedule.
    total_periods : int
        Number of payment periods to generate.
    months_per_period : int
        Interval between payment dates measured in months.

    Returns
    -------
    list of pandas.Timestamp
        Generated payment dates.

    Raises
    ------
    ValueError
        Raised when ``start_date`` cannot be parsed as a valid timestamp.
    """

    try:
        start_timestamp = pd.Timestamp(start_date).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("start_date must be a valid date") from exc

    return [
        start_timestamp + pd.DateOffset(months=months_per_period * payment_number)
        for payment_number in range(1, total_periods + 1)
    ]

def _prepare_sofr_table(rates_table):
    """Validate and normalize a SOFR rate table.

    Parameters
    ----------
    rates_table : pandas.DataFrame or None
        Input table expected to contain ``date`` and ``sofr`` columns.

    Returns
    -------
    pandas.DataFrame or None
        Cleaned SOFR table sorted by date with duplicate dates removed, or
        ``None`` when no table is supplied.

    Raises
    ------
    ValueError
        Raised when the input is not a DataFrame or when required columns are
        missing or invalid.
    """

    if rates_table is None:
        return None
    if not isinstance(rates_table, pd.DataFrame):
        raise ValueError("rates_table must be a pandas DataFrame")

    required_columns = {"date", "sofr"}
    missing_columns = required_columns.difference(set(rates_table.columns))
    if missing_columns:
        raise ValueError(
            f"rates_table must include columns: {sorted(required_columns)}"
        )

    prepared = rates_table[["date", "sofr"]].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
    prepared["sofr"] = pd.to_numeric(prepared["sofr"], errors="coerce")

    prepared.dropna(subset=["date", "sofr"], inplace=True)

    if prepared["date"].isna().any() or prepared["sofr"].isna().any():
        raise ValueError("rates_table columns 'date' and 'sofr' must be valid values")

    prepared = prepared.sort_values("date", kind="stable").drop_duplicates(
        subset=["date"], keep="last"
    )
    return prepared.reset_index(drop=True)


def _resolve_sofr_rate(as_of_date, prepared_rates_table):
    """Resolve a SOFR rate for a specific quarter-end date.

    Parameters
    ----------
    as_of_date : str or pandas.Timestamp
        Quarter-end date being priced.
    prepared_rates_table : pandas.DataFrame or None
        Normalized SOFR table from :func:`_prepare_sofr_table`.

    Returns
    -------
    tuple of float and str or None
        Resolved SOFR value and a status label of ``"actual"`` or
        ``"assumed"``. Returns ``None`` when no rate table is available.
    """

    if prepared_rates_table is None or prepared_rates_table.empty:
        return None

    as_of_date = pd.Timestamp(as_of_date).normalize()
    prior_rates = prepared_rates_table[prepared_rates_table["date"] == as_of_date]
    rates_status = "actual"

    if prior_rates.empty:
        rates_status = "assumed"
        return float(prepared_rates_table.iloc[0]["sofr"]), rates_status
    return float(prior_rates.iloc[-1]["sofr"]), rates_status

