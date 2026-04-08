# %%

"""SOFR data access helpers.

This module fetches Secured Overnight Financing Rate observations from FRED and
returns them in a format that can be consumed by the private credit model.
"""

import requests
import pandas as pd

# fred_api_key = os.getenv("FRED_API_KEY")

def get_sofr_data(api_key, frequency='D', end_date=None):
    """Fetch SOFR observations from the FRED API.

    Parameters
    ----------
    api_key : str
        FRED API key.
    frequency : {'D', 'M', 'Q', 'Y'}, default='D'
        Output frequency. Daily returns all observations. Monthly, quarterly,
        and yearly outputs are filtered to period-end observations.
    end_date : str or pandas.Timestamp or None, default=None
        Optional date used to extend the returned series beyond the latest FRED
        observation. If provided and later than the maximum available SOFR
        date, the function appends daily rows through ``end_date`` and rolls
        forward the most recent SOFR value.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``date``, ``sofr``, and ``rate_status`` columns, where
        ``sofr`` is stored as a decimal rather than a percentage and
        ``rate_status`` is either ``"actual"`` or ``"assumed"``.

    Raises
    ------
    ValueError
        Raised when ``frequency`` is not one of ``'D'``, ``'M'``, ``'Q'``, or
        ``'Y'``, or when ``end_date`` cannot be parsed as a valid date.

    Notes
    -----
    The FRED endpoint returns SOFR as a percentage value. This function divides
    by ``100`` so downstream modeling uses decimal rates.
    """

    if not api_key:
        raise ValueError("FRED_API_KEY is required to fetch SOFR data")

    # Fred API endpoint for SOFR data
    fred_api_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=SOFR&api_key={api_key}&file_type=json"
    
    # Fetch the data from the FRED API
    response = requests.get(fred_api_url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if 'observations' not in data:
        error_message = data.get('error_message') or data.get('message') or 'FRED response did not include observations'
        raise ValueError(f"Unable to load SOFR data: {error_message}")
    
    # Extract the observations from the data
    observations = data['observations']
    
    # Convert the observations into a pandas dataframe
    sofr_df = pd.DataFrame(observations)
    
    # Convert the date column to datetime format and the value column to numeric
    sofr_df['date'] = pd.to_datetime(sofr_df['date'])

    # Drop the 'realtime_start' and 'realtime_end' columns as they are not needed for our analysis
    sofr_df = sofr_df.drop(columns=['realtime_start', 'realtime_end'])

    # Convert the 'value' column to numeric, coercing errors to NaN (in case there are any non-numeric values)
    sofr_df['value'] = pd.to_numeric(sofr_df['value'], errors='coerce')

    sofr_df['rate_status'] = 'actual'

    # Fill missing values and missing dates within the actual observation window.
    # 1) Forward-fill NaN values where the date row exists but the rate is missing.
    sofr_df = sofr_df.sort_values('date').reset_index(drop=True)
    sofr_df['value'] = sofr_df['value'].ffill()

    # 2) Reindex to a continuous daily series so any gaps in the date sequence
    #    are filled by rolling forward the previous known value.
    min_date = sofr_df['date'].min()
    max_actual_date = sofr_df['date'].max()
    if pd.notna(min_date) and pd.notna(max_actual_date):
        full_dates = pd.date_range(start=min_date, end=max_actual_date, freq='D')
        sofr_df = (
            sofr_df.set_index('date')
            .reindex(full_dates)
            .rename_axis('date')
            .reset_index()
        )
        sofr_df['value'] = sofr_df['value'].ffill()
        sofr_df['rate_status'] = sofr_df['rate_status'].fillna('actual')

    if end_date is not None:
        try:
            normalized_end_date = pd.Timestamp(end_date).normalize()
        except (TypeError, ValueError) as exc:
            raise ValueError("end_date must be a valid date") from exc

        max_date = sofr_df['date'].max()
        if pd.notna(max_date):
            max_date = pd.Timestamp(max_date).normalize()
            if normalized_end_date > max_date:
                future_dates = pd.date_range(
                    start=max_date + pd.Timedelta(days=1),
                    end=normalized_end_date,
                    freq='D',
                )
                if len(future_dates) > 0:
                    latest_sofr = sofr_df.loc[sofr_df['date'] == max_date, 'value'].iloc[-1]
                    assumed_rows = pd.DataFrame(
                        {
                            'date': future_dates,
                            'value': latest_sofr,
                            'rate_status': 'assumed',
                        }
                    )
                    sofr_df = pd.concat([sofr_df, assumed_rows], ignore_index=True)

    # Create a boolean column for dates that are the last day of the month
    sofr_df['is_month_end'] = sofr_df['date'].dt.is_month_end

    # Create a boolean column for dates that are the last day of the quarter
    sofr_df['is_quarter_end'] = sofr_df['date'].dt.is_quarter_end

    # Create a boolean column for dates that are the last day of the year
    sofr_df['is_year_end'] = sofr_df['date'].dt.is_year_end

    # Arrange by date descending
    sofr_df = sofr_df.sort_values(by='date', ascending=False).reset_index(drop=True)

    # rename value to sofr
    sofr_df = sofr_df.rename(columns={'value': 'sofr'})
    sofr_df['sofr'] = sofr_df['sofr'] / 100  # Convert from percentage to decimal

    if frequency == 'D':
        return sofr_df[['date', 'sofr', 'rate_status']].reset_index(drop=True)
    elif frequency == 'M':
        return sofr_df[sofr_df['is_month_end']][['date', 'sofr', 'rate_status']].reset_index(drop=True)
    elif frequency == 'Q':
        return sofr_df[sofr_df['is_quarter_end']][['date', 'sofr', 'rate_status']].reset_index(drop=True)
    elif frequency == 'Y':
        return sofr_df[sofr_df['is_year_end']][['date', 'sofr', 'rate_status']].reset_index(drop=True)
    else:
        raise ValueError("Invalid frequency. Please choose from 'D', 'M', 'Q', or 'Y'.")
