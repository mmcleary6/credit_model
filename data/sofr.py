# %%

import os
import requests
import pandas as pd

# fred_api_key = os.getenv("FRED_API_KEY")

# Fetching SOFR data from the FRED API
def get_sofr_data(api_key, frequency='D'):
    # api_key = fred_api_key  # Use the API key from the environment variable

    # Fred API endpoint for SOFR data
    fred_api_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=SOFR&api_key={api_key}&file_type=json"
    
    # Fetch the data from the FRED API
    response = requests.get(fred_api_url)
    data = response.json()
    
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

    # Create a boolean column for dates that are the last day of the month
    sofr_df['is_month_end'] = sofr_df['date'].dt.is_month_end

    # Create a boolean column for dates that are the last day of the quarter
    sofr_df['is_quarter_end'] = sofr_df['date'].dt.is_quarter_end

    # Create a boolean column for dates that are the last day of the year
    sofr_df['is_year_end'] = sofr_df['date'].dt.is_year_end

    # Arrange by date descending
    sofr_df = sofr_df.sort_values(by='date', ascending=False).reset_index(drop=True)

    if frequency == 'D':
        return sofr_df
    elif frequency == 'M':
        return sofr_df[sofr_df['is_month_end']].reset_index(drop=True)
    elif frequency == 'Q':
        return sofr_df[sofr_df['is_quarter_end']].reset_index(drop=True)
    elif frequency == 'Y':
        return sofr_df[sofr_df['is_year_end']].reset_index(drop=True)
    else:
        raise ValueError("Invalid frequency. Please choose from 'D', 'M', 'Q', or 'Y'.")
