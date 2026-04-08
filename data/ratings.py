# %%

import requests
import pandas as pd

# import os
# fred_api_key = os.getenv("FRED_API_KEY")

RATING_SERIES_IDS = {
	"AAA": "BAMLC0A1CAAAEY",
	"AA": "BAMLC0A2CAAEY",
	"A": "BAMLC0A3CAEY",
	"BBB": "BAMLC0A4CBBBEY",
	"BB": "BAMLH0A1HYBBEY",
	"B": "BAMLH0A2HYBEY",
	"CCC": "BAMLH0A3HYCEY"
}

def get_effective_yield(api_key, rating, frequency='D', end_date=None):
	"""Fetch rating-specific effective yield observations from the FRED API.

	Parameters
	----------
	api_key : str
		FRED API key.
	rating : {'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'}
		Rating bucket used to select the corresponding ICE BofA series.
	frequency : {'D', 'M', 'Q', 'Y'}, default='D'
		Output frequency. Daily returns all observations. Monthly, quarterly,
		and yearly outputs are filtered to period-end observations.
	end_date : str or pandas.Timestamp or None, default=None
		Optional date used to extend the returned series beyond the latest
		FRED observation. If provided and later than the maximum available
		date, the function appends daily rows through ``end_date`` and rolls
		forward the most recent yield value.

	Returns
	-------
	pandas.DataFrame
		DataFrame with ``date``, ``effective_yield``, and ``rate_status``
		columns, where ``effective_yield`` is stored as a decimal rather than
		a percentage and ``rate_status`` is either ``"actual"`` or
		``"assumed"``.
	"""

	if not api_key:
		raise ValueError("FRED_API_KEY is required to fetch rating yield data")

	normalized_rating = str(rating).strip().upper()
	series_id = RATING_SERIES_IDS.get(normalized_rating)
	if series_id is None:
		valid_ratings = ", ".join(RATING_SERIES_IDS.keys())
		raise ValueError(f"Invalid rating '{rating}'. Choose from: {valid_ratings}.")

	fred_api_url = (
		"https://api.stlouisfed.org/fred/series/observations"
		f"?series_id={series_id}&api_key={api_key}&file_type=json"
	)

	response = requests.get(fred_api_url, timeout=30)
	response.raise_for_status()
	data = response.json()

	if 'observations' not in data:
		error_message = (
			data.get('error_message')
			or data.get('message')
			or 'FRED response did not include observations'
		)
		raise ValueError(f"Unable to load {series_id} data: {error_message}")

	ratings_df = pd.DataFrame(data['observations'])
	ratings_df['date'] = pd.to_datetime(ratings_df['date'])
	ratings_df = ratings_df.drop(columns=['realtime_start', 'realtime_end'])
	ratings_df['value'] = pd.to_numeric(ratings_df['value'], errors='coerce')
	ratings_df['rate_status'] = 'actual'

	# Fill missing values and missing dates within the actual observation window.
	# 1) Forward-fill NaN values where the date row exists but the rate is missing.
	ratings_df = ratings_df.sort_values('date').reset_index(drop=True)
	ratings_df['value'] = ratings_df['value'].ffill()

	# 2) Reindex to a continuous daily series so any gaps in the date sequence
	#    are filled by rolling forward the previous known value.
	min_date = ratings_df['date'].min()
	max_actual_date = ratings_df['date'].max()
	if pd.notna(min_date) and pd.notna(max_actual_date):
		full_dates = pd.date_range(start=min_date, end=max_actual_date, freq='D')
		ratings_df = (
			ratings_df.set_index('date')
			.reindex(full_dates)
			.rename_axis('date')
			.reset_index()
		)
		ratings_df['value'] = ratings_df['value'].ffill()
		ratings_df['rate_status'] = ratings_df['rate_status'].fillna('actual')

	if end_date is not None:
		try:
			normalized_end_date = pd.Timestamp(end_date).normalize()
		except (TypeError, ValueError) as exc:
			raise ValueError("end_date must be a valid date") from exc

		max_date = ratings_df['date'].max()
		if pd.notna(max_date):
			max_date = pd.Timestamp(max_date).normalize()
			if normalized_end_date > max_date:
				future_dates = pd.date_range(
					start=max_date + pd.Timedelta(days=1),
					end=normalized_end_date,
					freq='D',
				)
				if len(future_dates) > 0:
					latest_value = ratings_df.loc[
						ratings_df['date'] == max_date,
						'value',
					].iloc[-1]
					assumed_rows = pd.DataFrame(
						{
							'date': future_dates,
							'value': latest_value,
							'rate_status': 'assumed',
						}
					)
					ratings_df = pd.concat(
						[ratings_df, assumed_rows],
						ignore_index=True,
					)

	ratings_df['is_month_end'] = ratings_df['date'].dt.is_month_end
	ratings_df['is_quarter_end'] = ratings_df['date'].dt.is_quarter_end
	ratings_df['is_year_end'] = ratings_df['date'].dt.is_year_end

	ratings_df = ratings_df.sort_values(by='date', ascending=False).reset_index(drop=True)
	ratings_df = ratings_df.rename(columns={'value': 'effective_yield'})
	ratings_df = ratings_df.rename(columns={'rate_status': 'yield_status'})
	ratings_df['effective_yield'] = ratings_df['effective_yield'] / 100

	if frequency == 'D':
		return ratings_df[['date', 'effective_yield', 'yield_status']].reset_index(drop=True)
	elif frequency == 'M':
		return ratings_df[ratings_df['is_month_end']][['date', 'effective_yield', 'yield_status']].reset_index(drop=True)
	elif frequency == 'Q':
		return ratings_df[ratings_df['is_quarter_end']][['date', 'effective_yield', 'yield_status']].reset_index(drop=True)
	elif frequency == 'Y':
		return ratings_df[ratings_df['is_year_end']][['date', 'effective_yield', 'yield_status']].reset_index(drop=True)
	else:
		raise ValueError("Invalid frequency. Please choose from 'D', 'M', 'Q', or 'Y'.")
	
# example usage
def get_yield_curve(api_key):
	"""Build a wide yield curve table across supported rating buckets.

	Parameters
	----------
	api_key : str
		FRED API key used to fetch the underlying rating-specific effective
		yield series.

	Returns
	-------
	pandas.DataFrame
		DataFrame with one row per observation date and one column per rating
		bucket. The returned columns are ``date``, ``AAA``, ``AA``, ``A``,
		``BBB``, ``BB``, ``B``, and ``CCC``. Yield values are decimals rather
		than percentages.

	Raises
	------
	ValueError
		Raised when ``api_key`` is missing or when one of the underlying rating
		series cannot be loaded.
	requests.HTTPError
		Raised when the FRED API request fails for any underlying series.

	Notes
	-----
	This function calls :func:`get_effective_yield` once for each supported
	rating bucket using daily frequency, concatenates the results, and pivots
	them into a wide table keyed by ``date``.
	"""
	# api_key = fred_api_key
	AAA = get_effective_yield(rating="AAA", api_key=api_key, frequency="D", end_date=None)
	AA = get_effective_yield(rating="AA", api_key=api_key, frequency="D", end_date=None)
	A = get_effective_yield(rating="A", api_key=api_key, frequency="D", end_date=None)
	BBB = get_effective_yield(rating="BBB", api_key=api_key, frequency="D", end_date=None)
	BB = get_effective_yield(rating="BB", api_key=api_key, frequency="D", end_date=None)
	B = get_effective_yield(rating="B", api_key=api_key, frequency="D", end_date=None)
	CCC = get_effective_yield(rating="CCC", api_key=api_key, frequency="D", end_date=None)

	AAA['rating'] = "AAA"
	AA['rating'] = "AA"
	A['rating'] = "A"
	BBB['rating'] = "BBB"
	BB['rating'] = "BB"
	B['rating'] = "B"
	CCC['rating'] = "CCC"

	yield_curve = pd.concat([AAA, AA, A, BBB, BB, B, CCC], ignore_index=True)
	yield_curve = yield_curve.pivot(index='date', columns='rating', values='effective_yield').reset_index().rename_axis(None, axis=1)	
	return yield_curve

