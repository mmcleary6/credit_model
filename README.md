# Private Credit Portfolio Model

## Shiny App

This repository includes a Python Shiny app in `app/app.py` with two pages:

- **Loan Schedule**: Editable dataframe for the schedule of investments.
- **Portfolio Outputs**: Interactive Plotly charts for portfolio cash flow and IRR outputs.

### Run locally

1. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

2. Start the app from the repository root:

	```bash
	python -m shiny run --reload app/app.py
	```

### Editable schedule columns

The app expects these columns in the investment schedule table:

- `investment_name`
- `investment_date`
- `maturity_date`
- `loan_size`
- `spread`
- `base_rate`
- `sofr_assumption`
- `cash_interest_rate`
- `pik_interest`
- `amortization`
- `oid`
- `exit_fee`
- `prepayment_date`

## Python API

The core modeling package exposes a small API in `asset_modeling.credit` and
`data.sofr`.

### `private_credit_loan_model(...)`

Builds a quarterly loan-level cash flow schedule for a single investment.

- Accepts an optional `as_of_date` argument and labels each quarter-end row as
  `actual` or `projected` in the returned DataFrame.
- Accepts either a SOFR history DataFrame with `date` and `sofr` columns or a
	fixed float assumption through `sofr_rates`.
- Returns a DataFrame with quarter-end balances, payment components,
	resolved-rate metadata, and a running IRR.

### `loan_portfolio(...)`

Aggregates a schedule of investments into three tables:

- portfolio-level quarterly totals,
- loan-level cash flows across all investments,
- a one-row-per-loan summary including total payment and terminal IRR.

The portfolio schedule expects these modeling columns:

- `investment_name`
- `investment_date`
- `maturity_date`
- `par_value`
- `spread`
- `base_rate`
- `sofr_assumption`
- `pik_interest`
- `amortization`
- `oid`
- `exit_fee`
- `prepayment_date`

### `get_sofr_data(...)`

Fetches SOFR data from FRED and returns a DataFrame with `date` and `sofr`
columns. The `sofr` values are converted from percentages to decimals and can
be returned at daily, monthly, quarterly, or yearly frequency.