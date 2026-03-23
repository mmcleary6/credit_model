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