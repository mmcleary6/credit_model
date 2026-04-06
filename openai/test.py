# %%
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from openai import OpenAI

DEFAULT_URL = (
	"https://www.sec.gov/Archives/edgar/data/1318605/000162828026003952/"
	"tsla-20251231.htm"
)
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_USER_AGENT = os.getenv(
	"SEC_USER_AGENT",
	"credit-model/1.0 contact@example.com",
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def fetch_filing_html(url: str) -> str:
	response = requests.get(
		url,
		headers={"User-Agent": DEFAULT_USER_AGENT},
		timeout=60,
	)
	response.raise_for_status()
	return response.text


def normalize_text(value: str) -> str:
	return " ".join(value.split()).lower()


def score_summary_cashflow_table(text: str, row_count: int) -> int:
	score = 0
	phrases = [
		"net cash provided by operating activities",
		"net cash used in investing activities",
		"net cash provided by financing activities",
	]
	for phrase in phrases:
		if phrase in text:
			score += 3
	if row_count <= 10:
		score += 2
	return score


def score_full_cashflow_table(text: str, row_count: int) -> int:
	score = 0
	phrases = [
		"cash flows from operating activities",
		"cash flows from investing activities",
		"cash flows from financing activities",
		"net cash provided by operating activities",
		"net cash used in investing activities",
		"net cash provided by financing activities",
	]
	for phrase in phrases:
		if phrase in text:
			score += 3
	if row_count >= 20:
		score += 2
	return score


def extract_cashflow_tables(html: str) -> tuple[pd.DataFrame, pd.DataFrame]:
	warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
	soup = BeautifulSoup(html, "lxml")

	best_summary: tuple[int, pd.DataFrame] | None = None
	best_full: tuple[int, pd.DataFrame] | None = None

	for table in soup.find_all("table"):
		try:
			dataframes = pd.read_html(StringIO(str(table)))
		except ValueError:
			continue

		if not dataframes:
			continue

		dataframe = dataframes[0]
		flat_text = normalize_text(
			" ".join(dataframe.astype(str).fillna("").to_numpy().flatten())
		)
		row_count = len(dataframe.index)

		summary_score = score_summary_cashflow_table(flat_text, row_count)
		if summary_score and (best_summary is None or summary_score > best_summary[0]):
			best_summary = (summary_score, dataframe)

		full_score = score_full_cashflow_table(flat_text, row_count)
		if full_score and (best_full is None or full_score > best_full[0]):
			best_full = (full_score, dataframe)

	if best_summary is None or best_full is None:
		raise RuntimeError("Unable to identify the cash flow tables in the filing.")

	return best_summary[1], best_full[1]


def strip_code_fences(text: str) -> str:
	stripped = text.strip()
	if stripped.startswith("```"):
		stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
		stripped = re.sub(r"\s*```$", "", stripped)
	return stripped.strip()


def clean_table_with_openai(
	client: OpenAI,
	raw_csv: str,
	model: str,
	table_type: str,
) -> list[dict[str, Any]]:
	if table_type == "summary":
		prompt = f"""
You are cleaning a raw SEC cash flow summary table.

Return JSON only as an array of objects with these keys:
- line_item
- y2025
- y2024
- y2023

Rules:
- Preserve negatives as negative integers.
- Ignore blank columns, repeated duplicated text, currency symbols, and spacer rows.
- Keep the original source order.

RAW CSV:
{raw_csv}
"""
	else:
		prompt = f"""
You are cleaning a raw SEC consolidated statement of cash flows.

Return JSON only as an array of objects with these keys:
- section
- line_item
- y2025
- y2024
- y2023

Rules:
- section must be one of operating, investing, financing, reconciliation, supplemental, other.
- Preserve negatives as negative integers.
- Ignore blank columns, repeated duplicated text, currency symbols, and spacer rows.
- Keep line items in source order.

RAW CSV:
{raw_csv}
"""

	response = client.responses.create(model=model, input=prompt)
	content = strip_code_fences(response.output_text)
	payload = json.loads(content)
	if not isinstance(payload, list):
		raise ValueError("Model output was not a JSON array.")
	return payload


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
	dataframe = pd.DataFrame(records)
	ordered_columns = [
		column
		for column in ["section", "line_item", "y2025", "y2024", "y2023"]
		if column in dataframe.columns
	]
	return dataframe[ordered_columns]


def write_outputs(
	filing_name: str,
	summary_records: list[dict[str, Any]],
	full_records: list[dict[str, Any]],
) -> tuple[Path, Path, Path, Path]:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	summary_json_path = OUTPUT_DIR / f"{filing_name}_cashflow_summary.json"
	summary_csv_path = OUTPUT_DIR / f"{filing_name}_cashflow_summary.csv"
	full_json_path = OUTPUT_DIR / f"{filing_name}_cashflow_full.json"
	full_csv_path = OUTPUT_DIR / f"{filing_name}_cashflow_full.csv"

	summary_json_path.write_text(json.dumps(summary_records, indent=2), encoding="utf-8")
	full_json_path.write_text(json.dumps(full_records, indent=2), encoding="utf-8")

	records_to_dataframe(summary_records).to_csv(summary_csv_path, index=False)
	records_to_dataframe(full_records).to_csv(full_csv_path, index=False)

	return summary_json_path, summary_csv_path, full_json_path, full_csv_path


def filing_slug_from_url(url: str) -> str:
	return Path(url).stem


def main() -> None:
	url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
	client = OpenAI()

	html = fetch_filing_html(url)
	summary_table, full_table = extract_cashflow_tables(html)

	summary_records = clean_table_with_openai(
		client=client,
		raw_csv=summary_table.to_csv(index=False),
		model=DEFAULT_MODEL,
		table_type="summary",
	)
	full_records = clean_table_with_openai(
		client=client,
		raw_csv=full_table.to_csv(index=False),
		model=DEFAULT_MODEL,
		table_type="full",
	)

	summary_json, summary_csv, full_json, full_csv = write_outputs(
		filing_name=filing_slug_from_url(url),
		summary_records=summary_records,
		full_records=full_records,
	)

	print("Saved files:")
	print(f"- {summary_json}")
	print(f"- {summary_csv}")
	print(f"- {full_json}")
	print(f"- {full_csv}")
	print("\nCash flow summary:")
	print(records_to_dataframe(summary_records).to_string(index=False))
	print("\nFull cash flow table:")
	print(records_to_dataframe(full_records).to_string(index=False))


if __name__ == "__main__":
	main()

