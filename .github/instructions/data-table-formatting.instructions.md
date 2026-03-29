---
description: "Use when creating, editing, or rendering a data table (DataGrid) in this Shiny app. Covers column widths, number formatting, header capitalization, centering, header color, and row font styling."
applyTo: "app/**/*.py"
---
# Data Table Formatting

Always follow these rules when rendering a `render.DataGrid` table.

## Column Widths

Every table must have content-width columns driven by the data rows, not the header text.

- Pass `width="fit-content"` to `render.DataGrid`.
- Use **separate rules** for headers and body — headers wrap so they don't force wide columns:

```css
/* Headers: allow wrapping so column width is set by data, not header length */
#<table_id> table thead th,
#<table_id> [role="columnheader"] {
    white-space: normal !important;
    width: max-content !important;
    min-width: max-content !important;
}

/* Body cells: no wrapping */
#<table_id> table tbody td,
#<table_id> [role="gridcell"] {
    white-space: nowrap !important;
    width: max-content !important;
    min-width: max-content !important;
}
```

## Numeric Formatting

- **Currency / large numbers**: `f"{value:,.2f}"` — commas, 2 decimal places.
- **Rates (IRR, spread, SOFR, PIK, OID, exit fee, amortization)**: `f"{value * 100:.2f}%"`.
- **TVPI**: `f"{value:.2f}x"`.
- Handle `NaN`/missing: return `""` for null values.

## Column Header Capitalization

Use the existing `_format_column_label()` / `_format_table_headers()` helpers. They:
- Replace underscores with spaces and title-case each word.
- Force all-caps for: `IRR`, `TVPI`, `PIK`, `SOFR`, `OID`.

Never rename headers manually inline; always call `_format_table_headers(df)` before passing to `DataGrid`.

## Centering Columns

Center `IRR`, `TVPI`, `Quarter End` (Quarter Date), `Investment Date`, `Maturity Date`, and `Base Rate` columns.

Use both the `styles=` argument on `DataGrid` (body cells) **and** CSS selectors (header cells):

```python
# Body centering via styles=
center_styles = []
for col_name in centered_columns:
    if col_name in display_df.columns:
        center_styles.append({
            "cols": [display_df.columns.get_loc(col_name)],
            "style": {"textAlign": "center"},
        })
```

```css
/* Header centering via CSS (1-indexed nth-child / aria-colindex) */
#<table_id> table thead th:nth-child(<n>),
#<table_id> [role="columnheader"][aria-colindex="<n>"] {
    text-align: center !important;
    justify-content: center !important;
}
```

## Header Styling

Table headers should be dark grey to match the navbar (`#343a40`) with white text, and a smaller font size:

```css
#<table_id> table thead th,
#<table_id> [role="columnheader"] {
    background-color: #343a40 !important;
    color: #ffffff !important;
    font-size: 0.8rem !important;
}
```

## Row Data Styling

Row data font should be slightly smaller and dark grey:

```css
#<table_id> table tbody td,
#<table_id> [role="gridcell"] {
    font-size: 0.8rem !important;
    color: #495057 !important;
}
```

## Table Name (Title)

The `ui.h4()` label above each table should use a smaller font size:

```python
ui.h4("Table Title", style="font-size: 0.95rem;")
```
