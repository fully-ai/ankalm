# Excel & Spreadsheet Support Guide — AnkaLM

## Capabilities (`ankalm/multimodal/excel_engine.py`)
- **Supported Formats:** `.xlsx`, `.xlsm` (macros strictly disabled via `keep_vba=False`), `.csv`, `.tsv`.
- **Structural Inspection:** Reads sheet names, hidden sheets, merged cell ranges, named ranges, comments, hyperlinks, charts count, missing cells, and formula error rows (`#DIV/0!`, `#VALUE!`, `#REF!`, `#N/A`).
- **Deterministic Local Formula Interpreter (`AnkaLMFormulaInterpreter`):**
  Recalculates 26+ Excel functions locally (`SUM`, `AVERAGE`, `MIN`, `MAX`, `COUNT`, `COUNTA`, `IF`, `AND`, `OR`, `SUMIF`, `SUMIFS`, `COUNTIF`, `COUNTIFS`, `VLOOKUP`, `HLOOKUP`, `XLOOKUP`, `INDEX`, `MATCH`, `CONCAT`, `TEXT`, `DATE`, `YEAR`, `MONTH`, `DAY`, `ROUND`, `IFERROR`) and labels every formula cell as `recalculated_by_ankalm`, `cached_value_only`, `unsupported_formula`, or `formula_error`.
- **Non-Destructive Manipulation (`manipulate_workbook_safely`):**
  Supports adding/renaming sheets, updating cells, and adding formulas while writing to `<stem>_modified.xlsx` and refusing to overwrite the original file unless `allow_overwrite=True` is explicitly provided.
