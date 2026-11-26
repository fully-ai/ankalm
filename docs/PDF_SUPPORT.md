# PDF Support & Question Answering Guide — AnkaLM

## Capabilities (`ankalm/multimodal/pdf_engine.py`)
- **Multi-Page & Multi-Column Extraction:** Inspects font size spans and bounding boxes `(x0, y0, x1, y1)` across every page, sorting two-column academic/technical layouts in proper column-first reading order.
- **Automatic Local OCR Fallback:** Detects scanned or image-only PDF pages (`len(native_text) < 25`), renders page pixmaps at 150 DPI, and executes `AnkaLMLocalOCR` with confidence reporting.
- **Structured PDF Table Extraction:** Extracts tables with headers, rows, cell coordinates, and exports directly to **CSV** (`to_csv()`) and **JSON** (`to_json()`).
- **Cited Turkish PDF Q&A:** Answers questions on dates (`DATE_REGEX`), prices/financial figures (`PRICE_REGEX`), specific page tables (`"2. sayfadaki tabloyu açıkla"`), keyword occurrences, and map-reduce hierarchical summaries with exact `[PDF: <file> | Sayfa <n>]` citations.
