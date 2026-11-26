# ANKALM DOCUMENT TEST REPORT
## PDF, Excel, Formula Interpreter, DOCX, PPTX, HTML, and Cross-Document Verification

## 1. PDF Test Matrix (`tests/test_multimodal_documents.py`)
- **Multi-Page Text & Heading Hierarchy:** Verified on 2-page Turkish research report PDF (`rapor_2026.pdf`), accurately extracting font-size headings (`18.0pt`, `16.0pt`) and page dimensions (`595.0 x 842.0 pt`).
- **Structured Table Extraction & Export:** Extracted 4-row × 3-column performance table on Page 2 and verified both **CSV** (`to_csv()`) and **JSON** (`to_json()`) exports.
- **Turkish PDF Question Answering:**
  - Date extraction (`"Bu belgede geçen tarihleri listele"`): Extracted `26.11.2026` with citation `[PDF: rapor_2026.pdf | Sayfa 1]`.
  - Financial figure extraction (`"Belgedeki tüm fiyatları bul"`): Extracted `45.000 TL` and `1.250 TL` with citation `[PDF: rapor_2026.pdf | Sayfa 1]`.
  - Page-specific table explanation (`"2. sayfadaki tabloyu açıkla"`): Returned structured table summary and CSV payload with citation `[PDF Tablo: rapor_2026.pdf | Sayfa 2 | Tablo p2_t1]`.

## 2. Excel & Formula Interpreter Test Matrix (`tests/test_multimodal_documents.py`)
- **Macro Safety:** Verified `macros_disabled = True` (`keep_vba=False`).
- **Deterministic Formula Recalculation (`AnkaLMFormulaInterpreter`):**
  - `Satışlar!E2` (`=C2*D2`): Recalculated **`5000.0`** (`recalculated_by_ankalm`)
  - `Satışlar!C5` (`=SUM(C2:C4)`): Recalculated **`50.0`** (`recalculated_by_ankalm`)
  - `Satışlar!D5` (`=AVERAGE(D2:D4)`): Recalculated **`333.333333`** (`recalculated_by_ankalm`)
  - `Satışlar!E5` (`=SUM(E2:E4)`): Recalculated **`14500.0`** (`recalculated_by_ankalm`)
  - `Satışlar!C6` (`=IF(C5>40, "Yüksek", "Normal")`): Recalculated **`"Yüksek"`** (`recalculated_by_ankalm`)
  - `Satışlar!D6` (`=COUNTIF(C2:C4, ">=15")`): Recalculated **`2`** (`recalculated_by_ankalm`)
  - `Satışlar!E6` (`=VLOOKUP("Şubat", A2:D4, 4, FALSE)`): Recalculated **`200`** (`recalculated_by_ankalm`)
- **Non-Destructive Editing:** Verified that attempting to overwrite the source file without `allow_overwrite=True` raises `PermissionError`, and safe edits write to `finans_2026_modified.xlsx` with a complete diff preview.

## 3. DOCX, PPTX, HTML & Prompt-Injection Defense
- Verified `.docx` heading/table extraction, `.pptx` slide/title extraction, `.html` `<script>` stripping, and neutralization of embedded `<|system|>` and `"Ignore all previous instructions"` / `"Önceki tüm talimatları unut"` payloads.
