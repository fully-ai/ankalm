# AnkaLM AI Studio GUI Guide — AnkaLM

## Launching the Local AI Studio (`start.bat` / `./start.sh`)
Binds exclusively to `http://127.0.0.1:8420` (`web/index.html`, `web/styles.css`, `web/app.js`).

## 8 Integrated Workspaces
1. **💬 Sohbet (General Chat):** Turkish conversation with Dark/Light themes, SQLite persistence, streaming, Markdown/code rendering, copy/edit/regenerate/export/import, Hybrid RAG toggle, and Speculative Decoding (`Edge Draft`) toggle.
2. **📄 PDF/Belge (PDF, DOCX, PPTX, HTML Workspace):** Upload documents, inspect 15-stage pipeline telemetry (pages, tables, OCR, security warnings), and ask page-cited Turkish questions.
3. **📊 Excel (Spreadsheet Workspace):** Upload `.xlsx`/`.csv` workbooks, inspect sheet statistics and formula recalculation status (`recalculated_by_ankalm`), and run `Sheet!Range`-cited queries.
4. **👁️ Görüntü/OCR (Vision, OCR & Audio Workspace):** Upload `PNG`, `JPEG`, `WebP`, `BMP`, `TIFF`, `GIF`, or `WAV` files; inspect high-resolution tiles, `AnkaLM-Vision` scene classification, and `AnkaLMLocalOCR` bounding boxes.
5. **🛠️ Araçlar/Kod (Tools & Python Sandbox Workspace):** Execute `calculator`, isolated `python_sandbox`, `unit_converter`, `json_validator`, and `chart_generator`.
6. **🗂️ Modeller/MoE (Model Family Manager):** Inspect all 9 AnkaLM model profiles (including 8B/4B blueprints, Dense, Sparse MoE 4-expert routing shares, and 2GB Edge profiles).
7. **📈 Eğitim (Training Control Center):** Live telemetry cards, HTML5 Canvas loss curves, and start/stop/pause/resume/save controls.
8. **🧪 Test Lab (Turkish Evaluation & Hardware Lab):** 16-category Turkish benchmark scores and hardware report.
