# ANKALM MULTIMODAL STATE
## PDF, Excel, Image, Vision, OCR, DOCX, PPTX, HTML, Audio, Hybrid RAG & Tool Sandbox Status

- **PDF Engine (`ankalm/multimodal/pdf_engine.py`):** Multi-page, multi-column, font-hierarchy headings, automatic 150-DPI Local OCR fallback on scanned pages, structured table extraction (CSV/JSON export), map-reduce hierarchical summary, and page-cited Turkish Q&A.
- **Excel Engine (`ankalm/multimodal/excel_engine.py`):** `.xlsx`/`.xlsm` (macros disabled), `.csv`, `.tsv`; 26-function deterministic `AnkaLMFormulaInterpreter`; `Sheet!Range`-cited Q&A; and safe non-destructive editing (`manipulate_workbook_safely`).
- **Vision & OCR (`ankalm/model/vision.py`, `ankalm/multimodal/image_engine.py`, `ankalm/multimodal/ocr_engine.py`):**
  - `AnkaLM-Vision-Base` (`checkpoints/vision_final`): Multi-Scale (`8x8` fine + `16x16` coarse) ViT + Multimodal Projector (`907,787` params) trained on 400 deduplicated images (`320` train / `80` unseen perturbed validation) + exported to DirectML GPU (`exports/ankalm_vision_dml.onnx`, `6,277.9 img/s`).
  - `AnkaLMLocalOCR`: Projection-profile deskewing, table grid line detection, Turkish/English glyph template recognizer, and confidence/uncertainty reporting.
- **Other Formats & Security (`ankalm/multimodal/doc_parsers.py`, `ankalm/multimodal/audio_engine.py`):** `.docx`, `.pptx`, `.html`, `.xml`, `.odt`, `.epub`, `.wav` VAD segmentation, and untrusted prompt-injection sanitization.
- **Hybrid RAG (`ankalm/chat/hybrid_rag.py`):** Okapi BM25 + contrastively trained `AnkaLM-BiEncoder-Embed-v1` + Turkish morphological query rewriting + cross-feature reranking (`Recall@1`, `Recall@5`, `Recall@10`, `MRR`, `NDCG@10` evaluated on 30 held-out queries in `logs/rag_30_query_benchmark.json`).
- **Controlled Tools (`ankalm/tools/registry.py`):** 9+ registered JSON-schema tools including deterministic AST `calculator`, isolated subprocess `python_sandbox`, `unit_converter`, `json_validator`, `chart_generator`, and `cross_document_comparator`.
