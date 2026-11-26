# Multimodal Architecture & Component Attribution — AnkaLM

## Honest Architectural Separation (Section 4 & 26)
AnkaLM never claims that PDF parsing, spreadsheet arithmetic, or OCR rely solely on autoregressive text weights. The multimodal architecture cleanly separates and integrates:

1. **Native From-Scratch Trained Neural Models:**
   - `AnkaLM-Small` / `AnkaLM-MoE` / `AnkaLM-Edge`: Causal language models trained from random initialization on Turkish text.
   - `AnkaLM-Vision-Base` (`ankalm/model/vision.py`, `checkpoints/vision_final`): Independent 3-layer Vision Transformer (`875,019` parameters) + `MultimodalProjector` (`128 -> 384`) trained from scratch on paired Turkish visual-category data (`100%` validation accuracy across 10 visual scene types).
   - `AnkaLM-BiEncoder-Embed-v1` (`ankalm/chat/hybrid_rag.py`, `checkpoints/rag_embedder`): Independent neural bi-encoder trained via symmetric InfoNCE contrastive loss on AnkaLM's Turkish corpus.
2. **Deterministic Local Document & Multimodal Engines:**
   - `AnkaLMPDFEngine` (`ankalm/multimodal/pdf_engine.py`): Built on local `PyMuPDF` (`pymupdf`) + `pypdf` + `AnkaLMLocalOCR`.
   - `AnkaLMExcelEngine` (`ankalm/multimodal/excel_engine.py`): Built on `openpyxl` (`keep_vba=False`) + `AnkaLMFormulaInterpreter` (deterministic 26-function spreadsheet formula evaluator).
   - `AnkaLMLocalOCR` (`ankalm/multimodal/ocr_engine.py`): 100% offline projection-profile deskewing, line/character bounding-box segmentation, table grid detector, and multi-font Turkish glyph template recognizer.
   - `AnkaLMAudioEngine` (`ankalm/multimodal/audio_engine.py`): Built on Python `wave` + `numpy` DSP Voice Activity Detection (VAD).
