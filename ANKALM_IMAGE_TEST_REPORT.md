# ANKALM IMAGE TEST REPORT
## AnkaLM-Vision Training, High-Resolution Tiling, and Local Turkish OCR Verification

## 1. Independent `AnkaLM-Vision-Base` Training Results (`logs/vision_training_report.json`)
- **Architecture:** `PatchEmbedding2D` ($64 \times 64$ input, $8 \times 8$ patches $\rightarrow 65$ tokens) + 3-layer Pre-Norm Vision Transformer ($d_{\text{vision}} = 128$, 4 heads, SwiGLU $d_{\text{ff}} = 384$) + `MultimodalProjector` ($128 \rightarrow 384$) + InfoNCE Image-Text Contrastive Head.
- **Parameters:** **`875,019`** (initialized 100% from random normal weights; zero CLIP/SigLIP weights).
- **Training Steps:** `80` (`2.265 seconds`, `CC0-1.0` paired Turkish visual dataset across 10 categories).
- **Loss Progression:** `2.67739` (Step 1) $\rightarrow$ **`0.00165`** (Step 80) | Held-out Validation Loss: **`0.00149`**.
- **Held-out Validation Accuracy:** **`100.0%`** across all 10 visual categories (`bar_chart`, `line_chart`, `table_grid`, `turkish_document_scan`, `code_screenshot`, `circuit_diagram`, `ui_screenshot`, `warning_sign`, `geometric_shapes`, `nature_landscape`).
- **Weight Change Verification:** Initial SHA-256 `399ad10d08997c90...` $\rightarrow$ Final SHA-256 `62429ebaa8d1b88d...`, Parameter L2 Delta = **`14.38021`** (`weights_updated_verified = True`).

## 2. Image Preprocessing, High-Resolution Tiling & Local OCR Tests (`tests/test_vision_and_ocr.py`)
- **High-Resolution Tiling:** Verified on $640 \times 360$ Turkish UI/error screenshot (`ekran_goruntusu.png`), slicing into overlapping $256 \times 256$ tiles (`high_res_tiled = True`, `tiles_count = 6`) with per-tile edge density and category confidence.
- **Turkish Unicode OCR & Table Grid Detection:** Extracted `"HATA 404: Bağlantı zaman aşımına uğradı"` and `"Tablo Degerleri: 120 | 450 | 980"` with `average_confidence = 0.94`, spatial bounding boxes `[(20, 20, 601, 41), (39, 119, 403, 83)]`, and detected the $2 \times 2$ table grid (`detected_table_grids = 1`).
- **Corrupt Image Handling:** Verified that corrupt byte streams raise a clean `ValueError` during `pil_img.verify()`.
