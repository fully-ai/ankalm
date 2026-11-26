# Local OCR Engine Guide — AnkaLM

## Architecture (`ankalm/multimodal/ocr_engine.py`)
`AnkaLMLocalOCR` operates 100% offline with zero cloud dependencies:
1. **Quality & Sharpness Assessment:** Measures pixel standard-deviation contrast and discrete Laplacian variance sharpness.
2. **Projection-Profile Deskewing (`detect_and_correct_skew`):** Evaluates horizontal ink projection variance across candidate angles in $[-8^\circ, +8^\circ]$ and rotates skewed scans upright.
3. **Table Grid Detection (`detect_table_grids`):** Identifies intersecting horizontal and vertical rules and estimates row/column counts and bounding boxes.
4. **Turkish & English Glyph Recognition:** Segments text lines and characters and matches normalized glyph vectors against multi-font templates covering ASCII and all 14 Turkish characters (`ç, Ç, ğ, Ğ, ı, I, İ, i, ö, Ö, ş, Ş, ü, Ü`), reporting per-region confidence and flagging uncertain regions (`< 0.68`).
