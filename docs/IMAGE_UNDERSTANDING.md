# Image Understanding Guide — AnkaLM

## Capabilities (`ankalm/multimodal/image_engine.py`)
- **Supported Formats:** `PNG`, `JPEG`, `WebP`, `BMP`, `TIFF`, `GIF` (including multi-frame inspection) with decompression-bomb protection (`MAX_IMAGE_PIXELS = 36,000,000`).
- **High-Resolution Tiling (`extract_high_res_tiles`):** Slices high-resolution screenshots and diagrams into overlapping $256 \times 256$ tiles (with 32px overlap) so fine visual and textual detail is never lost to downsampling.
- **Spatial Quadrant Activity:** Measures edge gradient density across top-left, top-right, bottom-left, bottom-right, and center regions.
- **Dual Vision + OCR Synthesis:** Combines `AnkaLM-Vision-Base` scene predictions with `AnkaLMLocalOCR` text and table-grid extraction, explicitly reporting which insight came from which subsystem.
