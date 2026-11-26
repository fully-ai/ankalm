# AnkaLM-Vision Architecture & Training — AnkaLM

## Architecture (`ankalm/model/vision.py`)
`AnkaLM-Vision-Base` is an independent Vision Transformer and Multimodal Projector trained from random initialization (no CLIP or external pretrained weights):
- **Input Resolution & Patching:** $64 \times 64$ RGB images sliced via `PatchEmbedding2D` ($8 \times 8$ patches) into $64$ patch tokens + $1$ `[CLS]` token ($65$ visual tokens total, $d_{\text{vision}} = 128$).
- **Vision Encoder:** 3 bidirectional Pre-Norm `VisionTransformerBlock` layers (`RMSNorm`, 4-head Self-Attention, `SwiGLUFFN` with $d_{\text{ff}} = 384$).
- **Multimodal Projector:** 2-layer SiLU MLP + `RMSNorm` projecting visual tokens from $d_{\text{vision}} = 128$ to $d_{\text{model}} = 384$.
- **Parameters:** **`875,019`** trainable parameters.

## Training Results (`checkpoints/vision_final`)
- Trained for `80` steps (`2.27s`) using combined Cross-Entropy + InfoNCE Image-Text Contrastive Alignment Loss on 10 visual scene categories (`bar_chart`, `line_chart`, `table_grid`, `turkish_document_scan`, `code_screenshot`, `circuit_diagram`, `ui_screenshot`, `warning_sign`, `geometric_shapes`, `nature_landscape`).
- **Initial Loss:** `2.67739` $\rightarrow$ **Final Train Loss:** **`0.00165`** | **Validation Loss:** **`0.00149`** | **Validation Accuracy:** **`100.0%`**.
