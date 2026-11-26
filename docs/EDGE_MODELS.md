# AnkaLM Edge Model Family & Knowledge Distillation — AnkaLM

## Overview
The AnkaLM Edge family (`configs/edge_2gb.yaml`, `ankalm/training/distillation.py`) targets fast local inference on 2 GB VRAM / low-memory systems and acts as the draft engine for Speculative Decoding.

## Controlled Knowledge Distillation (`checkpoints/edge_final`)
- **Teacher:** `AnkaLM-Small` (`small_final`, `15,735,168` parameters, independently trained from random initialization)
- **Student:** `AnkaLM-Edge` (`3,196,608` parameters, `4.92x` parameter compression)
- **Distillation Objective:** $\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{CE}}(z_S, y) + 0.5 \cdot T^2 D_{\text{KL}}(\text{softmax}(z_S/T) \parallel \text{softmax}(z_T/T))$ at $T = 2.0$.
- **Measured Results (`100` steps, `307,200` tokens):**
  - Distillation total loss reduced from `6.0456` $\rightarrow$ **`2.34236`**.
  - Student validation loss / perplexity: **`5.13229`** / **`169.40`** (vs Teacher `4.69501` / `109.40`).
  - Measured Capability Retention Ratio ($\mathcal{L}_T / \mathcal{L}_S$): **`91.48%`** (`0.9148`).
  - Parameter L2 Delta from Random Initialization: **`32.463378`** (`weights_updated_verified = True`).
