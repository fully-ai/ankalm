# Mixture-of-Experts Architecture (`AnkaLM-MoE`) — AnkaLM

## Overview
`AnkaLM-MoE` (`ankalm/model/moe.py`, `configs/moe.yaml`) is an independent Sparse Mixture-of-Experts Decoder-Only Transformer branch trained from random initialization while preserving the dense `AnkaLM-Small` baseline.

## Key Architectural Specifications
- **Layers:** 6 Pre-Norm `AnkaLMMoEBlock` layers ($d_{\text{model}}=256, d_{\text{ff}}=512$, 4 query heads, 2 KV heads)
- **Expert Configuration per Layer:**
  - **1 Always-Active Shared Expert (`SwiGLUFFN`):** Captures universal Turkish syntax and morphological foundations on every token.
  - **4 Routed Experts (`SwiGLUFFN`):** Activated sparsely via learned **Top-2** softmax routing (`MoERouter`).
- **Total Stored Parameters:** **`15,082,752`** (All 4 routed experts + 1 shared expert reside in memory)
- **Active Parameters Per Token:** **`10,364,160`** (`68.72%` active fraction per token)
- **Load Balancing & Router Stability:**
  - Switch/GShard Load-Balancing Auxiliary Loss ($\mathcal{L}_{\text{aux}} = E \sum_{i=1}^E f_i P_i$, coefficient `0.02`)
  - Router Z-Loss ($\mathcal{L}_z = \frac{1}{N}\sum (\log \sum_i e^{z_i})^2$, coefficient `0.001`)
  - Capacity factor `1.25` with overflow monitoring and collapse detection.

## Measured MoE Training & Routing Diagnostics (`checkpoints/moe_final`)
- **Pretraining Steps:** `100` (`409,600` processed tokens, `3,020.0 tokens/sec` in BF16)
- **Training Loss:** `9.06406` $\rightarrow$ **`3.53257`**
- **Validation Loss / Perplexity:** `9.0693` (`8,684.33` PPL) $\rightarrow$ **`5.05921`** (**`157.47` PPL**)
- **Aggregate Expert Selection Shares:** `[25.76%, 26.71%, 25.02%, 22.50%]` (`[3166, 3282, 3075, 2765]` tokens routed)
- **Dead Experts / Expert Collapse:** `0` dead experts (`dead_experts_global = []`), `collapse_detected = False`.
