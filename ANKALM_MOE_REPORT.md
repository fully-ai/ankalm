# ANKALM MOE REPORT
## Sparse Mixture-of-Experts Research, Routing Balance, and Active vs. Stored Parameter Analysis

## 1. Architecture & Parameter Accounting (`ankalm/model/moe.py`)
- **Model Name:** `AnkaLM-MoE` (`checkpoints/moe_final`)
- **Total Stored Parameters:** **`15,082,752`** (All 4 routed SwiGLU experts + 1 shared SwiGLU expert per layer reside in RAM/VRAM: `30.27 MB` in BF16)
- **Active Parameters Per Token:** **`10,364,160`** (`68.72%` of total stored parameters computed per token)
- **Routing Policy:** Learned linear router (`MoERouter`) with **Top-2** softmax gating over 4 routed experts + 1 always-on shared expert (`0.5 * shared + 0.5 * routed`).
- **Auxiliary Losses:** Switch/GShard Load-Balancing Loss ($\alpha = 0.02$) + Router Z-Loss ($\beta = 0.001$) + Capacity Factor `1.25`.

## 2. Measured Pretraining & Routing Results (`logs/moe_training_and_routing_report.json`)
- **Steps Executed:** `100` (`409,600` processed tokens, 5 epochs)
- **Throughput:** **`3,020.0 tokens/sec`** in BF16 (+18.3% faster than dense `AnkaLM-Small` at `2,553.1 tokens/sec`)
- **Training Loss:** `9.06406` $\rightarrow$ **`3.53257`**
- **Validation Loss / Perplexity:** `9.0693` (`8,684.33` PPL) $\rightarrow$ **`5.05921`** (**`157.47` PPL**)
- **Parameter L2 Delta from Random Init:** **`38.410184`** (`weights_updated_verified = True`)
- **Measured Expert Utilization Distribution Across Validation Batch:**
  - Expert 0: `3,166` tokens (**`25.76%`**)
  - Expert 1: `3,282` tokens (**`26.71%`**)
  - Expert 2: `3,075` tokens (**`25.02%`**)
  - Expert 3: `2,765` tokens (**`22.50%`**)
  - **Dead Experts:** `0` (`dead_experts_global = []`)
  - **Expert Collapse Detected:** `False` (`collapse_detected_any_layer = False`)
