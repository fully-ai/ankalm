# ANKALM MOE STATE
## Sparse Mixture-of-Experts Checkpoint, Routing Distribution, and Collapse Diagnostics

- **Checkpoint Location:** `checkpoints/moe_final` (`configs/moe.yaml`)
- **Total Stored Parameters:** **`15,082,752`** (`30.27 MB` in BF16)
- **Active Parameters Per Token:** **`10,364,160`** (`68.72%` active fraction)
- **Expert Topology:** 6 layers $\times$ (1 Shared SwiGLU Expert + 4 Routed SwiGLU Experts with **Top-2** learned softmax gating)
- **Router Regularization:** Switch/GShard Load-Balancing Auxiliary Loss ($\alpha = 0.02$) + Router Z-Loss ($\beta = 0.001$) + Capacity Factor `1.25`
- **Cumulative Steps / Tokens:** `130` steps (`532,480` processed tokens)
- **Measured Expert Selection Shares:** `[25.76%, 26.71%, 25.02%, 22.50%]`
- **Dead Experts / Collapse Status:** `0` dead experts (`dead_experts_global = []`), `collapse_detected_any_layer = False`.
