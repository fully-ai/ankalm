# Ultra-Low VRAM (2 GB Target) & Optimization Guide — AnkaLM

## Execution Profiles (`ankalm/inference/speculative.py`)
AnkaLM never claims an 8B dense model (`~16 GB` in FP16) fits inside 2 GB VRAM. Instead, AnkaLM provides measured hardware-matched execution profiles:

| Profile Name | Target Budget | Model & Checkpoint | Precision | Weights Memory | First-Token Latency | Throughput (tok/s) | KV Cache Speedup | Fits 2GB Budget |
| :--- | :---: | :--- | :---: | ---: | ---: | ---: | ---: | :---: |
| `edge_2gb_fp16` | **2 GB** | `AnkaLM-Edge` (`edge_final`, 3.20M) | FP16 | **6.85 MB** | **2.75 ms** | **759.49 tok/s** | +198.8% | **YES** |
| `edge_2gb_int8` | **2 GB** | `AnkaLM-Edge` (`edge_final`, 3.20M) | INT8 | **9.08 MB** | 3.04 ms | 512.86 tok/s | +30.8% | **YES** |
| `edge_2gb_int4` | **2 GB** | `AnkaLM-Edge` (`edge_final`, 3.20M) | INT4 | **8.38 MB** | 5.95 ms | 277.36 tok/s | +44.1% | **YES** |
| `balanced_4gb_fp16` | 4–8 GB | `AnkaLM-Small` (`small_final`, 15.74M) | FP16 | **32.01 MB** | 5.28 ms | **404.87 tok/s** | +288.8% | **YES** |
| `balanced_4gb_int4` | 4–8 GB | `AnkaLM-Small` (`small_final`, 15.74M) | INT4 | **22.78 MB** | 15.55 ms | 79.88 tok/s | +19.8% | **YES** |
| `moe_sparse_bf16` | 8+ GB | `AnkaLM-MoE` (`moe_final`, 15.08M) | BF16 | **30.27 MB** | 14.73 ms | 121.88 tok/s | +62.5% | **YES** |

## Speculative Decoding (`AnkaLM-Edge` Draft $\rightarrow$ `AnkaLM-Small` Target)
- **Lookahead ($K$):** `3` draft tokens per target verification step
- **Measured Acceptance Rate:** **`43.75%`** (`14 / 32` draft tokens accepted)
- **Target Forward Calls:** Reduced from `24` to **`11`**
- **Measured Speedup:** **`1.246x`** (`178.2 tok/s` vs `143.1 tok/s` baseline) with **`100%` exact greedy output equivalence** (`exact_quality_equivalence = True`).
