# ANKALM LOW-VRAM REPORT
## 2 GB VRAM Target Execution Profiles, Knowledge Distillation, Quantization, and Speculative Decoding

## 1. Distilled `AnkaLM-Edge` Model (`logs/distillation_result.json`)
- **Teacher:** `AnkaLM-Small` (`small_final`, `15,735,168` parameters, Val PPL `109.40`)
- **Student:** `AnkaLM-Edge` (`edge_final`, `3,196,608` parameters, **`4.92x` parameter reduction**)
- **Distillation Steps:** `100` (`307,200` tokens, $T = 2.0, \alpha_{\text{KD}} = 0.5$)
- **Distillation Loss:** `6.0456` $\rightarrow$ **`2.34236`** | Student Val Loss / PPL: **`5.13229`** / **`169.40`**
- **Measured Capability Retention Ratio:** **`91.48%`** (`0.9148`)

## 2. Measured Low-VRAM Execution Profiles (`logs/low_vram_and_speculative_report.json`)
| Profile Name | Target Budget | Model & Checkpoint | Precision | Stored Weight Memory | First-Token Latency | Generation Speed | KV Cache Speedup | Fits 2GB Budget |
| :--- | :---: | :--- | :---: | ---: | ---: | ---: | ---: | :---: |
| `edge_2gb_fp16` | **2.0 GB** | `AnkaLM-Edge` (3.20M) | FP16 | **6.847 MB** | **2.75 ms** | **759.49 tok/s** | +198.8% | **YES** |
| `edge_2gb_int8` | **2.0 GB** | `AnkaLM-Edge` (3.20M) | INT8 | **9.081 MB** | 3.04 ms | 512.86 tok/s | +30.8% | **YES** |
| `edge_2gb_int4` | **2.0 GB** | `AnkaLM-Edge` (3.20M) | INT4 | **8.377 MB** | 5.95 ms | 277.36 tok/s | +44.1% | **YES** |
| `balanced_4gb_fp16` | 4.0 GB | `AnkaLM-Small` (15.74M) | FP16 | **32.013 MB** | 5.28 ms | **404.87 tok/s** | +288.8% | **YES** |
| `balanced_4gb_int4` | 4.0 GB | `AnkaLM-Small` (15.74M) | INT4 | **22.776 MB** | 15.55 ms | 79.88 tok/s | +19.8% | **YES** |
| `moe_sparse_bf16` | 8.0 GB | `AnkaLM-MoE` (15.08M) | BF16 | **30.268 MB** | 14.73 ms | 121.88 tok/s | +62.5% | **YES** |

## 3. Speculative Decoding Benchmark (`AnkaLM-Edge` Draft $\rightarrow$ `AnkaLM-Small` Target)
- **Lookahead Window ($K$):** `3` tokens
- **Draft Token Acceptance Rate:** **`43.75%`** (`14 / 32` draft tokens accepted)
- **Target Model Forward Passes:** Reduced from `24` to **`11`** (-54.2% target forward passes)
- **Throughput:** **`178.2 tokens/sec`** vs `143.1 tokens/sec` baseline (**`1.246x` speedup**)
- **Exact Greedy Quality Equivalence:** **`True`** (100% lossless token match)
