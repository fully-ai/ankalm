# ANKALM MODEL REPORT
## Complete Model Family Specifications, Parameter Counts, and Training Lineage

| Profile Key | Model Name | Family | Layers | $d_{\text{model}}$ | $d_{\text{ff}}$ | Heads ($Q / KV$) | Total Stored Params | Active Params / Token | Est. Training Memory (BF16) | Local Status |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | :--- |
| `micro` | `AnkaLM-Micro` | Dense | 2 | 128 | 384 | 4 / 2 | 1,442,432 | 1,442,432 | 0.31 GB | Trained (`micro_final`) |
| `edge` | `AnkaLM-Edge` | Edge (<2GB) | 4 | 192 | 512 | 4 / 2 | 3,196,608 | 3,196,608 | 0.40 GB | Distilled (`edge_final`) |
| `tiny` | `AnkaLM-Tiny` | Dense | 4 | 256 | 768 | 4 / 2 | 5,245,184 | 5,245,184 | 0.44 GB | Trained (`tiny_final`) |
| `moe` | `AnkaLM-MoE` | Sparse MoE (4E+1S) | 6 | 256 | 512 | 4 / 2 | 15,082,752 | 10,364,160 | 0.61 GB | Trained (`moe_final`) |
| `small` | `AnkaLM-Small` | Dense | 8 | 384 | 1024 | 6 / 2 | 15,735,168 | 15,735,168 | 0.69 GB | Trained (`small_final`, `small_sft`) |
| `extended` | `AnkaLM-Extended` | Dense | 12 | 512 | 1536 | 8 / 4 | 41,955,840 | 41,955,840 | 1.56 GB | Benchmarked (`830.7 tok/s`) |
| `edge_300m` | `AnkaLM-Edge-300M` | Edge Large | 16 | 1024 | 3584 | 16 / 4 | 226,526,208 | 226,526,208 | 4.44 GB | Meta-Verified Blueprint |
| `balanced_4b` | `AnkaLM-4B` | Balanced | 28 | 3072 | 10240 | 24 / 8 | 3,372,395,520 | 3,372,395,520 | 66.50 GB | `architecture_config_only` |
| `flagship_8b` | `AnkaLM-8B` | Flagship | 32 | 4096 | 15360 | 32 / 8 | 7,415,795,712 | 7,415,795,712 | 135.00 GB | `architecture_config_only` |

## Low-Memory Optimizer Benchmark Results (`AnkaLM-Tiny`, `logs/optimizer_benchmarks.json`)
- **Standard FP32 `AdamW`:** Optimizer State = `40.0177 MB` (`0.0%` reduction) | Loss: `9.0658 -> 5.9736`
- **Block-Quantized `AdamW8bit`:** Optimizer State = **`10.0047 MB`** (**`-75.0%` memory reduction**) | Loss: `9.0658 -> 6.0729`
- **Factored `Adafactor`:** Optimizer State = **`0.1152 MB`** (**`-99.71%` memory reduction**) | Loss: `9.0658 -> 9.1306` (requires longer warmup schedule)
