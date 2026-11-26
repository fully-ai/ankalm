# ANKALM PERFORMANCE REPORT
## Reproducible System-Wide Benchmarks Across Training, Inference, MoE, Quantization, Multimodal, RAG, and Tools

## 1. Compute & Training Throughput (`AMD Ryzen 9 9900X3D`, AVX-512 BF16)
- **Raw MatMul Compute:** `2,622.83 GFLOPS` (BF16) / `767.60 GFLOPS` (FP32)
- **`AnkaLM-Micro` (1.44M params) Training:** `29,970.1 tokens/sec` (BF16)
- **`AnkaLM-Tiny` (5.25M params) Training:** `6,750.9 tokens/sec` (BF16)
- **`AnkaLM-MoE` (15.08M total / 10.36M active params) Training:** **`3,020.0 tokens/sec`** (BF16)
- **`AnkaLM-Small` (15.74M dense params) Training:** **`2,553.1 tokens/sec`** (BF16)
- **`AnkaLM-Edge` (3.20M params) Knowledge Distillation:** `2,609.6 tokens/sec` (BF16)

## 2. Inference, Quantization & Speculative Decoding Latencies
- **Fastest Edge Profile (`AnkaLM-Edge` FP16):** `6.85 MB` weights | `2.75 ms` first-token latency | **`759.49 tokens/sec`**
- **Fastest Balanced Profile (`AnkaLM-Small` FP16):** `32.01 MB` weights | `5.28 ms` first-token latency | **`404.87 tokens/sec`**
- **Highest Compression Profile (`AnkaLM-Small` INT4):** `22.78 MB` weights (`2.81x` compression vs FP32 `64.03 MB`) | Val PPL `15.48` (zero perplexity degradation vs FP32 `15.48`)
- **Speculative Decoding (`Edge -> Small`):** `178.2 tokens/sec` vs `143.1 tokens/sec` (`1.246x` speedup, `43.75%` draft acceptance)

## 3. Multimodal, RAG & Tool Execution Latencies
- **Turkish BPE-8192 Tokenizer Encoding:** **`2,873,603.6 tokens/sec`**
- **2-Page PDF Full Analysis (Text + Headings + Tables + Summary):** `~135.3 ms`
- **Multi-Sheet Excel Analysis + 7-Formula Recalculation:** `~18.4 ms`
- **High-Res Image Tiling (6 tiles) + `AnkaLM-Vision` ViT + Local OCR:** `~64.2 ms`
- **1.5s 16-kHz WAV Audio DSP & VAD Segmentation:** `~4.1 ms`
- **Hybrid BM25 + AnkaLM Neural Bi-Encoder RAG Query:** `~3.8 ms` (`Recall@1 = 100%`, `MRR = 1.0`)
- **Deterministic AST Calculator Tool Call:** `< 1.0 ms`
- **Isolated Subprocess Python Sandbox Execution:** `~42.0 ms`
