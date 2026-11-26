# ANKALM PERFORMANCE STATE
## Hardware, Training, DirectML GPU, Quantization, and Speculative Decoding Benchmarks

- **GPU DirectML (`AMD Radeon RX 9060 XT`, 15.92 GB VRAM, `DmlExecutionProvider`):**
  - `AnkaLM-Vision-Base` Batch-16 ONNX Forward: **`6,277.9 images/sec`**
  - `AnkaLM-Edge` (3.20M params) Batch-8$\times$128 ONNX Forward: **`97,916.6 tokens/sec`**
  - `AnkaLM-Small` (15.74M params) Batch-8$\times$128 ONNX Teacher Forward: Measured in `logs/gpu_verification_report.json`
- **CPU AVX-512 BF16 (`AMD Ryzen 9 9900X3D`, 20–24 Threads):**
  - Raw MatMul Compute: **`4,598.6 GFLOPS` (4.60 TFLOPS) BF16** | **`1,223.7 GFLOPS` (1.22 TFLOPS) FP32**
  - `AnkaLM-Small` Pretraining Throughput: `2,553.1–2,766.0 tokens/sec`
  - `AnkaLM-MoE` Pretraining Throughput: **`3,020.0 tokens/sec`**
- **2GB Low-VRAM & Speculative Decoding:**
  - `AnkaLM-Edge` FP16: `6.85 MB` weights | `2.75 ms` first-token latency | **`759.49 tokens/sec`**
  - `AnkaLM-Small` FP16: `32.01 MB` weights | `5.28 ms` first-token latency | **`404.87 tokens/sec`**
  - Speculative Decoding (`Edge -> Small`): `43.75%` draft acceptance | **`1.246x` speedup** (`178.2 tok/s`) | `100%` exact greedy quality equivalence.
- **NVMe Storage Bandwidth:** `1,296.9 MB/s` sequential write (`fsync`) | `3,580.3 MB/s` sequential read.
