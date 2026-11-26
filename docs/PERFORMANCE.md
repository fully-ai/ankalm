# Performance Report — AnkaLM

## Measured Hardware & Training Throughput
- **CPU:** AMD Ryzen 9 9900X3D (12 Cores / 24 Threads)
- **Raw MatMul Compute:** `2,622.83 GFLOPS` (BF16) / `767.6 GFLOPS` (FP32)
- **AnkaLM-Micro (1.44M params):** `28,811.8 tokens/sec` (BF16) vs `23,464.8 tokens/sec` (FP32)
- **AnkaLM-Tiny (5.25M params):** `7,580.1 tokens/sec` (BF16) vs `6,192.2 tokens/sec` (FP32)
- **AnkaLM-Small (15.74M params):** `2,716.9 tokens/sec` (BF16) vs `2,166.0 tokens/sec` (FP32)
- **AnkaLM-Extended (41.96M params):** `830.7 tokens/sec` (BF16)

## Measured Inference & Storage Latency (`AnkaLM-Small`)
- **Checkpoint Load Time:** `11.4 ms` (FP32) / `13.8 ms` (FP16)
- **First-Token Latency:** `~8.5 ms`
- **Autoregressive Generation Speed (with KV Cache):** `247.5 tokens/sec` (FP32) / **`332.6 tokens/sec` (FP16)**
- **Peak RAM Footprint:** `1,457.99 MB` during AnkaLM-Small BF16 pretraining
