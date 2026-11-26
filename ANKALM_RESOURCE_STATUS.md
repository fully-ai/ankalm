# ANKALM RESOURCE STATUS
## CPU, RAM, NVMe Disk I/O, and Resource-Aware Safeguard Configuration

## 1. Measured Host Resources
- **CPU:** `AMD Ryzen 9 9900X3D 12-Core Processor` (12 Physical Cores / 24 Logical Threads, AVX-512)
- **Measured CPU Thread Scaling (BF16 / FP32 MatMul):**
  - 4 Threads: `2,349.8 GFLOPS` BF16 | `402.1 GFLOPS` FP32
  - 8 Threads: `4,031.7 GFLOPS` BF16 | `989.7 GFLOPS` FP32
  - 12 Threads: `3,526.0 GFLOPS` BF16 | `937.1 GFLOPS` FP32
  - 16 Threads: `4,173.9 GFLOPS` BF16 | `1,073.3 GFLOPS` FP32
  - 20–24 Threads: **`4,598.6 GFLOPS` (4.60 TFLOPS) BF16** | **`1,223.7 GFLOPS` (1.22 TFLOPS) FP32**
- **Selected Optimal Training Thread Pool:** **`20` threads** (`logical_cores - 4`), reserving 4 threads for Windows OS responsiveness and background I/O.
- **System Memory (RAM):** `31.10 GB` Total | `~11.3–12.4 GB` Free Available | Safety Reserve: `1.80–2.00 GB` minimum free RAM enforced by `ResourceUtilizationController`.
- **NVMe Storage Performance:**
  - Free Disk Space: **`211.4+ GB`** / `930.65 GB` (Safety Reserve: `2.00 GB` minimum free disk)
  - Sequential Write (with `os.fsync`): **`1,296.9 MB/s`**
  - Sequential Read: **`3,580.3 MB/s`**
