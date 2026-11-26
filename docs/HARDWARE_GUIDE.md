# Hardware Guide — AnkaLM

## Discovered System Specification
- **CPU:** AMD Ryzen 9 9900X3D 12-Core Processor (12 physical / 24 logical cores)
- **RAM:** 31.1 GB Total (12.44 GB Free at discovery)
- **Disk:** 216.22 GB Free
- **Detected GPUs:** AMD Radeon RX 9060 XT (15.92 GB VRAM, Driver `32.0.32029.2010`) & AMD Radeon(TM) Graphics (0.5 GB VRAM)
- **Verified Active PyTorch Backend:** CPU AVX-512 BF16 (`2,622.83 GFLOPS` BF16 / `767.6 GFLOPS` FP32)

## Memory Estimation Table (`ankalm doctor`)
| Profile | Parameters | Precision | Weights (MB) | Gradients (MB) | AdamW State (MB) | Activations (MB) | Total Est. (MB) |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `AnkaLM-Micro` | 1,442,432 | BF16 | 5.50 | 5.50 | 11.00 | 73.50 | **351.50 MB** |
| `AnkaLM-Tiny` | 5,245,184 | BF16 | 20.01 | 20.01 | 40.02 | 172.00 | **508.04 MB** |
| `AnkaLM-Small` | 15,735,168 | BF16 | 60.03 | 60.03 | 120.05 | 212.00 | **708.11 MB** |
| `AnkaLM-Extended` | 41,955,840 | BF16 | 160.05 | 160.05 | 320.10 | 576.00 | **1,472.20 MB** |
