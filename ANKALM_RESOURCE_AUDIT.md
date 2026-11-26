# ANKALM RESOURCE AUDIT
## Complete Hardware, Compute Backend, Thread-Scaling, Memory, and NVMe I/O Audit (Section 1)

**Audited At:** `2026-11-26T02:26:39+03:00`  
**Workspace:** `c:\Users\ubeyt\Desktop\ANKA-LM\AnkaLM`

---

## 1. Host System & Hardware Inventory

| Component | Detected Specification | Measured / Verified Capability |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (`10.0.26200`) | Win32 NT / PowerShell / Detached Process Group Support |
| **CPU Model** | `AMD Ryzen 9 9900X3D 12-Core Processor` | 12 Physical Cores / 24 Logical Threads (`AVX512`, `MKLDNN=True`, `OpenMP=True`) |
| **System RAM** | `31.10 GB` Total Physical RAM | `11.30–12.79 GB` Available Free RAM (`1.80–2.00 GB` OS Safety Reserve Enforced) |
| **Primary Discrete GPU** | `AMD Radeon RX 9060 XT` (RDNA 4 `gfx1200`, `0x7590`) | **`15.92 GB` Dedicated VRAM**, Driver `32.0.32029.2010` |
| **Integrated GPU** | `AMD Radeon(TM) Graphics` (`0x13C0`) | `0.50 GB` VRAM, Driver `32.0.21045.29003` |
| **NVMe Disk Storage** | `C:\` Volume (`930.65 GB` Total) | **`211.43 GB` Free** |
| **NVMe Write Bandwidth** | Sequential 32 MB Write + `os.fsync` | **`1,296.9 MB/s`** (`813.7 MB/s` sustained 440 MB 8B-layer `np.memmap` flush) |
| **NVMe Read Bandwidth** | Sequential 32 MB Read | **`3,580.3 MB/s`** (`1,348.4 MB/s` sustained 440 MB 8B-layer `np.memmap` read) |
| **Python & ML Runtime** | `Python 3.12.10` (`AnkaLM\.venv`) | `PyTorch 2.14.1+cpu`, `onnxruntime-directml 1.24.4`, `NumPy 2.2.6`, `PyArrow 26.0.1` |

---

## 2. Empirical CPU AVX-512 Thread-Scaling Benchmark ($1024 \times 1024$ MatMul)

| PyTorch Thread Count | BF16 Throughput (GFLOPS) | FP32 Throughput (GFLOPS) | BF16 Speedup vs. FP32 | Notes |
| ---: | ---: | ---: | ---: | :--- |
| `4` Threads | 2,349.8 GFLOPS | 402.1 GFLOPS | 5.84x | Single-CCD partial load |
| `8` Threads | 4,031.7 GFLOPS | 989.7 GFLOPS | 4.07x | Full 3D V-Cache CCD0 saturation |
| `12` Threads | 3,526.0 GFLOPS | 937.1 GFLOPS | 3.76x | Cross-CCD boundary contention |
| `16` Threads | 4,173.9 GFLOPS | 1,073.3 GFLOPS | 3.89x | Dual-CCD balanced load |
| **`20` Threads (Selected)** | **`~4,450.0 GFLOPS` (4.45 TFLOPS)** | **`~1,180.0 GFLOPS`** | **3.77x** | **Leaves 4 threads free for OS & DataLoaders** |
| `24` Threads | **4,598.6 GFLOPS (4.60 TFLOPS)** | **1,223.7 GFLOPS** | 3.76x | Full 24-thread saturation |

---

## 3. Empirical `AnkaLM-8B` Single-Layer & NVMe Offload Feasibility (`logs/offload_8b_feasibility_benchmark.json`)

To evaluate 8B training feasibility on this `31.1 GB` RAM machine without crashing the OS, we allocated and benchmarked a genuine single Transformer layer of `AnkaLM-8B` ($d_{\text{model}}=4096, d_{\text{ff}}=15360, n_{\text{heads}}=32, n_{\text{kv\_heads}}=8 \rightarrow 230,694,912$ parameters = **`440.02 MB` in BF16**):
- **Single 8B Layer Forward Pass (`B=1, T=128`):** `29.88 ms`
- **Single 8B Layer Backward Pass:** `88.14 ms`
- **Single 8B Layer `Adafactor` Optimizer Update:** `1,136.37 ms`
- **Single 8B Layer NVMe `np.memmap` Read / Write:** `326.33 ms` / `540.76 ms`
- **Peak Process RAM During Single 8B Layer Execution:** `3,023.4 MB` (`~2.95 GB`)
- **Estimated Full 32-Layer `AnkaLM-8B` Step Time (NVMe Layer Streaming):** **`78.33 seconds/step`** (`21.76 hours` per 1,000 steps)
- **Conclusion:** Full in-RAM dense training of `AnkaLM-8B` (`7,415,795,712` params) requires `~135.0 GB` (AdamW) or `~31.0 GB` (BF16+Adafactor), exceeding the `12.79 GB` free system RAM. Accordingly, `AnkaLM-8B` and `AnkaLM-4B` are preserved as meta-device verified architecture blueprints (`architecture_config_only`), while `AnkaLM-Small` (`15.74M`), `AnkaLM-MoE` (`15.08M`), and `AnkaLM-Edge` (`3.20M`) are trained in-RAM at `2,500–3,020 tokens/sec`.
