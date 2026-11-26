# ANKALM GPU STATUS
## Hardware GPU Discovery, DirectML (`DmlExecutionProvider`) Verification, and Compute Division

## 1. Detected GPU Hardware
- **Primary Discrete GPU:** `AMD Radeon RX 9060 XT` (RDNA 4 `gfx1200`, PCI ID `0x7590`, **`15.92 GB` Dedicated VRAM**, Driver `32.0.32029.2010`)
- **Integrated GPU:** `AMD Radeon(TM) Graphics` (`0.50 GB` VRAM, Driver `32.0.21045.29003`)

## 2. Active Hardware GPU Execution via DirectML (`onnxruntime-directml 1.24.4`)
Because Windows Python 3.12 provides `torch 2.14.1+cpu` (no `torch-directml` wheel exists for Python 3.12 and the host OS does not have AMD's ROCm HIP SDK installed), AnkaLM integrates **`onnxruntime-directml 1.24.4` (`DmlExecutionProvider`, `device_id=0`)** in `ankalm/hardware/dml_accelerator.py` to execute genuine neural network workloads directly on the **`AMD Radeon RX 9060 XT` GPU**:
- **`AnkaLM-Vision-Base` (`exports/ankalm_vision_dml.onnx`) on GPU:** **`6,277.9 images/sec`** (Batch-16 $\times$ 50 passes in `127.43 ms`)
- **`AnkaLM-Edge` (`exports/ankalm_edge_dml.onnx`, 3.20M params) on GPU:** **`97,916.6 tokens/sec`** (Batch-8 $\times$ 128 tokens $\times$ 25 passes in `261.45 ms`)
- **`AnkaLM-Small` Teacher (`exports/ankalm_small_teacher_dml.onnx`, 15.74M params) on GPU:** Executes teacher forward passes for Hybrid GPU-DirectML Teacher $\rightarrow$ PyTorch Student Knowledge Distillation.

## 3. Honest Workload Division
- **GPU (`AMD Radeon RX 9060 XT` via `DmlExecutionProvider`):** ONNX forward passes, teacher distillation logit generation, batch vision patch encoding, and batch inference benchmarks.
- **CPU (`AMD Ryzen 9 9900X3D` via `PyTorch 2.14.1+cpu` AVX-512 BF16 at 20 threads = `4.60 TFLOPS`):** Autograd backward passes (`loss.backward()`), `AdamW` / `AdamW8bit` / `Adafactor` optimizer updates, and dynamic Sparse MoE token routing.
