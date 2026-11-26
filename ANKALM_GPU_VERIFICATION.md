# ANKALM GPU VERIFICATION REPORT
## DirectML (`DmlExecutionProvider`) Hardware GPU Execution & Hybrid Distillation Acceptance Test (Section 1.1–1.3)

**Verified At:** `2026-11-26T02:26:39+03:00`  
**Raw JSON Report:** `logs/gpu_verification_report.json`  
**Implementation:** `ankalm/hardware/dml_accelerator.py`

---

## 1. Backend Investigation Summary

1. **Detected GPU:** `AMD Radeon RX 9060 XT` (RDNA 4 `gfx1200`, PCI Device `0x7590`, **`15.92 GB` Dedicated VRAM**, Windows Driver `32.0.32029.2010`).
2. **PyTorch Native CUDA/ROCm Status on Windows 11:**
   - `torch.__version__` = `2.14.1+cpu` (`torch.cuda.is_available() == False`, `torch.version.hip == None`).
   - `torch-directml` has no published wheel for Python 3.12 on PyPI, and the host Windows installation does not include the AMD ROCm HIP SDK (`hipInfo.exe` / `rocm-smi` absent).
   - Following Section 1.2 (*"Do not install incompatible drivers. Do not overwrite working system libraries. If one backend fails, investigate other compatible methods rather than immediately concluding that GPU acceleration is impossible. However, never claim GPU training is active when operations are running on the CPU."*), we investigated and enabled **`onnxruntime-directml 1.24.4` (`DmlExecutionProvider`, `device_id=0`)**.

---

## 2. Measured Hardware GPU Execution on `AMD Radeon RX 9060 XT` (`DmlExecutionProvider`)

We exported our from-scratch trained AnkaLM models to ONNX (`opset_version=17`) and executed genuine neural network forward passes on the physical **`AMD Radeon RX 9060 XT` GPU** via DirectX 12 / DirectML (`DmlExecutionProvider`):

| Model Executed on RX 9060 XT GPU | ONNX Artifact | Parameters | Batch $\times$ Sequence / Resolution | Measured GPU Throughput |
| :--- | :--- | ---: | :--- | ---: |
| **`AnkaLM-Vision-Base` (Multi-Scale ViT)** | `exports/ankalm_vision_dml.onnx` | 973,323 | Batch 16 $\times$ ($3 \times 64 \times 64$) | **`6,647.3 images/sec`** |
| **`AnkaLM-Edge` (2GB Student LM)** | `exports/ankalm_edge_dml.onnx` | 3,196,608 | Batch 8 $\times$ 128 tokens | **`101,277.1 tokens/sec`** |
| **`AnkaLM-Small` (Dense Teacher LM)** | `exports/ankalm_small_teacher_dml.onnx` | 15,735,168 | Batch 8 $\times$ 128 tokens | **`46,955.0 tokens/sec`** |

---

## 3. Genuine Hybrid GPU-DirectML Teacher $\rightarrow$ PyTorch Student Training Acceptance Test

In Knowledge Distillation (`AnkaLM-Small` Teacher $\rightarrow$ `AnkaLM-Edge` Student), the 15.74M-parameter Teacher only executes forward passes (`t_logits = teacher(xb)`). We executed a 15-step Hybrid GPU-DirectML Distillation training run where:
- **GPU (`AMD Radeon RX 9060 XT` via `DmlExecutionProvider`):** Computed the 15.74M-parameter Teacher's $(8 \times 128 \times 8192)$ logit tensors at **`46,955.0 tokens/sec`**.
- **CPU (`AMD Ryzen 9 9900X3D` via PyTorch AVX-512 BF16, 20 threads):** Computed the 3.20M-parameter Student forward pass, soft-target KL divergence + Cross-Entropy loss, backward autograd gradients, and AdamW weight updates.

### Measured Acceptance Test Results (`logs/gpu_verification_report.json`)
- **Steps Executed:** `15` (`15,360` tokens)
- **Initial Distillation Loss (Step 1):** `2.43103`
- **Final Distillation Loss (Step 15):** **`2.19735`** (`distillation_loss_decreased = True`)
- **Student Parameter L2 Delta Across 15 Steps:** **`6.86517`**
- **Checkpoint Save & Reload Verification (`checkpoints/edge_final`):** **`True`** (`global_step = 115`)
