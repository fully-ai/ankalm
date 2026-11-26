# ANKALM NEXT STEPS & CONTINUATION GUIDE
## Exact Commands for Independent Continual Training, Scaling, and Local Operation

## 1. One-Command Local Launch
- **Launch AnkaLM AI Studio Web Interface (`http://127.0.0.1:8420`):**
  ```bat
  start.bat
  ```
- **Resume Pretraining `AnkaLM-Small` from Latest Checkpoint:**
  ```bat
  resume.bat
  ```
- **Hardware & Memory Diagnostics:**
  ```bat
  doctor.bat
  ```

## 2. Advanced CLI Continuation Commands
- **Continue `AnkaLM-Small` Pretraining to 500 Steps:**
  ```powershell
  .\.venv\Scripts\python.exe -m ankalm.cli train --config configs/small.yaml --resume small_final --max-steps 500
  ```
- **Continue `AnkaLM-MoE` Sparse Mixture-of-Experts Training to 300 Steps:**
  ```powershell
  .\.venv\Scripts\python.exe -m ankalm.cli train --config configs/moe.yaml --resume moe_final --max-steps 300
  ```
- **Run Additional Knowledge Distillation into `AnkaLM-Edge`:**
  ```powershell
  .\.venv\Scripts\python.exe -m ankalm.cli distill --teacher small_final --student edge_final --steps 200
  ```
- **Re-Train or Expand `AnkaLM-Vision`:**
  ```powershell
  .\.venv\Scripts\python.exe -m ankalm.cli train-vision --steps 120
  ```
- **Run Full Automated Quality Assurance Suite (`35` tests):**
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/ -v
  ```

## 3. Scaling Roadmap for Larger Hardware
1. **Enabling Native ROCm GPU Acceleration:** When running on Linux with ROCm or Windows with a PyTorch ROCm build targeting `AMD Radeon RX 9060 XT (15.92 GB VRAM)`, `ankalm doctor` will automatically verify `cuda:0` (`rocm`) and unlock `AnkaLM-Extended` (`41.96M` params) and `AnkaLM-Edge-300M` (`226.53M` params) for fast in-VRAM pretraining.
2. **Multi-GPU / High-Memory Server Training for `AnkaLM-4B` & `AnkaLM-8B`:** Use `configs/balanced_4b.yaml` (`optimizer_type: adamw_8bit`) on $\ge 64$ GB VRAM systems and `configs/flagship_8b.yaml` (`optimizer_type: adafactor`) on $\ge 128$ GB multi-GPU nodes.
