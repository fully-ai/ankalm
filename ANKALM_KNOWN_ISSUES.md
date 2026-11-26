# ANKALM KNOWN ISSUES & HONEST LIMITATIONS

1. **8B / 4B Full In-RAM Training Constraint on 31.1 GB RAM Host:**
   - `AnkaLM-8B` (`7,415,795,712` parameters) requires `~135.0 GB` RAM/VRAM with AdamW or `~31.0 GB` with BF16+Adafactor, exceeding the host machine's `~11.5–12.4 GB` available free RAM.
   - `AnkaLM-8B` and `AnkaLM-4B` are therefore maintained as meta-device verified architecture blueprints (`architecture_config_only`) with empirically benchmarked NVMe layer-streaming feasibility (`logs/offload_8b_feasibility_benchmark.json`), while `AnkaLM-Small` (`15.74M`), `AnkaLM-MoE` (`15.08M`), and `AnkaLM-Edge` (`3.20M`) are genuinely trained in RAM.
2. **Windows Python 3.12 PyTorch Autograd vs. DirectML GPU Division:**
   - On Windows Python 3.12, `torch 2.14.1+cpu` lacks native ROCm/HIP kernels and `torch-directml` has no Python 3.12 wheel.
   - AnkaLM bridges this cleanly via `onnxruntime-directml 1.24.4` (`DmlExecutionProvider`), running ONNX forward passes, teacher distillation logit computation, and vision batch encoding on the physical **`AMD Radeon RX 9060 XT` (15.92 GB VRAM) GPU**, while PyTorch autograd backward passes and optimizer updates run on the **`AMD Ryzen 9 9900X3D` AVX-512 BF16 (`4.60 TFLOPS`)** CPU.
3. **Compact Model World-Knowledge Boundaries:**
   - While our from-scratch trained models achieve strong convergence on our Turkish corpus (`val_perplexity = 122.26` on held-out pretraining text and `1.5522` on response-masked Turkish instructions), compact `3.2M–15.7M` models should be paired with `HybridRAGEngine` and `AnkaLMToolRegistry` (`calculator`, `python_sandbox`, `pdf_analyzer`, `excel_analyzer`) for exact factual retrieval and numerical calculations.
4. **Overnight Supervisor Power Dependency:**
   - The detached `training_supervisor.py` process runs independently of the terminal session while the computer remains powered on and awake, but cannot survive a Windows OS shutdown, sleep/hibernation, or power loss (upon reboot, run `resume.bat` or `run_overnight.bat` to continue from the last atomic checkpoint).
