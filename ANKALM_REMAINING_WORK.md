# ANKALM REMAINING WORK & PRIORITIZED BACKLOG
## Future Scaling Roadmap After Current Autonomous Session (Section 16.1 & 18)

While the complete AnkaLM ecosystem (Dense, Sparse MoE, 2GB Distilled Edge, Multi-Scale Vision, Local OCR, PDF/Excel/DOCX/PPTX/HTML/WAV Intelligence, Hybrid BM25+BiEncoder RAG, 15-Tool Sandbox, Speculative Decoding, AI Studio GUI, and Detached Overnight Supervisor) is operational and verified across 36 automated tests, the following prioritized research tasks remain for longer multi-day compute horizons:

1. **Priority 1 — Multi-Million Token Continual Overnight Pretraining (`run_overnight.bat`):**
   - Continue running the detached `training_supervisor.py` process across tens of thousands of steps on `AnkaLM-Small` (`configs/small.yaml`) and `AnkaLM-MoE` (`configs/moe.yaml`), expanding the raw Turkish corpus in `datasets/raw/` with additional open-licensed Turkish books, legal/scientific articles, and programming documentation.
2. **Priority 2 — Native ROCm / HIP Kernel Training on `AMD Radeon RX 9060 XT (15.92 GB VRAM)`:**
   - Currently, `onnxruntime-directml` drives the RX 9060 XT GPU at `46,955–101,277 tokens/sec` for forward passes and teacher distillation while PyTorch AVX-512 BF16 handles backward autograd. Installing a native Linux ROCm or Windows HIP PyTorch build supporting RDNA 4 (`gfx1200`) will enable full in-VRAM forward+backward training of **`AnkaLM-Extended` (`41.96M` params)** and **`AnkaLM-Edge-300M` (`226.53M` params)** inside the 16 GB VRAM budget.
3. **Priority 3 — Multi-Node / High-VRAM Training of `AnkaLM-4B` & `AnkaLM-8B` Blueprints:**
   - Train the meta-verified `AnkaLM-4B` (`3.37B` params, `configs/balanced_4b.yaml` with `AdamW8bit`) and `AnkaLM-8B` (`7.42B` params, `configs/flagship_8b.yaml` with `Adafactor`) on $\ge 64–128$ GB multi-GPU infrastructure.
4. **Priority 4 — Natural Photographic Vision Corpus Expansion:**
   - Expand `AnkaLM-Vision-Base` (`973,323` params) from its 400-image procedural Turkish diagram/chart/document/UI dataset to large-scale open-licensed photographic Turkish caption and OCR datasets.
