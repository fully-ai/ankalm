# 🦅 AnkaLM — Independent Turkish Foundation Model Family, Sparse MoE, Multimodal Document/Vision Engine & Local AI Ecosystem

[![Release](https://img.shields.io/github/v/release/fully-ai/ankalm?include_prereleases&style=flat-square&color=f59e0b)](https://github.com/fully-ai/ankalm/releases)
[![CI](https://img.shields.io/github/actions/workflow/status/fully-ai/ankalm/ci.yml?branch=main&style=flat-square)](https://github.com/fully-ai/ankalm/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-0ea5e9?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-38bdf8?style=flat-square&logo=python&logoColor=white)](pyproject.toml)

**AnkaLM** is a 100% local, Turkish-first causal language model family, Sparse Mixture-of-Experts (MoE) trainer, 2 GB low-VRAM distilled edge runtime, multi-scale vision transformer, 15-stage multimodal document intelligence engine (PDF, OCR, Excel formula interpreter, DOCX, PPTX, HTML, WAV), Hybrid BM25 + Neural Bi-Encoder RAG system, and sandboxed tool-execution workspace built and trained from random initialization (`N(0, 0.02)`).

---

## 🎯 Project Purpose & Core Guarantees

1. **Zero External Pretrained Weights:** Every trained checkpoint in AnkaLM is initialized strictly from random weights (`N(0, 0.02)`) and trained locally with cryptographically verified SHA-256 parameter fingerprints and positive L2 weight deltas (`weights_updated_verified = True`). No Llama, Qwen, Gemma, Mistral, or GPT weights are downloaded or wrapped.
2. **Strict Separation of Trained Checkpoints vs. Architecture Blueprints:** Models that fit within local workstation RAM/VRAM (`AnkaLM-Small` 15.7M, `AnkaLM-MoE` 15.1M/10.4M active, `AnkaLM-Edge` 3.2M, `AnkaLM-Vision-Base` 973K, `AnkaLM-BiEncoder-Embed-v1` 656K, `AnkaLM-Tiny` 5.2M, `AnkaLM-Micro` 1.4M) are genuinely trained from scratch. High-parameter flagship profiles (`AnkaLM-4B` at 3.37B parameters and `AnkaLM-8B` at 7.42B parameters) are verified on PyTorch's `meta` device and explicitly tracked as **`architecture_config_only`** blueprints rather than claimed as pretrained weights.
3. **First-Class Turkish Morphology & Unicode Fidelity:** Custom Byte-Level BPE Turkish tokenizer (`vocab_size = 8,192`, `2.87M tok/s`) preserving `ç, Ç, ğ, Ğ, ı, I, İ, i, ö, Ö, ş, Ş, ü, Ü` and uppercase/lowercase dotted/dotless `İ`/`ı` vs. `I`/`i` transitions with 100% lossless roundtrip fidelity.
4. **100% Offline & Localhost-Bound:** Serves an OpenAI-compatible REST API and interactive AI Studio UI strictly on `127.0.0.1:8420` with zero cloud calls or telemetry.

---

## 🏗️ Architectural Overview & Subsystems

```text
AnkaLM/
├── ankalm/
│   ├── core/            # Model profiles (Micro -> 8B), training configs & maturity constants
│   ├── data/            # Turkish NFC normalizer, language ID, SHA-256 & Jaccard deduplicator, zero-leakage splitter
│   ├── tokenizer/       # Custom Turkish Byte-Level BPE & Unigram tokenizer engine (vocab_size=8192)
│   ├── model/           # Pre-Norm GQA Decoder Transformer (RMSNorm, RoPE, SwiGLU), Sparse MoE & Multi-Scale ViT
│   ├── optimization/    # Cosine LR scheduler, AdamW, blockwise 8-bit AdamW8bit & factored Adafactor
│   ├── training/        # Pretraining loop, response-masked SFT, teacher-student distillation & resource scheduler
│   ├── inference/       # KV-cached streaming generator & Speculative Decoding (Edge Draft -> Small Target)
│   ├── quantization/    # FP32, BF16, FP16, dynamic INT8 & group-wise INT4 weight quantization
│   ├── multimodal/      # 15-stage PDF/OCR, 26-fn Excel Formula Interpreter, Image tiling, DOCX/PPTX/HTML & WAV VAD
│   ├── chat/            # SQLite conversation store & Hybrid Okapi BM25 + Contrastive Bi-Encoder RAG
│   ├── tools/           # 15-tool registry with isolated subprocess Python sandbox (AST import/dunder guard)
│   ├── evaluation/      # 16-category Turkish benchmark suite & regression analyzer
│   ├── hardware/        # CPU AVX-512 / GPU DirectML discovery, memory estimator & ONNX DirectML accelerator
│   ├── storage/         # Atomic checkpoint rotation (latest -> previous -> best) & full-state RNG resume
│   └── api/             # FastAPI localhost server (127.0.0.1:8420) & Training Control Center
├── configs/             # YAML profiles: micro, tiny, small, moe, edge_2gb, balanced_4b, flagship_8b
├── datasets/            # Raw open-licensed Turkish corpus, cleaned JSONL splits & quality manifests
├── tokenizer/           # Trained 8,192-vocab Turkish BPE tokenizer artifact & benchmarks
├── checkpoints/         # Provenance metadata.json & config.yaml manifests (binary weights gitignored)
├── logs/                # Reproducible JSON training, MoE routing, RAG, GPU & offload benchmark reports
├── tests/               # 36 automated pytest unit & integration tests across 11 test modules
└── docs/                # 18 architectural, training, recovery, tokenizer, hardware & evaluation guides
```

### Core Component Highlights
- **Dense GQA Transformer (`ankalm/model/transformer.py`):** Pre-Norm decoder-only architecture with `RMSNorm`, Rotary Position Embeddings (`RoPE`), Grouped-Query Causal Attention (`GQA` with dynamic KV cache), `SwiGLUFFN`, tied input/output embeddings, and scaled residual projections (`1 / sqrt(2 * n_layers)`).
- **Sparse Mixture-of-Experts (`ankalm/model/moe.py`):** 4 Routed SwiGLU Experts + 1 Always-Active Shared Expert with Top-2 softmax routing, load-balancing auxiliary loss, router z-loss stabilization, and per-layer expert utilization diagnostics (`[25.8%, 26.7%, 25.0%, 22.5%]` balanced routing, `0` dead experts).
- **2 GB Low-VRAM Edge Distillation & Speculative Decoding (`ankalm/inference/speculative.py`):** Temperature-scaled KL-divergence teacher-student knowledge distillation from `AnkaLM-Small` into `AnkaLM-Edge` (`3.20M` params, `6.85 MB` in FP16, `91.48%` capability retention), plus lossless speculative decoding (`1.246x` speedup, `100%` exact greedy token equivalence).
- **Multi-Scale Vision Transformer (`ankalm/model/vision.py`):** From-scratch trained `AnkaLM-Vision-Base` (`973,323` params) combining fine `8x8` patch embeddings and coarse `16x16` global context pooling with a multimodal language projector and InfoNCE alignment head.

---

## 📊 Model Family Inventory: Genuinely Trained Models vs. Architecture Blueprints

All trained models below were initialized from random weights (`N(0, 0.02)`) and trained locally. Binary `.safetensors` and `.pt` weight files are excluded from Git via `.gitignore` to keep the repository lightweight, while every checkpoint's `checkpoints/<slot>/metadata.json`, `config.yaml`, and `logs/*.json` provenance report is tracked in version control.

### 1. Genuinely Trained Models (Verified Weight Updates from Random Initialization)

| Checkpoint Slot | Model Name | Architecture | Total Stored Params | Active Params / Token | Optimizer Steps | Processed Tokens | Final Train Loss | Val / Eval Loss | Val / Eval PPL | Weight L2 Delta | Maturity Status |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `small_final` | `AnkaLM-Small` | 8L Dense GQA (`d=384`) | **15,735,168** | **15,735,168** | 280 | 1,146,880 | 1.4098 | 4.8334 | 125.63 | 49.7301 | `pretrained` |
| `small_pretrained` | `AnkaLM-Small` | 8L Dense GQA (`d=384`) | 15,735,168 | 15,735,168 | 200 | 819,200 | 1.9430 | 4.8061 | 122.26 | 47.0732 | `pretrained` |
| `small_sft` | `AnkaLM-Small-Instruct` | Response-Masked SFT | **15,735,168** | **15,735,168** | 265 (200 + 65 SFT) | 840,840 | 0.5560 | **0.4397** | **1.55** | 15.6473 | `instruction_tuned` |
| `moe_final` | `AnkaLM-MoE` | Sparse MoE (4E + 1 Shared, Top-2) | **15,082,752** | **10,364,160** | 130 | 532,480 | 3.3624 | 5.0230 | 151.87 | 39.6277 | `moe_trained` |
| `edge_final` | `AnkaLM-Edge` | Distilled 2 GB Edge Student | **3,196,608** | **3,196,608** | 115 | 322,560 | 2.1974 | 5.1323 | 169.40 | 39.3285 | `distilled_edge` |
| `vision_final` | `AnkaLM-Vision-Base` | Multi-Scale ViT (`8x8` + `16x16`) | **973,323** | **973,323** | 90 | 400 unique imgs | 0.0016 | **0.0016** | **100% Acc** | 17.6346 | `vision_trained` |
| `rag_embedder` | `AnkaLM-BiEncoder-Embed-v1` | Contrastive InfoNCE Bi-Encoder | **656,129** | **656,129** | 60 | 1,440 pairs | 0.3121 | 0.3121 | Recall@1 100% | Verified | `pretrained` |
| `tiny_final` | `AnkaLM-Tiny` | 6L Dense GQA (`d=256`) | 5,245,184 | 5,245,168 | 100 | 409,600 | 3.4608 | 5.0491 | 155.89 | 30.0804 | `experimental_base` |
| `micro_final` | `AnkaLM-Micro` | 4L Dense GQA (`d=128`) | 1,442,432 | 1,442,432 | 60 | 122,880 | 6.3268 | 6.6677 | 786.59 | 19.5769 | `smoke_trained` |

### 2. Architecture-Only Blueprints (Not Trained — Explicit Hardware Honesty)

Full dense training of a 3.37B or 7.42B model requires `66.5 GB` to `135.0 GB` of RAM/VRAM, exceeding a single 32 GB RAM / 16 GB VRAM workstation. These profiles are validated on PyTorch's `meta` device (`checkpoints/balanced_4b_blueprint/metadata.json`, `checkpoints/flagship_8b_blueprint/metadata.json`) and an empirical single-layer `AnkaLM-8B` (`226.5M` params, `432 MB` BF16) NVMe memory-mapped offload benchmark is recorded in `logs/offload_8b_feasibility_benchmark.json`:

| Blueprint Slot | Model Profile | Architecture Specification | Exact Parameter Count | Est. Training RAM | Status |
| :--- | :--- | :--- | ---: | ---: | :--- |
| `balanced_4b_blueprint` | `AnkaLM-4B` (`configs/balanced_4b.yaml`) | 28L Dense GQA (`d=3072, ff=9216, H=24, KV=8`) | **3,372,395,520** | ~66.5 GB | `architecture_config_only` (Untrained) |
| `flagship_8b_blueprint` | `AnkaLM-8B` (`configs/flagship_8b.yaml`) | 32L Dense GQA (`d=4096, ff=15360, H=32, KV=8`) | **7,415,795,712** | ~135.0 GB | `architecture_config_only` (Untrained) |

---

## 📄 Multimodal Document, Spreadsheet, Vision, RAG & Tool Intelligence

- **PDF Intelligence & Local OCR (`ankalm/multimodal/pdf_engine.py`, `ankalm/multimodal/ocr_engine.py`):** Native multi-page and multi-column PDF text extraction, font-hierarchy heading detection, automatic 150-DPI Local Turkish OCR fallback on scanned pages, projection-profile deskewing, table-grid extraction (CSV/JSON export), map-reduce hierarchical summarization, and page-cited Turkish Q&A.
- **Excel & 26-Function Formula Interpreter (`ankalm/multimodal/excel_engine.py`):** Parses `.xlsx`, `.xlsm` (with macros safely disabled), `.csv`, and `.tsv`; evaluates 26 spreadsheet functions (`SUM`, `AVERAGE`, `MIN`, `MAX`, `COUNT`, `COUNTA`, `IF`, `COUNTIF`, `SUMIF`, `VLOOKUP`, `INDEX`, `MATCH`, `ROUND`, `ABS`, `IFERROR`, etc.) via `AnkaLMFormulaInterpreter`; provides `Sheet!Range`-cited numerical insights and safe non-destructive workbook editing.
- **Image Understanding & High-Res Tiling (`ankalm/multimodal/image_engine.py`):** Overlapping high-resolution tiling (`> 256x256`), table-grid and UI layout analysis, OCR confidence reporting, and `AnkaLM-Vision-Base` scene classification across 10 visual categories (`100%` unseen validation accuracy).
- **DOCX, PPTX, HTML, XML, Source Code & WAV Audio (`ankalm/multimodal/doc_parsers.py`, `ankalm/multimodal/audio_engine.py`):** Structured paragraph/slide/table extraction, prompt-injection sanitization for untrusted documents, and 16-kHz WAV Voice Activity Detection (VAD).
- **Hybrid BM25 + Bi-Encoder RAG (`ankalm/chat/hybrid_rag.py`):** Combines Okapi BM25 lexical scoring with contrastively trained `AnkaLM-BiEncoder-Embed-v1` neural embeddings, Turkish morphological query rewriting, and cross-feature reranking (`Recall@1 = 1.0000`, `MRR = 1.0000` on the 30-query held-out benchmark in `logs/rag_30_query_benchmark.json`).
- **Controlled 15-Tool Registry & Isolated Python Sandbox (`ankalm/tools/registry.py`):** Deterministic AST calculator, isolated subprocess Python execution (`python -I`, AST import/dunder guard, strict timeout, no network access), unit converter, JSON validator/auto-repair, PNG chart generator, and cross-document PDF-vs-Excel comparator.

---

## 🖥️ GPU Backend Support & Honest Hardware Limitations

- **CPU Autograd Backend (Linux & Windows):** PyTorch AVX-512 BF16 (`4.60 TFLOPS` measured across 20 threads on AMD Ryzen 9 9900X3D) executes full forward/backward autograd, gradient clipping, and `AdamW` / `AdamW8bit` / `Adafactor` optimizer steps at `2,500–3,020 tokens/sec`.
- **DirectML GPU Acceleration (Windows DX12):** On Windows workstations with DirectX 12 GPUs (verified on `AMD Radeon RX 9060 XT 15.92 GB VRAM` via `onnxruntime-directml`), `DirectMLGPUAccelerator` (`ankalm/hardware/dml_accelerator.py`) executes ONNX-exported neural forward passes on the physical GPU (`101,277 tok/s` batch forward on `AnkaLM-Edge`, `46,955 tok/s` teacher distillation logits on `AnkaLM-Small`, `6,647 img/s` on `AnkaLM-Vision-Base`).
- **Current Limitation:** Because standard PyTorch on Windows/RDNA 4 (`gfx1200`) does not bundle native ROCm backward kernels by default, backward autograd runs on AVX-512 BF16 CPU while DirectML accelerates forward passes and teacher distillation. On Linux without DirectML, AnkaLM automatically falls back to `CPUExecutionProvider` (or CUDA/ROCm if installed).

---

## ⚙️ Requirements & Installation

- **OS:** Linux (Ubuntu/Arch/Debian) or Windows 10/11 (64-bit)
- **Python:** Python `>= 3.10` (tested on Python 3.12 and 3.14)
- **RAM:** Minimum 4 GB RAM (8+ GB recommended for pretraining `AnkaLM-Small` and `AnkaLM-MoE`)

### Linux Installation
```bash
git clone https://github.com/fully-ai/ankalm.git
cd ankalm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```
*(Or simply run `./install.sh`.)*

### Windows Installation
```bat
git clone https://github.com/fully-ai/ankalm.git
cd ankalm
install.bat
```

---

## 🚀 Usage, Training, Checkpoint Resume & Testing

### 1. Run Automated Test Suite (36 Tests)
```bash
pytest -v
```

### 2. Hardware Diagnostics & Data Pipeline
```bash
ankalm doctor
ankalm data inspect
ankalm data clean
ankalm data dedupe
ankalm data validate
ankalm tokenizer train --vocab-size 8192
ankalm tokenizer inspect
```

### 3. Train Models From Scratch & Resume From Checkpoints
Because binary weight files (`.safetensors` / `.pt`) are excluded from Git, run the training commands below to materialize trained checkpoints locally in `checkpoints/`:

```bash
# Smoke pretraining on AnkaLM-Micro (1.44M params)
ankalm train --config configs/micro.yaml --max-steps 60

# Pretrain AnkaLM-Small (15.74M params) from random initialization
ankalm train --config configs/small.yaml --max-steps 200

# Resume training seamlessly from the latest atomic checkpoint
ankalm train --config configs/small.yaml --resume latest --max-steps 280

# Supervised Turkish Instruction Tuning (SFT)
ankalm train --config configs/small.yaml --sft --sft-steps 65

# Detached Overnight Supervisor (auto-rotates checkpoints & external backups)
python3 training_supervisor.py --config configs/small.yaml --chunk-steps 20 --max-total-steps 1000
python3 training_status.py
python3 training_stop.py
```

### 4. Evaluation, Quantization, CLI Chat & Local Web Studio
```bash
ankalm evaluate --checkpoint small_sft
ankalm quantize --checkpoint small_sft
ankalm chat --checkpoint small_sft
ankalm serve --host 127.0.0.1 --port 8420
```

---

## ⚠️ Known Limitations & Development Roadmap

- **Known Limitations ([ANKALM_KNOWN_ISSUES.md](ANKALM_KNOWN_ISSUES.md)):**
  - `AnkaLM-Small` (`15.74M` params) and `AnkaLM-MoE` (`15.08M` params) are compact workstation-trained research models; while they learn Turkish syntax, domain vocabulary, and instruction templates, they do not possess the broad world knowledge of multi-billion-parameter cloud LLMs.
  - `AnkaLM-4B` (`3.37B`) and `AnkaLM-8B` (`7.42B`) are `architecture_config_only` blueprints awaiting `>= 64–128 GB` multi-GPU training hardware.
  - `AnkaLMLocalOCR` is optimized for clean/moderately skewed Turkish digital documents and tables (`>= 150 DPI`), not handwritten cursive text.
- **Roadmap ([ANKALM_REMAINING_WORK.md](ANKALM_REMAINING_WORK.md)):**
  1. Multi-million-token continual overnight pretraining on expanded open-licensed Turkish corpora.
  2. Native Linux ROCm / HIP backward kernel integration for in-VRAM training of `AnkaLM-Extended` (`41.96M`) and `AnkaLM-Edge-300M` (`226.53M`).
  3. Multi-GPU pretraining of the `AnkaLM-4B` and `AnkaLM-8B` blueprints.
  4. Expansion of `AnkaLM-Vision-Base` to large-scale open-licensed photographic Turkish caption datasets.

---

## 📚 Documentation Index

- [ANKALM_MASTER_ENGINEERING_REPORT.md](ANKALM_MASTER_ENGINEERING_REPORT.md) — Complete engineering, training, MoE, multimodal, and honesty audit report
- [ANKALM_MODEL_STATE.md](ANKALM_MODEL_STATE.md) & [ANKALM_MODEL_REPORT.md](ANKALM_MODEL_REPORT.md) — Model family parameter ledger & 8B NVMe offload benchmark
- [ANKALM_MOE_REPORT.md](ANKALM_MOE_REPORT.md) — Sparse MoE routing balance, shared expert & auxiliary loss analysis
- [ANKALM_MULTIMODAL_REPORT.md](ANKALM_MULTIMODAL_REPORT.md), [ANKALM_DOCUMENT_TEST_REPORT.md](ANKALM_DOCUMENT_TEST_REPORT.md) & [ANKALM_IMAGE_TEST_REPORT.md](ANKALM_IMAGE_TEST_REPORT.md) — PDF, Excel, OCR, Vision & RAG verification reports
- [ANKALM_GPU_VERIFICATION.md](ANKALM_GPU_VERIFICATION.md) & [ANKALM_LOW_VRAM_REPORT.md](ANKALM_LOW_VRAM_REPORT.md) — DirectML GPU acceleration, 2 GB Edge distillation & speculative decoding
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) & [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md) — System & Transformer architecture specifications
- [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) & [docs/TRAINING_RECOVERY.md](docs/TRAINING_RECOVERY.md) — Pretraining, SFT, and atomic checkpoint recovery
- [SECURITY.md](SECURITY.md), [PRIVACY.md](PRIVACY.md) & [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

---

## 📜 License

Licensed under the **Apache License, Version 2.0** — see [LICENSE](LICENSE) for full details.
