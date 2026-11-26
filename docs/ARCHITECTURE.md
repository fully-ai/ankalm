# System Architecture — AnkaLM

## Component Overview
- `ankalm/core`: Configuration dataclasses (`AnkaLMConfig`, `ModelProfileConfig`, `TrainingProfileConfig`, `StorageConfig`), immutable special token IDs, and maturity level definitions.
- `ankalm/hardware`: Hardware discovery (`detector.py`), memory estimator, and forward/backward/optimizer throughput benchmarks (`benchmark.py`).
- `ankalm/data`: Multi-format streaming ingestion (`acquisition.py`), NFC Turkish cleaning & language ID (`cleaning.py`), SHA-256 + shingle Jaccard deduplication (`dedupe.py`), manifests (`manifest.py`), packed sequence datasets (`dataset.py`), and Turkish instruction tuning data (`instruction_data.py`).
- `ankalm/tokenizer`: Rust-backed HuggingFace `tokenizers` Byte-Level BPE & Unigram training (`trainer.py`), Turkish Unicode preservation (`turkish_normalizer.py`), and fingerprint verification (`tokenizer.py`).
- `ankalm/model`: Independent Decoder-Only Transformer (`layers.py`, `transformer.py`).
- `ankalm/optimization`: Cosine warmup LR scheduler and weight-decay parameter grouping (`scheduler.py`).
- `ankalm/storage`: Atomic checkpoint rotation (`checkpoint.py`) and disk quota management (`manager.py`).
- `ankalm/training`: Pretraining (`trainer.py`), Supervised Instruction Tuning (`sft_trainer.py`), and single-process PID supervisor (`process_manager.py`).
- `ankalm/evaluation`: 16-category Turkish Evaluation Lab (`suite.py`).
- `ankalm/inference` & `ankalm/quantization`: KV-cached streaming generator (`engine.py`) and FP32/BF16/FP16/INT8/INT4 quantizer (`quantizer.py`).
- `ankalm/chat` & `ankalm/api` & `web`: SQLite conversation DB (`database.py`), local RAG (`rag.py`), localhost FastAPI server (`server.py`), and dark/light Web UI.
