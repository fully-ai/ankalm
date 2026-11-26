# Changelog — AnkaLM

## [0.1.0] - 2026-11-26
### Added
- Created complete **AnkaLM** zero-to-trained Turkish language model ecosystem from scratch.
- Implemented hardware discovery (`ankalm doctor`) detecting AMD Ryzen 9 9900X3D (AVX-512 BF16 at 2,622.83 GFLOPS) and AMD Radeon RX 9060 XT (15.92 GB VRAM).
- Built multi-format Turkish dataset pipeline (`.txt`, `.md`, `.jsonl`, `.json`, `.csv`, `.parquet`, structured conversations) with NFC Turkish normalization, language ID, SHA-256 exact deduplication, shingle Jaccard near-deduplication, and zero-leakage train/val/eval splitting.
- Trained and benchmarked Byte-Level BPE (`vocab_size=8192`) and Unigram Turkish tokenizers preserving `ç, Ç, ğ, Ğ, ı, I, İ, i, ö, Ö, ş, Ş, ü, Ü`.
- Implemented independent Decoder-Only Transformer (`AnkaLMForCausalLM`) with `RMSNorm`, `RoPE`, `GroupedQueryCausalAttention` (GQA + KV Cache), and `SwiGLUFFN`.
- Pretrained `AnkaLM-Micro` (1.44M params), `AnkaLM-Tiny` (5.25M params), and `AnkaLM-Small` (15.74M params, 200 steps, 819,200 tokens) from random initialization, including verified checkpoint resume (`step 150 -> 200`).
- Executed Turkish Supervised Instruction Tuning (`AnkaLM-Small-Instruct`, 80 steps, perplexity reduced from 2,403.42 to 12.27).
- Built 16-category Turkish Evaluation Lab, FP32/BF16/FP16/INT8/INT4 Quantization Suite, FastAPI localhost server, SQLite Chat UI + Local RAG, Training Control Center Dashboard, CLI, Windows/Linux scripts, and automated `pytest` suite.
