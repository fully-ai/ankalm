# ANKALM DATA STATE
## Turkish Pretraining, Instruction Tuning, Vision, and RAG Evaluation Datasets

## 1. Turkish Pretraining Corpus (`datasets/manifests/dataset_manifest.json`)
- **Total Verified Unique Documents:** `726` (`48,793` unique BPE-8192 tokens)
- **Formats Ingested:** `.txt`, `.md`, `.jsonl` (including Wikipedia `CC-BY-SA-4.0` & structured dialogues), `.json`, `.csv`, `.parquet` (`PyArrow 26.0.1` + `NumPy 2.2.6`).
- **Splits (`zero_leakage_verified = True`):**
  - `train.jsonl`: `602` docs (`39,371` unique tokens)
  - `val.jsonl`: `50` docs (`3,873` unique tokens)
  - `test.jsonl`: `50` docs (`3,396` unique tokens)

## 2. Turkish Instruction Tuning Dataset (`datasets/processed/sft_instructions.jsonl`)
- Covers all 13 required Turkish instruction categories (conversation, Q&A, instruction following, explanation, summarization, rewriting, translation, mathematics, programming, structured output, JSON, multi-turn dialogue, uncertainty handling) with prompt-prefix loss masking (`ignore_index = -100`).

## 3. Multi-Scale Vision Dataset (`ankalm/model/vision.py`)
- **Size:** `400` procedural `CC0-1.0` Turkish visual images across 10 categories (`320` train + `80` unseen perturbed validation images).
- **Leakage Guard:** SHA-256 pixel tensor hashing verifies `0` duplicate images and `0` train/validation pixel overlap (`zero_train_test_pixel_leakage_verified = True`).
