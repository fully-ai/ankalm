# Model Card — AnkaLM-Small & AnkaLM-Small-Instruct

## Model Details
- **Model Family:** AnkaLM (`AnkaLM-Micro`, `AnkaLM-Tiny`, `AnkaLM-Small`, `AnkaLM-Small-Instruct`)
- **Architecture:** Independent Pre-Norm Decoder-Only Transformer with `RMSNorm`, `RotaryEmbedding` (RoPE), `GroupedQueryCausalAttention` (GQA, 6 query heads / 2 KV heads), `SwiGLUFFN`, and tied input/output embeddings.
- **Parameters:** `15,735,168` (`AnkaLM-Small`)
- **Initialization:** 100% Random Normal Initialization ($\mathcal{N}(0, 0.02)$ with $1/\sqrt{2L}$ residual output scaling). Zero pretrained weights from third-party models.
- **Language:** Turkish (`tr`) and Bilingual Turkish-English (`tr-en`)
- **License:** Apache-2.0

## Training Data & Pretraining Metrics
- **Corpus:** 726 cleaned, deduplicated Turkish documents (`48,793` unique tokens) from open-licensed Turkish Wikipedia (`CC-BY-SA-4.0`) and AnkaLM multi-domain Turkish datasets (`CC-BY-4.0` / `CC0-1.0`), plus 48 multi-category Turkish instruction dialogues (`4,120` unique tokens).
- **Pretraining (`small_final`):** 200 AdamW steps (`819,200` processed tokens, including verified checkpoint resume from step 150 to 200). Held-out evaluation loss: `4.80614` (`122.259` perplexity). Parameter L2 Delta from random init: `47.073166`.
- **Instruction Tuning (`small_sft`):** 80 SFT steps (`54,397` tokens). Instruction validation loss reduced from `7.78465` (`2,403.42` PPL) to `2.50686` (`12.27` PPL).

## Intended Use & Limitations
- Intended for local Turkish language modeling research, offline assistance, educational inspection, and continual local training via `resume.bat`.
- As a compact 15.7M-parameter model trained from scratch on a single local computer, it should not be relied upon for safety-critical advice or frontier-scale world knowledge.
