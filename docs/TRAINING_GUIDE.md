# Training Guide — AnkaLM

## From-Scratch Pretraining
AnkaLM pretrains a causal decoder-only Transformer using next-token cross-entropy loss (`F.cross_entropy`) over sequence-packed documents (`PackedTokenDataset`).
- **Optimizer:** `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.08$ applied strictly to 2D parameter matrices; 1D RMSNorm weights have $0.0$ decay).
- **Schedule:** Linear warmup followed by cosine decay (`CosineWarmupScheduler`).
- **Mixed Precision:** Automatic `torch.autocast` (`bf16` on AVX-512 CPU or compatible GPU, delivering +25.4% higher throughput than `fp32`).
- **Gradient Health:** Gradient norm clipping at `1.0` with finite-value verification before every optimizer step.

## Supervised Instruction Tuning (SFT)
After pretraining completes (`checkpoints/small_final`), run SFT on the 13-category Turkish instruction dataset (`datasets/processed/sft_instructions.jsonl`):
```bat
.\.venv\Scripts\python.exe -m ankalm.cli train --config configs/small.yaml --sft --sft-steps 80
```
Saves the distinct instruction-tuned checkpoint to `checkpoints/small_sft` (`maturity_level = instruction_tuned`) while preserving `checkpoints/small_final` and `checkpoints/small_pretrained`.
