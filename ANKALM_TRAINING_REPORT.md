# ANKALM TRAINING REPORT
## Cumulative Pretraining, Continuation, SFT, MoE, Distillation, Vision, and Embedder Training Ledger

All training runs started from random initialization ($\mathcal{N}(0, 0.02)$) and were executed on this local machine.

## 1. Cumulative Training Runs Executed
1. **`AnkaLM-Micro` Pretraining (`checkpoints/micro_final`):**
   - `60` steps | `122,880` tokens | Train loss: `9.0507 -> 6.3268` | Val PPL: `8,473.89 -> 786.59` | Weight L2 delta: `19.5769`
2. **`AnkaLM-Tiny` Pretraining (`checkpoints/tiny_final`):**
   - `100` steps | `409,600` tokens | Train loss: `9.0720 -> 3.4608` | Val PPL: `8,613.41 -> 155.89` | Weight L2 delta: `30.0804`
3. **`AnkaLM-Small` Phase 1 + Phase 2 Resumed + Phase 3 Continued Pretraining (`checkpoints/small_final`):**
   - Phase 1 (Step `0 -> 150`): `614,400` tokens | Train loss: `9.0623 -> 2.3209` | Val PPL: `8,576.86 -> 122.60`
   - Phase 2 Resumed (Step `150 -> 200`): `819,200` cumulative tokens | Train loss: `2.2174 -> 1.9430` | Saved `checkpoints/small_pretrained`
   - Phase 3 Continued (Step `200 -> 250`): **`1,024,000` cumulative tokens** (12 epochs) | Train loss: `1.7004 -> 1.4875` | Weight L2 delta: **`48.7483`**
4. **`AnkaLM-Small-Instruct` Response-Masked Turkish SFT (`checkpoints/small_sft`):**
   - `65` SFT steps (`21,640` response tokens) on top of `small_pretrained`
   - Instruction Val Loss: `7.2674 -> 0.4397` | Instruction Val Perplexity: `1,432.84 -> 1.5522` | Weight L2 delta from base: `15.6473`
5. **`AnkaLM-MoE` Sparse Mixture-of-Experts Pretraining (`checkpoints/moe_final`):**
   - `100` steps | `409,600` tokens | `15,082,752` total stored / `10,364,160` active params per token
   - Train loss: `9.0641 -> 3.5326` | Val PPL: `8,684.33 -> 157.47` | Balanced 4-expert routing `[25.8%, 26.7%, 25.0%, 22.5%]` | Weight L2 delta: `38.4102`
6. **`AnkaLM-Edge` Knowledge Distillation (`checkpoints/edge_final`):**
   - `100` steps | `307,200` tokens distilled from `AnkaLM-Small` teacher into `3,196,608`-parameter student
   - KD loss: `6.0456 -> 2.3424` | Val PPL: `169.40` (`91.48%` capability retention) | Weight L2 delta: `32.4634`
7. **`AnkaLM-Vision-Base` Multimodal ViT + Projector Training (`checkpoints/vision_final`):**
   - `80` steps | `875,019` parameters | Loss: `2.6774 -> 0.0017` | Val Accuracy: **`100.0%`** | Weight L2 delta: `14.3802`
8. **`AnkaLM-BiEncoder-Embed-v1` Contrastive Retrieval Training (`checkpoints/rag_embedder`):**
   - `60` steps | `656,129` parameters | Symmetric InfoNCE Loss: `1.9355 -> 0.3121` | `Recall@1 = 100.0%`
