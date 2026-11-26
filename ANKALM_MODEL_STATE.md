# ANKALM MODEL STATE
## Model Family Inventory, Lineage, and 8B NVMe Offload Feasibility

## 1. Model Family Lineage
- **`AnkaLM-Small` (`15,735,168` params, Dense Baseline):** Trained from random initialization (`small_init` $\rightarrow$ `small_step_150` $\rightarrow$ `small_pretrained` at step 200 $\rightarrow$ `small_final` at step 280, `1,146,880` tokens).
- **`AnkaLM-Small-Instruct` (`15,735,168` params, SFT):** Fine-tuned from `small_pretrained` (`small_sft`, 65 response-masked Turkish instruction steps, Val Loss `0.4397`, Val PPL `1.5522`).
- **`AnkaLM-MoE` (`15,082,752` total / `10,364,160` active params, Sparse MoE):** Trained from random initialization (`moe_final`, step 130, `532,480` tokens, 4 Routed + 1 Shared Expert, Top-2 gating).
- **`AnkaLM-Edge` (`3,196,608` params, 2GB Low-VRAM Student):** Distilled from `AnkaLM-Small` teacher (`edge_final`, step 115, including Hybrid GPU-DirectML teacher distillation).
- **`AnkaLM-Vision-Base` (`907,787` params, Multi-Scale ViT):** Trained from random initialization (`vision_final`, 90 steps, 400 deduplicated images, zero train/val leakage).
- **`AnkaLM-BiEncoder-Embed-v1` (`656,129` params, RAG Embedder):** Trained from random initialization (`checkpoints/rag_embedder/embedder.pt`, 60 steps InfoNCE contrastive learning).
- **`AnkaLM-8B` (`7,415,795,712` params) & `AnkaLM-4B` (`3,372,395,520` params):** Meta-device verified architecture blueprints (`architecture_config_only`).

## 2. Empirical 8B NVMe Offload Feasibility (`logs/offload_8b_feasibility_benchmark.json`)
A genuine single layer of `AnkaLM-8B` (`d_model=4096, d_ff=15360`, `226,500,608` parameters = `432.02 MB` in BF16) was allocated, stepped with `Adafactor`, and streamed to NVMe via `np.memmap`. While layer-by-layer NVMe streaming executes without crashing RAM, a full 32-layer 8B step requires streaming `~27.6 GB` of layer shards per step. Thus, `AnkaLM-Small`, `AnkaLM-MoE`, and `AnkaLM-Edge` remain the active in-RAM training targets on this workstation.
