# Live & Persisted Training State — AnkaLM

## Active Checkpoint Inventory (`checkpoints/`)
| Checkpoint Slot | Model Name | Total Stored Params | Active Params / Token | Global Step | Processed Tokens | Train Loss | Val Loss / PPL | Maturity Level |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `small_final` | `AnkaLM-Small` | 15,735,168 | 15,735,168 | **250** | **1,024,000** | **1.4875** | 4.8249 / 124.58 | `pretrained` |
| `small_pretrained` | `AnkaLM-Small` | 15,735,168 | 15,735,168 | 200 | 819,200 | 1.9430 | 4.8061 / 122.26 | `pretrained` |
| `small_sft` | `AnkaLM-Small-Instruct` | 15,735,168 | 15,735,168 | 265 | 840,840 | 0.5560 | **0.4397 / 1.55** | `instruction_tuned` |
| `moe_final` | `AnkaLM-MoE` | **15,082,752** | **10,364,160** | 100 | 409,600 | 3.5326 | 5.0592 / 157.47 | `moe_trained` |
| `edge_final` | `AnkaLM-Edge` | **3,196,608** | **3,196,608** | 100 | 307,200 | 2.3424 | 5.1323 / 169.40 | `distilled_edge` |
| `vision_final` | `AnkaLM-Vision-Base` | **875,019** | **875,019** | 80 | 1,600 imgs | 0.0017 | **0.0015 (100% Acc)** | `vision_trained` |
| `rag_embedder` | `AnkaLM-BiEncoder-Embed-v1` | 656,129 | 656,129 | 60 | 1,440 pairs | 0.3121 | Contrastive InfoNCE | `pretrained` |
| `tiny_final` | `AnkaLM-Tiny` | 5,245,184 | 5,245,184 | 100 | 409,600 | 3.4608 | 5.0491 / 155.89 | `experimental_base` |
| `micro_final` | `AnkaLM-Micro` | 1,442,432 | 1,442,432 | 60 | 122,880 | 6.3268 | 6.6677 / 786.59 | `smoke_trained` |
| `flagship_8b_blueprint` | `AnkaLM-8B` | 7,415,795,712 | 7,415,795,712 | 0 | 0 | N/A | Meta-Verified Blueprint | `architecture_config_only` |
| `balanced_4b_blueprint` | `AnkaLM-4B` | 3,372,395,520 | 3,372,395,520 | 0 | 0 | N/A | Meta-Verified Blueprint | `architecture_config_only` |
