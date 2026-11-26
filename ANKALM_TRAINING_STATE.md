# ANKALM TRAINING STATE
## Active Checkpoints, Optimizer States, and Overnight Supervisor Configuration

| Checkpoint Slot | Model Name | Total Stored Params | Active Params / Token | Global Step | Processed Tokens | Train Loss | Val Loss | Val Perplexity | Weight L2 Delta | Maturity Status |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `small_final` | `AnkaLM-Small` | 15,735,168 | 15,735,168 | **280** | **1,146,880** | **~1.35** | ~4.83 | ~125.0 | **>49.5** | `pretrained` |
| `small_pretrained` | `AnkaLM-Small` | 15,735,168 | 15,735,168 | 200 | 819,200 | 1.9430 | 4.8061 | 122.26 | 47.0732 | `pretrained` |
| `small_sft` | `AnkaLM-Small-Instruct` | 15,735,168 | 15,735,168 | 265 | 840,840 | 0.5560 | **0.4397** | **1.55** | 15.6473 | `instruction_tuned` |
| `moe_final` | `AnkaLM-MoE` | **15,082,752** | **10,364,160** | **130** | **532,480** | **~3.10** | ~5.02 | ~151.0 | **>39.5** | `moe_trained` |
| `edge_final` | `AnkaLM-Edge` | **3,196,608** | **3,196,608** | **115** | **322,560** | **2.21** | 5.1323 | 169.40 | **>33.0** | `distilled_edge` |
| `vision_final` | `AnkaLM-Vision-Base` | **907,787** | **907,787** | **90** | 2,160 imgs | **<0.01** | **<0.01** | **100% Acc** | **>14.5** | `vision_trained` |
| `tiny_final` | `AnkaLM-Tiny` | 5,245,184 | 5,245,184 | 100 | 409,600 | 3.4608 | 5.0491 | 155.89 | 30.0804 | `experimental_base` |
| `micro_final` | `AnkaLM-Micro` | 1,442,432 | 1,442,432 | 60 | 122,880 | 6.3268 | 6.6677 | 786.59 | 19.5769 | `smoke_trained` |
| `balanced_4b_blueprint` | `AnkaLM-4B` | 3,372,395,520 | 3,372,395,520 | 0 | 0 | N/A | N/A | Blueprint | 0.0 | `architecture_config_only` |
| `flagship_8b_blueprint` | `AnkaLM-8B` | 7,415,795,712 | 7,415,795,712 | 0 | 0 | N/A | N/A | Blueprint | 0.0 | `architecture_config_only` |

## Overnight Supervisor Controls
- **Start / Detach:** `run_overnight.bat` (or `python training_supervisor.py --detach`)
- **Check Live Status:** `status.bat` (or `python training_status.py`)
- **Graceful Stop:** `stop_training.bat` (or `python training_stop.py`)
- **Resume:** `resume.bat` (or `python training_resume.py`)
