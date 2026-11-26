# ANKALM SYSTEM STATE
## Live Workspace, Checkpoints, Backups, and Active Supervisor Summary

- **Project Root:** `c:\Users\ubeyt\Desktop\ANKA-LM\AnkaLM`
- **External Checkpoint Backup Root:** `c:\Users\ubeyt\Desktop\ANKA-LM\AnkaLM_Checkpoint_Backups`
- **Python Environment:** `c:\Users\ubeyt\Desktop\ANKA-LM\AnkaLM\.venv` (`Python 3.12.10`, `PyTorch 2.14.1+cpu`, `onnxruntime-directml 1.24.4`, `NumPy 2.2.6`, `PyArrow 26.0.1`)
- **Primary Trained Dense Checkpoint:** `checkpoints/small_final` (`AnkaLM-Small`, `15,735,168` params, step `280`, `1,146,880` processed tokens, `maturity_level = pretrained`)
- **Primary Instruction-Tuned Checkpoint:** `checkpoints/small_sft` (`AnkaLM-Small-Instruct`, `15,735,168` params, Val Loss `0.4397`, Val PPL `1.5522`, `maturity_level = instruction_tuned`)
- **Primary Sparse MoE Checkpoint:** `checkpoints/moe_final` (`AnkaLM-MoE`, `15,082,752` total / `10,364,160` active params, step `130`, `532,480` processed tokens, `maturity_level = moe_trained`)
- **Primary 2GB Low-VRAM Edge Checkpoint:** `checkpoints/edge_final` (`AnkaLM-Edge`, `3,196,608` params, step `115`, `maturity_level = distilled_edge`)
- **Primary Multimodal Vision Checkpoint:** `checkpoints/vision_final` (`AnkaLM-Vision-Base` Multi-Scale ViT, `907,787` params, `maturity_level = vision_trained`)
- **Primary Neural RAG Embedder Checkpoint:** `checkpoints/rag_embedder/embedder.pt` (`AnkaLM-BiEncoder-Embed-v1`, `656,129` params)
- **Overnight Supervisor Scripts:** `training_supervisor.py`, `training_status.py`, `training_stop.py`, `training_resume.py`, `run_overnight.bat`, `status.bat`, `stop_training.bat`
