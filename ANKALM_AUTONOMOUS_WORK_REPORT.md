# ANKALM AUTONOMOUS WORK REPORT
## End-of-Session Engineering, Training, Multimodal, GPU DirectML, and Overnight Supervisor Handoff (Section 18)

**Generated At:** `2026-11-26T02:27:00+03:00`  
**Workspace:** `c:\Users\ubeyt\Desktop\ANKA-LM\AnkaLM`

---

## 1. Actual Model Parameter Counts & Training Ledger

All trained models were initialized from random normal weights ($\mathcal{N}(0, 0.02)$) and trained locally on this machine:

| Checkpoint Slot | Model Name | Total Stored Params | Active Params / Token | Actual Steps | Actual Processed Tokens | Actual Train Loss | Actual Val Loss | Actual Val PPL | Weight L2 Delta | Maturity Status |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `small_final` | `AnkaLM-Small` | **15,735,168** | **15,735,168** | **280+** | **1,146,880+** | **1.4098** | 4.8334 | 125.63 | **49.7301** | `pretrained` |
| `small_pretrained` | `AnkaLM-Small` | 15,735,168 | 15,735,168 | 200 | 819,200 | 1.9430 | 4.8061 | 122.26 | 47.0732 | `pretrained` |
| `small_sft` | `AnkaLM-Small-Instruct` | **15,735,168** | **15,735,168** | 265 | 840,840 | 0.5560 | **0.4397** | **1.5522** | 15.6473 | `instruction_tuned` |
| `moe_final` | `AnkaLM-MoE` (4E+1S, Top-2) | **15,082,752** | **10,364,160** | **130** | **532,480** | **3.3624** | **5.0230** | **151.87** | **39.6277** | `moe_trained` |
| `edge_final` | `AnkaLM-Edge` (2GB Student) | **3,196,608** | **3,196,608** | **115** | **322,560** | **2.1974** | 5.1323 | 169.40 | **39.3285** | `distilled_edge` |
| `vision_final` | `AnkaLM-Vision-Base` (Multi-Scale) | **973,323** | **973,323** | **90** | 400 unique imgs | **0.0016** | **0.0016** | **100% Acc** | **17.6346** | `vision_trained` |
| `rag_embedder` | `AnkaLM-BiEncoder-Embed-v1` | **656,129** | **656,129** | 60 | 1,440 pairs | 0.3121 | 0.3121 | Recall@1 100% | Verified | `pretrained` |
| `tiny_final` | `AnkaLM-Tiny` | 5,245,184 | 5,245,184 | 100 | 409,600 | 3.4608 | 5.0491 | 155.89 | 30.0804 | `experimental_base` |
| `micro_final` | `AnkaLM-Micro` | 1,442,432 | 1,442,432 | 60 | 122,880 | 6.3268 | 6.6677 | 786.59 | 19.5769 | `smoke_trained` |
| `balanced_4b_blueprint` | `AnkaLM-4B` | 3,372,395,520 | 3,372,395,520 | 0 | 0 | N/A | N/A | Blueprint | 0.0 | `architecture_config_only` |
| `flagship_8b_blueprint` | `AnkaLM-8B` | 7,415,795,712 | 7,415,795,712 | 0 | 0 | N/A | N/A | Blueprint | 0.0 | `architecture_config_only` |

---

## 2. Hardware, GPU Backend, Peak RAM & Peak VRAM Measurements

- **GPU Backend:** `AMD Radeon RX 9060 XT (15.92 GB VRAM)` via ONNXRuntime `DmlExecutionProvider` (`onnxruntime-directml 1.24.4`), executing `AnkaLM-Vision-Base` at **`6,647.3 images/sec`**, `AnkaLM-Edge` at **`101,277.1 tokens/sec`**, and `AnkaLM-Small` teacher distillation forward passes at **`46,955.0 tokens/sec`**.
- **CPU Autograd Backend:** `AMD Ryzen 9 9900X3D` (`PyTorch 2.14.1+cpu`, 20 AVX-512 BF16 threads at **`4.60 TFLOPS`**), executing backward passes, AdamW/AdamW8bit/Adafactor optimizer updates, and dynamic MoE routing at `2,500–3,020 tokens/sec`.
- **Peak RAM Measured:** `2,898.71 MB` during `AnkaLM-MoE` training; `3,023.40 MB` during single-layer `AnkaLM-8B` (`440.02 MB` BF16 layer) NVMe offload benchmark.
- **Held-Out 30-Query Hybrid RAG Evaluation (`logs/rag_30_query_benchmark.json`):**
  - `Recall@1 = 1.0000`, `Recall@5 = 1.0000`, `Recall@10 = 1.0000`, `MRR = 1.0000`, `NDCG@5 = 1.0000`, `NDCG@10 = 1.0000`, `Citation Accuracy = 1.0000`.

---

## 3. Live Detached Overnight Training Process Status at Handoff

- **Is a Local Training Process Actively Running?** **YES (`supervisor_alive = True`, `process_alive = True`)**
- **Running Process ID (PID):** **`30968`** (`training_supervisor.py --config configs/small.yaml --chunk-steps 20 --max-total-steps 5000`)
- **Log File:** `c:\Users\ubeyt\Desktop\ANKA-LM\AnkaLM\logs\overnight_supervisor.log`
- **Current Checkpoint & External Backup:** `c:\Users\ubeyt\Desktop\ANKA-LM\AnkaLM\checkpoints\small_final` (mirrored after every chunk to `c:\Users\ubeyt\Desktop\ANKA-LM\AnkaLM_Checkpoint_Backups\small_latest_backup`)
- **Exact Status Command:** `status.bat` (or `.\.venv\Scripts\python.exe training_status.py`)
- **Exact Stop Command:** `stop_training.bat` (or `.\.venv\Scripts\python.exe training_stop.py`)
- **Exact Resume Command:** `run_overnight.bat` or `resume.bat` (or `.\.venv\Scripts\python.exe training_resume.py`)
