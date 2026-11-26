# ANKALM OVERNIGHT TRAINING REPORT
## Persistent Detached Training Supervisor Architecture, Checkpointing, and Recovery (Section 14 & 18)

**Generated At:** `2026-11-26T02:27:00+03:00`  
**Supervisor Script:** `training_supervisor.py`  
**Status & Control Scripts:** `training_status.py`, `training_stop.py`, `training_resume.py`, `run_overnight.bat`, `run_overnight.ps1`, `run_overnight.sh`, `status.bat`, `stop_training.bat`

---

## 1. Supervisor Architecture & Safeguards

`training_supervisor.py` is a detached OS background supervisor (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` on Windows, `start_new_session=True` on Linux) that continues training AnkaLM independently of the interactive terminal while the computer remains powered on:
1. **Single-Instance PID Guard:** Inspects `logs/overnight_supervisor.json` via `psutil.pid_exists(pid)` before starting, preventing duplicate supervisors or concurrent writes to the same checkpoint directory.
2. **Resource-Aware Safeguards (`ResourceUtilizationController`):** Before every training chunk (`20` steps per chunk), verifies that free system RAM $\ge 1.80$ GB and free disk space $\ge 2.00$ GB. If memory or disk pressure is high, it safely pauses, runs `gc.collect()`, logs the event, and resumes when resources recover.
3. **Optimal Compute Configuration:** Configures `20` AVX-512 threads (`~4.45 TFLOPS` BF16) on the 24-thread `AMD Ryzen 9 9900X3D`, leaving 4 logical threads free for Windows desktop responsiveness.
4. **Atomic Checkpointing & External Backups:** At the end of every chunk, saves atomic checkpoints (`checkpoints/small_final`, `checkpoints/latest`, `checkpoints/previous`, `checkpoints/best`, `checkpoints/session_final`) AND mirrors a verified backup outside the source tree to `c:\Users\ubeyt\Desktop\ANKA-LM\AnkaLM_Checkpoint_Backups\`.
5. **Graceful Interruption (`training_stop.py` / `stop_training.bat`):** Signals `logs/stop_training.signal`, allowing the active optimizer step to complete, saving an atomic checkpoint, flushing `logs/overnight_supervisor.log`, and exiting cleanly with code `0`.

---

## 2. Live Running Detached Supervisor Status (Active at Handoff)

- **Supervisor Status:** **`running` (`supervisor_alive = True`, `process_alive = True`)**
- **Active Detached Process ID (PID):** **`30968`**
- **Active Model & Config:** `AnkaLM-Small` (`15,735,168` parameters, `configs/small.yaml`)
- **Resumed Step & Target:** Resumed from step `282` (`1,155,072` tokens) $\rightarrow$ actively training step `283+` toward `max_total_steps = 5000` (`20` steps per atomic checkpoint chunk)
- **Live Resource Telemetry (`PID 30968`):** `71.2%` CPU utilization (`20` AVX-512 BF16 threads at `2,431.4 tokens/sec`), `1,729.5 MB` process RAM (`12.32 GB` free system RAM available), `207.49 GB` free disk space
- **Supervisor Log File:** `c:\Users\ubeyt\Desktop\ANKA-LM\AnkaLM\logs\overnight_supervisor.log`
- **Supervisor State JSON:** `c:\Users\ubeyt\Desktop\ANKA-LM\AnkaLM\logs\overnight_supervisor.json`
- **External Checkpoint Backup Location:** `c:\Users\ubeyt\Desktop\ANKA-LM\AnkaLM_Checkpoint_Backups\small_latest_backup`

---

## 3. Exact Operational Commands

- **Start / Resume Detached Overnight Training:**
  ```bat
  run_overnight.bat
  ```
  *(Or `.\.venv\Scripts\python.exe training_resume.py --config configs/small.yaml --chunk-steps 20 --max-total-steps 5000`)*
- **Inspect Live Overnight Training Status & PID:**
  ```bat
  status.bat
  ```
  *(Or `.\.venv\Scripts\python.exe training_status.py`)*
- **Gracefully Stop Overnight Training & Save Final Checkpoint:**
  ```bat
  stop_training.bat
  ```
  *(Or `.\.venv\Scripts\python.exe training_stop.py`)*
- **Run Continuous Interactive CLI Training:**
  ```powershell
  .\.venv\Scripts\python.exe -m ankalm.cli train --config configs/small.yaml --continuous
  ```
