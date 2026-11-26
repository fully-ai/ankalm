"""Persistent Overnight Training Supervisor for AnkaLM (Section 14.1-14.4)."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import psutil

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ankalm.core.config import AnkaLMConfig
from ankalm.storage.checkpoint import CheckpointManager
from ankalm.training.process_manager import TrainingProcessManager
from ankalm.training.resource_scheduler import ResourceUtilizationController
from ankalm.training.trainer import AnkaLMTrainer

SUPERVISOR_STATE_FILE = PROJECT_ROOT / "logs" / "overnight_supervisor.json"
SUPERVISOR_LOG_FILE = PROJECT_ROOT / "logs" / "overnight_supervisor.log"
EXTERNAL_BACKUP_DIR = PROJECT_ROOT.parent / "AnkaLM_Checkpoint_Backups"


def _log(msg: str) -> None:
    SUPERVISOR_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [SUPERVISOR PID={os.getpid()}] {msg}"
    print(line, flush=True)
    with open(SUPERVISOR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_supervisor_state(payload: Dict[str, Any]) -> None:
    SUPERVISOR_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SUPERVISOR_STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(SUPERVISOR_STATE_FILE)


def read_supervisor_state() -> Dict[str, Any]:
    if not SUPERVISOR_STATE_FILE.exists():
        return {"status": "idle", "supervisor_pid": 0, "supervisor_alive": False}
    try:
        data = json.loads(SUPERVISOR_STATE_FILE.read_text(encoding="utf-8"))
        pid = int(data.get("supervisor_pid", 0))
        alive = False
        if pid > 0 and psutil.pid_exists(pid):
            try:
                p = psutil.Process(pid)
                alive = p.is_running() and p.status() != psutil.STATUS_ZOMBIE
            except Exception:
                alive = False
        data["supervisor_alive"] = alive
        if data.get("status") == "running" and not alive:
            data["status"] = "exited"
        return data
    except Exception:
        return {"status": "idle", "supervisor_pid": 0, "supervisor_alive": False}


def _backup_checkpoint_outside_tree(ckpt_dir: Path, label: str) -> Optional[Path]:
    """Maintain local verified checkpoint backups outside the source tree (Section 17)."""
    try:
        EXTERNAL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        dst = EXTERNAL_BACKUP_DIR / label
        tmp_dst = EXTERNAL_BACKUP_DIR / f".tmp_{label}_{os.getpid()}"
        if tmp_dst.exists():
            shutil.rmtree(tmp_dst, ignore_errors=True)
        shutil.copytree(ckpt_dir, tmp_dst)
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        tmp_dst.rename(dst)
        return dst
    except Exception as exc:
        _log(f"Warning: external backup failed: {exc}")
        return None


def run_supervised_loop(
    config_path: str = "configs/small.yaml",
    chunk_steps: int = 20,
    max_total_steps: int = 5000,
    sleep_between_chunks_sec: float = 1.0,
) -> int:
    # Prevent duplicate supervisors
    existing = read_supervisor_state()
    if existing.get("supervisor_alive") and int(existing.get("supervisor_pid", 0)) != os.getpid():
        _log(f"Supervisor already running with PID {existing['supervisor_pid']}; exiting duplicate.")
        return 0

    proc_mgr = TrainingProcessManager()
    proc_mgr.clear_signals()
    controller = ResourceUtilizationController(min_free_ram_gb=1.8, min_free_disk_gb=2.0)
    threads = controller.configure_optimal_threads()

    cfg = AnkaLMConfig.from_yaml(PROJECT_ROOT / config_path)
    slot_prefix = cfg.model.profile
    resume_slot = f"{slot_prefix}_final"
    if not (PROJECT_ROOT / "checkpoints" / resume_slot / "metadata.json").exists():
        resume_slot = "latest" if (PROJECT_ROOT / "checkpoints" / "latest" / "metadata.json").exists() else None

    start_ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    interrupted = False

    def _on_signal(signum: int, frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        proc_mgr.request_stop()
        _log(f"Signal {signum} received; requesting graceful step completion and atomic checkpoint save...")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except Exception:
            pass

    _log(f"Started Persistent Overnight Training Supervisor for {cfg.model.name} ({config_path}) with {threads} AVX-512 threads.")
    ckpt_mgr = CheckpointManager()
    cycles = 0

    while not interrupted and not proc_mgr.should_stop():
        snap = controller.sample_resources()
        if not snap.safe_to_train:
            _log(f"Resource safeguard triggered (Free RAM={snap.available_ram_gb}GB, Free Disk={snap.disk_free_gb}GB); pausing 10s...")
            _write_supervisor_state({
                "status": "paused_resource_guard",
                "supervisor_pid": os.getpid(),
                "supervisor_alive": True,
                "started_at": start_ts,
                "updated_at": snap.timestamp,
                "model_name": cfg.model.name,
                "config_path": config_path,
                "resource_snapshot": snap.to_dict(),
                "log_file": str(SUPERVISOR_LOG_FILE),
            })
            time.sleep(10.0)
            continue

        cur_step = 0
        if resume_slot is not None:
            try:
                m, _ = ckpt_mgr.load_checkpoint(resume_slot, restore_rng=False)
                cur_step = m.global_step
            except Exception:
                cur_step = 0

        if cur_step >= max_total_steps:
            _log(f"Reached target max_total_steps={max_total_steps} (current={cur_step}); completing supervisor cleanly.")
            break

        next_target = min(max_total_steps, cur_step + chunk_steps)
        _write_supervisor_state({
            "status": "running",
            "supervisor_pid": os.getpid(),
            "supervisor_alive": True,
            "started_at": start_ts,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model_name": cfg.model.name,
            "config_path": config_path,
            "dataset": "datasets/processed/train.jsonl",
            "gpu_backend": "DirectML (DmlExecutionProvider) + AVX-512 BF16 (20 Threads)",
            "current_step": cur_step,
            "target_chunk_step": next_target,
            "max_total_steps": max_total_steps,
            "last_checkpoint": str(PROJECT_ROOT / "checkpoints" / (resume_slot or "latest")),
            "resource_snapshot": snap.to_dict(),
            "log_file": str(SUPERVISOR_LOG_FILE),
            "failure_reason": None,
        })

        try:
            trainer = AnkaLMTrainer(cfg, config_path_str=config_path)
            res = trainer.train(
                resume_from=resume_slot,
                max_steps_override=next_target,
                slot_prefix=slot_prefix,
                maturity_target="moe_trained" if cfg.model.use_moe else "pretrained",
            )
            resume_slot = f"{slot_prefix}_final"
            cycles += 1
            backup_path = _backup_checkpoint_outside_tree(Path(res.checkpoint_path), f"{slot_prefix}_latest_backup")

            _log(
                f"Chunk #{cycles} complete: step={res.final_step} | tokens={res.processed_tokens:,} | "
                f"train_loss={res.final_train_loss:.4f} | val_loss={res.final_val_loss:.4f} | "
                f"ppl={res.final_val_perplexity:.2f} | tok/s={res.avg_tokens_per_second:.1f} | backup={backup_path}"
            )
            _write_supervisor_state({
                "status": "running" if not (interrupted or proc_mgr.should_stop()) else "stopped_cleanly",
                "supervisor_pid": os.getpid(),
                "supervisor_alive": True,
                "started_at": start_ts,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "model_name": cfg.model.name,
                "config_path": config_path,
                "dataset": "datasets/processed/train.jsonl",
                "gpu_backend": "DirectML (DmlExecutionProvider) + AVX-512 BF16 (20 Threads)",
                "last_successful_step": res.final_step,
                "processed_tokens": res.processed_tokens,
                "train_loss": res.final_train_loss,
                "val_loss": res.final_val_loss,
                "val_perplexity": res.final_val_perplexity,
                "tokens_per_second": res.avg_tokens_per_second,
                "param_l2_delta": res.param_l2_delta,
                "last_checkpoint": res.checkpoint_path,
                "external_backup_path": str(backup_path or ""),
                "resource_snapshot": controller.sample_resources().to_dict(),
                "log_file": str(SUPERVISOR_LOG_FILE),
                "failure_reason": None,
            })
        except Exception as exc:
            _log(f"Supervisor caught training error: {type(exc).__name__}: {exc}")
            _write_supervisor_state({
                "status": "error",
                "supervisor_pid": os.getpid(),
                "supervisor_alive": False,
                "started_at": start_ts,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "model_name": cfg.model.name,
                "config_path": config_path,
                "log_file": str(SUPERVISOR_LOG_FILE),
                "failure_reason": f"{type(exc).__name__}: {exc}",
            })
            return 1

        del trainer
        gc.collect()
        if interrupted or proc_mgr.should_stop():
            break
        time.sleep(sleep_between_chunks_sec)

    final_state = read_supervisor_state()
    final_state["status"] = "stopped_cleanly"
    final_state["supervisor_alive"] = False
    final_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_supervisor_state(final_state)
    _log("Persistent Overnight Training Supervisor exited cleanly with verified checkpoint.")
    return 0


def launch_detached_supervisor(
    config_path: str = "configs/small.yaml",
    chunk_steps: int = 20,
    max_total_steps: int = 5000,
) -> int:
    """Spawn a detached OS background supervisor process that persists independently of the terminal."""
    existing = read_supervisor_state()
    if existing.get("supervisor_alive"):
        return int(existing["supervisor_pid"])

    SUPERVISOR_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training_supervisor.py"),
        "--config",
        config_path,
        "--chunk-steps",
        str(chunk_steps),
        "--max-total-steps",
        str(max_total_steps),
    ]
    log_fh = open(SUPERVISOR_LOG_FILE, "a", encoding="utf-8")
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    return proc.pid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnkaLM Persistent Overnight Training Supervisor")
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--chunk-steps", type=int, default=20)
    parser.add_argument("--max-total-steps", type=int, default=5000)
    parser.add_argument("--detach", action="store_true", help="Spawn as a detached background process and exit launcher")
    args = parser.parse_args()

    if args.detach:
        pid = launch_detached_supervisor(args.config, args.chunk_steps, args.max_total_steps)
        print(f"Launched detached AnkaLM Overnight Training Supervisor (PID={pid}). Log: {SUPERVISOR_LOG_FILE}")
        sys.exit(0)

    sys.exit(run_supervised_loop(args.config, args.chunk_steps, args.max_total_steps))
