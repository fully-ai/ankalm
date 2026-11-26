"""Supervised single-process training state, PID liveness checks, and control signals."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import psutil

from ankalm.core.constants import PROJECT_ROOT


@dataclass
class LiveTrainingState:
    status: str  # idle, running, paused, completed, stopped, error
    pid: int
    process_alive: bool
    updated_at: str
    active_model: str
    model_profile: str
    parameter_count: int
    tokenizer_vocab_size: int
    dataset_name: str
    global_step: int
    max_steps: int
    epoch: int
    processed_tokens: int
    train_loss: float
    val_loss: float
    perplexity: float
    learning_rate: float
    tokens_per_second: float
    gpu_utilization_pct: float
    vram_usage_mb: float
    ram_usage_mb: float
    cpu_usage_pct: float
    checkpoint_status: str
    latest_checkpoint_path: str
    maturity_level: str
    param_l2_delta: float
    config_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainingProcessManager:
    """
    Guarantees single-process training supervision, verifies OS PID liveness before reporting
    'running', and provides safe stop/pause/checkpoint signals for the CLI and Dashboard.
    """

    def __init__(self, logs_dir: Optional[Path] = None) -> None:
        self.logs_dir = logs_dir or (PROJECT_ROOT / "logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.logs_dir / "training_state.json"
        self.metrics_file = self.logs_dir / "training_metrics.jsonl"
        self.stop_signal_file = self.logs_dir / "stop_training.signal"
        self.pause_signal_file = self.logs_dir / "pause_training.signal"
        self.save_signal_file = self.logs_dir / "save_checkpoint.signal"

    def clear_signals(self) -> None:
        for p in (self.stop_signal_file, self.pause_signal_file, self.save_signal_file):
            if p.exists():
                p.unlink(missing_ok=True)

    def request_stop(self) -> None:
        self.stop_signal_file.write_text(str(time.time()), encoding="utf-8")

    def request_pause(self, paused: bool = True) -> None:
        if paused:
            self.pause_signal_file.write_text(str(time.time()), encoding="utf-8")
        else:
            self.pause_signal_file.unlink(missing_ok=True)

    def request_immediate_checkpoint(self) -> None:
        self.save_signal_file.write_text(str(time.time()), encoding="utf-8")

    def should_stop(self) -> bool:
        return self.stop_signal_file.exists()

    def is_paused(self) -> bool:
        return self.pause_signal_file.exists()

    def consume_save_request(self) -> bool:
        if self.save_signal_file.exists():
            self.save_signal_file.unlink(missing_ok=True)
            return True
        return False

    def is_pid_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        if not psutil.pid_exists(pid):
            return False
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    def write_state(self, state: LiveTrainingState) -> None:
        tmp = self.logs_dir / f".tmp_state_{os.getpid()}.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
        tmp.replace(self.state_file)

    def read_state(self) -> Dict[str, Any]:
        """Read current training state and verify OS process liveness (Section 1.2 & 9)."""
        if not self.state_file.exists():
            return {
                "status": "idle",
                "pid": 0,
                "process_alive": False,
                "active_model": "None",
                "global_step": 0,
                "max_steps": 0,
                "processed_tokens": 0,
            }
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            pid = int(data.get("pid", 0))
            alive = self.is_pid_alive(pid)
            data["process_alive"] = alive
            # Never report a training job as running when its process has exited (Section 1.2)
            if data.get("status") in ("running", "paused") and not alive:
                data["status"] = "stopped"
            return data
        except Exception:
            return {"status": "idle", "pid": 0, "process_alive": False}

    def ensure_no_competing_training(self) -> None:
        current = self.read_state()
        if current.get("process_alive") and current.get("pid") != os.getpid():
            raise RuntimeError(
                f"Another AnkaLM training process is already running (PID {current.get('pid')}). "
                "Stop it first before starting a new training run."
            )

    def start_background_training(
        self,
        config_path: str = "configs/small.yaml",
        resume_from: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> int:
        """Spawn a single supervised background training process."""
        self.ensure_no_competing_training()
        self.clear_signals()
        cmd = [sys.executable, "-m", "ankalm.cli", "train", "--config", config_path]
        if resume_from:
            cmd.extend(["--resume", resume_from])
        if max_steps is not None:
            cmd.extend(["--max-steps", str(max_steps)])

        log_out = open(self.logs_dir / "background_train.log", "a", encoding="utf-8")
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_out,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        return proc.pid
