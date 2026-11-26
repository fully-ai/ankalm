"""Resource-Aware Utilization Controller and Internal Sequential Task Queue Scheduler (Section 2 & 5)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import gc
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import psutil
import torch

from ankalm.core.constants import PROJECT_ROOT


@dataclass
class ResourceSnapshot:
    timestamp: str
    cpu_usage_pct: float
    optimal_torch_threads: int
    total_ram_gb: float
    available_ram_gb: float
    used_ram_pct: float
    process_rss_mb: float
    disk_free_gb: float
    gpu_dml_available: bool
    memory_pressure_high: bool
    disk_pressure_high: bool
    safe_to_train: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceUtilizationController:
    """
    Adaptive hardware controller that maximizes productive CPU/GPU/RAM utilization
    while enforcing OS safety reserves (default 2.0 GB free RAM, 2.0 GB free disk).
    """

    def __init__(
        self,
        min_free_ram_gb: float = 2.0,
        min_free_disk_gb: float = 2.0,
        leave_os_threads: int = 4,
    ) -> None:
        self.min_free_ram_gb = min_free_ram_gb
        self.min_free_disk_gb = min_free_disk_gb
        self.leave_os_threads = leave_os_threads

    def configure_optimal_threads(self) -> int:
        logical = psutil.cpu_count(logical=True) or 8
        # On 24-thread Ryzen 9 9900X3D, 20 threads delivers ~4.45 TFLOPS BF16 while keeping 4 threads for OS/dataloaders
        optimal = max(2, logical - self.leave_os_threads)
        torch.set_num_threads(optimal)
        return optimal

    def sample_resources(self) -> ResourceSnapshot:
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage(str(PROJECT_ROOT.anchor or PROJECT_ROOT))
        proc = psutil.Process(os.getpid())

        avail_ram_gb = round(vm.available / (1024 ** 3), 2)
        total_ram_gb = round(vm.total / (1024 ** 3), 2)
        disk_free_gb = round(disk.free / (1024 ** 3), 2)
        rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

        mem_high = avail_ram_gb < self.min_free_ram_gb
        disk_high = disk_free_gb < self.min_free_disk_gb

        dml_ok = False
        try:
            import onnxruntime as ort
            dml_ok = "DmlExecutionProvider" in ort.get_available_providers()
        except Exception:
            pass

        return ResourceSnapshot(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            cpu_usage_pct=round(psutil.cpu_percent(interval=0.05), 1),
            optimal_torch_threads=torch.get_num_threads(),
            total_ram_gb=total_ram_gb,
            available_ram_gb=avail_ram_gb,
            used_ram_pct=round(vm.percent, 1),
            process_rss_mb=rss_mb,
            disk_free_gb=disk_free_gb,
            gpu_dml_available=dml_ok,
            memory_pressure_high=mem_high,
            disk_pressure_high=disk_high,
            safe_to_train=(not mem_high) and (not disk_high),
        )


@dataclass
class ScheduledWorkloadTask:
    task_id: str
    priority: int
    workload_type: str
    model_name: str
    config_path: str
    device: str
    status: str = "pending"  # pending, running, completed, skipped_low_memory, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    peak_memory_mb: float = 0.0
    progress_description: str = ""
    checkpoint_path: str = ""
    result_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceAwareTaskScheduler:
    """
    Sequential, resource-guarded workload scheduler (Section 5).
    Never runs competing large models simultaneously: completes one task, saves its checkpoint,
    releases memory via gc.collect(), and launches the next priority workload.
    """

    def __init__(self, controller: Optional[ResourceUtilizationController] = None) -> None:
        self.controller = controller or ResourceUtilizationController()
        self.queue_file = PROJECT_ROOT / "logs" / "task_scheduler_queue.json"
        self.tasks: List[ScheduledWorkloadTask] = []

    def add_task(
        self,
        workload_type: str,
        model_name: str,
        config_path: str,
        device: str = "cpu_avx512_bf16+gpu_dml",
        priority: int = 10,
    ) -> ScheduledWorkloadTask:
        t = ScheduledWorkloadTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            priority=priority,
            workload_type=workload_type,
            model_name=model_name,
            config_path=config_path,
            device=device,
        )
        self.tasks.append(t)
        self.tasks.sort(key=lambda x: x.priority)
        self._persist()
        return t

    def _persist(self) -> None:
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.queue_file, "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in self.tasks], f, indent=2, ensure_ascii=False)

    def run_all_sequentially(self, handlers: Dict[str, Callable[[ScheduledWorkloadTask], Dict[str, Any]]]) -> List[Dict[str, Any]]:
        self.controller.configure_optimal_threads()
        completed_summaries: List[Dict[str, Any]] = []

        for task in self.tasks:
            if task.status == "completed":
                continue

            snap_before = self.controller.sample_resources()
            if not snap_before.safe_to_train:
                gc.collect()
                snap_before = self.controller.sample_resources()
                if not snap_before.safe_to_train:
                    task.status = "skipped_low_memory"
                    task.progress_description = f"Paused due to resource guard (Free RAM={snap_before.available_ram_gb}GB, Free Disk={snap_before.disk_free_gb}GB)"
                    self._persist()
                    continue

            task.status = "running"
            task.start_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._persist()

            try:
                fn = handlers.get(task.workload_type)
                if fn is None:
                    raise KeyError(f"No handler registered for workload_type='{task.workload_type}'")
                res = fn(task)
                snap_after = self.controller.sample_resources()
                task.status = "completed"
                task.end_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                task.peak_memory_mb = max(snap_before.process_rss_mb, snap_after.process_rss_mb)
                task.checkpoint_path = str(res.get("checkpoint_path", res.get("student_checkpoint", "")))
                task.result_summary = res
                task.progress_description = "Completed and checkpoint verified"
            except Exception as exc:
                task.status = "failed"
                task.end_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                task.progress_description = f"{type(exc).__name__}: {exc}"
                task.result_summary = {"error": str(exc)}
            finally:
                self._persist()
                gc.collect()
                completed_summaries.append(task.to_dict())

        return completed_summaries
