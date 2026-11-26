"""Storage inspection, disk exhaustion guards, and safe cleanup utilities (Section 19)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
from typing import Any, Dict, List

import psutil

from ankalm.core.config import StorageConfig
from ankalm.core.constants import PROJECT_ROOT
from ankalm.storage.checkpoint import CheckpointManager


@dataclass
class DirectoryStorageInfo:
    key: str
    path: str
    exists: bool
    file_count: int
    size_bytes: int
    size_mb: float


@dataclass
class StorageSummary:
    disk_total_gb: float
    disk_used_gb: float
    disk_free_gb: float
    total_project_mb: float
    directories: Dict[str, DirectoryStorageInfo]
    checkpoints_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _measure_dir(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return (0, 0)
    if path.is_file():
        return (1, path.stat().st_size)
    files = 0
    total_bytes = 0
    for p in path.rglob("*"):
        if p.is_file():
            files += 1
            total_bytes += p.stat().st_size
    return (files, total_bytes)


class StorageManager:
    """Manage AnkaLM directories, report storage footprint, and guard against disk exhaustion."""

    def __init__(self, config: StorageConfig | None = None, root: Path = PROJECT_ROOT) -> None:
        self.config = config or StorageConfig()
        self.root = root
        self.paths = self.config.resolve(root)

    def verify_sufficient_disk_space(self, required_mb: float = 500.0) -> bool:
        disk = psutil.disk_usage(str(self.root.anchor or self.root))
        free_gb = disk.free / (1024 ** 3)
        required_gb = max(self.config.min_free_disk_gb, required_mb / 1024.0)
        if free_gb < required_gb:
            raise OSError(
                f"Insufficient free disk space: {free_gb:.2f} GB available, at least {required_gb:.2f} GB required."
            )
        return True

    def inspect_storage(self) -> StorageSummary:
        disk = psutil.disk_usage(str(self.root.anchor or self.root))
        dirs_info: Dict[str, DirectoryStorageInfo] = {}
        total_bytes = 0
        for key, p in self.paths.items():
            f_cnt, b_cnt = _measure_dir(p)
            total_bytes += b_cnt
            dirs_info[key] = DirectoryStorageInfo(
                key=key,
                path=str(p),
                exists=p.exists(),
                file_count=f_cnt,
                size_bytes=b_cnt,
                size_mb=round(b_cnt / (1024 * 1024), 3),
            )
        ckpt_mgr = CheckpointManager(self.paths["checkpoints_dir"])
        ckpts = ckpt_mgr.list_checkpoints()
        return StorageSummary(
            disk_total_gb=round(disk.total / (1024 ** 3), 2),
            disk_used_gb=round(disk.used / (1024 ** 3), 2),
            disk_free_gb=round(disk.free / (1024 ** 3), 2),
            total_project_mb=round(total_bytes / (1024 * 1024), 2),
            directories=dirs_info,
            checkpoints_count=len(ckpts),
        )

    def clean_cache(self, preserve_conversations: bool = True) -> int:
        """Clean temporary cache files while preserving the SQLite conversation database."""
        cache_dir = self.paths["cache_dir"]
        if not cache_dir.exists():
            return 0
        removed_bytes = 0
        conv_db = self.paths["conversations_db"]
        for item in cache_dir.iterdir():
            if preserve_conversations and item.resolve() == conv_db.resolve():
                continue
            if item.is_file():
                sz = item.stat().st_size
                item.unlink(missing_ok=True)
                removed_bytes += sz
            elif item.is_dir() and item.name != "rag":
                _, sz = _measure_dir(item)
                shutil.rmtree(item, ignore_errors=True)
                removed_bytes += sz
        return removed_bytes
