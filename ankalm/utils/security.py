"""Security, privacy, archive traversal protection, and path validation utilities."""
from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

from ankalm.core.constants import PROJECT_ROOT

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password|authorization)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{8,}['\"]?"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
]


def redact_secrets(text: str) -> str:
    """Redact potential credentials or tokens from logs and exports."""
    if not text:
        return text
    result = text
    for pattern in SECRET_PATTERNS:
        result = pattern.sub("[REDACTED_SECRET]", result)
    return result


def validate_safe_path(target_path: str | Path, allowed_root: Path = PROJECT_ROOT) -> Path:
    """Ensure target_path resolves strictly inside allowed_root (prevents path traversal)."""
    root_resolved = allowed_root.resolve()
    target_resolved = (root_resolved / target_path).resolve() if not Path(target_path).is_absolute() else Path(target_path).resolve()
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PermissionError(
            f"Unsafe path access blocked: '{target_path}' resolves outside allowed root '{root_resolved}'"
        ) from exc
    return target_resolved


def safe_extract_zip(zip_path: str | Path, dest_dir: str | Path) -> list[Path]:
    """Extract a ZIP archive safely with zip-slip / directory traversal protection."""
    dest_root = Path(dest_dir).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_target = (dest_root / member.filename).resolve()
            try:
                member_target.relative_to(dest_root)
            except ValueError as exc:
                raise PermissionError(f"Archive traversal blocked in ZIP member: {member.filename}") from exc
            zf.extract(member, dest_root)
            extracted.append(member_target)
    return extracted


def safe_extract_tar(tar_path: str | Path, dest_dir: str | Path) -> list[Path]:
    """Extract a TAR archive safely with path traversal and symlink escape protection."""
    dest_root = Path(dest_dir).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk():
                raise PermissionError(f"Symlink/hardlink blocked in TAR member: {member.name}")
            member_target = (dest_root / member.name).resolve()
            try:
                member_target.relative_to(dest_root)
            except ValueError as exc:
                raise PermissionError(f"Archive traversal blocked in TAR member: {member.name}") from exc
            tf.extract(member, dest_root, filter="data")
            extracted.append(member_target)
    return extracted


def verify_localhost_binding(host: str) -> bool:
    """Enforce localhost-only binding by default (Section 17 & 20)."""
    allowed = {"127.0.0.1", "localhost", "::1"}
    if host not in allowed:
        raise ValueError(
            f"External host binding '{host}' is forbidden by AnkaLM local-only security policy. Use 127.0.0.1."
        )
    return True
