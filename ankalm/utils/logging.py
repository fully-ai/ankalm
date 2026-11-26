"""Structured and console logging utilities for AnkaLM."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ankalm.core.constants import PROJECT_ROOT
from ankalm.utils.security import redact_secrets


def get_logger(name: str = "ankalm", log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    default_log = log_file or (PROJECT_ROOT / "logs" / "ankalm.log")
    default_log.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(default_log, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def setup_json_logger(log_path: Path) -> Path:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def append_jsonl_metric(log_path: Path, record: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    line = redact_secrets(json.dumps(payload, ensure_ascii=False))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
