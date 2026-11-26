"""Dataset manifest tracking: checksums, licenses, token counts, language distributions, and processing history."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from ankalm.core.constants import PROJECT_ROOT

APPROVED_LICENSES = {
    "CC0-1.0": "Creative Commons Zero v1.0 Universal (Public Domain Dedication)",
    "Public-Domain": "Public Domain (Expired Copyright / Open Heritage)",
    "CC-BY-4.0": "Creative Commons Attribution 4.0 International",
    "CC-BY-SA-4.0": "Creative Commons Attribution-ShareAlike 4.0 International",
    "Apache-2.0": "Apache License 2.0",
    "MIT": "MIT License",
}


@dataclass
class DatasetManifestEntry:
    name: str
    category: str
    source: str
    license: str
    license_verified: bool
    format: str
    file_path: str
    file_size_bytes: int
    sha256_checksum: str
    document_count: int
    token_count: int
    character_count: int
    language_distribution: Dict[str, int]
    average_quality_score: float
    processing_history: List[str] = field(default_factory=list)


@dataclass
class DatasetManifestReport:
    generated_at: str
    total_datasets: int
    total_documents: int
    total_tokens: int
    total_characters: int
    total_size_bytes: int
    split_counts: Dict[str, int]
    split_token_counts: Dict[str, int]
    zero_leakage_verified: bool
    entries: List[DatasetManifestEntry]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_and_save_manifests(
    entries: List[DatasetManifestEntry],
    split_counts: Dict[str, int],
    split_token_counts: Dict[str, int],
    zero_leakage_verified: bool,
    manifests_dir: Optional[Path] = None,
) -> DatasetManifestReport:
    target_dir = manifests_dir or (PROJECT_ROOT / "datasets" / "manifests")
    target_dir.mkdir(parents=True, exist_ok=True)

    report = DatasetManifestReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        total_datasets=len(entries),
        total_documents=sum(e.document_count for e in entries),
        total_tokens=sum(e.token_count for e in entries),
        total_characters=sum(e.character_count for e in entries),
        total_size_bytes=sum(e.file_size_bytes for e in entries),
        split_counts=split_counts,
        split_token_counts=split_token_counts,
        zero_leakage_verified=zero_leakage_verified,
        entries=entries,
    )

    manifest_file = target_dir / "dataset_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    return report
