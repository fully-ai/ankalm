"""Automated tests for AnkaLM Data Cleaning, Deduplication, Storage, and Security (Section 21)."""
from __future__ import annotations

from pathlib import Path
import zipfile
import pytest

from ankalm.core.constants import PROJECT_ROOT
from ankalm.data.cleaning import clean_document, identify_turkish_language
from ankalm.data.dedupe import DeduplicationEngine
from ankalm.storage.manager import StorageManager
from ankalm.utils.security import (
    redact_secrets,
    safe_extract_zip,
    validate_safe_path,
    verify_localhost_binding,
)


def test_turkish_cleaning_and_language_id() -> None:
    raw = "<p>Türkiye'de <b>doğal dil işleme</b> araştırmaları hızla gelişmektedir ve önemli sonuçlar üretmektedir.</p>"
    res = clean_document(raw)
    assert res.is_valid is True
    assert "<p>" not in res.cleaned_text
    assert res.language in ("tr", "tr-en")
    assert res.quality_score >= 0.5

    spam = "Hemen tıkla bedava kazan casino bahis fırsatı http://a.com http://b.com http://c.com http://d.com"
    spam_res = clean_document(spam)
    assert spam_res.is_valid is False
    assert spam_res.rejection_reason == "spam_detected"


def test_exact_and_near_deduplication_and_leakage_guard() -> None:
    deduper = DeduplicationEngine(near_dup_threshold=0.8)
    doc1 = "Türkçe sondan eklemeli ve ünlü uyumlarına dayalı zengin bir dildir."
    doc2 = "Türkçe sondan eklemeli ve ünlü uyumlarına dayalı zengin bir dildir."
    doc3 = "Türkçe, sondan eklemeli ve ünlü uyumlarına dayalı çok zengin bir dildir!"

    dup1, _ = deduper.is_duplicate(doc1, "d1")
    dup2, reason2 = deduper.is_duplicate(doc2, "d2")
    dup3, reason3 = deduper.is_duplicate(doc3, "d3")

    assert dup1 is False
    assert dup2 is True and reason2 == "exact_duplicate"
    assert dup3 is True and "near_duplicate" in reason3
    assert deduper.check_leakage_against(doc1) is True


def test_security_path_validation_zip_slip_and_localhost() -> None:
    safe = validate_safe_path("configs/small.yaml", PROJECT_ROOT)
    assert safe.exists()

    with pytest.raises(PermissionError, match="Unsafe path access blocked"):
        validate_safe_path("../../Windows/System32", PROJECT_ROOT)

    assert verify_localhost_binding("127.0.0.1") is True
    with pytest.raises(ValueError, match="forbidden by AnkaLM local-only security policy"):
        verify_localhost_binding("0.0.0.0")

    redacted = redact_secrets("api_key = 'sk-1234567890abcdefghijklmnop'")
    assert "sk-12345" not in redacted
    assert "[REDACTED_SECRET]" in redacted


def test_zip_slip_archive_traversal_blocked(tmp_path: Path) -> None:
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../../escaped.txt", "malicious content")

    with pytest.raises(PermissionError, match="Archive traversal blocked"):
        safe_extract_zip(evil_zip, tmp_path / "extract_target")


def test_storage_manager_inspection() -> None:
    sm = StorageManager()
    assert sm.verify_sufficient_disk_space(required_mb=10.0) is True
    summary = sm.inspect_storage()
    assert summary.disk_free_gb > 0
    assert summary.checkpoints_count >= 1
