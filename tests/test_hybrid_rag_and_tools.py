"""Automated tests for Hybrid BM25 + AnkaLM Bi-Encoder RAG, Controlled Tool Calling, and Python Sandbox (Section 20)."""
from __future__ import annotations

from pathlib import Path
import openpyxl
import pymupdf

from ankalm.chat.hybrid_rag import HybridRAGEngine
from ankalm.tools.registry import ToolCallRequest, get_default_tool_registry


def test_hybrid_bm25_and_neural_rag_retrieval_and_metrics(tmp_path: Path) -> None:
    rag = HybridRAGEngine(tmp_path / "hybrid_rag")

    d1 = rag.index_structured_chunks(
        filename="ankara_cografya.txt",
        file_type="TXT",
        raw_bytes=b"ankara_doc_1",
        workspace="research",
        tags=["coğrafya", "türkiye"],
        chunks=[
            {
                "title": "Ankara Başkent",
                "content": "Ankara, Türkiye'nin başkenti ve İç Anadolu Bölgesi'nin en kalabalık ikinci şehridir.",
                "citation": "[TXT: ankara_cografya.txt | Paragraf #1]",
            }
        ],
    )
    assert d1["duplicate_detected"] is False

    # Duplicate ingestion check
    d1_dup = rag.index_structured_chunks(
        filename="ankara_cografya.txt",
        file_type="TXT",
        raw_bytes=b"ankara_doc_1",
        workspace="research",
        chunks=[],
    )
    assert d1_dup["duplicate_detected"] is True

    rag.index_structured_chunks(
        filename="moe_mimari.md",
        file_type="MD",
        raw_bytes=b"moe_doc_2",
        workspace="research",
        tags=["yapay_zeka", "moe"],
        chunks=[
            {
                "title": "Sparse MoE Mimarisi",
                "content": "AnkaLM-MoE modelinde 4 yönlendirilen uzman ve 1 paylaşılan uzman Top-2 yönlendirme ile çalışır.",
                "citation": "[MD: moe_mimari.md | Bölüm #1]",
            }
        ],
    )

    metrics = rag.evaluate_retrieval_quality([
        ("Türkiye'nin başkenti hangi şehirdir?", "Ankara"),
        ("AnkaLM-MoE kaç uzman ile çalışır?", "4 yönlendirilen uzman"),
    ])
    assert metrics["recall_at_1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["citation_accuracy"] == 1.0


def test_controlled_tools_calculator_sandbox_and_security(tmp_path: Path) -> None:
    registry = get_default_tool_registry()

    # 1. Calculator
    r_calc = registry.execute(ToolCallRequest("calculator", {"expression": "sqrt(225) + mean([10, 20, 30]) * 3"}))
    assert r_calc.success is True
    assert r_calc.result["result"] == 75.0

    # 2. Unit converter
    r_unit = registry.execute(ToolCallRequest("unit_converter", {"value": 100, "from_unit": "c", "to_unit": "f"}))
    assert r_unit.success is True
    assert r_unit.result["converted"] == 212.0

    # 3. JSON validator & auto-repair
    r_json = registry.execute(
        ToolCallRequest(
            "json_validator",
            {"content": '```json\n{"model": "AnkaLM", "yerel": true,}\n```', "required_keys": ["model", "yerel"]},
        )
    )
    assert r_json.success is True
    assert r_json.result["valid"] is True
    assert r_json.result["was_repaired"] is True

    # 4. Isolated Python sandbox (valid math script)
    r_py = registry.execute(
        ToolCallRequest("python_sandbox", {"code": "import math\nvals = [x**2 for x in range(1, 6)]\nprint(sum(vals))"})
    )
    assert r_py.success is True
    assert r_py.result["stdout"] == "55"

    # 5. Python sandbox blocks forbidden network/OS imports
    r_forbidden = registry.execute(ToolCallRequest("python_sandbox", {"code": "import socket\nsocket.socket()"}))
    assert r_forbidden.success is False
    assert "PermissionError" in (r_forbidden.error or "")

    # 6. Python sandbox enforces timeout on infinite loop
    r_timeout = registry.execute(ToolCallRequest("python_sandbox", {"code": "while True:\n    pass"}, timeout_sec=0.6))
    assert r_timeout.success is False
    assert "TimeoutError" in (r_timeout.error or "")

    # 7. Permission enforcement for destructive tools
    r_perm = registry.execute(
        ToolCallRequest("destructive_file_replace", {"target_path": "configs/custom.yaml"}, user_authorized=False)
    )
    assert r_perm.success is False
    assert "Permission denied" in (r_perm.error or "")
