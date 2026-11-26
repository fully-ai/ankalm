"""Automated tests for AnkaLM Local API, Chat Database, and Local RAG (Section 21: Application)."""
from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from ankalm.api.server import create_app
from ankalm.chat.database import ChatDatabase
from ankalm.chat.rag import LocalRAGEngine


def test_chat_database_crud_search_export_import(tmp_path: Path) -> None:
    db = ChatDatabase(tmp_path / "test_chat.sqlite3")
    conv = db.create_conversation(title="Türkçe Biçimbilim Sohbeti")
    cid = conv["id"]

    m1 = db.add_message(cid, "user", "İstanbul ve Iğdır kelimelerindeki ünlü uyumu nedir?")
    m2 = db.add_message(cid, "assistant", "Türkçede kalın ve ince ünlülerin uyumudur.")

    hits = db.search_conversations("Iğdır")
    assert len(hits) == 1
    assert hits[0]["id"] == cid

    db.rename_conversation(cid, title="Güncellenmiş Başlık")
    updated = db.get_conversation(cid)
    assert updated["title"] == "Güncellenmiş Başlık"
    assert len(updated["messages"]) == 2

    # Edit message truncates subsequent messages
    after_edit = db.edit_message(m1["id"], "Yeni düzenlenen soru?", truncate_after=True)
    assert len(after_edit["messages"]) == 1

    exported = db.export_conversation(cid)
    imported = db.import_conversation(exported)
    assert imported["id"] != cid
    assert len(imported["messages"]) == 1

    assert db.delete_conversation(cid) is True


def test_local_rag_engine_ingestion_and_citations(tmp_path: Path) -> None:
    rag = LocalRAGEngine(tmp_path / "rag")
    doc_text = (
        "AnkaLM yerel araştırma laboratuvarı mimarisi RoPE, RMSNorm, SwiGLU ve GQA bileşenlerini kullanır.\n\n"
        "Model ağırlıkları tamamen rastgele başlangıç değerlerinden eğitilmiştir."
    ).encode("utf-8")
    meta = rag.ingest_document("ankalm_notlari.md", doc_text)
    assert meta["chunk_count"] >= 1

    results = rag.retrieve("AnkaLM hangi mimari bileşenleri kullanır?", top_k=2)
    assert len(results) >= 1
    assert "Yerel RAG Kaynağı: ankalm_notlari.md" in results[0]["citation"]


def test_fastapi_local_endpoints() -> None:
    app = create_app()
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["local_only"] is True
    assert r_health.json()["inference_ready"] is True

    r_models = client.get("/v1/models")
    assert r_models.status_code == 200
    models_list = r_models.json()["data"]
    assert len(models_list) >= 2
    assert any(m["inference_ready"] for m in models_list)

    r_tok = client.post("/api/tokenize", json={"text": "Pijamalı hasta yağız şoföre çabucak güvendi."})
    assert r_tok.status_code == 200
    t_ids = r_tok.json()["token_ids"]
    r_detok = client.post("/api/detokenize", json={"token_ids": t_ids})
    assert r_detok.status_code == 200
    assert "şoföre" in r_detok.json()["text"]

    r_chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "small_final",
            "messages": [{"role": "user", "content": "Türkiye'nin başkenti neresidir?"}],
            "max_tokens": 16,
            "stream": False,
        },
    )
    assert r_chat.status_code == 200
    body = r_chat.json()
    assert len(body["choices"][0]["message"]["content"]) > 0
