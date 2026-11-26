"""Local document parser and local RAG engine with explicit source citations (Section 15.1)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from ankalm.core.constants import PROJECT_ROOT
from ankalm.data.cleaning import turkish_lower

SUPPORTED_ATTACHMENT_EXTENSIONS = {
    ".txt", ".md", ".json", ".jsonl", ".csv", ".pdf",
    ".py", ".js", ".ts", ".rs", ".c", ".cpp", ".h",
    ".html", ".css", ".yaml", ".yml", ".sh", ".bat", ".ps1", ".sql",
}


def parse_local_attachment(filename: str, raw_bytes: bytes) -> str:
    """Parse TXT, Markdown, JSON, CSV, PDF (via local pypdf), and code files locally."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_ATTACHMENT_EXTENSIONS:
        raise ValueError(f"Unsupported attachment file extension: {ext}")

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for idx, page in enumerate(reader.pages):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages.append(f"[Sayfa {idx + 1}]\n{txt}")
        return "\n\n".join(pages)

    text = raw_bytes.decode("utf-8", errors="replace")
    if ext == ".json":
        try:
            obj = json.loads(text)
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            return text
    if ext == ".csv":
        reader = csv.reader(io.StringIO(text))
        rows = [" | ".join(r) for r in reader if any(cell.strip() for cell in r)]
        return "\n".join(rows)
    return text


def _compute_local_hashed_embedding(text: str, dim: int = 256) -> List[float]:
    """
    Deterministic local feature-hashed word + character 3-gram L2-normalized embedding.
    Operates 100% offline with zero external model dependencies and preserves Turkish morphology.
    """
    norm = turkish_lower(text)
    words = re.findall(r"\w+", norm, re.UNICODE)
    vec = [0.0] * dim

    # Word unigrams and bigrams
    features: List[Tuple[str, float]] = [(w, 1.5) for w in words]
    for i in range(len(words) - 1):
        features.append((f"{words[i]}_{words[i+1]}", 1.2))
    # Character 3-grams and 4-grams for agglutinative Turkish root matching
    for w in words:
        if len(w) >= 3:
            for j in range(len(w) - 2):
                features.append((w[j : j + 3], 0.6))
        if len(w) >= 4:
            for j in range(len(w) - 3):
                features.append((w[j : j + 4], 0.8))

    for feat, weight in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=4).digest()
        val = int.from_bytes(digest, "little")
        idx = val % dim
        sign = 1.0 if ((val >> 16) & 1) == 0 else -1.0
        vec[idx] += sign * weight

    norm_sq = sum(v * v for v in vec)
    if norm_sq > 1e-12:
        inv = 1.0 / math.sqrt(norm_sq)
        vec = [round(v * inv, 6) for v in vec]
    return vec


def _cosine_sim(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


class LocalRAGEngine:
    """
    Local Document Retrieval-Augmented Generation (RAG) engine.
    Keeps retrieved document snippets explicitly separated from learned AnkaLM weights via source citations.
    """

    def __init__(self, rag_dir: Optional[Path] = None) -> None:
        self.rag_dir = rag_dir or (PROJECT_ROOT / "cache" / "rag")
        self.rag_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.rag_dir / "rag_index.sqlite3"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES rag_documents(id) ON DELETE CASCADE
                );
                """
            )

    def chunk_text(self, text: str, chunk_size: int = 360, overlap: int = 60) -> List[str]:
        cleaned = text.replace("\r\n", "\n").strip()
        if not cleaned:
            return []
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) + 2 <= chunk_size:
                current = f"{current}\n\n{p}".strip() if current else p
            else:
                if current:
                    chunks.append(current)
                if len(p) <= chunk_size:
                    current = p
                else:
                    # Slice long paragraph with character overlap
                    step = max(1, chunk_size - overlap)
                    for i in range(0, len(p), step):
                        sub = p[i : i + chunk_size].strip()
                        if sub:
                            chunks.append(sub)
                    current = ""
        if current:
            chunks.append(current)
        return chunks

    def ingest_document(self, filename: str, raw_bytes: bytes, chunk_size: int = 360) -> Dict[str, Any]:
        parsed_text = parse_local_attachment(filename, raw_bytes)
        chunks = self.chunk_text(parsed_text, chunk_size=chunk_size)
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        ext = Path(filename).suffix.lower().lstrip(".") or "txt"

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO rag_documents (id, filename, file_type, char_count, chunk_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, filename, ext, len(parsed_text), len(chunks), now),
            )
            for idx, ch in enumerate(chunks):
                cid = f"chk_{uuid.uuid4().hex[:10]}"
                emb = _compute_local_hashed_embedding(ch)
                conn.execute(
                    "INSERT INTO rag_chunks (id, document_id, chunk_index, content, embedding_json) VALUES (?, ?, ?, ?, ?)",
                    (cid, doc_id, idx, ch, json.dumps(emb)),
                )
        return {
            "id": doc_id,
            "filename": filename,
            "file_type": ext,
            "char_count": len(parsed_text),
            "chunk_count": len(chunks),
            "created_at": now,
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, filename, file_type, char_count, chunk_count, created_at FROM rag_documents ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_document(self, document_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM rag_documents WHERE id = ?", (document_id,))
            return cur.rowcount > 0

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.08) -> List[Dict[str, Any]]:
        """Retrieve top_k relevant local document chunks with source citations."""
        if not query or not query.strip():
            return []
        q_emb = _compute_local_hashed_embedding(query)
        scored: List[Dict[str, Any]] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding_json, d.filename
                FROM rag_chunks c
                JOIN rag_documents d ON c.document_id = d.id
                """
            ).fetchall()
            for r in rows:
                emb = json.loads(r["embedding_json"])
                sim = _cosine_sim(q_emb, emb)
                if sim >= min_score:
                    scored.append({
                        "chunk_id": r["id"],
                        "document_id": r["document_id"],
                        "filename": r["filename"],
                        "chunk_index": r["chunk_index"],
                        "content": r["content"],
                        "similarity": round(sim, 4),
                        "citation": f"[Yerel RAG Kaynağı: {r['filename']} | Parça #{r['chunk_index'] + 1} | Benzerlik: {sim:.2f}]",
                    })
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]
