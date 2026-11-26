"""Independent Hybrid Retrieval System: Okapi BM25 + Keyword/Metadata Filter + AnkaLM's Own Trained Bi-Encoder Embedding Model (Section 9.2 & 9.3)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ankalm.core.constants import PROJECT_ROOT
from ankalm.data.cleaning import turkish_lower

WORD_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class AnkaLMEmbeddingModel(nn.Module):
    """
    AnkaLM's Own Independent Neural Bi-Encoder for Turkish Semantic Retrieval.
    Trained from random initialization using symmetric InfoNCE contrastive loss on AnkaLM's Turkish corpus.
    Never downloads external pretrained embedding weights.
    """

    def __init__(self, vocab_buckets: int = 4096, d_embed: int = 128, n_heads: int = 4) -> None:
        super().__init__()
        self.vocab_buckets = vocab_buckets
        self.d_embed = d_embed
        self.embed = nn.Embedding(vocab_buckets, d_embed)
        self.attn = nn.MultiheadAttention(d_embed, n_heads, batch_first=True, bias=False)
        self.norm1 = nn.LayerNorm(d_embed)
        self.ffn = nn.Sequential(
            nn.Linear(d_embed, d_embed * 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_embed * 2, d_embed, bias=False),
        )
        self.norm2 = nn.LayerNorm(d_embed)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))
        self.apply(self._init_w)

    def _init_w(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def text_to_bucket_ids(self, text: str, max_len: int = 48) -> List[int]:
        norm = turkish_lower(text)
        words = WORD_TOKEN_RE.findall(norm)
        tokens: List[str] = list(words[:max_len])
        # Add character 4-gram stems for Turkish morphological invariance
        for w in words[:16]:
            if len(w) >= 4:
                tokens.append(w[:4])
        if not tokens:
            return [0]
        ids = []
        for t in tokens[:max_len]:
            h = int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=4).digest(), "little")
            ids.append((h % (self.vocab_buckets - 1)) + 1)
        return ids

    def encode_tensor(self, batch_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(batch_ids)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        # Mean pooling over non-padded tokens
        mask_f = mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return F.normalize(pooled, dim=-1)

    @torch.no_grad()
    def encode_text(self, text: str) -> List[float]:
        self.eval()
        ids = self.text_to_bucket_ids(text)
        t_ids = torch.tensor([ids], dtype=torch.long)
        mask = torch.ones_like(t_ids, dtype=torch.bool)
        vec = self.encode_tensor(t_ids, mask)[0]
        return [round(float(v), 6) for v in vec.tolist()]


def train_ankalm_embedding_model(
    steps: int = 60,
    batch_size: int = 24,
    lr: float = 3e-3,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train AnkaLMEmbeddingModel from scratch on (query_prefix, passage_body) pairs from datasets/processed/train.jsonl."""
    target_dir = save_dir or (PROJECT_ROOT / "checkpoints" / "rag_embedder")
    target_dir.mkdir(parents=True, exist_ok=True)
    pt_path = target_dir / "embedder.pt"

    torch.manual_seed(42)
    model = AnkaLMEmbeddingModel()
    train_jsonl = PROJECT_ROOT / "datasets" / "processed" / "train.jsonl"

    pairs: List[Tuple[List[int], List[int]]] = []
    if train_jsonl.exists():
        with open(train_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                txt = json.loads(line).get("text", "")
                words = txt.split()
                if len(words) >= 10:
                    q_str = " ".join(words[:6])
                    p_str = " ".join(words[4:36])
                    pairs.append((model.text_to_bucket_ids(q_str, 24), model.text_to_bucket_ids(p_str, 36)))

    if len(pairs) < 8:
        pairs = [
            (model.text_to_bucket_ids("Türkiye başkenti Ankara İç Anadolu"), model.text_to_bucket_ids("Ankara Türkiye'nin başkenti ve İç Anadolu Bölgesi şehridir.")),
            (model.text_to_bucket_ids("Türkçe sondan eklemeli ünlü uyumu"), model.text_to_bucket_ids("Türkçe Ural-Altay dil ailesine mensup sondan eklemeli bir dildir.")),
            (model.text_to_bucket_ids("Fotosentez bitki güneş glikoz oksijen"), model.text_to_bucket_ids("Bitkiler güneş ışığı ile karbondioksit ve sudan glikoz ve oksijen üretir.")),
            (model.text_to_bucket_ids("Transformer öz-dikkat nedensel maskeleme"), model.text_to_bucket_ids("Transformer mimarisinde öz-dikkat ve nedensel maskeleme kullanılır.")),
        ] * 8

    def _pad_batch(seqs: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_l = max(len(s) for s in seqs)
        padded = [s + [0] * (max_l - len(s)) for s in seqs]
        t = torch.tensor(padded, dtype=torch.long)
        return (t, t != 0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    init_loss = 0.0
    final_loss = 0.0
    model.train()

    for s in range(1, steps + 1):
        idx = torch.randperm(len(pairs))[: min(batch_size, len(pairs))].tolist()
        q_seqs = [pairs[i][0] for i in idx]
        p_seqs = [pairs[i][1] for i in idx]
        q_ids, q_mask = _pad_batch(q_seqs)
        p_ids, p_mask = _pad_batch(p_seqs)

        opt.zero_grad(set_to_none=True)
        q_emb = model.encode_tensor(q_ids, q_mask)
        p_emb = model.encode_tensor(p_ids, p_mask)
        scale = model.logit_scale.exp().clamp(max=50.0)
        logits = scale * (q_emb @ p_emb.t())
        targets = torch.arange(logits.size(0), dtype=torch.long)
        loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
        if s == 1:
            init_loss = float(loss.item())
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    model.eval()
    meta = {
        "model_name": "AnkaLM-BiEncoder-Embed-v1",
        "trained_from_scratch": True,
        "d_embed": model.d_embed,
        "steps": steps,
        "initial_contrastive_loss": round(init_loss, 5),
        "final_contrastive_loss": round(final_loss, 5),
    }
    torch.save({"state_dict": model.state_dict(), "metadata": meta}, pt_path)
    return meta


class HybridRAGEngine:
    """
    Production Local Hybrid Retrieval-Augmented Generation (RAG) Engine (Section 9.2 & 9.3).
    Combines:
    1. Okapi BM25 lexical scoring (k1=1.5, b=0.75) with Turkish morphology stem matching.
    2. Exact keyword & metadata/workspace/tag filtering.
    3. AnkaLM's own contrastively trained neural Bi-Encoder semantic embeddings (`AnkaLMEmbeddingModel`).
    4. Exact document deduplication, collection management, and source citation tracking.
    """

    def __init__(self, rag_dir: Optional[Path] = None) -> None:
        self.rag_dir = rag_dir or (PROJECT_ROOT / "cache" / "rag")
        self.rag_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.rag_dir / "hybrid_rag.sqlite3"
        self.embedder = self._load_or_train_embedder()
        self._init_db()

    def _load_or_train_embedder(self) -> AnkaLMEmbeddingModel:
        pt_path = PROJECT_ROOT / "checkpoints" / "rag_embedder" / "embedder.pt"
        if not pt_path.exists():
            train_ankalm_embedding_model(steps=50)
        model = AnkaLMEmbeddingModel()
        if pt_path.exists():
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    workspace TEXT NOT NULL DEFAULT 'general',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    sha256_hash TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citation TEXT NOT NULL,
                    tokens_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_docs_workspace ON documents(workspace);
                CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(sha256_hash);
                """
            )

    def _tokenize_bm25(self, text: str) -> List[str]:
        norm = turkish_lower(text)
        words = WORD_TOKEN_RE.findall(norm)
        tokens = list(words)
        # Add 5-char prefixes for Turkish agglutinative root matching (e.g. 'kitaplarımızdan' -> 'kitap')
        for w in words:
            if len(w) >= 6:
                tokens.append(w[:5])
        return tokens

    def index_structured_chunks(
        self,
        filename: str,
        file_type: str,
        raw_bytes: bytes,
        chunks: List[Dict[str, Any]],
        workspace: str = "general",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Index format-specific structured chunks with duplicate detection and hybrid BM25 + neural vectors."""
        sha_hash = hashlib.sha256(raw_bytes).hexdigest()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, filename, workspace, chunk_count FROM documents WHERE sha256_hash = ? AND workspace = ?",
                (sha_hash, workspace),
            ).fetchone()
            if existing:
                return {
                    "id": existing["id"],
                    "filename": existing["filename"],
                    "file_type": file_type,
                    "workspace": existing["workspace"],
                    "chunk_count": existing["chunk_count"],
                    "duplicate_detected": True,
                }

            doc_id = f"doc_{uuid.uuid4().hex[:10]}"
            now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            total_chars = sum(len(c.get("content", "")) for c in chunks)

            conn.execute(
                """
                INSERT INTO documents (id, filename, file_type, workspace, tags_json, sha256_hash, char_count, chunk_count, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    filename,
                    file_type,
                    workspace,
                    json.dumps(tags or [], ensure_ascii=False),
                    sha_hash,
                    total_chars,
                    len(chunks),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                ),
            )

            for idx, ch in enumerate(chunks):
                cid = f"chk_{uuid.uuid4().hex[:10]}"
                content = str(ch.get("content", ""))
                title = str(ch.get("title", filename))
                citation = str(ch.get("citation", f"[{file_type}: {filename} | Parça #{idx + 1}]"))
                bm25_tokens = self._tokenize_bm25(f"{title} {content}")
                neural_emb = self.embedder.encode_text(f"{title} {content}")

                conn.execute(
                    """
                    INSERT INTO chunks (id, document_id, chunk_index, title, content, citation, tokens_json, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cid,
                        doc_id,
                        idx,
                        title,
                        content,
                        citation,
                        json.dumps(bm25_tokens, ensure_ascii=False),
                        json.dumps(neural_emb),
                    ),
                )

        return {
            "id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "workspace": workspace,
            "char_count": total_chars,
            "chunk_count": len(chunks),
            "duplicate_detected": False,
        }

    def list_documents(self, workspace: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if workspace:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE workspace = ? ORDER BY created_at DESC", (workspace,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["tags"] = json.loads(d.pop("tags_json", "[]"))
                d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
                out.append(d)
            return out

    def delete_document(self, document_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            return cur.rowcount > 0

    def search(
        self,
        query: str,
        top_k: int = 4,
        workspace: Optional[str] = None,
        tag_filter: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """
        Execute Hybrid Retrieval combining Okapi BM25, AnkaLM Neural Bi-Encoder Cosine Similarity,
        and Exact Keyword/Phrase Boosting with workspace & tag filters.
        """
        if not query or not query.strip():
            return []

        q_tokens = self._tokenize_bm25(query)
        q_emb = self.embedder.encode_text(query)
        q_norm_str = turkish_lower(query.strip())

        with self._connect() as conn:
            sql = """
                SELECT c.id, c.document_id, c.chunk_index, c.title, c.content, c.citation,
                       c.tokens_json, c.embedding_json, d.filename, d.file_type, d.workspace, d.tags_json
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
            """
            params: Tuple[Any, ...] = ()
            if workspace:
                sql += " WHERE d.workspace = ?"
                params = (workspace,)
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        # Filter by tag if specified
        candidate_rows = []
        for r in rows:
            if tag_filter:
                tags = json.loads(r["tags_json"] or "[]")
                if tag_filter not in tags:
                    continue
            candidate_rows.append(r)

        if not candidate_rows:
            return []

        doc_tokens_list = [json.loads(r["tokens_json"]) for r in candidate_rows]
        n_docs = len(doc_tokens_list)
        avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(n_docs, 1)

        # Document frequencies for query terms
        df: Dict[str, int] = {}
        for qt in set(q_tokens):
            df[qt] = sum(1 for dt in doc_tokens_list if qt in dt)

        raw_bm25_scores: List[float] = []
        neural_scores: List[float] = []
        exact_boosts: List[float] = []

        for r, dt in zip(candidate_rows, doc_tokens_list):
            dl = max(1, len(dt))
            tf_map: Dict[str, int] = {}
            for tok in dt:
                tf_map[tok] = tf_map.get(tok, 0) + 1

            bm25 = 0.0
            for qt in q_tokens:
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                n_q = df.get(qt, 0)
                idf = math.log(1.0 + ((n_docs - n_q + 0.5) / (n_q + 0.5)))
                denom = tf + k1 * (1.0 - b + b * (dl / max(avg_dl, 1.0)))
                bm25 += idf * ((tf * (k1 + 1.0)) / max(denom, 1e-9))
            raw_bm25_scores.append(bm25)

            emb = json.loads(r["embedding_json"])
            cos_sim = sum(a * b_val for a, b_val in zip(q_emb, emb))
            neural_scores.append(max(0.0, cos_sim))

            c_low = turkish_lower(r["content"])
            exact = 1.0 if q_norm_str in c_low else (0.5 if any(w in c_low for w in q_norm_str.split() if len(w) >= 4) else 0.0)
            exact_boosts.append(exact)

        max_bm25 = max(raw_bm25_scores) if raw_bm25_scores and max(raw_bm25_scores) > 1e-9 else 1.0
        scored_results: List[Dict[str, Any]] = []

        for idx, r in enumerate(candidate_rows):
            bm25_norm = raw_bm25_scores[idx] / max_bm25
            n_sim = neural_scores[idx]
            ex_b = exact_boosts[idx]
            hybrid_score = (0.55 * bm25_norm) + (0.35 * n_sim) + (0.10 * ex_b)

            if hybrid_score >= 0.05:
                scored_results.append({
                    "chunk_id": r["id"],
                    "document_id": r["document_id"],
                    "filename": r["filename"],
                    "file_type": r["file_type"],
                    "workspace": r["workspace"],
                    "chunk_index": r["chunk_index"],
                    "title": r["title"],
                    "content": r["content"],
                    "citation": r["citation"],
                    "bm25_score": round(raw_bm25_scores[idx], 4),
                    "neural_similarity": round(n_sim, 4),
                    "hybrid_score": round(hybrid_score, 4),
                })

        scored_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored_results[:top_k]

    def rewrite_turkish_query(self, query: str) -> str:
        """
        Turkish Morphological Query Rewriting (Section 12):
        Strips conversational question fillers and appends root stems for agglutinative recall.
        """
        q_norm = turkish_lower(query.strip())
        fillers = {"nedir", "nelerdir", "nasıl", "nerede", "hangi", "hakkında", "bana", "lütfen", "açıkla", "anlat", "bul"}
        words = [w for w in WORD_TOKEN_RE.findall(q_norm) if w not in fillers]
        if not words:
            words = WORD_TOKEN_RE.findall(q_norm)
        stems = []
        for w in words:
            if len(w) >= 6:
                stems.append(w[:5])
        return " ".join(words + stems)

    def _rerank_candidates(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Second-stage Cross-Feature Reranker combining Stage-1 Hybrid Score with
        query bigram proximity and stem coverage density (Section 12).
        """
        q_words = [w for w in WORD_TOKEN_RE.findall(turkish_lower(query)) if len(w) >= 3]
        if not q_words or not candidates:
            return candidates[:top_k]
        q_bigrams = {f"{q_words[i]} {q_words[i+1]}" for i in range(len(q_words) - 1)}

        for cand in candidates:
            c_low = turkish_lower(f"{cand['title']} {cand['content']}")
            word_hits = sum(1 for qw in q_words if qw in c_low or (len(qw) >= 5 and qw[:5] in c_low))
            coverage = word_hits / len(q_words)
            bigram_bonus = 0.12 if any(bg in c_low for bg in q_bigrams) else 0.0
            rerank_score = (0.75 * cand["hybrid_score"]) + (0.20 * coverage) + bigram_bonus
            cand["rerank_score"] = round(rerank_score, 4)

        candidates.sort(key=lambda x: x.get("rerank_score", x["hybrid_score"]), reverse=True)
        return candidates[:top_k]

    def evaluate_retrieval_quality(self, evaluation_queries: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Evaluate Recall@1, Recall@3, Recall@5, Recall@10, Precision@1, MRR, NDCG@5, NDCG@10,
        and Citation Accuracy given list of (query_string, expected_substring_in_citation_or_content) (Section 12).
        """
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        hits_at_10 = 0
        rr_sum = 0.0
        ndcg5_sum = 0.0
        ndcg10_sum = 0.0
        citation_valid_cnt = 0

        for q_str, expected_target in evaluation_queries:
            rewritten = self.rewrite_turkish_query(q_str)
            stage1 = self.search(rewritten, top_k=15)
            res = self._rerank_candidates(q_str, stage1, top_k=10)
            exp_low = turkish_lower(expected_target)
            found_rank = 0
            for rank, item in enumerate(res, start=1):
                combined = turkish_lower(f"{item['citation']} {item['content']}")
                if exp_low in combined:
                    found_rank = rank
                    break
            if found_rank == 1:
                hits_at_1 += 1
            if 1 <= found_rank <= 3:
                hits_at_3 += 1
            if 1 <= found_rank <= 5:
                hits_at_5 += 1
                ndcg5_sum += 1.0 / math.log2(found_rank + 1.0)
            if 1 <= found_rank <= 10:
                hits_at_10 += 1
                rr_sum += 1.0 / found_rank
                ndcg10_sum += 1.0 / math.log2(found_rank + 1.0)
            if res and res[0]["citation"].startswith("[") and "]" in res[0]["citation"]:
                citation_valid_cnt += 1

        n = max(1, len(evaluation_queries))
        return {
            "queries_evaluated": len(evaluation_queries),
            "recall_at_1": round(hits_at_1 / n, 4),
            "recall_at_3": round(hits_at_3 / n, 4),
            "recall_at_5": round(hits_at_5 / n, 4),
            "recall_at_10": round(hits_at_10 / n, 4),
            "precision_at_1": round(hits_at_1 / n, 4),
            "mrr": round(rr_sum / n, 4),
            "ndcg_at_5": round(ndcg5_sum / n, 4),
            "ndcg_at_10": round(ndcg10_sum / n, 4),
            "citation_accuracy": round(citation_valid_cnt / n, 4),
        }
