"""SQLite-backed local conversation history with search, edit, rename, delete, export, and import."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from ankalm.core.constants import PROJECT_ROOT


class ChatDatabase:
    """100% local SQLite database for AnkaLM conversations and message history."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or (PROJECT_ROOT / "cache" / "conversations.sqlite3")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    model_checkpoint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    rag_citations TEXT NOT NULL DEFAULT '[]',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
                """
            )

    def create_conversation(
        self,
        title: str = "Yeni Türkçe Sohbet",
        system_prompt: str = "Sen AnkaLM, yerel donanımda sıfırdan eğitilmiş Türkçe yapay zeka asistanısın.",
        model_checkpoint: str = "latest",
    ) -> Dict[str, Any]:
        cid = f"conv_{uuid.uuid4().hex[:12]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, system_prompt, model_checkpoint, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (cid, title, system_prompt, model_checkpoint, now, now),
            )
        return self.get_conversation(cid)

    def list_conversations(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, system_prompt, model_checkpoint, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, system_prompt, model_checkpoint, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Conversation not found: {conversation_id}")
            conv = dict(row)
            msg_rows = conn.execute(
                "SELECT id, conversation_id, role, content, rag_citations, metrics_json, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC, rowid ASC",
                (conversation_id,),
            ).fetchall()
            messages = []
            for m in msg_rows:
                md = dict(m)
                md["rag_citations"] = json.loads(md.get("rag_citations") or "[]")
                md["metrics"] = json.loads(md.get("metrics_json") or "{}")
                del md["metrics_json"]
                messages.append(md)
            conv["messages"] = messages
            return conv

    def rename_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        conv = self.get_conversation(conversation_id)
        new_title = title if title is not None else conv["title"]
        new_sys = system_prompt if system_prompt is not None else conv["system_prompt"]
        new_ckpt = model_checkpoint if model_checkpoint is not None else conv["model_checkpoint"]
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, system_prompt = ?, model_checkpoint = ?, updated_at = ? WHERE id = ?",
                (new_title, new_sys, new_ckpt, now, conversation_id),
            )
        return self.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            return cur.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        rag_citations: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        citations_str = json.dumps(rag_citations or [], ensure_ascii=False)
        metrics_str = json.dumps(metrics or {}, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, rag_citations, metrics_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (mid, conversation_id, role, content, citations_str, metrics_str, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        return {
            "id": mid,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "rag_citations": rag_citations or [],
            "metrics": metrics or {},
            "created_at": now,
        }

    def edit_message(self, message_id: str, new_content: str, truncate_after: bool = True) -> Dict[str, Any]:
        """Edit an existing message and optionally remove subsequent messages for regeneration."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT rowid, id, conversation_id, role FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Message not found: {message_id}")
            cid = row["conversation_id"]
            rid = row["rowid"]
            conn.execute("UPDATE messages SET content = ? WHERE id = ?", (new_content, message_id))
            if truncate_after:
                conn.execute(
                    "DELETE FROM messages WHERE conversation_id = ? AND rowid > ?",
                    (cid, rid),
                )
        return self.get_conversation(cid)

    def delete_message(self, message_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
            return cur.rowcount > 0

    def search_conversations(self, query: str) -> List[Dict[str, Any]]:
        """Search conversation titles and message contents supporting Turkish characters."""
        q_norm = query.strip().lower()
        if not q_norm:
            return self.list_conversations()
        matches: List[Dict[str, Any]] = []
        for conv_summary in self.list_conversations():
            full = self.get_conversation(conv_summary["id"])
            in_title = q_norm in full["title"].lower()
            in_msgs = any(q_norm in m["content"].lower() for m in full["messages"])
            if in_title or in_msgs:
                matches.append(full)
        return matches

    def export_conversation(self, conversation_id: str) -> Dict[str, Any]:
        conv = self.get_conversation(conversation_id)
        return {
            "ankalm_export_version": "1.0",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "conversation": conv,
        }

    def import_conversation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        conv_data = payload.get("conversation", payload)
        created = self.create_conversation(
            title=conv_data.get("title", "İçe Aktarılan Sohbet"),
            system_prompt=conv_data.get("system_prompt", "Sen AnkaLM Türkçe yapay zeka asistanısın."),
            model_checkpoint=conv_data.get("model_checkpoint", "latest"),
        )
        cid = created["id"]
        for msg in conv_data.get("messages", []):
            self.add_message(
                conversation_id=cid,
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                rag_citations=msg.get("rag_citations", []),
                metrics=msg.get("metrics", {}),
            )
        return self.get_conversation(cid)
