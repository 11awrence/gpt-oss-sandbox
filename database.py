from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "./data/chat_history.db"))

def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
                content TEXT,
                created_at TEXT NOT NULL,
                tool_call_id TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id, id);
            CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at);
        """)

@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DATABASE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def save_conversation(conversation_id: str, title: str, system_prompt: str, messages: list[dict], created_at: str, updated_at: str) -> None:
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO conversations (id, title, system_prompt, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                system_prompt = excluded.system_prompt,
                updated_at = excluded.updated_at
        """, (conversation_id, title, system_prompt, created_at, updated_at))
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        for msg in messages:
            conn.execute("""
                INSERT INTO messages (conversation_id, role, content, created_at, tool_call_id)
                VALUES (?, ?, ?, ?, ?)
            """, (
                conversation_id,
                msg.get("role"),
                msg.get("content"),
                updated_at,
                msg.get("tool_call_id"),
            ))

def load_all_conversations() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT id, title, system_prompt, created_at, updated_at FROM conversations ORDER BY updated_at DESC").fetchall()
        result = []
        for row in rows:
            msgs = conn.execute(
                "SELECT role, content, tool_call_id FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (row["id"],),
            ).fetchall()
            result.append({
                "conversation_id": row["id"],
                "title": row["title"],
                "messages": [
                    {k: m[k] for k in ("role", "content", "tool_call_id") if m[k] is not None}
                    for m in msgs
                ],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return result

def delete_conversation(conversation_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cur.rowcount > 0
