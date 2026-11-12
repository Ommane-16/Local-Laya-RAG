"""SQLite metadata store."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from config import DB_PATH

class MetadataStore:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER,
                    text TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")

    def add_chunk(self, chunk: dict) -> bool:
        text = str(chunk.get("text", "")).strip()
        if not text:
            return False
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            import hashlib
            chunk_id = hashlib.sha256(
                f'{chunk.get("source","")}|{chunk.get("page")}|{text}'.encode()
            ).hexdigest()
        with self._connect() as conn:
            cur = conn.execute("""
                INSERT OR IGNORE INTO chunks
                (chunk_id, source, page, chunk_index, text)
                VALUES (?, ?, ?, ?, ?)
            """, (chunk_id, chunk.get("source", "unknown"),
                  chunk.get("page"), chunk.get("chunk_index"), text))
            return cur.rowcount > 0

    def add_chunks(self, chunks: list[dict]) -> int:
        return sum(self.add_chunk(item) for item in chunks)

    def get_all_chunks(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT chunk_id, source, page, chunk_index, text
                FROM chunks ORDER BY source, page, chunk_index
            """).fetchall()
        return [dict(row) for row in rows]

    def count_chunks(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def delete_source(self, source: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            return cur.rowcount

    def clear(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
