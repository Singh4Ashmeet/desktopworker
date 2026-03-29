from __future__ import annotations

import asyncio
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from config import SidConfig
from db_utils import configure_connection, optimize_connection
from memory.embeddings import LocalEmbeddingModel


@dataclass(slots=True)
class FactRecord:
    id: int
    key: str
    value: str
    category: str
    created_at: str
    updated_at: str
    active: bool
    superseded_by: int | None


class FactStore:
    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self.path = config.memory_dir / "facts.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.tuning = configure_connection(self._conn)
        self._embedder = LocalEmbeddingModel(config)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                superseded_by INTEGER NULL
            )
            """
        )
        self._migrate_legacy_schema()
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_key_active ON facts(key, active)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_category_active ON facts(category, active)"
        )
        self._conn.commit()

    def _migrate_legacy_schema(self) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        if "active" not in columns:
            self._conn.execute(
                "ALTER TABLE facts ADD COLUMN active INTEGER NOT NULL DEFAULT 1"
            )
        if "superseded_by" not in columns:
            self._conn.execute(
                "ALTER TABLE facts ADD COLUMN superseded_by INTEGER NULL"
            )

    async def set_fact(self, key: str, value: str, category: str = "general") -> None:
        async with self._lock:
            await self._set_fact_async(key, value, category)

    async def _set_fact_async(self, key: str, value: str, category: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cur = await asyncio.to_thread(
            self._conn.execute,
            """
            INSERT INTO facts (key, value, category, created_at, updated_at, active, superseded_by)
            VALUES (?, ?, ?, ?, ?, 1, NULL)
            """,
            (key, value, category, now, now),
        )
        new_id = int(cur.lastrowid)
        existing = await asyncio.to_thread(
            self._conn.execute,
            """
            SELECT id, key, value, category, created_at, updated_at, active, superseded_by
            FROM facts
            WHERE category = ? AND active = 1 AND id != ?
            ORDER BY updated_at DESC
            LIMIT 3
            """,
            (category, new_id),
        )
        rows = existing.fetchall()
        new_text = f"{key} {value}"
        for row in rows:
            similarity = await self._semantic_similarity(
                new_text, f"{row['key']} {row['value']}"
            )
            if similarity > 0.85:
                await asyncio.to_thread(
                    self._conn.execute,
                    "UPDATE facts SET active = 0, superseded_by = ?, updated_at = ? WHERE id = ?",
                    (new_id, now, row["id"]),
                )

        await asyncio.to_thread(self._conn.commit)

    async def get_fact(self, key: str) -> str | None:
        async with self._lock:
            return await asyncio.to_thread(self._get_fact_sync, key)

    def _get_fact_sync(self, key: str) -> str | None:
        row = self._conn.execute(
            """
            SELECT value
            FROM facts
            WHERE key = ? AND active = 1
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (key,),
        ).fetchone()
        return str(row["value"]) if row else None

    async def list_facts(
        self, category: str | None = None, active_only: bool = True
    ) -> list[FactRecord]:
        async with self._lock:
            return await asyncio.to_thread(self._list_facts_sync, category, active_only)

    def _list_facts_sync(
        self, category: str | None, active_only: bool
    ) -> list[FactRecord]:
        where = []
        params: list[object] = []
        if category:
            where.append("category = ?")
            params.append(category)
        if active_only:
            where.append("active = 1")
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = self._conn.execute(
            f"""
            SELECT id, key, value, category, created_at, updated_at, active, superseded_by
            FROM facts
            {clause}
            ORDER BY updated_at DESC
            """,
            params,
        ).fetchall()
        return [FactRecord(**dict(row)) for row in rows]

    async def list_superseded(self) -> list[dict[str, object]]:
        async with self._lock:
            return await asyncio.to_thread(self._list_superseded_sync)

    def _list_superseded_sync(self) -> list[dict[str, object]]:
        rows = self._conn.execute(
            """
            SELECT old.id AS old_id, old.key AS old_key, old.value AS old_value,
                   new.id AS new_id, new.key AS new_key, new.value AS new_value
            FROM facts AS old
            LEFT JOIN facts AS new ON new.id = old.superseded_by
            WHERE old.active = 0 AND old.superseded_by IS NOT NULL
            ORDER BY old.updated_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    async def delete_fact(self, key: str) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._delete_fact_sync, key)

    def _delete_fact_sync(self, key: str) -> bool:
        cur = self._conn.execute(
            "UPDATE facts SET active = 0, updated_at = ? WHERE key = ? AND active = 1",
            (datetime.now(timezone.utc).isoformat(), key),
        )
        self._conn.commit()
        return cur.rowcount > 0

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(optimize_connection, self._conn)
            await asyncio.to_thread(self._conn.close)

    async def _semantic_similarity(self, left: str, right: str) -> float:
        try:
            return await self._embedder.cosine_similarity(left, right)
        except Exception:
            return _cosine_similarity(left, right)


def _cosine_similarity(a: str, b: str) -> float:
    va = _token_vector(a)
    vb = _token_vector(b)
    if not va or not vb:
        return 0.0
    common = set(va) & set(vb)
    dot = sum(va[t] * vb[t] for t in common)
    mag_a = math.sqrt(sum(v * v for v in va.values()))
    mag_b = math.sqrt(sum(v * v for v in vb.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _token_vector(text: str) -> dict[str, float]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    out: dict[str, float] = {}
    for token in tokens:
        out[token] = out.get(token, 0.0) + 1.0
    return out


__all__ = ["FactStore", "FactRecord"]
