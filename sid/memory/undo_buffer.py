from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import SidConfig
from db_utils import configure_connection, optimize_connection


@dataclass(slots=True)
class UndoOperation:
    id: int
    timestamp: str
    op_type: str
    src_path: str
    dst_path: str
    backup_blob: bytes | None
    reversed: int


class UndoBuffer:
    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self.db_path = config.memory_dir / "undo.db"
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self.tuning = configure_connection(self._conn)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS undo_ops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                op_type TEXT NOT NULL,
                src_path TEXT NOT NULL,
                dst_path TEXT NOT NULL,
                backup_blob BLOB,
                reversed INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._migrate_legacy_schema()
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_undo_timestamp ON undo_ops(timestamp DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_undo_reversed ON undo_ops(reversed)"
        )
        self._conn.commit()

    def _migrate_legacy_schema(self) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(undo_ops)").fetchall()
        }
        if "timestamp" not in columns:
            self._conn.execute(
                "ALTER TABLE undo_ops ADD COLUMN timestamp TEXT NOT NULL DEFAULT ''"
            )
            self._conn.execute(
                "UPDATE undo_ops SET timestamp = datetime('now') WHERE timestamp = ''"
            )
        if "op_type" not in columns and "operation" in columns:
            self._conn.execute(
                "ALTER TABLE undo_ops ADD COLUMN op_type TEXT NOT NULL DEFAULT 'delete'"
            )
            self._conn.execute(
                "UPDATE undo_ops SET op_type = operation WHERE op_type = 'delete'"
            )
        if "backup_blob" not in columns and "backup_path" in columns:
            self._conn.execute("ALTER TABLE undo_ops ADD COLUMN backup_blob BLOB")
        if "reversed" not in columns:
            self._conn.execute(
                "ALTER TABLE undo_ops ADD COLUMN reversed INTEGER NOT NULL DEFAULT 0"
            )

    async def record_operation(
        self, op_type: str, src_path: str, dst_path: str = ""
    ) -> int:
        async with self._lock:
            return await asyncio.to_thread(
                self._record_operation_sync, op_type, src_path, dst_path
            )

    async def record_backup(
        self, op_type: str, src_path: str, dst_path: str = ""
    ) -> int:
        return await self.record_operation(op_type, src_path, dst_path)

    def _record_operation_sync(
        self, op_type: str, src_path: str, dst_path: str = ""
    ) -> int:
        src = Path(src_path).expanduser().resolve()
        backup_blob = self._serialize_path(src) if src.exists() else None
        timestamp = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO undo_ops (timestamp, op_type, src_path, dst_path, backup_blob, reversed)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (
                timestamp,
                op_type,
                str(src),
                str(Path(dst_path).expanduser().resolve()) if dst_path else "",
                backup_blob,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    async def undo_last(self) -> bool:
        results = await self.undo_last_n(1)
        return bool(results and results[0])

    async def undo_last_n(self, n: int) -> list[bool]:
        async with self._lock:
            return await asyncio.to_thread(self._undo_last_n_sync, n)

    def _undo_last_n_sync(self, n: int) -> list[bool]:
        rows = self._conn.execute(
            """
            SELECT id, timestamp, op_type, src_path, dst_path, backup_blob, reversed
            FROM undo_ops
            WHERE reversed = 0
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(1, n),),
        ).fetchall()

        results: list[bool] = []
        for row in rows:
            op = UndoOperation(**dict(row))
            ok = self._restore_operation(op)
            if ok:
                self._conn.execute(
                    "UPDATE undo_ops SET reversed = 1 WHERE id = ?", (op.id,)
                )
            results.append(ok)
        self._conn.commit()
        return results

    def _restore_operation(self, op: UndoOperation) -> bool:
        if not op.backup_blob:
            return False
        target = Path(op.src_path)
        moved_target = Path(op.dst_path) if op.dst_path else target

        if op.op_type == "move" and moved_target.exists():
            if moved_target.is_dir():
                shutil.rmtree(moved_target, ignore_errors=True)
            else:
                moved_target.unlink(missing_ok=True)

        return self._restore_blob(op.backup_blob, target)

    def _serialize_path(self, path: Path) -> bytes:
        payload: dict[str, Any]
        if path.is_file():
            payload = {
                "kind": "file",
                "name": path.name,
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        elif path.is_dir():
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        zf.write(file_path, arcname=str(file_path.relative_to(path)))
            payload = {
                "kind": "dir",
                "name": path.name,
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
        else:
            payload = {"kind": "missing", "name": path.name, "data": ""}
        return json.dumps(payload).encode("utf-8")

    def _restore_blob(self, blob: bytes, target: Path) -> bool:
        payload = json.loads(blob.decode("utf-8"))
        kind = payload.get("kind")
        data = base64.b64decode(payload.get("data", "") or b"")

        if kind == "file":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return True

        if kind == "dir":
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                zf.extractall(target)
            return True

        return False

    async def list_pending_crash_recovery(self) -> list[UndoOperation]:
        async with self._lock:
            fetched = await asyncio.to_thread(self._list_pending_sync)
        return [UndoOperation(**dict(row)) for row in fetched]

    def _list_pending_sync(self):
        return self._conn.execute(
            """
            SELECT id, timestamp, op_type, src_path, dst_path, backup_blob, reversed
            FROM undo_ops
            WHERE reversed = 0
            ORDER BY id DESC
            """
        ).fetchall()

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(optimize_connection, self._conn)
            await asyncio.to_thread(self._conn.close)


__all__ = ["UndoBuffer", "UndoOperation"]
