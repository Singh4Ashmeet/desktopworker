from __future__ import annotations

import sqlite3
import time


def configure_connection(conn: sqlite3.Connection) -> dict[str, float]:
    """Enable WAL tuning and return simple before/after read timings."""
    before = _benchmark_read(conn)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA synchronous=NORMAL")
    after = _benchmark_read(conn)
    return {"before_ms": before, "after_ms": after, "improvement_ms": before - after}


def optimize_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA optimize")


def _benchmark_read(conn: sqlite3.Connection) -> float:
    start = time.perf_counter()
    conn.execute("SELECT 1").fetchone()
    return (time.perf_counter() - start) * 1000.0
