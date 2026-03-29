from __future__ import annotations

from pathlib import Path

import pytest

from config import SidConfig
from memory.fact_store import FactStore
from memory.undo_buffer import UndoBuffer
from memory.vector_store import VectorStore


def make_config(tmp_path: Path) -> SidConfig:
    cfg = SidConfig(
        sid_dir=str(tmp_path / ".sid"),
        allowlist_paths=[str(tmp_path)],
        use_local_llm=True,
        use_offline_tts=True,
    )
    cfg.validate_startup()
    return cfg


@pytest.mark.asyncio
async def test_fact_store_persist_and_recall(tmp_path: Path):
    cfg = make_config(tmp_path)
    store = FactStore(cfg)
    await store.set_fact("user_name", "Alex", "personal")
    value = await store.get_fact("user_name")
    await store.close()
    assert value == "Alex"


@pytest.mark.asyncio
async def test_vector_store_query_returns_relevant(tmp_path: Path):
    cfg = make_config(tmp_path)
    vs = VectorStore(cfg)
    await vs.add_interaction("open chrome", "opened browser", ["open_app"])
    await vs.add_interaction("organize downloads", "sorted files", ["search_files"])
    hits = await vs.query_relevant("downloads", n=3)
    await vs.flush()
    assert any("downloads" in hit.lower() for hit in hits)


@pytest.mark.asyncio
async def test_undo_buffer_restores_deleted_file(tmp_path: Path):
    cfg = make_config(tmp_path)
    undo = UndoBuffer(cfg)
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")

    await undo.record_operation("delete", str(file_path), str(file_path))
    file_path.unlink()
    ok = await undo.undo_last()
    await undo.close()

    assert ok is True
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == "hello"
