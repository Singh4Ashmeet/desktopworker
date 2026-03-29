from __future__ import annotations

from pathlib import Path

import pytest

from config import SidConfig
from memory.undo_buffer import UndoBuffer
from security.action_log import ActionLog
from security.permissions import PermissionChecker, PermissionTier
from tools.registry import ToolRegistry


@pytest.fixture
def tool_env(tmp_path: Path):
    sid_dir = tmp_path / ".sid"
    config = SidConfig(
        sid_dir=str(sid_dir),
        allowlist_paths=[str(tmp_path)],
        use_local_llm=True,
        use_offline_tts=True,
    )
    config.validate_startup()
    action_log = ActionLog(config)
    undo = UndoBuffer(config)
    registry = ToolRegistry(config, undo, action_log)
    checker = PermissionChecker(config)
    return config, registry, checker, undo


@pytest.mark.asyncio
async def test_search_files_finds_existing_file(tool_env, tmp_path: Path):
    _, registry, checker, _ = tool_env
    target = tmp_path / "alpha_test_file.txt"
    target.write_text("hello", encoding="utf-8")

    result = await registry.execute(
        "search_files",
        {"query": "alpha_test_file", "directory": str(tmp_path)},
        checker,
    )

    assert result["success"] is True
    paths = result["data"]["paths"]
    assert str(target.resolve()) in paths


@pytest.mark.asyncio
async def test_run_command_captures_stdout(tool_env):
    _, registry, checker, _ = tool_env
    result = await registry.execute(
        "run_command", {"command": "python -c \"print('sid-ok')\""}, checker
    )
    assert result["success"] is True
    assert "sid-ok" in result["data"]["stdout"]


@pytest.mark.asyncio
async def test_run_command_blocks_danger_commands(tool_env):
    _, _, checker, _ = tool_env
    blocked, reason = checker.is_blocked("run_command", {"command": "rm -rf /"})
    assert blocked is True
    assert "Blocked" in reason


@pytest.mark.asyncio
async def test_move_file_saves_to_undo_buffer(tool_env, tmp_path: Path):
    _, registry, checker, undo = tool_env
    src = tmp_path / "a.txt"
    dst = tmp_path / "b.txt"
    src.write_text("x", encoding="utf-8")

    result = await registry.execute(
        "move_file", {"source": str(src), "destination": str(dst)}, checker
    )
    assert result["success"] is True
    assert dst.exists()

    row = undo._conn.execute("SELECT COUNT(*) AS c FROM undo_ops").fetchone()
    assert row[0] >= 1


@pytest.mark.asyncio
async def test_delete_file_requires_caution_tier(tool_env, tmp_path: Path):
    config, _, checker, _ = tool_env
    p = tmp_path / "danger.txt"
    p.write_text("x", encoding="utf-8")
    tier = checker.check("delete_file", {"path": str(p)})
    assert tier in {PermissionTier.CAUTION, PermissionTier.DANGER}
    assert tier != PermissionTier.SAFE
