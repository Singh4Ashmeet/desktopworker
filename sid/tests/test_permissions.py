from __future__ import annotations

from pathlib import Path

from config import SidConfig
from security.permissions import PermissionChecker, PermissionTier


def make_checker(tmp_path: Path) -> PermissionChecker:
    config = SidConfig(
        sid_dir=str(tmp_path / ".sid"),
        allowlist_paths=[str(tmp_path)],
        use_local_llm=True,
        use_offline_tts=True,
    )
    config.validate_startup()
    return PermissionChecker(config)


def test_read_file_is_safe_tier(tmp_path: Path):
    checker = make_checker(tmp_path)
    tier = checker.check("read_file", {"path": str(tmp_path / "a.txt")})
    assert tier == PermissionTier.SAFE


def test_delete_is_danger_tier(tmp_path: Path):
    checker = make_checker(tmp_path)
    tier = checker.check("delete_file", {"path": str(tmp_path / "a.txt")})
    assert tier == PermissionTier.DANGER


def test_rm_rf_is_blocked(tmp_path: Path):
    checker = make_checker(tmp_path)
    blocked, _ = checker.is_blocked("run_command", {"command": "rm -rf /"})
    assert blocked is True


def test_sandbox_blocks_system_paths(tmp_path: Path):
    checker = make_checker(tmp_path)
    blocked, reason = checker.is_blocked(
        "write_file", {"path": "C:\\Windows\\system.ini"}
    )
    assert blocked is True
    assert "Blocked" in reason
