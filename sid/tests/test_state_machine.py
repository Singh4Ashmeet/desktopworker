from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from config import SidConfig
from state import InvalidStateTransition, SidState, StateManager


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
async def test_state_machine_valid_transitions_persist(tmp_path: Path):
    config = make_config(tmp_path)
    manager = StateManager(config)

    await manager.transition(SidState.SLEEPING, "boot")
    await manager.transition(SidState.WAKING, "voice")
    await manager.transition(SidState.LISTENING, "ready")

    payload = json.loads(config.state_file.read_text(encoding="utf-8"))
    assert payload["state"] == SidState.LISTENING.value


@pytest.mark.asyncio
async def test_state_machine_invalid_transition_is_rejected(tmp_path: Path):
    config = make_config(tmp_path)
    manager = StateManager(config)
    await manager.transition(SidState.SLEEPING, "boot")
    with pytest.raises(InvalidStateTransition):
        await manager.transition(SidState.ACTING, "skip_listening")


@pytest.mark.asyncio
async def test_state_machine_recovers_inflight_state_to_sleeping(tmp_path: Path):
    config = make_config(tmp_path)
    config.state_file.write_text(
        json.dumps({"state": "ACTING", "reason": "crash"}), encoding="utf-8"
    )
    manager = StateManager(config)
    restored = await manager.load_persisted_state()
    assert restored == SidState.SLEEPING


@pytest.mark.asyncio
async def test_simultaneous_wake_requests_are_serialised(tmp_path: Path):
    config = make_config(tmp_path)
    manager = StateManager(config)
    await manager.transition(SidState.SLEEPING, "boot")

    first, second = await asyncio.gather(
        manager.request_wake("voice"),
        manager.request_wake("hotkey"),
    )
    assert sorted([first, second]) == [False, True]
