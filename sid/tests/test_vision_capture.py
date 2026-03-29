from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from config import SidConfig
from vision.capture import ScreenCaptureDependencyError, ScreenCaptureLoop


def make_config(tmp_path: Path) -> SidConfig:
    cfg = SidConfig(
        sid_dir=str(tmp_path / ".sid"),
        allowlist_paths=[str(tmp_path)],
        use_local_llm=True,
        use_offline_tts=True,
    )
    cfg.validate_startup()
    return cfg


class FakeWorldModel:
    def snapshot(self) -> dict[str, int]:
        return {"idle_minutes": 0}

    async def update_screen_stability(self, _stable_ticks: int) -> None:
        return None

    async def update_from_screen(self, _analysis) -> None:
        return None


@pytest.mark.asyncio
async def test_screen_capture_loop_disables_cleanly_when_mss_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    loop = ScreenCaptureLoop(make_config(tmp_path), FakeWorldModel(), object())

    async def no_sleep(_delay: float) -> None:
        return None

    def missing_capture() -> tuple[str, str]:
        raise ScreenCaptureDependencyError("mss is required for screen capture")

    monkeypatch.setattr("vision.capture.asyncio.sleep", no_sleep)
    monkeypatch.setattr(loop, "_capture", missing_capture)

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(loop.run(), timeout=1.0)

    assert "Screen capture disabled" in caplog.text
    assert "Screen capture loop error" not in caplog.text
