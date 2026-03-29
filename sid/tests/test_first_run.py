from __future__ import annotations

import logging
from pathlib import Path

import pytest

from config import SidConfig
from main import first_run_setup


def make_config(tmp_path: Path) -> SidConfig:
    cfg = SidConfig(
        sid_dir=str(tmp_path / ".sid"),
        allowlist_paths=[str(tmp_path)],
        use_local_llm=True,
        use_offline_tts=True,
    )
    cfg.validate_startup()
    return cfg


class FakeFactStore:
    def __init__(self) -> None:
        self.facts: list[tuple[str, str, str]] = []

    async def set_fact(self, key: str, value: str, namespace: str) -> None:
        self.facts.append((key, value, namespace))


class FakeTTS:
    def __init__(self, fail_wake_chime: bool = False) -> None:
        self.fail_wake_chime = fail_wake_chime
        self.spoken: list[str] = []
        self.played = 0

    async def speak(self, text: str, force_offline: bool = False) -> None:
        del force_offline
        self.spoken.append(text)

    async def play_wake_chime(self) -> None:
        self.played += 1
        if self.fail_wake_chime:
            raise EOFError()


class FakeLLMClient:
    async def ensure_runtime(self) -> str:
        return "llama3.2:3b"


class FakeTTSHealth:
    available = True


@pytest.mark.asyncio
async def test_first_run_setup_survives_wake_chime_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    config = make_config(tmp_path)
    answers = iter(["Ashmeet Singh", "n", "n"])

    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    fact_store = FakeFactStore()
    tts = FakeTTS(fail_wake_chime=True)

    with caplog.at_level(logging.ERROR):
        await first_run_setup(
            config,
            fact_store,
            tts,
            FakeLLMClient(),
            tts_health=FakeTTSHealth(),
        )

    assert config.initialized_flag.exists()
    assert ("user_name", "Ashmeet Singh", "personal") in fact_store.facts
    assert tts.played == 1
    assert tts.spoken == [
        "What should I call you?",
        "I'm Sid. Local, offline, and ready.",
    ]
    assert "Failed to play wake chime" in caplog.text
