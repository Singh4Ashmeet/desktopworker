from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import pytest

from audio.tts import TTSEngine
from config import SidConfig


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
async def test_verify_offline_tts_uses_espeak_alias_on_windows_style_setups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = make_config(tmp_path)
    engine = TTSEngine(config)

    def fake_which(binary: str) -> str | None:
        if binary == "espeak":
            return "C:/Program Files/eSpeak/command_line/espeak.exe"
        return None

    monkeypatch.setattr("audio.tts.shutil.which", fake_which)

    try:
        health = await engine.verify_offline_tts()
    finally:
        engine.close()

    assert health.available is True
    assert health.backend == "espeak"
    assert "eSpeak fallback ready" in health.message


@pytest.mark.asyncio
async def test_verify_offline_tts_finds_standard_espeak_install_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = make_config(tmp_path)
    engine = TTSEngine(config)
    install_dir = tmp_path / "Program Files" / "eSpeak NG"
    install_dir.mkdir(parents=True, exist_ok=True)
    binary = install_dir / "espeak-ng.exe"
    binary.write_text("", encoding="utf-8")

    monkeypatch.setenv("ProgramFiles", str(tmp_path / "Program Files"))
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)

    def fake_which(_binary: str) -> str | None:
        return None

    monkeypatch.setattr("audio.tts.shutil.which", fake_which)

    try:
        health = await engine.verify_offline_tts()
    finally:
        engine.close()

    assert health.available is True
    assert health.backend == "espeak"
    assert "eSpeak fallback ready" in health.message


@pytest.mark.asyncio
async def test_speak_disables_tts_cleanly_after_all_backends_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    engine = TTSEngine(make_config(tmp_path))
    fallback_attempted = threading.Event()
    calls = {"piper": 0, "espeak": 0}

    def fake_piper(_text: str) -> None:
        calls["piper"] += 1
        raise RuntimeError("Piper voice assets are not ready")

    def fake_espeak(_text: str) -> None:
        calls["espeak"] += 1
        fallback_attempted.set()
        raise RuntimeError("winget install eSpeak.eSpeak")

    monkeypatch.setattr(engine, "_speak_piper", fake_piper)
    monkeypatch.setattr(engine, "_speak_espeak", fake_espeak)

    with caplog.at_level(logging.WARNING):
        await engine.speak("hello")
        assert await asyncio.to_thread(fallback_attempted.wait, 1.0)
        await asyncio.sleep(0.1)
        await engine.speak("again")
        await asyncio.sleep(0.1)

    engine.close()

    assert "All local TTS backends failed" not in caplog.text
    assert "Speech output disabled" in caplog.text
    assert calls == {"piper": 1, "espeak": 1}


def test_play_wave_file_falls_back_cleanly_for_empty_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    engine = TTSEngine(make_config(tmp_path))
    wav_path = tmp_path / "wake.wav"
    wav_path.write_bytes(b"")
    fallback_calls = {"count": 0}

    def fake_chime() -> None:
        fallback_calls["count"] += 1

    def unexpected_playback(_wav_bytes: bytes) -> None:
        raise AssertionError("Empty wave assets should not be parsed for playback")

    monkeypatch.setattr(engine, "_play_system_chime", fake_chime)
    monkeypatch.setattr(engine, "_play_wav_bytes", unexpected_playback)

    try:
        engine._play_wave_file(wav_path)
    finally:
        engine.close()

    assert fallback_calls["count"] == 1
