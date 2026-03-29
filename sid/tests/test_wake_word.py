from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from audio.wake_word import WakeWordDetector
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


def test_load_model_uses_defaults_when_configured_model_is_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    config = make_config(tmp_path)
    model_path = tmp_path / "wakeword.onnx"
    model_path.write_bytes(b"")
    config.wake_word_model = str(model_path)

    loop = asyncio.new_event_loop()
    detector = WakeWordDetector(config, loop, lambda _source: None)
    calls: list[dict[str, object]] = []

    def fake_model(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return "default-model"

    try:
        with caplog.at_level(logging.WARNING):
            model = detector._load_model(fake_model)
    finally:
        loop.close()

    assert model == "default-model"
    assert calls == [{"args": (), "kwargs": {}}]
    assert "is empty" in caplog.text


def test_load_model_falls_back_when_custom_model_load_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    config = make_config(tmp_path)
    model_path = tmp_path / "wakeword.onnx"
    model_path.write_bytes(b"not-a-real-model")
    config.wake_word_model = str(model_path)

    loop = asyncio.new_event_loop()
    detector = WakeWordDetector(config, loop, lambda _source: None)
    calls: list[dict[str, object]] = []

    def fake_model(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        if kwargs:
            raise RuntimeError("bad model")
        return "default-model"

    try:
        with caplog.at_level(logging.WARNING):
            model = detector._load_model(fake_model)
    finally:
        loop.close()

    assert model == "default-model"
    assert calls == [
        {
            "args": (),
            "kwargs": {
                "wakeword_models": [str(model_path)],
                "inference_framework": "onnx",
            },
        },
        {"args": (), "kwargs": {}},
    ]
    assert "failed to load" in caplog.text
