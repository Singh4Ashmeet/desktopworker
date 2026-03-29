from __future__ import annotations

from pathlib import Path

from audio.stt import STTEngine
from config import SidConfig


def make_config(tmp_path: Path) -> SidConfig:
    cfg = SidConfig(
        sid_dir=str(tmp_path / ".sid"),
        allowlist_paths=[str(tmp_path)],
        use_local_llm=True,
        use_offline_tts=True,
        vad_confidence_threshold=0.7,
    )
    cfg.validate_startup()
    return cfg


def test_vad_gate_logic_respects_threshold(tmp_path: Path):
    stt = STTEngine(make_config(tmp_path))
    assert stt.passes_vad_gate(0.71) is True
    assert stt.passes_vad_gate(0.69) is False
