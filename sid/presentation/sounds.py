from __future__ import annotations

import asyncio
import logging
from pathlib import Path


class SoundPlayer:
    def __init__(self, assets_dir: Path) -> None:
        self.assets_dir = assets_dir

    async def play(self, filename: str) -> None:
        path = self.assets_dir / filename
        await asyncio.to_thread(self._play_sync, path)

    def _play_sync(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            import simpleaudio as sa  # type: ignore
        except Exception:
            return
        try:
            wave = sa.WaveObject.from_wave_file(str(path))
            wave.play()
        except Exception:
            logging.exception("Failed to play sound file: %s", path)


__all__ = ["SoundPlayer"]
