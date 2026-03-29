from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import SidConfig


@dataclass(slots=True)
class WakeWordStats:
    detections: int = 0
    failures: int = 0


class WakeWordDetector:
    def __init__(
        self,
        config: SidConfig,
        loop: asyncio.AbstractEventLoop,
        trigger_wake: Callable[[str], asyncio.Future | asyncio.Task | None],
    ) -> None:
        self.config = config
        self.loop = loop
        self.trigger_wake = trigger_wake
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stats = WakeWordStats()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_guarded, name="sid-wakeword", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run_guarded(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_detector()
            except Exception:
                self.stats.failures += 1
                logging.exception("Wake word detector crashed; restarting in 2s")
                time.sleep(2)

    def _run_detector(self) -> None:
        try:
            from openwakeword.model import Model  # type: ignore
            import numpy as np  # type: ignore
            import pyaudio  # type: ignore
        except Exception:
            logging.warning("openwakeword dependencies unavailable; wake detector idle")
            while not self._stop.is_set():
                time.sleep(0.5)
            return

        model = self._load_model(Model)
        chunk_size = 1280
        cooldown = self.config.wake_word_cooldown_seconds
        threshold = self.config.wake_word_sensitivity

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=chunk_size,
        )

        last_fire = 0.0
        try:
            while not self._stop.is_set():
                audio_bytes = stream.read(chunk_size, exception_on_overflow=False)
                pcm = np.frombuffer(audio_bytes, dtype=np.int16)
                scores = model.predict(pcm)
                score = max(scores.values()) if scores else 0.0
                now = time.monotonic()
                if score >= threshold and (now - last_fire) >= cooldown:
                    self.stats.detections += 1
                    last_fire = now
                    self.loop.call_soon_threadsafe(self.trigger_wake, "voice")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def _load_model(self, model_cls):
        model_path = Path(self.config.wake_word_model)
        if model_path.exists():
            try:
                if model_path.stat().st_size <= 0:
                    logging.warning(
                        "Wake word model at %s is empty; using openWakeWord defaults",
                        model_path,
                    )
                else:
                    return model_cls(
                        wakeword_models=[str(model_path)], inference_framework="onnx"
                    )
            except OSError as exc:
                logging.warning(
                    "Wake word model at %s could not be inspected (%s); using defaults",
                    model_path,
                    exc,
                )
            except Exception as exc:
                logging.warning(
                    "Wake word model at %s failed to load (%s); using defaults",
                    model_path,
                    exc,
                )
        else:
            logging.warning(
                "Wake word model not found at %s; using openWakeWord defaults",
                model_path,
            )
        return model_cls()


__all__ = ["WakeWordDetector", "WakeWordStats"]
