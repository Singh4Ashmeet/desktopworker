from __future__ import annotations

import asyncio
import concurrent.futures
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config import SidConfig
from model_manager import DownloadSpec, ModelDownloadManager
from audio.silero_vad import SileroVAD, SileroVADSession


@dataclass(slots=True)
class STTResult:
    text: str
    language: str
    confidence: float
    duration_ms: int


@dataclass(slots=True)
class VADSessionStats:
    positives: int = 0
    false_positives: int = 0

    @property
    def false_positive_rate(self) -> float:
        if self.positives == 0:
            return 0.0
        return self.false_positives / self.positives


class STTEngine:
    """Fully local faster-whisper transcription with Silero VAD gating."""

    WHISPER_REPO_MAP = {
        "tiny.en": "tiny.en",
        "base.en": "base.en",
        "small.en": "small.en",
    }

    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self._model = None
        self._model_lock = asyncio.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="sid-stt"
        )
        self._download_manager = ModelDownloadManager(config)
        self.stats = VADSessionStats()
        self._silero_vad: SileroVAD | None = None

    def passes_vad_gate(self, probability: float) -> bool:
        """Check if speech probability exceeds configured VAD threshold."""
        return probability >= self.config.vad_confidence_threshold

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._model_lock:
            if self._model is not None:
                return self._model
            await self._ensure_model_assets()
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                self._executor, self._load_model_sync
            )
        return self._model

    async def _ensure_model_assets(self) -> None:
        model_dir = Path(self.config.whisper_model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        vad_path = Path(self.config.vad_model_path)
        if (
            not vad_path.exists()
            and self.config.allow_model_downloads
            and self.config.model_download_consent
        ):
            spec = DownloadSpec(
                name="silero-vad",
                url="https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx",
                destination=vad_path,
            )
            await self._download_manager.ensure_file(spec)

    async def _ensure_vad(self) -> SileroVAD:
        """Ensure Silero VAD is initialized."""
        if self._silero_vad is None:
            self._silero_vad = SileroVAD(
                self.config,
                speech_threshold=self.config.vad_confidence_threshold,
            )
            await self._silero_vad.ensure_model()
        return self._silero_vad

    def _load_model_sync(self) -> Any:
        from faster_whisper import WhisperModel  # type: ignore

        cache_dir = str(Path(self.config.whisper_model_dir))
        model_name = self.WHISPER_REPO_MAP.get(
            self.config.whisper_model, self.config.whisper_model
        )
        expected = (
            Path(cache_dir)
            / f"models--Systran--faster-whisper-{model_name.replace('.', '-')}"
        )
        if not self.config.model_download_consent and not expected.exists():
            raise RuntimeError(
                f"Whisper model {model_name} is not cached locally and downloads are disabled"
            )
        return WhisperModel(
            model_name, device="auto", compute_type="int8", download_root=cache_dir
        )

    async def transcribe_file(
        self, audio_path: str, on_partial: Callable[[str], Any] | None = None
    ) -> STTResult:
        model = await self._ensure_model()
        started = time.perf_counter()

        def _do_transcribe() -> STTResult:
            segments, info = model.transcribe(audio_path, beam_size=4, vad_filter=True)
            texts: list[str] = []
            for seg in segments:
                part = (seg.text or "").strip()
                if not part:
                    continue
                texts.append(part)
                if on_partial:
                    on_partial(" ".join(texts))
            duration_ms = int((time.perf_counter() - started) * 1000)
            return STTResult(
                text=" ".join(texts).strip(),
                language=str(getattr(info, "language", "en") or "en"),
                confidence=float(getattr(info, "language_probability", 0.0) or 0.0),
                duration_ms=duration_ms,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _do_transcribe)

    async def transcribe_stream(
        self,
        audio_chunks: list[bytes],
        sample_rate: int = 16000,
        on_partial: Callable[[str], Any] | None = None,
    ) -> STTResult:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            await asyncio.to_thread(
                self._write_wav_sync, temp_path, audio_chunks, sample_rate
            )
            return await self.transcribe_file(str(temp_path), on_partial=on_partial)
        finally:
            temp_path.unlink(missing_ok=True)

    async def listen_once(
        self,
        timeout_seconds: int | None = None,
        on_partial: Callable[[str], Any] | None = None,
    ) -> STTResult:
        chunks = await self._record_until_silence(
            timeout_seconds or self.config.stt_max_record_seconds
        )
        if not chunks:
            return STTResult(text="", language="en", confidence=0.0, duration_ms=0)
        result = await self.transcribe_stream(chunks, on_partial=on_partial)
        if result.text.strip():
            return result
        self.stats.false_positives += 1
        return result

    async def _record_until_silence(self, timeout_seconds: int) -> list[bytes]:
        """Record audio using Silero VAD for speech detection."""
        try:
            import pyaudio  # type: ignore
        except Exception as exc:
            raise RuntimeError("pyaudio is required for microphone capture") from exc

        # Ensure Silero VAD is ready
        vad = await self._ensure_vad()

        # Silero VAD frame size: 512 samples @ 16kHz = 30ms
        chunk_size = 512
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=chunk_size,
        )

        # Create VAD session
        vad_session = SileroVADSession(
            vad,
            silence_frames_threshold=max(1, self.config.stt_silence_ms // 30),
        )
        vad_session.reset()

        frames: list[bytes] = []
        started = time.monotonic()
        should_end = False

        try:
            while time.monotonic() - started < timeout_seconds and not should_end:
                frame = await asyncio.to_thread(
                    stream.read, chunk_size, exception_on_overflow=False
                )
                is_speaking, should_end = vad_session.process_frame(frame)
                if is_speaking:
                    self.stats.positives += 1
                frames.append(frame)
            return frames
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def _write_wav_sync(
        self, path: Path, chunks: list[bytes], sample_rate: int
    ) -> None:
        import wave

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for chunk in chunks:
                wav_file.writeframes(chunk)


__all__ = ["STTEngine", "STTResult", "VADSessionStats"]
