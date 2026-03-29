"""Silero VAD - Voice Activity Detection using ONNX Runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from config import SidConfig
from model_manager import DownloadSpec, ModelDownloadManager


class SileroVAD:
    """
    Silero Voice Activity Detection via ONNX Runtime.

    Processes audio in 30ms frames (512 samples @ 16kHz).
    Returns speech probability [0.0, 1.0].
    """

    # Silero VAD requires specific sample rates
    SAMPLE_RATE = 16000
    FRAME_SIZE_MS = 30
    FRAME_SIZE_SAMPLES = int(SAMPLE_RATE * FRAME_SIZE_MS / 1000)  # 512 samples

    # VAD thresholds
    DEFAULT_SPEECH_THRESHOLD = 0.65
    DEFAULT_SILENCE_THRESHOLD = 0.35

    def __init__(
        self,
        config: SidConfig,
        speech_threshold: float | None = None,
        silence_threshold: float | None = None,
    ) -> None:
        self.config = config
        self.speech_threshold = speech_threshold or config.vad_confidence_threshold
        self.silence_threshold = silence_threshold or self.DEFAULT_SILENCE_THRESHOLD

        self._session: Any | None = None
        self._state: np.ndarray | None = None
        self._context: np.ndarray | None = None

    async def ensure_model(self) -> None:
        """Ensure Silero VAD model is loaded."""
        if self._session is not None:
            return

        onnx_path = Path(self.config.vad_model_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        # Download if not present
        if not onnx_path.exists():
            if self.config.allow_model_downloads and self.config.model_download_consent:
                download_mgr = ModelDownloadManager(self.config)
                spec = DownloadSpec(
                    name="silero-vad",
                    url="https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx",
                    destination=onnx_path,
                )
                await download_mgr.ensure_file(spec)
            else:
                raise RuntimeError(
                    f"Silero VAD model not found at {onnx_path}. "
                    "Enable model downloads or download manually."
                )

        # Load ONNX model
        try:
            import onnxruntime as ort  # type: ignore

            self._session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self._init_state()
            logging.info("Silero VAD model loaded: %s", onnx_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Silero VAD: {exc}") from exc

    def _init_state(self) -> None:
        """Initialize VAD state tensors."""
        if self._session is None:
            return
        # Silero VAD state: 2 LSTM layers, hidden size 64
        self._state = np.zeros((2, 1, 64), dtype=np.float32)
        self._context = np.array([0], dtype=np.int64)

    def reset_state(self) -> None:
        """Reset VAD state (call when starting new audio session)."""
        self._init_state()

    def process_frame(self, audio_frame: bytes) -> float:
        """
        Process a single audio frame and return speech probability.

        Args:
            audio_frame: Raw PCM audio bytes (int16, mono, 16kHz)

        Returns:
            Speech probability in [0.0, 1.0]
        """
        if self._session is None:
            raise RuntimeError("Silero VAD not initialized. Call ensure_model() first.")

        # Convert bytes to numpy array (int16 -> float32, normalized)
        audio_np = np.frombuffer(audio_frame, dtype=np.int16).astype(np.float32)
        audio_np = audio_np / 32768.0  # Normalize to [-1, 1]

        if len(audio_np) != self.FRAME_SIZE_SAMPLES:
            # Pad or truncate to expected frame size
            if len(audio_np) < self.FRAME_SIZE_SAMPLES:
                audio_np = np.pad(
                    audio_np, (0, self.FRAME_SIZE_SAMPLES - len(audio_np))
                )
            else:
                audio_np = audio_np[: self.FRAME_SIZE_SAMPLES]

        # Run inference
        input_feed = {
            "input": audio_np.reshape(1, -1),
            "sr": np.array([self.SAMPLE_RATE], dtype=np.int64),
            "state": self._state,
            "context": self._context,
        }

        outputs = self._session.run(None, input_feed)
        probability = float(outputs[0][0, 0])
        self._state = outputs[1]
        self._context = outputs[2]

        return probability

    def is_speaking(self, probability: float) -> bool:
        """Check if probability indicates speech."""
        return probability >= self.speech_threshold

    def is_silent(self, probability: float) -> bool:
        """Check if probability indicates silence."""
        return probability < self.silence_threshold


class SileroVADSession:
    """
    High-level VAD session manager.

    Accumulates audio frames until speech is detected,
    then continues until silence is detected.
    """

    def __init__(
        self,
        vad: SileroVAD,
        silence_frames_threshold: int = 23,  # ~700ms at 30ms/frame
    ) -> None:
        self.vad = vad
        self.silence_frames_threshold = silence_frames_threshold

        self._frames: list[bytes] = []
        self._speech_started = False
        self._silence_count = 0
        self._total_speech_frames = 0

    def reset(self) -> None:
        """Reset session state."""
        self._frames = []
        self._speech_started = False
        self._silence_count = 0
        self._total_speech_frames = 0
        self.vad.reset_state()

    def process_frame(self, audio_frame: bytes) -> tuple[bool, bool]:
        """
        Process a frame and return (is_speaking, should_end).

        Args:
            audio_frame: Raw PCM audio bytes

        Returns:
            Tuple of (currently_speaking, should_end_session)
        """
        probability = self.vad.process_frame(audio_frame)
        is_speaking = self.vad.is_speaking(probability)

        if is_speaking:
            if not self._speech_started:
                self._speech_started = True
                self._silence_count = 0
            self._total_speech_frames += 1
            self._silence_count = 0
            self._frames.append(audio_frame)
            return True, False

        # Not speaking
        if self._speech_started:
            self._silence_count += 1
            self._frames.append(audio_frame)  # Include trailing silence

            if self._silence_count >= self.silence_frames_threshold:
                return False, True  # End of speech

        return False, False

    def get_audio_frames(self) -> list[bytes]:
        """Get all accumulated audio frames."""
        return self._frames

    def get_total_duration_ms(self) -> int:
        """Get total duration of recorded audio in milliseconds."""
        return len(self._frames) * SileroVAD.FRAME_SIZE_MS


__all__ = ["SileroVAD", "SileroVADSession"]
