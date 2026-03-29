from __future__ import annotations

import asyncio
import io
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from config import SidConfig


@dataclass(slots=True)
class TTSHealthResult:
    available: bool
    message: str
    latency_ms: int = 0
    backend: str = ""


class TTSEngine:
    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._cancel_playback = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop, name="sid-tts", daemon=True
        )
        self._current_process: subprocess.Popen[bytes] | None = None
        self._fallback_reason = ""
        self._disabled_reason = ""
        self._preferred_backend = "piper"
        self._worker.start()

    async def speak(self, text: str, force_offline: bool = False) -> None:
        del force_offline
        if not self.config.use_offline_tts or self._disabled_reason:
            return
        cleaned = (text or "").strip()
        if cleaned:
            await asyncio.to_thread(self._queue.put, cleaned)

    async def cancel_speech(self) -> None:
        self._cancel_playback.set()
        await asyncio.to_thread(self._terminate_current_process)
        await asyncio.to_thread(self._drain_queue)

    async def verify_offline_tts(self) -> TTSHealthResult:
        result = await asyncio.to_thread(self._verify_offline_tts_sync)
        self._disabled_reason = "" if result.available else result.message
        if result.backend in {"piper", "espeak"}:
            self._preferred_backend = result.backend
        return result

    async def play_wake_chime(self) -> None:
        await asyncio.to_thread(self._play_wave_file, self._resolve_asset("wake.wav"))

    async def play_done_chime(self) -> None:
        await asyncio.to_thread(self._play_wave_file, self._resolve_asset("done.wav"))

    async def play_error_chime(self) -> None:
        await asyncio.to_thread(self._play_wave_file, self._resolve_asset("error.wav"))

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=3)
        self._terminate_current_process()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self._disabled_reason or not self.config.use_offline_tts:
                continue
            self._cancel_playback.clear()
            try:
                if self._preferred_backend == "espeak":
                    self._speak_espeak(item)
                else:
                    self._speak_piper(item)
            except Exception as exc:
                self._fallback_reason = str(exc)
                if self._preferred_backend != "espeak":
                    logging.warning("Piper unavailable, falling back to eSpeak: %s", exc)
                try:
                    self._speak_espeak(item)
                    self._preferred_backend = "espeak"
                except Exception as fallback_exc:
                    self._disable_speech(str(fallback_exc))

    def _speak_piper(self, text: str) -> None:
        binary = self._resolve_binary(self.config.piper_binary)
        voice_path = Path(self.config.piper_voice_path)
        voice_cfg = Path(self.config.piper_voice_config_path)
        if not binary or not voice_path.exists() or not voice_cfg.exists():
            raise RuntimeError("Piper voice assets are not ready")

        process = subprocess.Popen(
            [
                binary,
                "--model",
                str(voice_path),
                "--config",
                str(voice_cfg),
                "--output_raw",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._current_process = process
        assert process.stdin is not None
        process.stdin.write(text.encode("utf-8"))
        process.stdin.close()

        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:
            process.terminate()
            raise RuntimeError("sounddevice is required for Piper playback") from exc

        stream = sd.RawOutputStream(
            samplerate=22050,
            channels=1,
            dtype="int16",
            blocksize=self.config.tts_chunk_size,
        )
        stream.start()
        try:
            assert process.stdout is not None
            while not self._cancel_playback.is_set():
                chunk = process.stdout.read(self.config.tts_chunk_size)
                if not chunk:
                    break
                stream.write(chunk)
        finally:
            stream.stop()
            stream.close()
            if self._cancel_playback.is_set():
                self._terminate_current_process()
            process.wait(timeout=5)
            self._current_process = None
            self._cancel_playback.clear()

    def _speak_espeak(self, text: str) -> None:
        binary = self._resolve_espeak_binary()
        if not binary:
            raise RuntimeError(self.espeak_install_command())
        proc = subprocess.run(
            [binary, "--stdout", text], capture_output=True, check=False, timeout=15
        )
        if proc.returncode != 0:
            raise RuntimeError(
                proc.stderr.decode("utf-8", errors="ignore") or "espeak-ng failed"
            )
        self._play_wav_bytes(proc.stdout)

    def _verify_offline_tts_sync(self) -> TTSHealthResult:
        if not self.config.use_offline_tts:
            return TTSHealthResult(
                False, "Offline speech is disabled in config.", backend="disabled"
            )

        started = time.perf_counter()
        binary = self._resolve_binary(self.config.piper_binary)
        voice_path = Path(self.config.piper_voice_path)
        voice_cfg = Path(self.config.piper_voice_config_path)
        piper_issue = ""
        if not binary:
            piper_issue = "Piper binary missing."
        elif not voice_path.exists():
            piper_issue = f"Piper voice missing: {voice_path}"
        elif not voice_cfg.exists():
            piper_issue = f"Piper voice config missing: {voice_cfg}"
        else:
            try:
                proc = subprocess.run(
                    [
                        binary,
                        "--model",
                        str(voice_path),
                        "--config",
                        str(voice_cfg),
                        "--output_raw",
                    ],
                    input=b"ready",
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
            except Exception as exc:
                piper_issue = f"Piper health check failed: {exc}"
            else:
                if proc.returncode == 0 and proc.stdout:
                    return TTSHealthResult(
                        True,
                        "Piper ready",
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        backend="piper",
                    )
                piper_issue = "Piper smoke test failed"

        espeak_binary = self._resolve_espeak_binary()
        if espeak_binary:
            return TTSHealthResult(
                True,
                f"eSpeak fallback ready via {Path(espeak_binary).name}.",
                backend="espeak",
            )

        install_hint = self.espeak_install_command()
        return TTSHealthResult(
            False,
            f"No local TTS backend available. {piper_issue} Install eSpeak with {install_hint}.",
            backend="disabled",
        )

    def espeak_install_command(self) -> str:
        platform_name = shutil.which("winget")
        if Path("/usr/bin/apt").exists():
            return "apt install espeak-ng"
        if shutil.which("brew"):
            return "brew install espeak"
        if platform_name:
            return "winget install eSpeak.eSpeak"
        return "Install espeak-ng from your platform package manager"

    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        if not wav_bytes:
            self._play_system_chime()
            return
        try:
            import sounddevice as sd  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            self._play_system_chime()
            return
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                if wav_file.getsampwidth() != 2:
                    raise ValueError("Only 16-bit PCM wave chimes are supported")
                frames = wav_file.readframes(wav_file.getnframes())
                if not frames:
                    raise ValueError("Wave chime contains no audio frames")
                audio = np.frombuffer(frames, dtype=np.int16)
                if wav_file.getnchannels() > 1:
                    audio = audio.reshape(-1, wav_file.getnchannels())
                sd.play(audio, wav_file.getframerate())
                sd.wait()
        except (EOFError, ValueError, wave.Error) as exc:
            logging.warning("Unable to play Sid chime: %s", exc)
            self._play_system_chime()

    def _play_wave_file(self, path: Path) -> None:
        if not path.exists():
            logging.debug("Sid chime asset missing: %s", path)
            self._play_system_chime()
            return
        wav_bytes = path.read_bytes()
        if not wav_bytes:
            logging.warning("Sid chime asset is empty: %s", path)
            self._play_system_chime()
            return
        self._play_wav_bytes(wav_bytes)

    def _play_system_chime(self) -> None:
        if os.name != "nt":
            return
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_OK)
        except Exception:
            return

    def _resolve_binary(self, value: str) -> str | None:
        candidate = shutil.which(value)
        if candidate:
            return candidate
        path = Path(value).expanduser()
        if path.exists():
            return str(path)
        return None

    def _resolve_espeak_binary(self) -> str | None:
        candidates: list[str] = []
        for value in (self.config.espeak_binary, "espeak-ng", "espeak"):
            if value and value not in candidates:
                candidates.append(value)
        for env_var in ("ProgramFiles", "ProgramFiles(x86)"):
            base = os.getenv(env_var)
            if not base:
                continue
            for value in (
                Path(base) / "eSpeak NG" / "espeak-ng.exe",
                Path(base) / "eSpeak NG" / "command_line" / "espeak-ng.exe",
                Path(base) / "eSpeak" / "command_line" / "espeak.exe",
            ):
                candidate = str(value)
                if candidate not in candidates:
                    candidates.append(candidate)
        for value in candidates:
            resolved = self._resolve_binary(value)
            if resolved:
                return resolved
        return None

    def _disable_speech(self, reason: str) -> None:
        message = reason.strip() or "No local TTS backend available."
        if self._disabled_reason == message:
            return
        self._disabled_reason = message
        self._drain_queue()
        logging.warning("Speech output disabled: %s", message)

    def _terminate_current_process(self) -> None:
        if self._current_process and self._current_process.poll() is None:
            self._current_process.terminate()

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _resolve_asset(self, filename: str) -> Path:
        local_asset = (
            Path(__file__).resolve().parents[1] / "assets" / "sounds" / filename
        )
        if local_asset.exists():
            return local_asset
        return Path(self.config.sid_dir) / "assets" / "sounds" / filename


__all__ = ["TTSEngine", "TTSHealthResult"]
