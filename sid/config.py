from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _expand_path(value: str) -> str:
    return str(Path(os.path.expandvars(os.path.expanduser(value))).resolve())


def _default_allowlist() -> list[str]:
    temp_dir = os.getenv("TEMP") or os.getenv("TMP") or "/tmp"
    return [
        str(Path.home() / "Documents"),
        str(Path.home() / "Downloads"),
        str(Path.home() / "Desktop"),
        str(Path(temp_dir)),
        str(Path.home() / ".sid"),
    ]


class SidConfig(BaseModel):
    user_name: str = "boss"
    projects_dir: str = "~/Projects"
    sid_dir: str = "~/.sid"
    log_level: str = "INFO"

    use_local_llm: bool = True
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = ""
    ollama_model_candidates: list[str] = Field(
        default_factory=lambda: [
            "llama3.2:3b",
            "llama3.1:8b",
            "mistral:7b",
            "qwen2.5:1.5b",
        ]
    )
    ollama_context_limit: int = 32768
    ollama_start_command: str = "ollama serve"
    react_max_steps: int = 12
    react_dry_run: bool = False
    context_warn_ratio: float = 0.70
    context_drop_ratio: float = 0.90

    whisper_model: str = "base.en"
    whisper_model_dir: str = "~/.sid/models/whisper"
    vad_confidence_threshold: float = 0.65
    vad_model_path: str = "~/.sid/models/vad/silero_vad.onnx"
    stt_max_record_seconds: int = 30
    stt_silence_ms: int = 700

    wake_word_sensitivity: float = 0.5
    wake_word_model: str = "~/.sid/models/wakeword/hey_sid.tflite"
    wake_word_models_dir: str = "~/.sid/models/wakeword"
    wake_word_cooldown_seconds: float = 1.5
    hotkey: str = "ctrl+space"

    tts_speed: float = 1.0
    use_offline_tts: bool = True
    piper_binary: str = "piper"
    piper_voice_name: str = "en_US-lessac-medium"
    piper_voice_path: str = "~/.sid/models/piper/en_US-lessac-medium.onnx"
    piper_voice_config_path: str = "~/.sid/models/piper/en_US-lessac-medium.onnx.json"
    piper_voice_checksum: str = ""
    espeak_binary: str = "espeak-ng"
    tts_chunk_size: int = 4096

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_dir: str = "~/.sid/models/embeddings"
    embedding_dim: int = 384
    memory_decay_lambda: float = 0.01

    vision_model_name: str = "vikhyatk/moondream2"
    vision_model_dir: str = "~/.sid/models/vision/moondream2"
    enable_vision: bool = True
    screen_capture_interval: float = 1.0
    screen_phash_threshold: int = 8
    screen_stability_backoff_ticks: int = 5
    ocr_cache_ttl_seconds: int = 30
    active_window_debounce_ms: int = 75
    transient_window_ignore_ms: int = 200
    enable_proactive: bool = True
    enable_focus_breaks: bool = False

    searxng_url: str = "http://localhost:8080"
    web_request_timeout_seconds: int = 30
    allow_model_downloads: bool = True
    model_download_consent: bool = False

    hud_position: str = "bottom-right"
    hud_opacity: float = 0.92

    allowlist_paths: list[str] = Field(default_factory=_default_allowlist)
    require_confirm_delete: bool = True
    enable_undo_buffer: bool = True
    undo_buffer_max_mb: int = 500
    tool_default_timeout_seconds: int = 30
    danger_tool_timeout_seconds: int = 15
    sandbox_max_memory_mb: int = 512
    sandbox_max_cpu_percent: int = 85

    enable_morning_briefing: bool = False
    briefing_time: str = "08:30"

    cache_list_dir_ttl_seconds: int = 5
    cache_process_list_ttl_seconds: int = 2
    cache_system_stats_ttl_seconds: int = 3
    cache_window_list_ttl_seconds: int = 3

    permitted_runtime_hosts: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1"]
    )
    explicit_fetch_hosts: list[str] = Field(default_factory=list)

    @field_validator(
        "projects_dir",
        "sid_dir",
        "whisper_model_dir",
        "vad_model_path",
        "wake_word_model",
        "wake_word_models_dir",
        "piper_voice_path",
        "piper_voice_config_path",
        "embedding_model_dir",
        "vision_model_dir",
        mode="before",
    )
    @classmethod
    def _expand_single_path(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return _expand_path(value)

    @field_validator("allowlist_paths", mode="before")
    @classmethod
    def _expand_allowlist(cls, value: Any) -> list[str]:
        if value is None or value == "":
            value = _default_allowlist()
        if isinstance(value, str):
            value = [value]
        return [_expand_path(str(v)) for v in value]

    @property
    def sid_path(self) -> Path:
        return Path(self.sid_dir).expanduser().resolve()

    @property
    def config_file(self) -> Path:
        return self.sid_path / "config.toml"

    @property
    def logs_dir(self) -> Path:
        return self.sid_path / "logs"

    @property
    def memory_dir(self) -> Path:
        return self.sid_path / "memory"

    @property
    def undo_dir(self) -> Path:
        return self.sid_path / "undo"

    @property
    def models_dir(self) -> Path:
        return self.sid_path / "models"

    @property
    def training_data_dir(self) -> Path:
        return self.sid_path / "training_data"

    @property
    def initialized_flag(self) -> Path:
        return self.sid_path / "initialized"

    @property
    def state_file(self) -> Path:
        return self.sid_path / "state.json"

    @property
    def audit_manifest_file(self) -> Path:
        return self.logs_dir / "manifest.json"

    @property
    def hud_positions_file(self) -> Path:
        return self.sid_path / "hud_positions.json"

    @property
    def current_context_limit(self) -> int:
        return self.ollama_context_limit

    def ensure_dirs(self) -> None:
        for path in [
            self.sid_path,
            self.logs_dir,
            self.memory_dir,
            self.undo_dir,
            self.models_dir,
            self.training_data_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def validate_startup(self) -> None:
        self.ensure_dirs()
        if not self.allowlist_paths:
            raise ValueError("allowlist_paths must include at least one directory")
        if self.react_max_steps < 1:
            raise ValueError("react_max_steps must be at least 1")
        if not 0.0 <= self.vad_confidence_threshold <= 1.0:
            raise ValueError("vad_confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.wake_word_sensitivity <= 1.0:
            raise ValueError("wake_word_sensitivity must be in [0, 1]")
        if (
            self.tool_default_timeout_seconds < 1
            or self.danger_tool_timeout_seconds < 1
        ):
            raise ValueError("tool timeouts must be at least 1 second")
        if self.context_warn_ratio <= 0 or self.context_warn_ratio >= 1:
            raise ValueError("context_warn_ratio must be between 0 and 1")
        if (
            self.context_drop_ratio <= self.context_warn_ratio
            or self.context_drop_ratio >= 1
        ):
            raise ValueError(
                "context_drop_ratio must be greater than warn ratio and less than 1"
            )

    @classmethod
    def load(cls, path: str | None = None, **overrides: Any) -> "SidConfig":
        sid_path = Path(_expand_path(path or "~/.sid"))
        sid_path.mkdir(parents=True, exist_ok=True)
        config_path = sid_path / "config.toml"
        payload: dict[str, Any] = {}
        if config_path.exists():
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        payload.update(overrides)
        payload.setdefault("sid_dir", str(sid_path))
        config = cls(**payload)
        if not config_path.exists():
            config.save()
        return config

    def save(self) -> None:
        self.ensure_dirs()
        serialized = self._to_toml()
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=str(self.sid_path)
        ) as handle:
            handle.write(serialized)
            temp_name = handle.name
        Path(temp_name).replace(self.config_file)

    def _to_toml(self) -> str:
        data = self.model_dump()
        s = self._toml_string
        lines: list[str] = [
            f"user_name = {s(data['user_name'])}",
            f"projects_dir = {s(data['projects_dir'])}",
            f"sid_dir = {s(data['sid_dir'])}",
            f"log_level = {s(data['log_level'])}",
            f"use_local_llm = {str(data['use_local_llm']).lower()}",
            f"ollama_host = {s(data['ollama_host'])}",
            f"ollama_model = {s(data['ollama_model'])}",
            f"ollama_model_candidates = {self._toml_array(data['ollama_model_candidates'])}",
            f"ollama_context_limit = {data['ollama_context_limit']}",
            f"ollama_start_command = {s(data['ollama_start_command'])}",
            f"react_max_steps = {data['react_max_steps']}",
            f"react_dry_run = {str(data['react_dry_run']).lower()}",
            f"context_warn_ratio = {data['context_warn_ratio']}",
            f"context_drop_ratio = {data['context_drop_ratio']}",
            f"whisper_model = {s(data['whisper_model'])}",
            f"whisper_model_dir = {s(data['whisper_model_dir'])}",
            f"vad_confidence_threshold = {data['vad_confidence_threshold']}",
            f"vad_model_path = {s(data['vad_model_path'])}",
            f"stt_max_record_seconds = {data['stt_max_record_seconds']}",
            f"stt_silence_ms = {data['stt_silence_ms']}",
            f"wake_word_sensitivity = {data['wake_word_sensitivity']}",
            f"wake_word_model = {s(data['wake_word_model'])}",
            f"wake_word_models_dir = {s(data['wake_word_models_dir'])}",
            f"wake_word_cooldown_seconds = {data['wake_word_cooldown_seconds']}",
            f"hotkey = {s(data['hotkey'])}",
            f"tts_speed = {data['tts_speed']}",
            f"use_offline_tts = {str(data['use_offline_tts']).lower()}",
            f"piper_binary = {s(data['piper_binary'])}",
            f"piper_voice_name = {s(data['piper_voice_name'])}",
            f"piper_voice_path = {s(data['piper_voice_path'])}",
            f"piper_voice_config_path = {s(data['piper_voice_config_path'])}",
            f"piper_voice_checksum = {s(data['piper_voice_checksum'])}",
            f"espeak_binary = {s(data['espeak_binary'])}",
            f"tts_chunk_size = {data['tts_chunk_size']}",
            f"embedding_model_name = {s(data['embedding_model_name'])}",
            f"embedding_model_dir = {s(data['embedding_model_dir'])}",
            f"embedding_dim = {data['embedding_dim']}",
            f"memory_decay_lambda = {data['memory_decay_lambda']}",
            f"vision_model_name = {s(data['vision_model_name'])}",
            f"vision_model_dir = {s(data['vision_model_dir'])}",
            f"enable_vision = {str(data['enable_vision']).lower()}",
            f"screen_capture_interval = {data['screen_capture_interval']}",
            f"screen_phash_threshold = {data['screen_phash_threshold']}",
            f"screen_stability_backoff_ticks = {data['screen_stability_backoff_ticks']}",
            f"ocr_cache_ttl_seconds = {data['ocr_cache_ttl_seconds']}",
            f"active_window_debounce_ms = {data['active_window_debounce_ms']}",
            f"transient_window_ignore_ms = {data['transient_window_ignore_ms']}",
            f"enable_proactive = {str(data['enable_proactive']).lower()}",
            f"enable_focus_breaks = {str(data['enable_focus_breaks']).lower()}",
            f"searxng_url = {s(data['searxng_url'])}",
            f"web_request_timeout_seconds = {data['web_request_timeout_seconds']}",
            f"allow_model_downloads = {str(data['allow_model_downloads']).lower()}",
            f"model_download_consent = {str(data['model_download_consent']).lower()}",
            f"hud_position = {s(data['hud_position'])}",
            f"hud_opacity = {data['hud_opacity']}",
            f"allowlist_paths = {self._toml_array(data['allowlist_paths'])}",
            f"require_confirm_delete = {str(data['require_confirm_delete']).lower()}",
            f"enable_undo_buffer = {str(data['enable_undo_buffer']).lower()}",
            f"undo_buffer_max_mb = {data['undo_buffer_max_mb']}",
            f"tool_default_timeout_seconds = {data['tool_default_timeout_seconds']}",
            f"danger_tool_timeout_seconds = {data['danger_tool_timeout_seconds']}",
            f"sandbox_max_memory_mb = {data['sandbox_max_memory_mb']}",
            f"sandbox_max_cpu_percent = {data['sandbox_max_cpu_percent']}",
            f"enable_morning_briefing = {str(data['enable_morning_briefing']).lower()}",
            f'briefing_time = "{data["briefing_time"]}"',
            f"cache_list_dir_ttl_seconds = {data['cache_list_dir_ttl_seconds']}",
            f"cache_process_list_ttl_seconds = {data['cache_process_list_ttl_seconds']}",
            f"cache_system_stats_ttl_seconds = {data['cache_system_stats_ttl_seconds']}",
            f"cache_window_list_ttl_seconds = {data['cache_window_list_ttl_seconds']}",
            f"permitted_runtime_hosts = {self._toml_array(data['permitted_runtime_hosts'])}",
            f"explicit_fetch_hosts = {self._toml_array(data['explicit_fetch_hosts'])}",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _toml_array(values: list[str]) -> str:
        return "[" + ", ".join(SidConfig._toml_string(value) for value in values) + "]"

    @staticmethod
    def _toml_string(value: str) -> str:
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'


__all__ = ["SidConfig"]
