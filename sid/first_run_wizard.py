"""First-run wizard for Sid AI Assistant.

Guides users through:
1. Model download consent
2. LLM model selection
3. Whisper model selection
4. Piper voice selection
5. Embedding model selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import SidConfig
from model_manager import DownloadSpec, ModelDownloadManager


@dataclass(slots=True)
class ModelOption:
    """A model selection option."""

    id: str
    name: str
    description: str
    size_mb: int
    ram_mb: int  # Estimated RAM usage
    recommended: bool = False


# LLM Model Options (via Ollama)
LLM_OPTIONS = [
    ModelOption(
        id="llama3.2:3b",
        name="Llama 3.2 3B",
        description="Fast inference, lower quality. Best for 8GB RAM systems.",
        size_mb=2000,
        ram_mb=4000,
    ),
    ModelOption(
        id="llama3.1:8b",
        name="Llama 3.1 8B",
        description="Better reasoning and tool use. Requires 16GB+ RAM.",
        size_mb=4700,
        ram_mb=8000,
        recommended=True,
    ),
    ModelOption(
        id="mistral:7b",
        name="Mistral 7B",
        description="Balanced performance, good for general tasks.",
        size_mb=4100,
        ram_mb=8000,
    ),
    ModelOption(
        id="qwen2.5:1.5b",
        name="Qwen 2.5 1.5B",
        description="Tiny model for low-resource machines.",
        size_mb=1000,
        ram_mb=2000,
    ),
]

# Whisper Model Options
WHISPER_OPTIONS = [
    ModelOption(
        id="tiny.en",
        name="Whisper Tiny",
        description="Fastest transcription, lower accuracy.",
        size_mb=150,
        ram_mb=1000,
    ),
    ModelOption(
        id="base.en",
        name="Whisper Base",
        description="Good balance of speed and accuracy.",
        size_mb=150,
        ram_mb=1500,
        recommended=True,
    ),
    ModelOption(
        id="small.en",
        name="Whisper Small",
        description="More accurate, slower transcription.",
        size_mb=500,
        ram_mb=2000,
    ),
]

# Piper Voice Options
PIPER_VOICES = {
    "en_US-lessac-medium": {
        "name": "Lessac (US)",
        "description": "Natural American English voice",
        "url": "https://github.com/rhasspy/piper-voices/raw/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "config_url": "https://github.com/rhasspy/piper-voices/raw/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "sha256": "",  # Will be computed on first download
        "size_mb": 80,
        "recommended": True,
    },
    "en_GB-alan-medium": {
        "name": "Alan (British)",
        "description": "British English male voice",
        "url": "https://github.com/rhasspy/piper-voices/raw/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "config_url": "https://github.com/rhasspy/piper-voices/raw/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
        "sha256": "",
        "size_mb": 80,
        "recommended": False,
    },
    "en_US-amy-medium": {
        "name": "Amy (US)",
        "description": "American English female voice",
        "url": "https://github.com/rhasspy/piper-voices/raw/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "config_url": "https://github.com/rhasspy/piper-voices/raw/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        "sha256": "",
        "size_mb": 80,
        "recommended": False,
    },
}

# Embedding Model Options
EMBEDDING_OPTIONS = [
    ModelOption(
        id="sentence-transformers/all-MiniLM-L6-v2",
        name="MiniLM-L6 (fast)",
        description="22MB, 384-dim. Fast on CPU, good quality.",
        size_mb=22,
        ram_mb=500,
        recommended=True,
    ),
    ModelOption(
        id="sentence-transformers/all-mpnet-base-v2",
        name="MPNet-base (accurate)",
        description="420MB, 768-dim. Better quality, more RAM.",
        size_mb=420,
        ram_mb=2000,
    ),
]

# Piper binary download info
PIPER_BINARY = {
    "windows": {
        "url": "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_windows_amd64.zip",
        "extract_to": "piper",
    },
    "linux": {
        "url": "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_x86_64",
        "extract_to": None,
    },
    "macos": {
        "url": "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_macos_x64",
        "extract_to": None,
    },
}


class FirstRunWizard:
    """Interactive first-run setup wizard."""

    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self.download_mgr = ModelDownloadManager(config)

    async def run_wizard(self) -> None:
        """Run the complete first-run wizard."""
        print("=" * 60)
        print("Welcome to Sid AI Assistant!")
        print("=" * 60)
        print()

        # Step 1: Model download consent
        await self._step_consent()

        # Step 2: LLM model selection
        await self._step_select_llm()

        # Step 3: Whisper model selection
        await self._step_select_whisper()

        # Step 4: Piper voice selection
        await self._step_select_piper_voice()

        # Step 5: Embedding model selection
        await self._step_select_embedding()

        # Step 6: Download models
        await self._download_selected_models()

        print()
        print("=" * 60)
        print("Setup complete! Sid is ready to use.")
        print("=" * 60)

    async def _step_consent(self) -> None:
        """Get user consent for model downloads."""
        print("\n[Step 1/6] Model Downloads")
        print("-" * 40)
        print("Sid needs to download AI models to run locally.")
        print("All models are free, open-source, and stored on your machine.")
        print()
        print("Estimated total download size: ~5GB (full setup)")
        print("Minimum footprint: ~500MB (tiny models only)")
        print()

        response = (
            input("Do you consent to downloading models? [Y/n]: ").strip().lower()
        )
        if response in ("", "y", "yes"):
            self.config.allow_model_downloads = True
            self.config.model_download_consent = True
        else:
            print(
                "Model downloads declined. You will need to manually download models."
            )
            self.config.allow_model_downloads = False
            self.config.model_download_consent = False

    async def _step_select_llm(self) -> None:
        """Select LLM model."""
        print("\n[Step 2/6] LLM Model Selection")
        print("-" * 40)
        print("Select the main AI model for Sid's reasoning:")
        print()

        for i, opt in enumerate(LLM_OPTIONS, 1):
            rec = " [RECOMMENDED]" if opt.recommended else ""
            print(f"  {i}. {opt.name}{rec}")
            print(f"     {opt.description}")
            print(f"     Size: {opt.size_mb}MB | RAM: {opt.ram_mb}MB")
            print()

        choice = input(f"Enter choice [1-{len(LLM_OPTIONS)}] (default: 2): ").strip()
        idx = int(choice) - 1 if choice else 1
        idx = max(0, min(idx, len(LLM_OPTIONS) - 1))

        self.config.ollama_model = LLM_OPTIONS[idx].id
        print(f"Selected: {LLM_OPTIONS[idx].name}")

    async def _step_select_whisper(self) -> None:
        """Select Whisper model."""
        print("\n[Step 3/6] Speech Recognition Model")
        print("-" * 40)
        print("Select the speech-to-text model:")
        print()

        for i, opt in enumerate(WHISPER_OPTIONS, 1):
            rec = " [RECOMMENDED]" if opt.recommended else ""
            print(f"  {i}. {opt.name}{rec}")
            print(f"     {opt.description}")
            print(f"     Size: {opt.size_mb}MB | RAM: {opt.ram_mb}MB")
            print()

        choice = input(
            f"Enter choice [1-{len(WHISPER_OPTIONS)}] (default: 2): "
        ).strip()
        idx = int(choice) - 1 if choice else 1
        idx = max(0, min(idx, len(WHISPER_OPTIONS) - 1))

        self.config.whisper_model = WHISPER_OPTIONS[idx].id
        print(f"Selected: {WHISPER_OPTIONS[idx].name}")

    async def _step_select_piper_voice(self) -> None:
        """Select Piper voice."""
        print("\n[Step 4/6] Text-to-Speech Voice")
        print("-" * 40)
        print("Select Sid's voice:")
        print()

        voices = list(PIPER_VOICES.items())
        for i, (vid, vdata) in enumerate(voices, 1):
            rec = " [RECOMMENDED]" if vdata.get("recommended") else ""
            print(f"  {i}. {vdata['name']}{rec}")
            print(f"     {vdata['description']}")
            print(f"     Size: {vdata['size_mb']}MB")
            print()

        choice = input(f"Enter choice [1-{len(voices)}] (default: 1): ").strip()
        idx = int(choice) - 1 if choice else 0
        idx = max(0, min(idx, len(voices) - 1))

        voice_id = voices[idx][0]
        voice_data = PIPER_VOICES[voice_id]

        self.config.piper_voice_name = voice_id
        self.config.piper_voice_path = str(
            self.config.models_dir / "piper" / f"{voice_id}.onnx"
        )
        self.config.piper_voice_config_path = str(
            self.config.models_dir / "piper" / f"{voice_id}.onnx.json"
        )
        print(f"Selected: {voice_data['name']}")

    async def _step_select_embedding(self) -> None:
        """Select embedding model."""
        print("\n[Step 5/6] Memory Embedding Model")
        print("-" * 40)
        print("Select the model for memory embeddings:")
        print()

        for i, opt in enumerate(EMBEDDING_OPTIONS, 1):
            rec = " [RECOMMENDED]" if opt.recommended else ""
            print(f"  {i}. {opt.name}{rec}")
            print(f"     {opt.description}")
            print(f"     Size: {opt.size_mb}MB | RAM: {opt.ram_mb}MB")
            print()

        choice = input(
            f"Enter choice [1-{len(EMBEDDING_OPTIONS)}] (default: 1): "
        ).strip()
        idx = int(choice) - 1 if choice else 0
        idx = max(0, min(idx, len(EMBEDDING_OPTIONS) - 1))

        self.config.embedding_model_name = EMBEDDING_OPTIONS[idx].id
        print(f"Selected: {EMBEDDING_OPTIONS[idx].name}")

    async def _download_selected_models(self) -> None:
        """Download all selected models."""
        print("\n[Step 6/6] Downloading Models...")
        print("-" * 40)

        if not self.config.model_download_consent:
            print("Skipping downloads (consent not given).")
            return

        # Download Piper voice
        print("\nDownloading Piper voice...")
        voice_data = PIPER_VOICES.get(self.config.piper_voice_name)
        if voice_data:
            await self._download_piper_voice(voice_data)

        print("\nModel downloads complete!")

    async def _download_piper_voice(self, voice_data: dict[str, Any]) -> None:
        """Download Piper voice model and config."""
        piper_dir = self.config.models_dir / "piper"
        piper_dir.mkdir(parents=True, exist_ok=True)

        voice_id = self.config.piper_voice_name
        onnx_path = piper_dir / f"{voice_id}.onnx"
        json_path = piper_dir / f"{voice_id}.onnx.json"

        # Download ONNX model
        onnx_spec = DownloadSpec(
            name=f"Piper voice: {voice_data['name']}",
            url=voice_data["url"],
            destination=onnx_path,
            sha256=voice_data.get("sha256", ""),
        )
        await self.download_mgr.ensure_file(onnx_spec)

        # Download config JSON
        json_spec = DownloadSpec(
            name=f"Piper voice config: {voice_data['name']}",
            url=voice_data["config_url"],
            destination=json_path,
        )
        await self.download_mgr.ensure_file(json_spec)

        print(f"  Downloaded: {voice_data['name']}")

    def save_config(self) -> None:
        """Save configuration to disk."""
        self.config.save()
        print("\nConfiguration saved to:", self.config.config_file)


async def run_first_run_wizard(config: SidConfig) -> None:
    """Run the first-run wizard if needed."""
    if config.initialized_flag.exists():
        return  # Already initialized

    wizard = FirstRunWizard(config)
    await wizard.run_wizard()
    wizard.save_config()

    # Mark as initialized
    config.initialized_flag.write_text("initialized", encoding="utf-8")


__all__ = ["FirstRunWizard", "run_first_run_wizard", "PIPER_VOICES"]
