from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

from PIL import Image

from config import SidConfig
from model_manager import ModelDownloadManager


class MoondreamVisionEngine:
    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()
        self._download_manager = ModelDownloadManager(config)

    async def describe(self, image_b64: str, prompt: str) -> str:
        model, tokenizer = await self._ensure_model()
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

        def _run() -> str:
            if hasattr(model, "answer_question"):
                return str(model.answer_question(image, prompt, tokenizer))
            return str(
                getattr(model, "generate_caption", lambda *_args, **_kwargs: "")(image)
            )

        return await asyncio.to_thread(_run)

    async def _ensure_model(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        async with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return self._model, self._tokenizer
            model_dir = Path(self.config.vision_model_dir)
            if self.config.allow_model_downloads and self.config.model_download_consent:
                await self._download_manager.ensure_hf_snapshot(
                    self.config.vision_model_name, model_dir
                )
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            model = await asyncio.to_thread(
                AutoModelForCausalLM.from_pretrained,
                str(model_dir if model_dir.exists() else self.config.vision_model_name),
                trust_remote_code=True,
                local_files_only=not self.config.model_download_consent,
            )
            tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                str(model_dir if model_dir.exists() else self.config.vision_model_name),
                trust_remote_code=True,
                local_files_only=not self.config.model_download_consent,
            )
            self._model = model
            self._tokenizer = tokenizer
        return self._model, self._tokenizer


__all__ = ["MoondreamVisionEngine"]
