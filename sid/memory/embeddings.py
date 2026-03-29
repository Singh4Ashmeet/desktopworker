from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path
from typing import Sequence

from config import SidConfig
from model_manager import ModelDownloadManager


class LocalEmbeddingModel:
    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self._model = None
        self._lock = asyncio.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="sid-embed"
        )
        self._download_manager = ModelDownloadManager(config)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        model = await self._ensure_model()
        if model is None:
            return [self._fallback_vector(text) for text in texts]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: model.encode(list(texts), normalize_embeddings=True).tolist(),
        )

    async def cosine_similarity(self, left: str, right: str) -> float:
        vectors = await self.embed_texts([left, right])
        if len(vectors) != 2:
            return 0.0
        return float(sum(a * b for a, b in zip(vectors[0], vectors[1])))

    async def _ensure_model(self):
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception:
                self._model = None
                return self._model

            model_root = Path(self.config.embedding_model_dir)
            model_root.mkdir(parents=True, exist_ok=True)
            if self.config.allow_model_downloads and self.config.model_download_consent:
                await self._download_manager.ensure_hf_snapshot(
                    self.config.embedding_model_name, model_root / self._safe_name()
                )
            try:
                self._model = SentenceTransformer(
                    self.config.embedding_model_name,
                    cache_folder=str(model_root),
                    device="cpu",
                    local_files_only=not self.config.model_download_consent,
                )
            except Exception:
                self._model = None
        return self._model

    def _safe_name(self) -> str:
        return self.config.embedding_model_name.replace("/", "_").replace(":", "_")

    def _fallback_vector(self, text: str) -> list[float]:
        buckets = [0.0] * max(8, min(self.config.embedding_dim, 32))
        for token in text.lower().split():
            buckets[hash(token) % len(buckets)] += 1.0
        norm = sum(value * value for value in buckets) ** 0.5 or 1.0
        return [value / norm for value in buckets]


__all__ = ["LocalEmbeddingModel"]
