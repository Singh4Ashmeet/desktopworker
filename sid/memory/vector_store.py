from __future__ import annotations

import asyncio
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from config import SidConfig
from memory.embeddings import LocalEmbeddingModel


class VectorStore:
    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self.path = config.memory_dir / "chroma"
        self.path.mkdir(parents=True, exist_ok=True)
        self._fallback_file = self.path / "fallback_interactions.json"
        self._fallback_data: list[dict[str, Any]] = []
        self._client = None
        self._collection = None
        self._lock = asyncio.Lock()
        self._embedder = LocalEmbeddingModel(config)
        self._init_done = False
        self._reembed_task: asyncio.Task[None] | None = None

    @property
    def embedding_key(self) -> str:
        return f"{self.config.embedding_model_name}::{self.config.embedding_dim}"

    @property
    def collection_name(self) -> str:
        safe = self.embedding_key.replace("/", "_").replace(":", "_").replace("-", "_")
        return f"sid_interactions_{safe}"

    async def _ensure_init(self) -> None:
        if self._init_done:
            return
        async with self._lock:
            if self._init_done:
                return
            try:
                import chromadb  # type: ignore

                self._client = chromadb.PersistentClient(path=str(self.path))
                self._collection = self._client.get_or_create_collection(
                    self.collection_name,
                    metadata={
                        "embedding_model_name": self.config.embedding_model_name,
                        "embedding_dim": self.config.embedding_dim,
                    },
                )
                await self._queue_reembed_if_needed()
            except Exception:
                if self._fallback_file.exists():
                    self._fallback_data = json.loads(
                        self._fallback_file.read_text(encoding="utf-8")
                    )
            self._init_done = True

    async def add_interaction(
        self, user_command: str, sid_response: str, tools_used: list[str]
    ) -> None:
        await self._ensure_init()
        text = f"{user_command} -> {sid_response}"
        created_at = datetime.now(timezone.utc)
        metadata = {
            "timestamp": created_at.timestamp(),
            "tools_used": ",".join(tools_used),
            "task_type": tools_used[0] if tools_used else "general",
            "embedding_model_name": self.config.embedding_model_name,
            "embedding_dim": self.config.embedding_dim,
            "embedding_key": self.embedding_key,
        }
        record_id = str(uuid.uuid4())

        if self._collection is not None:
            embedding = (await self._embedder.embed_texts([text]))[0]
            await asyncio.to_thread(
                self._collection.add,
                ids=[record_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
            )
            return

        self._fallback_data.append(
            {"id": record_id, "text": text, "metadata": metadata}
        )
        await asyncio.to_thread(self._persist_fallback)

    async def query_relevant(
        self, user_input: str, n: int = 6, use_decay: bool = True
    ) -> list[str]:
        await self._ensure_init()
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)

        if self._collection is not None:
            embedding = (await self._embedder.embed_texts([user_input]))[0]
            payload = await asyncio.to_thread(
                self._collection.query,
                query_embeddings=[embedding],
                n_results=max(10, n * 3),
                include=["documents", "metadatas", "distances"],
            )
            docs = payload.get("documents", [[]])[0]
            metas = payload.get("metadatas", [[]])[0]
            distances = payload.get("distances", [[]])[0]
            ranked: list[tuple[float, str]] = []
            for doc, meta, distance in zip(docs, metas, distances):
                meta = meta or {}
                if meta.get("embedding_key") != self.embedding_key:
                    continue
                ts = _metadata_time(meta)
                if ts is None or ts < cutoff:
                    continue
                base_score = 1.0 / (1.0 + float(distance))
                ranked.append(
                    (
                        _apply_decay(
                            base_score, ts, self.config.memory_decay_lambda, use_decay
                        ),
                        str(doc),
                    )
                )
            ranked.sort(key=lambda item: item[0], reverse=True)
            return [item[1] for item in ranked[:n]]

        query_tokens = set(user_input.lower().split())
        ranked: list[tuple[float, str]] = []
        for row in self._fallback_data:
            meta = row.get("metadata", {})
            ts = _metadata_time(meta)
            if ts is None or ts < cutoff:
                continue
            text = str(row.get("text", ""))
            overlap = sum(
                1 for token in query_tokens if token and token in text.lower()
            )
            ranked.append(
                (
                    _apply_decay(
                        float(overlap), ts, self.config.memory_decay_lambda, use_decay
                    ),
                    text,
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[:n]]

    async def similarity(self, left: str, right: str) -> float:
        try:
            return await self._embedder.cosine_similarity(left, right)
        except Exception:
            left_tokens = set(left.lower().split())
            right_tokens = set(right.lower().split())
            if not left_tokens or not right_tokens:
                return 0.0
            return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    async def flush(self) -> None:
        if self._collection is None:
            await asyncio.to_thread(self._persist_fallback)
        if self._reembed_task is not None:
            await asyncio.gather(self._reembed_task, return_exceptions=True)

    async def _queue_reembed_if_needed(self) -> None:
        if self._client is None or self._reembed_task is not None:
            return
        collections = await asyncio.to_thread(self._client.list_collections)
        legacy_names = [
            collection.name
            for collection in collections
            if collection.name.startswith("sid_interactions_")
            and collection.name != self.collection_name
        ]
        if legacy_names:
            self._reembed_task = asyncio.create_task(
                self._reembed_from_collections(legacy_names)
            )

    async def _reembed_from_collections(self, collection_names: list[str]) -> None:
        if self._client is None or self._collection is None:
            return
        for name in collection_names:
            source = self._client.get_collection(name)
            payload = await asyncio.to_thread(source.get)
            documents = payload.get("documents") or []
            metadatas = payload.get("metadatas") or []
            ids = payload.get("ids") or []
            if not documents:
                continue
            embeddings = await self._embedder.embed_texts(
                [str(doc) for doc in documents]
            )
            rewritten_meta = []
            for meta in metadatas:
                item = dict(meta or {})
                item["embedding_model_name"] = self.config.embedding_model_name
                item["embedding_dim"] = self.config.embedding_dim
                item["embedding_key"] = self.embedding_key
                rewritten_meta.append(item)
            await asyncio.to_thread(
                self._collection.add,
                ids=[f"reembed-{uuid.uuid4()}" for _ in ids],
                documents=documents,
                embeddings=embeddings,
                metadatas=rewritten_meta,
            )

    def _persist_fallback(self) -> None:
        self._fallback_file.write_text(
            json.dumps(self._fallback_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _metadata_time(meta: dict[str, Any]) -> datetime | None:
    raw = meta.get("timestamp")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    try:
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    except Exception:
        return None


def _apply_decay(
    base_score: float, created_at: datetime, decay_lambda: float, use_decay: bool
) -> float:
    if not use_decay:
        return base_score
    age_days = max(
        0.0, (datetime.now(timezone.utc) - created_at).total_seconds() / 86400.0
    )
    return base_score * math.exp(-decay_lambda * age_days)


__all__ = ["VectorStore"]
