from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class OCRCacheEntry:
    text: str
    expires_at: float


class OCRCache:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, OCRCacheEntry] = {}
        self.hits = 0
        self.misses = 0

    def make_key(self, phash: str, bbox: tuple[int, int, int, int]) -> str:
        return f"{phash}:{bbox}"

    def get(self, phash: str, bbox: tuple[int, int, int, int]) -> str | None:
        key = self.make_key(phash, bbox)
        entry = self._entries.get(key)
        now = time.monotonic()
        if entry is None or entry.expires_at <= now:
            self.misses += 1
            self._entries.pop(key, None)
            return None
        self.hits += 1
        return entry.text

    def set(self, phash: str, bbox: tuple[int, int, int, int], text: str) -> None:
        key = self.make_key(phash, bbox)
        self._entries[key] = OCRCacheEntry(
            text=text, expires_at=time.monotonic() + self.ttl_seconds
        )

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
