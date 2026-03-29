from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CacheEntry:
    value: dict[str, Any]
    expires_at: float


class ToolResultCache:
    def __init__(self) -> None:
        self._items: dict[str, CacheEntry] = {}
        self.hits: dict[str, int] = {}
        self.misses: dict[str, int] = {}

    def make_key(self, tool_name: str, kwargs: dict[str, Any]) -> str:
        ordered = sorted((k, repr(v)) for k, v in kwargs.items())
        return f"{tool_name}|{ordered!r}"

    def get(self, tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        key = self.make_key(tool_name, kwargs)
        item = self._items.get(key)
        now = time.monotonic()
        if item is None or item.expires_at <= now:
            self.misses[tool_name] = self.misses.get(tool_name, 0) + 1
            if item is not None:
                self._items.pop(key, None)
            return None
        self.hits[tool_name] = self.hits.get(tool_name, 0) + 1
        return dict(item.value)

    def set(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        value: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        key = self.make_key(tool_name, kwargs)
        self._items[key] = CacheEntry(
            value=dict(value), expires_at=time.monotonic() + ttl_seconds
        )

    def invalidate_by_tool_prefix(self, tool_names: set[str]) -> None:
        for key in list(self._items):
            tool_name = key.split("|", 1)[0]
            if tool_name in tool_names:
                self._items.pop(key, None)
