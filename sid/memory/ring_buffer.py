from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(slots=True)
class Turn:
    role: str
    content: str


class RingBuffer:
    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._items: Deque[Turn] = deque(maxlen=max_turns)
        self._lock = asyncio.Lock()

    async def add(self, role: str, content: str) -> None:
        async with self._lock:
            self._items.append(Turn(role=role, content=content))

    async def get_all(self) -> list[Turn]:
        async with self._lock:
            return list(self._items)

    async def clear(self) -> None:
        async with self._lock:
            self._items.clear()


__all__ = ["RingBuffer", "Turn"]
