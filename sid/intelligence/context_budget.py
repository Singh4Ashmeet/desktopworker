from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def estimate_tokens(text: str) -> int:
    return max(1, (len(text or "") + 3) // 4)


@dataclass(slots=True)
class ContextBudgetReport:
    total_tokens: int
    limit: int
    warn_threshold: int
    drop_threshold: int


class ContextBudgetManager:
    def __init__(self, limit: int, warn_ratio: float, drop_ratio: float) -> None:
        self.limit = limit
        self.warn_threshold = int(limit * warn_ratio)
        self.drop_threshold = int(limit * drop_ratio)

    def measure(
        self, system_prompt: str, memories: list[str], messages: list[dict[str, Any]]
    ) -> ContextBudgetReport:
        total = estimate_tokens(system_prompt)
        total += sum(estimate_tokens(item) for item in memories)
        for message in messages:
            content = message.get("content", "")
            total += estimate_tokens(str(content))
        return ContextBudgetReport(
            total_tokens=total,
            limit=self.limit,
            warn_threshold=self.warn_threshold,
            drop_threshold=self.drop_threshold,
        )

    def summarize_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        if len(messages) < 6:
            return messages, None
        older = messages[1:-4]
        if not older:
            return messages, None
        summary_parts = []
        for message in older[:8]:
            content = str(message.get("content", ""))
            summary_parts.append(content[:120])
        summary = " | ".join(summary_parts)
        new_messages = [
            messages[0],
            {"role": "system", "content": f"Conversation summary: {summary}"},
        ] + messages[-4:]
        return new_messages, "summarized_oldest_turns"

    def drop_low_relevance_memories(
        self, memories: list[str]
    ) -> tuple[list[str], str | None]:
        if len(memories) <= 2:
            return memories, None
        kept = memories[: max(1, len(memories) // 2)]
        return kept, "dropped_low_relevance_memories"
