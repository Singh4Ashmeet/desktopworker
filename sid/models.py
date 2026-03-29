from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict


class ToolResult(TypedDict):
    success: bool
    output: str
    data: dict[str, Any] | None


class LLMTextChunk(TypedDict):
    type: Literal["text"]
    text: str


class ReactLoopResult(TypedDict):
    status: Literal["completed", "aborted", "failed"]
    steps_taken: int
    result: str
    aborted_reason: str | None


@dataclass(slots=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


LLMChunk = LLMTextChunk | ToolUseBlock
