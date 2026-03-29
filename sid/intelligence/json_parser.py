from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
BRACE_PAIRS = {"{": "}", "[": "]", "(": ")"}


@dataclass(slots=True)
class ParsedAction:
    thought: str
    action: str | None
    action_input: dict[str, Any]
    final_answer: str | None


def extract_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty model output")

    fence = JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # Try bracket-matching recovery: find { and count matching }
    recovered = _recover_json_by_bracket_matching(text)
    if recovered:
        try:
            return json.loads(recovered)
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON object found in model output")


def _recover_json_by_bracket_matching(text: str) -> str | None:
    """Attempt to recover valid JSON by matching braces, brackets, and quotes."""
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    stack: list[str] = []
    in_string = False
    escape_next = False
    end_idx = start_idx

    for i in range(start_idx, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            end_idx = i
            continue

        if char == "\\":
            escape_next = True
            end_idx = i
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            end_idx = i
            continue

        if in_string:
            end_idx = i
            continue

        if char in BRACE_PAIRS:
            stack.append(char)
            end_idx = i
        elif char in BRACE_PAIRS.values():
            if not stack:
                break
            expected_open = [k for k, v in BRACE_PAIRS.items() if v == char][0]
            if stack[-1] == expected_open:
                stack.pop()
                end_idx = i
            else:
                break

    if not stack:
        return text[start_idx : end_idx + 1]
    return None


def parse_react_json(raw: str) -> ParsedAction:
    payload = extract_json_object(raw)
    thought = str(payload.get("thought", "") or "")
    action = payload.get("action")
    final_answer = payload.get("final_answer")
    action_input = payload.get("action_input") or {}
    if action is not None and not isinstance(action_input, dict):
        raise ValueError("action_input must be an object")
    return ParsedAction(
        thought=thought,
        action=str(action) if action is not None else None,
        action_input=dict(action_input),
        final_answer=str(final_answer) if final_answer is not None else None,
    )


__all__ = ["ParsedAction", "extract_json_object", "parse_react_json"]
