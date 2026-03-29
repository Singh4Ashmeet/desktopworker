from __future__ import annotations

import re
from dataclasses import dataclass


INJECTION_PATTERNS = [
    re.compile(r"ignore\s+previous\s+instructions?", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"\[/INST\]", re.IGNORECASE),
    re.compile(r"^\s*system\s*:\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*assistant\s*:\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*developer\s*:\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\n\s*Human\s*:\s*", re.IGNORECASE),
    re.compile(r"\n\s*Assistant\s*:\s*", re.IGNORECASE),
]


@dataclass(slots=True)
class SanitizationResult:
    original: str
    sanitized: str
    changed: bool


def sanitize_external_text(role: str, text: str) -> SanitizationResult:
    original = text or ""
    sanitized = original
    for pattern in INJECTION_PATTERNS:
        sanitized = pattern.sub("[sanitized]", sanitized)
    sanitized = sanitized.replace("<<", "&lt;&lt;").replace(">>", "&gt;&gt;")
    wrapped = f'<external_content role="{role}">\n{sanitized}\n</external_content>'
    return SanitizationResult(
        original=original, sanitized=wrapped, changed=(wrapped != original)
    )
