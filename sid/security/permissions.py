from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any

from config import SidConfig


class PermissionTier(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    DANGER = "danger"
    BLOCKED = "blocked"


SAFE_TOOLS = {
    "read_file",
    "search_files",
    "list_dir",
    "get_file_info",
    "take_screenshot",
    "get_active_window",
    "describe_screen",
    "read_clipboard",
    "search_web",
    "fetch_url",
    "list_windows",
    "get_system_info",
    "check_process",
    "send_notification",
    "recall_fact",
    "search_memory",
    "list_facts",
    "get_cursor_position",
}

CAUTION_TOOLS = {
    "write_file",
    "create_folder",
    "copy_file",
    "open_app",
    "close_app",
    "kill_process",
    "run_command",
    "write_clipboard",
    "set_reminder",
    "remember_fact",
    "forget_fact",
    "move_file",
}

DANGER_TOOLS = {
    "delete_file",
    "undo_last_file_op",
}

DANGER_COMMAND_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bsudo\s+rm\b",
    r"\bdel\s+/f\b",
    r"\bformat\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bshutdown\b",
    r"\breboot\b",
]

CAUTION_COMMAND_PATTERNS = [
    r"\brm\b",
    r"\bdel\b",
    r"\bformat\b",
    r"\bsudo\b",
    r"chmod\s+777",
    r"\bmkfs\b",
    r"\bdd\b",
]

SENSITIVE_BLOCKED_PARTS = [
    ".ssh",
    ".gnupg",
    "AppData\\Local\\Google\\Chrome\\User Data",
    "AppData\\Local\\Microsoft\\Edge\\User Data",
]

SYSTEM_PATH_HINTS = [
    "/etc",
    "/sys",
    "/bin",
    "C:\\Windows",
    "C:\\System32",
]


class PermissionChecker:
    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self._sid_source_root = Path(__file__).resolve().parents[1]
        self._allowlist = [
            Path(item).expanduser().resolve() for item in config.allowlist_paths
        ]

    def check(self, tool_name: str, kwargs: dict[str, Any]) -> PermissionTier:
        blocked, _ = self.is_blocked(tool_name, kwargs)
        if blocked:
            return PermissionTier.BLOCKED

        if self._outside_allowlist(kwargs):
            return PermissionTier.DANGER

        if tool_name == "run_command":
            cmd = str(kwargs.get("command", ""))
            if any(
                re.search(pat, cmd, re.IGNORECASE) for pat in DANGER_COMMAND_PATTERNS
            ):
                return PermissionTier.DANGER
            if any(
                re.search(pat, cmd, re.IGNORECASE) for pat in CAUTION_COMMAND_PATTERNS
            ):
                return PermissionTier.CAUTION
            return PermissionTier.CAUTION

        if tool_name == "move_file":
            src = Path(str(kwargs.get("source", ""))).expanduser().resolve()
            dst = Path(str(kwargs.get("destination", ""))).expanduser().resolve()
            return (
                PermissionTier.CAUTION
                if src.parent != dst.parent
                else PermissionTier.SAFE
            )

        if tool_name in SAFE_TOOLS:
            return PermissionTier.SAFE
        if tool_name in DANGER_TOOLS:
            return PermissionTier.DANGER
        if tool_name in CAUTION_TOOLS:
            return PermissionTier.CAUTION
        return PermissionTier.CAUTION

    def is_blocked(self, tool_name: str, kwargs: dict[str, Any]) -> tuple[bool, str]:
        command = str(kwargs.get("command", ""))
        normalized_cmd = command.lower().strip()
        if tool_name == "run_command":
            if re.search(r"\brm\s+-rf\s+/$", normalized_cmd):
                return True, "Blocked: refusing catastrophic delete command."
            if "format c:" in normalized_cmd:
                return True, "Blocked: refusing catastrophic format command."
            if self._looks_like_exfiltration(command):
                return True, "Blocked: potential data exfiltration detected."

        for path in self._extract_paths(kwargs):
            raw = str(path)
            if any(part.lower() in raw.lower() for part in SENSITIVE_BLOCKED_PARTS):
                return True, "Blocked: sensitive path access denied."
            if self._looks_like_sid_source_path(path):
                return (
                    True,
                    "Blocked: modifying Sid source/config requires explicit user request.",
                )
            if any(raw.lower().startswith(h.lower()) for h in SYSTEM_PATH_HINTS):
                if tool_name in {
                    "delete_file",
                    "run_command",
                    "write_file",
                    "move_file",
                    "copy_file",
                }:
                    return True, "Blocked: modifying system paths is not allowed."

        return False, ""

    def _outside_allowlist(self, kwargs: dict[str, Any]) -> bool:
        paths = self._extract_paths(kwargs)
        if not paths:
            return False
        return any(not self._inside_allowlist(path) for path in paths)

    def _inside_allowlist(self, path: Path) -> bool:
        resolved = path.expanduser().resolve()
        for root in self._allowlist:
            if resolved == root or root in resolved.parents:
                return True
        return False

    def _extract_paths(self, kwargs: dict[str, Any]) -> list[Path]:
        keys = {
            "path",
            "source",
            "destination",
            "working_dir",
            "script_path",
            "directory",
        }
        paths: list[Path] = []
        for key, value in kwargs.items():
            if key not in keys:
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            paths.append(Path(value).expanduser())
        return paths

    def _looks_like_sid_source_path(self, path: Path) -> bool:
        resolved = path.expanduser().resolve()
        return (
            resolved == self._sid_source_root
            or self._sid_source_root in resolved.parents
        )

    def _looks_like_exfiltration(self, command: str) -> bool:
        lowered = command.lower()
        if "curl" in lowered or "wget" in lowered or "invoke-webrequest" in lowered:
            if any(
                token in lowered
                for token in [".ssh", "token", "password", "browser", "cookies"]
            ):
                return True
        return False


__all__ = ["PermissionTier", "PermissionChecker"]
