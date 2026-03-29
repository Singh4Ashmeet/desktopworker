from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import SidConfig


class ActionLog:
    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self.path = config.logs_dir / "actions.log"
        self.manifest_path = config.audit_manifest_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._verify_manifest()

    def log_startup(self, version: str) -> None:
        self._emit(
            {
                "event": "sid_started",
                "version": version,
                "config": {
                    "use_local_llm": self.config.use_local_llm,
                    "enable_vision": self.config.enable_vision,
                    "enable_proactive": self.config.enable_proactive,
                    "allowlist_paths": self.config.allowlist_paths,
                },
            }
        )

    def log_shutdown(self, uptime_seconds: int = 0) -> None:
        self._emit({"event": "sid_stopped", "uptime_seconds": uptime_seconds})
        self._rotate_if_needed()

    def log_tool_call(
        self,
        *,
        state: str,
        tool: str,
        args: dict[str, Any],
        tier: str,
        result: str,
        output_preview: str,
    ) -> None:
        self._emit(
            {
                "state": state,
                "tool": tool,
                "args": args,
                "tier": tier,
                "result": result,
                "output_preview": (output_preview or "")[:200],
            }
        )

    def log_sanitization(self, source: str, original: str, sanitized: str) -> None:
        self._emit(
            {
                "event": "sanitization",
                "source": source,
                "original": original,
                "sanitized": sanitized,
            }
        )

    def log_budget_event(self, action: str, total_tokens: int, limit: int) -> None:
        self._emit(
            {
                "event": "context_budget",
                "action": action,
                "total_tokens": total_tokens,
                "limit": limit,
            }
        )

    def log_startup_complete(self, timings: dict[str, float]) -> None:
        self._emit({"event": "startup_complete", "timings_ms": timings})

    def warn_manifest_mismatch(self, filename: str) -> None:
        self._emit({"event": "audit_manifest_mismatch", "file": filename})

    def log_session_stats(self, name: str, payload: dict[str, Any]) -> None:
        self._emit({"event": name, **payload})

    def _emit(self, payload: dict[str, Any]) -> None:
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        fd = os.open(
            str(self.path),
            os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_SYNC", 0),
            0o644,
        )
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        self._rotate_if_needed()

    def _rotate_if_needed(self) -> None:
        if not self.path.exists() or self.path.stat().st_size < 10 * 1024 * 1024:
            return
        rotated = self.path.with_name(
            f"actions-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"
        )
        self.path.replace(rotated)
        self._record_manifest(rotated)

    def _record_manifest(self, rotated: Path) -> None:
        manifest = self._load_manifest()
        digest = hashlib.sha256(rotated.read_bytes()).hexdigest()
        manifest[rotated.name] = digest
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _verify_manifest(self) -> None:
        manifest = self._load_manifest()
        for name, digest in manifest.items():
            candidate = self.path.parent / name
            if not candidate.exists():
                continue
            current = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if current != digest:
                self.warn_manifest_mismatch(name)

    def _load_manifest(self) -> dict[str, str]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}


__all__ = ["ActionLog"]
