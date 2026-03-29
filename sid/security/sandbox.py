from __future__ import annotations

from pathlib import Path

from config import SidConfig


class Sandbox:
    def __init__(self, config: SidConfig) -> None:
        self.roots = [Path(p).expanduser().resolve() for p in config.sandbox_paths]

    def is_allowed(self, path: str | Path) -> bool:
        p = Path(path).expanduser().resolve()
        for root in self.roots:
            if p == root or root in p.parents:
                return True
        return False

    def enforce(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        if not self.is_allowed(p):
            raise PermissionError(f"Path not in sandbox: {p}")
        return p


__all__ = ["Sandbox"]
