from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WorldModel:
    def __init__(self, config=None) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self._candidate_window: tuple[str | None, datetime] | None = None
        now = datetime.now(timezone.utc)
        self._active_window_since = now
        self._focus_session_started_at = now
        self.data: dict[str, Any] = {
            "active_app": None,
            "active_window_title": None,
            "active_window_bounds": None,
            "active_display_id": None,
            "active_file": None,
            "open_windows": [],
            "recent_files": [],
            "clipboard_preview": None,
            "time": None,
            "date": None,
            "day_of_week": None,
            "upcoming_events": [],
            "last_command": None,
            "last_command_time": None,
            "screen_description": None,
            "screen_stability_score": 0.0,
            "error_detected": False,
            "error_summary": None,
            "focus_score": 0.0,
            "continuous_work_minutes": 0,
            "idle_minutes": 0,
            "system": {"cpu_percent": 0.0, "memory_percent": 0.0, "disk_free_gb": 0.0},
        }

    async def update_from_screen(self, description: dict[str, Any] | str) -> None:
        async with self._lock:
            if isinstance(description, str):
                self.data["screen_description"] = description
                return
            self.data["screen_description"] = description.get("activity")
            self.data["active_app"] = description.get("active_app") or self.data.get(
                "active_app"
            )
            self.data["active_file"] = description.get("visible_file")
            self.data["error_detected"] = bool(description.get("error_detected", False))
            self.data["error_summary"] = description.get("error_summary")

    async def update_screen_stability(self, stable_ticks: int) -> None:
        async with self._lock:
            score = min(
                1.0,
                stable_ticks
                / max(
                    1,
                    (self.config.screen_stability_backoff_ticks if self.config else 4),
                ),
            )
            self.data["screen_stability_score"] = score

    async def update_active_window(self) -> None:
        try:
            import pygetwindow as gw  # type: ignore
        except Exception:
            return

        def _read_window() -> dict[str, Any] | None:
            active = gw.getActiveWindow()
            if active is None or not active.title:
                return None
            return {
                "title": active.title,
                "bounds": {
                    "x": int(active.left),
                    "y": int(active.top),
                    "width": int(active.width),
                    "height": int(active.height),
                },
                "open_windows": [w.title for w in gw.getAllWindows() if w.title],
            }

        payload = await asyncio.to_thread(_read_window)
        if payload is None:
            return

        title = str(payload["title"])
        now = datetime.now(timezone.utc)
        debounce_ms = self.config.active_window_debounce_ms if self.config else 75
        transient_ms = self.config.transient_window_ignore_ms if self.config else 200

        async with self._lock:
            current_title = self.data.get("active_window_title")
            if title != current_title:
                if self._candidate_window is None or self._candidate_window[0] != title:
                    self._candidate_window = (title, now)
                    return

                elapsed_ms = (now - self._candidate_window[1]).total_seconds() * 1000.0
                if elapsed_ms < debounce_ms:
                    return
                if elapsed_ms < transient_ms:
                    return

                self.data["active_window_title"] = title
                self.data["active_window_bounds"] = payload["bounds"]
                self.data["open_windows"] = payload["open_windows"][:100]
                self.data["active_app"] = (
                    title.split("-")[-1].strip() if "-" in title else title
                )
                self.data["active_display_id"] = _display_id_from_bounds(
                    payload["bounds"]
                )
                self._active_window_since = now
                self._focus_session_started_at = now
                self._candidate_window = None
            else:
                self.data["open_windows"] = payload["open_windows"][:100]

            idle_minutes = max(
                0, int((now - self._active_window_since).total_seconds() // 60)
            )
            self.data["idle_minutes"] = idle_minutes
            if idle_minutes > 5:
                self._focus_session_started_at = now
                self.data["continuous_work_minutes"] = 0
            else:
                self.data["continuous_work_minutes"] = max(
                    0,
                    int((now - self._focus_session_started_at).total_seconds() // 60),
                )
            stability = float(self.data.get("screen_stability_score", 0.0) or 0.0)
            self.data["focus_score"] = round(min(1.0, 0.6 + (0.4 * stability)), 3)

    async def update_system_stats(self) -> None:
        try:
            import psutil  # type: ignore
        except Exception:
            return

        def _stats() -> dict[str, float]:
            vm = psutil.virtual_memory()
            du = psutil.disk_usage(str(Path.home()))
            return {
                "cpu_percent": float(psutil.cpu_percent(interval=0.1)),
                "memory_percent": float(vm.percent),
                "disk_free_gb": float(du.free / (1024**3)),
            }

        stats = await asyncio.to_thread(_stats)
        async with self._lock:
            self.data["system"] = stats

    async def update_time(self) -> None:
        now = datetime.now()
        async with self._lock:
            self.data["time"] = now.strftime("%H:%M:%S")
            self.data["date"] = now.strftime("%Y-%m-%d")
            self.data["day_of_week"] = now.strftime("%A")

    async def update_clipboard(self) -> None:
        text = ""
        try:
            import pyperclip  # type: ignore

            text = await asyncio.to_thread(pyperclip.paste)
        except Exception:
            pass
        async with self._lock:
            self.data["clipboard_preview"] = (text or "")[:100]

    async def set_last_command(self, command: str) -> None:
        async with self._lock:
            self.data["last_command"] = command
            self.data["last_command_time"] = datetime.now(timezone.utc).isoformat()

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    async def update_loop(self) -> None:
        async def window_loop() -> None:
            while True:
                await self.update_active_window()
                await asyncio.sleep(0.075)

        async def system_loop() -> None:
            while True:
                await self.update_system_stats()
                await asyncio.sleep(30)

        async def time_loop() -> None:
            while True:
                await self.update_time()
                await asyncio.sleep(60)

        async def clipboard_loop() -> None:
            while True:
                await self.update_clipboard()
                await asyncio.sleep(1)

        tasks = [
            asyncio.create_task(window_loop()),
            asyncio.create_task(system_loop()),
            asyncio.create_task(time_loop()),
            asyncio.create_task(clipboard_loop()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()


def _display_id_from_bounds(bounds: dict[str, int]) -> str:
    return f"display-{bounds.get('x', 0)}-{bounds.get('y', 0)}"


__all__ = ["WorldModel"]
