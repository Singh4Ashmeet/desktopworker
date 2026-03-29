from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from config import SidConfig
from state import SidState, StateManager


class SystemTray:
    def __init__(self, config: SidConfig, state_manager: StateManager) -> None:
        self.config = config
        self.state_manager = state_manager
        self._icon = None
        self._running = False
        self._last_task: str = "No task has been run yet."
        self._frames: dict[SidState, list[tuple[object, float]]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_last_task(self, text: str) -> None:
        self._last_task = text

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._running = True
        await asyncio.to_thread(self._run_blocking)

    def _run_blocking(self) -> None:
        try:
            import pystray  # type: ignore
            from PIL import Image, ImageDraw  # type: ignore
        except Exception:
            logging.warning("pystray/Pillow not available; tray disabled")
            return

        self._frames = self._load_frames(Image, ImageDraw)

        def wake(_icon=None, _item=None) -> None:
            self._schedule_async(self.state_manager.request_wake("tray"))

        def last_task(_icon=None, _item=None) -> None:
            self._notify("Sid - Last task", self._last_task)

        def open_action_log(_icon=None, _item=None) -> None:
            self._open_path(self.config.logs_dir / "actions.log")

        def open_settings(_icon=None, _item=None) -> None:
            settings_path = Path.cwd() / ".env"
            if not settings_path.exists():
                settings_path = Path.cwd() / "config.env"
            self._open_path(settings_path)

        def quit_app(_icon=None, _item=None) -> None:
            self._schedule_async(self.state_manager.request_shutdown())
            self._running = False
            if self._icon:
                self._icon.stop()

        menu = pystray.Menu(
            pystray.MenuItem("Wake Sid", wake, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Last task", last_task),
            pystray.MenuItem("Action log", open_action_log),
            pystray.MenuItem("Settings", open_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", quit_app),
        )

        initial = self._frames[SidState.SLEEPING][0][0]
        self._icon = pystray.Icon("sid", initial, "Sid", menu)

        self._icon.visible = True
        if hasattr(self._icon, "run_detached"):
            self._icon.run_detached()
        else:
            threading.Thread(
                target=self._icon.run, name="sid-tray-icon", daemon=True
            ).start()
        self._icon.update_menu()
        animator = threading.Thread(
            target=self._animate_loop, name="sid-tray-anim", daemon=True
        )
        animator.start()

        while self._running:
            time.sleep(0.25)
        if self._icon:
            self._icon.stop()

    def _load_frames(self, Image, ImageDraw):
        tray_dir = Path(__file__).resolve().parents[1] / "assets" / "tray"

        def generated(color: str, arc: bool = False):
            out = []
            count = 8 if arc else 4
            for i in range(count):
                img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                if arc:
                    draw.arc(
                        (4, 4, 28, 28),
                        start=i * 45,
                        end=i * 45 + 250,
                        fill=color,
                        width=4,
                    )
                else:
                    alpha = 110 + (i * 30)
                    draw.ellipse(
                        (5, 5, 27, 27),
                        fill=(
                            int(color[1:3], 16),
                            int(color[3:5], 16),
                            int(color[5:7], 16),
                            alpha,
                        ),
                    )
                out.append(img)
            return out

        def load_or_generate(prefix: str, fallback_color: str, arc: bool = False):
            frames = []
            for frame in sorted(tray_dir.glob(f"{prefix}_*.png")):
                try:
                    frames.append(Image.open(frame).convert("RGBA"))
                except Exception:
                    continue
            return frames or generated(fallback_color, arc=arc)

        sleeping = load_or_generate("sleep", "#9ca3af")
        listening = load_or_generate("awake", "#22c55e")
        acting = load_or_generate("working", "#3b82f6", arc=True)
        cooldown = load_or_generate("sleep", "#64748b")

        return {
            SidState.OFF: [(sleeping[0], 0.5)],
            SidState.SLEEPING: [(img, 2.0 / max(1, len(sleeping))) for img in sleeping],
            SidState.WAKING: [(img, 0.3 / max(1, len(listening))) for img in listening],
            SidState.LISTENING: [
                (img, 0.3 / max(1, len(listening))) for img in listening
            ],
            SidState.ACTING: [(img, 0.8 / max(1, len(acting))) for img in acting],
            SidState.COOLDOWN: [(img, 1.2 / max(1, len(cooldown))) for img in cooldown],
        }

    def _animate_loop(self) -> None:
        idx_map: dict[SidState, int] = {state: 0 for state in self._frames}
        while self._running and self._icon is not None:
            state = self.state_manager.state
            frames = self._frames.get(state) or self._frames[SidState.SLEEPING]
            idx = idx_map.get(state, 0) % len(frames)
            image, delay = frames[idx]
            idx_map[state] = idx + 1
            self._icon.icon = image
            time.sleep(max(0.05, delay))

    def _schedule_async(self, coro) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _notify(self, title: str, message: str) -> None:
        try:
            from plyer import notification  # type: ignore

            notification.notify(title=title, message=message, timeout=5)
        except Exception:
            logging.info("%s: %s", title, message)

    def _open_path(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception:
            subprocess.run(["notepad", str(path)], check=False)

    def stop(self) -> None:
        self._running = False
        if self._icon:
            self._icon.stop()


__all__ = ["SystemTray"]
