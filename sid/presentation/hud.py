from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from config import SidConfig
from presentation.hud_content import get_state_html


@dataclass(slots=True)
class HUDState:
    visible: bool = False
    state: str = "SLEEP"
    transcript: str = ""
    task_lines: list[str] = field(default_factory=list)
    proactive_text: str = ""
    detail_title: str = ""
    detail_lines: list[str] = field(default_factory=list)
    display_hint: str | None = None
    active_window_bounds: dict[str, int] | None = None


class _HeadlessHUDBackend:
    def show(self, _html: str) -> None:
        return

    def hide(self) -> None:
        return

    def close(self) -> None:
        return

    def move_to_anchor(
        self, _display_hint: str | None, _bounds: dict[str, int] | None
    ) -> None:
        return


class _QtHUDBackend:
    def __init__(self, config: SidConfig) -> None:
        from PyQt6 import QtCore, QtWidgets
        from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt
        from PyQt6.QtWebEngineWidgets import QWebEngineView

        self._QtCore = QtCore
        self._QtWidgets = QtWidgets
        self._QPropertyAnimation = QPropertyAnimation
        self._QEasingCurve = QEasingCurve
        self._Qt = Qt
        self._QWebEngineView = QWebEngineView

        self._config = config
        self._app = None
        self._window = None
        self._view = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._screen_prefs = self._load_screen_preferences()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="sid-hud-qt", daemon=True
        )
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run(self) -> None:
        QtWidgets = self._QtWidgets
        Qt = self._Qt
        QWebEngineView = self._QWebEngineView

        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self._window = QtWidgets.QWidget()
        self._window.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self._window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._window.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._window.setWindowOpacity(self._config.hud_opacity)
        self._window.resize(380, 220)

        layout = QtWidgets.QVBoxLayout(self._window)
        layout.setContentsMargins(0, 0, 0, 0)
        self._view = QWebEngineView(self._window)
        self._view.setStyleSheet("background: transparent;")
        layout.addWidget(self._view)

        self._move_on_qt_thread(None, None)
        self._ready.set()
        self._window.hide()
        self._app.exec()

    def show(self, html: str) -> None:
        if not self._window or not self._view:
            return

        def _apply() -> None:
            if not self._window or not self._view:
                return
            self._view.setHtml(html)
            self._window.show()
            self._fade(self._window, 0.0, 1.0, 250)

        self._QtCore.QTimer.singleShot(0, _apply)

    def hide(self) -> None:
        if not self._window:
            return

        def _apply() -> None:
            if not self._window:
                return
            self._fade(self._window, 1.0, 0.0, 400, hide_on_finish=True)

        self._QtCore.QTimer.singleShot(0, _apply)

    def move_to_anchor(
        self, display_hint: str | None, bounds: dict[str, int] | None
    ) -> None:
        self._QtCore.QTimer.singleShot(
            0, lambda: self._move_on_qt_thread(display_hint, bounds)
        )

    def _move_on_qt_thread(
        self, display_hint: str | None, bounds: dict[str, int] | None
    ) -> None:
        if self._app is None or self._window is None:
            return
        screen = self._select_screen(display_hint, bounds)
        geometry = (
            screen.availableGeometry()
            if screen is not None
            else self._app.primaryScreen().availableGeometry()
        )
        screen_key = self._screen_key(screen)
        x, y = self._resolve_anchor_position(screen_key, geometry)
        self._window.move(x, y)
        self._persist_screen_position(screen_key, x, y)

    def _resolve_anchor_position(self, screen_key: str, geometry) -> tuple[int, int]:
        stored = self._screen_prefs.get(screen_key)
        if stored:
            return int(stored.get("x", geometry.x() + 20)), int(
                stored.get("y", geometry.y() + 20)
            )

        margin = 20
        position = str(self._config.hud_position).lower()
        if position == "bottom-left":
            return geometry.x() + margin, geometry.y() + geometry.height() - 180
        if position == "top-right":
            return geometry.x() + geometry.width() - 400, geometry.y() + margin
        if position == "top-left":
            return geometry.x() + margin, geometry.y() + margin
        return (
            geometry.x() + geometry.width() - 400,
            geometry.y() + geometry.height() - 180,
        )

    def _select_screen(self, display_hint: str | None, bounds: dict[str, int] | None):
        assert self._app is not None
        screens = list(self._app.screens()) or [self._app.primaryScreen()]
        if bounds:
            best = None
            best_area = -1
            for screen in screens:
                geo = screen.availableGeometry()
                left = max(geo.x(), int(bounds.get("x", geo.x())))
                top = max(geo.y(), int(bounds.get("y", geo.y())))
                right = min(
                    geo.x() + geo.width(),
                    int(bounds.get("x", 0)) + int(bounds.get("width", 0)),
                )
                bottom = min(
                    geo.y() + geo.height(),
                    int(bounds.get("y", 0)) + int(bounds.get("height", 0)),
                )
                area = max(0, right - left) * max(0, bottom - top)
                if area > best_area:
                    best_area = area
                    best = screen
            if best is not None and best_area >= 0:
                return best

        if display_hint:
            for screen in screens:
                if display_hint in self._screen_key(screen):
                    return screen
        return self._app.primaryScreen()

    def _screen_key(self, screen) -> str:
        if screen is None:
            return "primary"
        serial = ""
        try:
            serial = screen.serialNumber() or ""
        except Exception:
            serial = ""
        name = ""
        try:
            name = screen.name() or ""
        except Exception:
            name = ""
        geometry = screen.availableGeometry()
        return f"{serial or name or 'screen'}:{geometry.x()}:{geometry.y()}:{geometry.width()}:{geometry.height()}"

    def _fade(
        self,
        window,
        start: float,
        end: float,
        duration: int,
        hide_on_finish: bool = False,
    ) -> None:
        anim = self._QPropertyAnimation(window, b"windowOpacity")
        anim.setDuration(duration)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(self._QEasingCurve.Type.InOutQuad)
        if hide_on_finish:
            anim.finished.connect(window.hide)
        anim.start()
        window._sid_anim = anim  # type: ignore[attr-defined]

    def close(self) -> None:
        if self._app is None:
            return
        self._QtCore.QTimer.singleShot(0, self._app.quit)

    def _load_screen_preferences(self) -> dict[str, dict[str, int]]:
        path = self._config.hud_positions_file
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _persist_screen_position(self, screen_key: str, x: int, y: int) -> None:
        self._screen_prefs[screen_key] = {"x": x, "y": y}
        try:
            self._config.hud_positions_file.write_text(
                json.dumps(self._screen_prefs, indent=2), encoding="utf-8"
            )
        except Exception:
            logging.debug("Failed to persist HUD position", exc_info=True)


class HUD:
    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self._state = HUDState()
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        try:
            backend: _HeadlessHUDBackend | _QtHUDBackend = _QtHUDBackend(config)
            backend.start()
            self._backend = backend
        except Exception:
            logging.warning("PyQt6/WebEngine not available; HUD running headless")
            self._backend = _HeadlessHUDBackend()

    async def run(self) -> None:
        while True:
            action, payload = await self._queue.get()
            if action == "state":
                self._state.state = str(payload)
            elif action == "transcript":
                self._state.transcript = str(payload)
            elif action == "task":
                self._state.task_lines.append(str(payload))
                self._state.task_lines = self._state.task_lines[-5:]
            elif action == "proactive":
                self._state.proactive_text = str(payload)
            elif action == "detail":
                title, lines = payload
                self._state.detail_title = str(title)
                self._state.detail_lines = [str(line) for line in lines][-8:]
            elif action == "clear_detail":
                self._state.detail_title = ""
                self._state.detail_lines.clear()
            elif action == "anchor":
                display_hint, bounds = payload
                self._state.display_hint = display_hint
                self._state.active_window_bounds = bounds
                await asyncio.to_thread(
                    self._backend.move_to_anchor, display_hint, bounds
                )
            await self._render()

    async def _render(self) -> None:
        html = self.render_html()
        if self._state.state in {"AWAKE", "WORKING", "PROACTIVE"}:
            self._state.visible = True
            await asyncio.to_thread(self._backend.show, html)
        else:
            self._state.visible = False
            await asyncio.to_thread(self._backend.hide)

    async def set_state(self, state: str) -> None:
        mapped = {
            "SLEEPING": "SLEEP",
            "COOLDOWN": "SLEEP",
            "WAKING": "AWAKE",
            "LISTENING": "AWAKE",
            "ACTING": "WORKING",
        }.get(str(state).upper(), str(state).upper())
        await self._queue.put(("state", mapped))

    async def set_transcript(self, text: str) -> None:
        await self._queue.put(("transcript", text))

    async def add_task_line(self, line: str) -> None:
        await self._queue.put(("task", line))

    async def set_proactive_text(self, text: str) -> None:
        await self._queue.put(("proactive", text))

    async def set_detail_panel(self, title: str, lines: list[str]) -> None:
        await self._queue.put(("detail", (title, lines)))

    async def clear_detail_panel(self) -> None:
        await self._queue.put(("clear_detail", None))

    async def sync_to_world(self, world_snapshot: dict[str, Any]) -> None:
        await self._queue.put(
            (
                "anchor",
                (
                    world_snapshot.get("active_display_id"),
                    world_snapshot.get("active_window_bounds"),
                ),
            )
        )

    def render_html(self) -> str:
        state = self._state.state
        base = get_state_html(state)

        if state == "AWAKE":
            base = base.replace(
                '<div id="transcript" style="font-size:14px; opacity:0.7; min-height:20px; font-style:italic;"></div>',
                f'<div id="transcript" style="font-size:14px; opacity:0.7; min-height:20px; font-style:italic;">{self._escape(self._state.transcript)}</div>',
            )
        elif state == "WORKING":
            lines = "".join(
                f"<div>{self._escape(line)}</div>"
                for line in self._state.task_lines[-5:]
            )
            base = base.replace(
                '<div id="task-stream" style="font-size:13px; line-height:1.8; opacity:0.75;"></div>',
                f'<div id="task-stream" style="font-size:13px; line-height:1.8; opacity:0.75;">{lines}</div>',
            )
        elif state == "PROACTIVE":
            base = base.replace(
                '<span id="proactive-text" style="font-size:14px; line-height:1.6; opacity:0.85;"></span>',
                f'<span id="proactive-text" style="font-size:14px; line-height:1.6; opacity:0.85;">{self._escape(self._state.proactive_text)}</span>',
            )

        detail_panel = ""
        if self._state.detail_lines:
            title = self._escape(self._state.detail_title or "Details")
            detail_body = "".join(
                f'<div style="margin-bottom:4px;">{self._escape(line)}</div>'
                for line in self._state.detail_lines
            )
            detail_panel = f"""
            <div style="
              margin-top:10px;
              width:340px;
              background: rgba(12,16,24,0.92);
              border: 1px solid rgba(255,255,255,0.08);
              border-radius: 14px;
              padding: 10px 12px;
              color: white;
              font-family: -apple-system, 'SF Pro Display', sans-serif;
              font-size: 12px;
              line-height: 1.5;
            ">
              <div style="font-size:11px; opacity:0.55; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.08em;">{title}</div>
              {detail_body}
            </div>
            """
        return f'<div style="display:flex; flex-direction:column; align-items:flex-end;">{base}{detail_panel}</div>'

    def _escape(self, text: str) -> str:
        return (
            (text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def close(self) -> None:
        self._state.visible = False
        self._backend.close()


__all__ = ["HUD", "HUDState"]
