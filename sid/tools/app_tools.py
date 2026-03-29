from __future__ import annotations

import asyncio
import ctypes
import os
import subprocess
from typing import Any

from models import ToolResult
from tools.registry import ToolContext, tool

_CTX: ToolContext | None = None


APP_ALIASES = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "vscode": "code",
    "visual studio code": "code",
    "notepad": "notepad",
    "powershell": "powershell",
    "cmd": "cmd",
}


def init_context(ctx: ToolContext) -> None:
    global _CTX
    _CTX = ctx


@tool(
    name="open_app",
    description="Open an application by name.",
    parameters_schema={
        "type": "object",
        "properties": {"app_name": {"type": "string"}},
        "required": ["app_name"],
    },
)
async def open_app(app_name: str) -> ToolResult:
    mapped = APP_ALIASES.get(app_name.lower(), app_name)

    def _run() -> int:
        try:
            proc = subprocess.Popen(mapped, shell=True)
            return proc.pid
        except Exception:
            try:
                os.startfile(mapped)  # type: ignore[attr-defined]
                return -1
            except Exception as exc:
                raise RuntimeError(str(exc)) from exc

    try:
        pid = await asyncio.to_thread(_run)
        return {"success": True, "output": f"Opened {app_name}", "data": {"pid": pid}}
    except Exception as exc:
        return {
            "success": False,
            "output": f"Failed to open {app_name}: {exc}",
            "data": None,
        }


@tool(
    name="close_app",
    description="Close app by name.",
    parameters_schema={
        "type": "object",
        "properties": {"app_name": {"type": "string"}},
        "required": ["app_name"],
    },
)
async def close_app(app_name: str) -> ToolResult:
    try:
        import psutil  # type: ignore
    except Exception:
        return {"success": False, "output": "psutil not installed", "data": None}

    query = app_name.lower()

    def _run() -> int:
        count = 0
        for proc in psutil.process_iter(["name"]):
            name = (proc.info.get("name") or "").lower()
            if query in name:
                proc.terminate()
                count += 1
        return count

    count = await asyncio.to_thread(_run)
    if count == 0:
        return {
            "success": False,
            "output": f"No process found for {app_name}",
            "data": None,
        }
    return {
        "success": True,
        "output": f"Closed {count} process(es) for {app_name}",
        "data": {"count": count},
    }


@tool(
    name="switch_window",
    description="Activate a window by title substring.",
    parameters_schema={
        "type": "object",
        "properties": {"title_contains": {"type": "string"}},
        "required": ["title_contains"],
    },
)
async def switch_window(title_contains: str) -> ToolResult:
    try:
        import pygetwindow as gw  # type: ignore
    except Exception:
        return {"success": False, "output": "pygetwindow not installed", "data": None}

    def _run() -> bool:
        for window in gw.getAllWindows():
            title = window.title or ""
            if title_contains.lower() in title.lower():
                try:
                    window.activate()
                    return True
                except Exception:
                    return _activate_window_fallback(window)
        return False

    ok = await asyncio.to_thread(_run)
    return {
        "success": ok,
        "output": "Window switched" if ok else "Window not found",
        "data": None,
    }


def _activate_window_fallback(window: Any) -> bool:
    hwnd = getattr(window, "_hWnd", None)
    if not hwnd or os.name != "nt":
        return False

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    current_thread = kernel32.GetCurrentThreadId()
    foreground = user32.GetForegroundWindow()
    foreground_thread = (
        user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
    )
    target_thread = user32.GetWindowThreadProcessId(hwnd, None)
    attached: list[int] = []

    for thread_id in {foreground_thread, target_thread}:
        if thread_id and thread_id != current_thread:
            if user32.AttachThreadInput(thread_id, current_thread, True):
                attached.append(thread_id)

    try:
        user32.ShowWindow(hwnd, 9)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.SetFocus(hwnd)
        user32.SetActiveWindow(hwnd)
        return user32.GetForegroundWindow() == hwnd
    finally:
        for thread_id in attached:
            user32.AttachThreadInput(thread_id, current_thread, False)


@tool(
    name="list_windows",
    description="List top-level windows.",
    parameters_schema={"type": "object", "properties": {}},
)
async def list_windows() -> ToolResult:
    try:
        import pygetwindow as gw  # type: ignore
    except Exception:
        return {"success": False, "output": "pygetwindow not installed", "data": None}

    def _run() -> list[dict[str, Any]]:
        out = []
        for w in gw.getAllWindows():
            if not w.title:
                continue
            out.append(
                {
                    "title": w.title,
                    "app": w.title.split("-")[-1].strip()
                    if "-" in w.title
                    else w.title,
                    "pid": None,
                }
            )
        return out

    windows = await asyncio.to_thread(_run)
    preview = "\n".join(w["title"] for w in windows[:200])
    return {"success": True, "output": preview, "data": {"windows": windows}}


@tool(
    name="minimize_window",
    description="Minimize window by title substring.",
    parameters_schema={
        "type": "object",
        "properties": {"title_contains": {"type": "string"}},
        "required": ["title_contains"],
    },
)
async def minimize_window(title_contains: str) -> ToolResult:
    try:
        import pygetwindow as gw  # type: ignore
    except Exception:
        return {"success": False, "output": "pygetwindow not installed", "data": None}

    def _run() -> bool:
        for w in gw.getAllWindows():
            if title_contains.lower() in (w.title or "").lower():
                w.minimize()
                return True
        return False

    ok = await asyncio.to_thread(_run)
    return {
        "success": ok,
        "output": "Window minimized" if ok else "Window not found",
        "data": None,
    }


@tool(
    name="maximize_window",
    description="Maximize window by title substring.",
    parameters_schema={
        "type": "object",
        "properties": {"title_contains": {"type": "string"}},
        "required": ["title_contains"],
    },
)
async def maximize_window(title_contains: str) -> ToolResult:
    try:
        import pygetwindow as gw  # type: ignore
    except Exception:
        return {"success": False, "output": "pygetwindow not installed", "data": None}

    def _run() -> bool:
        for w in gw.getAllWindows():
            if title_contains.lower() in (w.title or "").lower():
                w.maximize()
                return True
        return False

    ok = await asyncio.to_thread(_run)
    return {
        "success": ok,
        "output": "Window maximized" if ok else "Window not found",
        "data": None,
    }
