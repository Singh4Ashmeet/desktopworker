from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from typing import Any

from models import ToolResult
from tools.registry import ToolContext, tool

_CTX: ToolContext | None = None


def init_context(ctx: ToolContext) -> None:
    global _CTX
    _CTX = ctx


def _ctx() -> ToolContext:
    if _CTX is None:
        raise RuntimeError("Tool context not initialized")
    return _CTX


@tool(
    name="take_screenshot",
    description="Take a screenshot and return base64 or saved path.",
    parameters_schema={
        "type": "object",
        "properties": {
            "region": {"type": "object"},
            "save_path": {"type": "string"},
        },
    },
)
async def take_screenshot(
    region: dict[str, int] | None = None, save_path: str | None = None
) -> ToolResult:
    def _capture() -> tuple[bytes, tuple[int, int]]:
        try:
            import mss  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise RuntimeError("mss and Pillow required") from exc

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            if region:
                monitor = {
                    "left": int(region.get("x", 0)),
                    "top": int(region.get("y", 0)),
                    "width": int(region.get("width", monitor["width"])),
                    "height": int(region.get("height", monitor["height"])),
                }
            grab = sct.grab(monitor)
            image = Image.frombytes("RGB", grab.size, grab.rgb)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            return buf.getvalue(), image.size

    try:
        data, size = await asyncio.to_thread(_capture)
    except Exception as exc:
        return {"success": False, "output": f"Screenshot failed: {exc}", "data": None}

    if save_path:
        p = Path(save_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(p.write_bytes, data)
        return {
            "success": True,
            "output": str(p),
            "data": {"path": str(p), "size": size},
        }

    b64 = base64.b64encode(data).decode("ascii")
    return {
        "success": True,
        "output": "screenshot captured",
        "data": {"image_b64": b64, "size": size},
    }


@tool(
    name="click_at",
    description="Click at screen coordinates.",
    parameters_schema={
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "button": {"type": "string"},
            "double": {"type": "boolean"},
        },
        "required": ["x", "y"],
    },
)
async def click_at(
    x: int, y: int, button: str = "left", double: bool = False
) -> ToolResult:
    try:
        import pyautogui  # type: ignore
    except Exception:
        return {"success": False, "output": "pyautogui not installed", "data": None}

    def _run() -> None:
        if double:
            pyautogui.doubleClick(x=x, y=y, button=button)
        else:
            pyautogui.click(x=x, y=y, button=button)

    await asyncio.to_thread(_run)
    return {"success": True, "output": f"Clicked at ({x}, {y})", "data": None}


@tool(
    name="type_text",
    description="Type text at current cursor focus.",
    parameters_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}, "interval": {"type": "number"}},
        "required": ["text"],
    },
)
async def type_text(text: str, interval: float = 0.02) -> ToolResult:
    try:
        import pyautogui  # type: ignore
    except Exception:
        return {"success": False, "output": "pyautogui not installed", "data": None}

    await asyncio.to_thread(pyautogui.write, text, interval)
    return {"success": True, "output": "Text typed", "data": None}


@tool(
    name="press_keys",
    description="Press hotkey combinations.",
    parameters_schema={
        "type": "object",
        "properties": {"keys": {"type": "array", "items": {"type": "string"}}},
        "required": ["keys"],
    },
)
async def press_keys(keys: list[str]) -> ToolResult:
    try:
        import pyautogui  # type: ignore
    except Exception:
        return {"success": False, "output": "pyautogui not installed", "data": None}

    if len(keys) == 1 and "+" in keys[0]:
        parts = [k.strip() for k in keys[0].split("+") if k.strip()]
    else:
        parts = keys

    await asyncio.to_thread(pyautogui.hotkey, *parts)
    return {"success": True, "output": f"Pressed keys: {'+'.join(parts)}", "data": None}


@tool(
    name="scroll",
    description="Scroll at position.",
    parameters_schema={
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "amount": {"type": "integer"},
            "direction": {"type": "string"},
        },
        "required": ["x", "y", "amount"],
    },
)
async def scroll(x: int, y: int, amount: int, direction: str = "down") -> ToolResult:
    try:
        import pyautogui  # type: ignore
    except Exception:
        return {"success": False, "output": "pyautogui not installed", "data": None}

    val = -abs(amount) if direction.lower() == "down" else abs(amount)
    await asyncio.to_thread(pyautogui.moveTo, x, y)
    await asyncio.to_thread(pyautogui.scroll, val)
    return {"success": True, "output": "Scrolled", "data": None}


@tool(
    name="get_cursor_position",
    description="Get current cursor position.",
    parameters_schema={"type": "object", "properties": {}},
)
async def get_cursor_position() -> ToolResult:
    try:
        import pyautogui  # type: ignore
    except Exception:
        return {"success": False, "output": "pyautogui not installed", "data": None}

    pos = await asyncio.to_thread(pyautogui.position)
    data = {"x": int(pos.x), "y": int(pos.y)}
    return {"success": True, "output": str(data), "data": data}


@tool(
    name="get_active_window",
    description="Get active window metadata.",
    parameters_schema={"type": "object", "properties": {}},
)
async def get_active_window() -> ToolResult:
    try:
        import pygetwindow as gw  # type: ignore
    except Exception:
        return {"success": False, "output": "pygetwindow not installed", "data": None}

    def _run() -> dict[str, Any] | None:
        w = gw.getActiveWindow()
        if w is None:
            return None
        return {
            "title": w.title,
            "app": w.title.split("-")[-1].strip() if w.title else None,
            "x": w.left,
            "y": w.top,
            "width": w.width,
            "height": w.height,
        }

    data = await asyncio.to_thread(_run)
    if not data:
        return {"success": False, "output": "No active window", "data": None}
    return {"success": True, "output": str(data), "data": data}


@tool(
    name="describe_screen",
    description="Describe current screen via vision analyzer.",
    parameters_schema={"type": "object", "properties": {}},
)
async def describe_screen() -> ToolResult:
    ctx = _ctx()
    if ctx.vision_describer is not None:
        text = await ctx.vision_describer()
        return {"success": True, "output": text, "data": {"description": text}}

    shot = await take_screenshot()
    if not shot.get("success"):
        return shot
    return {
        "success": True,
        "output": "Screen captured; no analyzer configured",
        "data": shot.get("data"),
    }
