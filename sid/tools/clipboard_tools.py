from __future__ import annotations

import asyncio
import tkinter as tk

from models import ToolResult
from tools.registry import ToolContext, tool

_CTX: ToolContext | None = None


def init_context(ctx: ToolContext) -> None:
    global _CTX
    _CTX = ctx


@tool(
    name="read_clipboard",
    description="Read text from system clipboard.",
    parameters_schema={"type": "object", "properties": {}},
)
async def read_clipboard() -> ToolResult:
    def _read() -> str:
        try:
            import pyperclip  # type: ignore

            return pyperclip.paste() or ""
        except Exception:
            root = tk.Tk()
            root.withdraw()
            try:
                return root.clipboard_get()
            finally:
                root.destroy()

    try:
        text = await asyncio.to_thread(_read)
        return {"success": True, "output": text, "data": {"text": text}}
    except Exception as exc:
        return {
            "success": False,
            "output": f"Clipboard read failed: {exc}",
            "data": None,
        }


@tool(
    name="write_clipboard",
    description="Write text to system clipboard.",
    parameters_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
async def write_clipboard(text: str) -> ToolResult:
    def _write() -> None:
        try:
            import pyperclip  # type: ignore

            pyperclip.copy(text)
            return
        except Exception:
            root = tk.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
            root.destroy()

    try:
        await asyncio.to_thread(_write)
        return {"success": True, "output": "Clipboard updated", "data": None}
    except Exception as exc:
        return {
            "success": False,
            "output": f"Clipboard write failed: {exc}",
            "data": None,
        }
