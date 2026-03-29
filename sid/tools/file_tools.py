from __future__ import annotations

import asyncio
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from models import ToolResult
from tools.registry import ToolContext, tool

_CTX: ToolContext | None = None


def init_context(ctx: ToolContext) -> None:
    global _CTX
    _CTX = ctx


def _get_ctx() -> ToolContext:
    if _CTX is None:
        raise RuntimeError("Tool context not initialized")
    return _CTX


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


@tool(
    name="search_files",
    description="Search files by substring and extension.",
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "directory": {"type": "string"},
            "extensions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["query"],
    },
)
async def search_files(
    query: str, directory: str = "~", extensions: list[str] | None = None
) -> ToolResult:
    base = _expand(directory)

    def _run() -> list[str]:
        result: list[str] = []
        ext_set = {e.lower().lstrip(".") for e in extensions or []}
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if query.lower() not in p.name.lower():
                continue
            if ext_set and p.suffix.lower().lstrip(".") not in ext_set:
                continue
            result.append(str(p))
            if len(result) >= 500:
                break
        return result

    matches = await asyncio.to_thread(_run)
    return {
        "success": True,
        "output": "\n".join(matches[:100]),
        "data": {"paths": matches},
    }


@tool(
    name="read_file",
    description="Read a text file.",
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_chars": {"type": "integer"},
        },
        "required": ["path"],
    },
)
async def read_file(path: str, max_chars: int = 8000) -> ToolResult:
    p = _expand(path)

    def _run() -> str:
        return p.read_text(encoding="utf-8", errors="ignore")[:max_chars]

    if not p.exists() or not p.is_file():
        return {"success": False, "output": f"File not found: {p}", "data": None}
    content = await asyncio.to_thread(_run)
    return {"success": True, "output": content, "data": {"path": str(p)}}


@tool(
    name="write_file",
    description="Write or append content to file.",
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "mode": {"type": "string", "enum": ["overwrite", "append"]},
        },
        "required": ["path", "content"],
    },
)
async def write_file(path: str, content: str, mode: str = "overwrite") -> ToolResult:
    p = _expand(path)

    def _run() -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with p.open("a", encoding="utf-8") as f:
                f.write(content)
        else:
            p.write_text(content, encoding="utf-8")

    await asyncio.to_thread(_run)
    return {"success": True, "output": f"Wrote file: {p}", "data": {"path": str(p)}}


@tool(
    name="move_file",
    description="Move file/folder from source to destination.",
    parameters_schema={
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "destination": {"type": "string"},
        },
        "required": ["source", "destination"],
    },
)
async def move_file(source: str, destination: str) -> ToolResult:
    ctx = _get_ctx()
    src = _expand(source)
    dst = _expand(destination)
    if not src.exists():
        return {"success": False, "output": f"Source not found: {src}", "data": None}

    await ctx.undo_buffer.record_operation("move", str(src), str(dst))

    def _run() -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    await asyncio.to_thread(_run)
    return {
        "success": True,
        "output": f"Moved {src} -> {dst}",
        "data": {"source": str(src), "destination": str(dst)},
    }


@tool(
    name="copy_file",
    description="Copy file/folder from source to destination.",
    parameters_schema={
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "destination": {"type": "string"},
        },
        "required": ["source", "destination"],
    },
)
async def copy_file(source: str, destination: str) -> ToolResult:
    src = _expand(source)
    dst = _expand(destination)
    if not src.exists():
        return {"success": False, "output": f"Source not found: {src}", "data": None}

    def _run() -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    await asyncio.to_thread(_run)
    return {
        "success": True,
        "output": f"Copied {src} -> {dst}",
        "data": {"source": str(src), "destination": str(dst)},
    }


@tool(
    name="delete_file",
    description="Delete file/folder path.",
    parameters_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)
async def delete_file(path: str) -> ToolResult:
    ctx = _get_ctx()
    p = _expand(path)
    if not p.exists():
        return {"success": False, "output": f"Path not found: {p}", "data": None}

    await ctx.undo_buffer.record_operation("delete", str(p), str(p))

    def _run() -> None:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)

    await asyncio.to_thread(_run)
    return {"success": True, "output": f"Deleted: {p}", "data": {"path": str(p)}}


@tool(
    name="create_folder",
    description="Create folder idempotently.",
    parameters_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)
async def create_folder(path: str) -> ToolResult:
    p = _expand(path)
    await asyncio.to_thread(p.mkdir, parents=True, exist_ok=True)
    return {"success": True, "output": f"Folder ready: {p}", "data": {"path": str(p)}}


@tool(
    name="list_dir",
    description="List directory entries.",
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "show_hidden": {"type": "boolean"},
        },
        "required": ["path"],
    },
)
async def list_dir(path: str, show_hidden: bool = False) -> ToolResult:
    p = _expand(path)
    if not p.exists() or not p.is_dir():
        return {"success": False, "output": f"Directory not found: {p}", "data": None}

    def _run() -> list[dict[str, Any]]:
        items = []
        for entry in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if not show_hidden and entry.name.startswith("."):
                continue
            st = entry.stat()
            items.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "is_dir": entry.is_dir(),
                    "size": st.st_size,
                    "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
                }
            )
        return items

    entries = await asyncio.to_thread(_run)
    preview = "\n".join(
        f"{e['name']}\t{'dir' if e['is_dir'] else e['size']}\t{e['modified']}"
        for e in entries[:200]
    )
    return {"success": True, "output": preview, "data": {"entries": entries}}


@tool(
    name="get_file_info",
    description="Get file metadata.",
    parameters_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)
async def get_file_info(path: str) -> ToolResult:
    p = _expand(path)
    if not p.exists():
        return {"success": False, "output": f"Path not found: {p}", "data": None}

    def _run() -> dict[str, Any]:
        st = p.stat()
        return {
            "path": str(p),
            "type": "directory" if p.is_dir() else "file",
            "size": st.st_size,
            "created": datetime.fromtimestamp(st.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "permissions": oct(st.st_mode),
        }

    info = await asyncio.to_thread(_run)
    return {"success": True, "output": str(info), "data": info}


@tool(
    name="undo_last_file_op",
    description="Undo last destructive file operation.",
    parameters_schema={
        "type": "object",
        "properties": {"n": {"type": "integer"}},
    },
)
async def undo_last_file_op(n: int = 1) -> ToolResult:
    ctx = _get_ctx()
    results = await ctx.undo_buffer.undo_last_n(max(1, n))
    ok = any(results)
    if not ok:
        return {"success": False, "output": "No undo operation available", "data": None}
    restored = sum(1 for item in results if item)
    return {
        "success": True,
        "output": f"Undo completed for {restored} operation(s)",
        "data": {"count": restored},
    }
