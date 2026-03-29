from __future__ import annotations

import asyncio
import os
import platform
import shlex
import subprocess
import time
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


async def _stream_lines(
    reader: asyncio.StreamReader, sink: list[str], hud: Any = None
) -> None:
    while True:
        line = await reader.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="ignore").rstrip()
        sink.append(text)
        if hud is not None:
            try:
                await hud.add_task_line(text)
            except Exception:
                pass


@tool(
    name="run_command",
    description="Run a shell command and capture stdout/stderr.",
    parameters_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "working_dir": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["command"],
    },
)
async def run_command(
    command: str, working_dir: str | None = None, timeout: int = 30
) -> ToolResult:
    ctx = _get_ctx()
    cwd = str(Path(working_dir).expanduser().resolve()) if working_dir else None
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    out_lines: list[str] = []
    err_lines: list[str] = []
    out_task = asyncio.create_task(_stream_lines(proc.stdout, out_lines, ctx.hud))
    err_task = asyncio.create_task(_stream_lines(proc.stderr, err_lines, None))

    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        out_task.cancel()
        err_task.cancel()
        return {
            "success": False,
            "output": f"Command timed out after {timeout}s",
            "data": None,
        }

    await asyncio.gather(out_task, err_task, return_exceptions=True)
    stdout = "\n".join(out_lines)
    stderr = "\n".join(err_lines)
    success = proc.returncode == 0
    output = stdout if success else (stderr or stdout or f"exit_code={proc.returncode}")

    return {
        "success": success,
        "output": output,
        "data": {"stdout": stdout, "stderr": stderr, "exit_code": proc.returncode},
    }


@tool(
    name="run_script",
    description="Run a script file with detected interpreter.",
    parameters_schema={
        "type": "object",
        "properties": {
            "script_path": {"type": "string"},
            "interpreter": {"type": "string"},
        },
        "required": ["script_path"],
    },
)
async def run_script(script_path: str, interpreter: str = "auto") -> ToolResult:
    path = Path(script_path).expanduser().resolve()
    if not path.exists():
        return {"success": False, "output": f"Script not found: {path}", "data": None}

    if interpreter == "auto":
        if path.suffix == ".py":
            interpreter = "python"
        elif path.suffix == ".ps1":
            interpreter = "powershell"
        elif path.suffix in {".sh", ".bash"}:
            interpreter = "bash"
        else:
            interpreter = ""

    cmd = _build_command_line([interpreter, str(path)] if interpreter else [str(path)])

    return await run_command(cmd, working_dir=str(path.parent))


def _build_command_line(parts: list[str]) -> str:
    filtered = [part for part in parts if part]
    if os.name == "nt":
        return subprocess.list2cmdline(filtered)
    return " ".join(shlex.quote(part) for part in filtered)


@tool(
    name="check_process",
    description="Check whether a process is running.",
    parameters_schema={
        "type": "object",
        "properties": {"name_or_pid": {"type": "string"}},
        "required": ["name_or_pid"],
    },
)
async def check_process(name_or_pid: str) -> ToolResult:
    try:
        import psutil  # type: ignore
    except Exception:
        return {"success": False, "output": "psutil not installed", "data": None}

    def _run() -> dict[str, Any] | None:
        if str(name_or_pid).isdigit():
            pid = int(name_or_pid)
            if not psutil.pid_exists(pid):
                return None
            p = psutil.Process(pid)
            return {
                "running": p.is_running(),
                "pid": p.pid,
                "cpu_percent": p.cpu_percent(interval=0.1),
                "memory_mb": p.memory_info().rss / (1024 * 1024),
            }

        query = name_or_pid.lower()
        for p in psutil.process_iter(["pid", "name"]):
            name = (p.info.get("name") or "").lower()
            if query in name:
                proc = psutil.Process(int(p.info["pid"]))
                return {
                    "running": proc.is_running(),
                    "pid": proc.pid,
                    "cpu_percent": proc.cpu_percent(interval=0.1),
                    "memory_mb": proc.memory_info().rss / (1024 * 1024),
                }
        return None

    info = await asyncio.to_thread(_run)
    if info is None:
        return {
            "success": False,
            "output": f"Process not found: {name_or_pid}",
            "data": None,
        }
    return {"success": True, "output": str(info), "data": info}


@tool(
    name="kill_process",
    description="Terminate process by name or pid.",
    parameters_schema={
        "type": "object",
        "properties": {"name_or_pid": {"type": "string"}},
        "required": ["name_or_pid"],
    },
)
async def kill_process(name_or_pid: str) -> ToolResult:
    try:
        import psutil  # type: ignore
    except Exception:
        return {"success": False, "output": "psutil not installed", "data": None}

    def _run() -> bool:
        if str(name_or_pid).isdigit():
            pid = int(name_or_pid)
            if not psutil.pid_exists(pid):
                return False
            psutil.Process(pid).terminate()
            return True

        query = name_or_pid.lower()
        killed = False
        for p in psutil.process_iter(["pid", "name"]):
            name = (p.info.get("name") or "").lower()
            if query in name:
                psutil.Process(int(p.info["pid"])).terminate()
                killed = True
        return killed

    ok = await asyncio.to_thread(_run)
    return {
        "success": ok,
        "output": "Process terminated" if ok else "No matching process found",
        "data": {"target": name_or_pid},
    }


@tool(
    name="list_processes",
    description="List running processes.",
    parameters_schema={"type": "object", "properties": {}},
)
async def list_processes() -> ToolResult:
    try:
        import psutil  # type: ignore
    except Exception:
        return {"success": False, "output": "psutil not installed", "data": None}

    def _run() -> list[dict[str, Any]]:
        out = []
        for proc in psutil.process_iter(["pid", "name"]):
            out.append(
                {"pid": proc.info.get("pid"), "name": proc.info.get("name") or ""}
            )
        return out[:500]

    items = await asyncio.to_thread(_run)
    preview = "\n".join(f"{item['pid']}: {item['name']}" for item in items[:100])
    return {"success": True, "output": preview, "data": {"processes": items}}


@tool(
    name="get_system_info",
    description="Get CPU, memory, disk and uptime.",
    parameters_schema={"type": "object", "properties": {}},
)
async def get_system_info() -> ToolResult:
    try:
        import psutil  # type: ignore
    except Exception:
        return {"success": False, "output": "psutil not installed", "data": None}

    def _run() -> dict[str, Any]:
        boot = psutil.boot_time()
        uptime = max(0.0, time.time() - boot)
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage(str(Path.home())).percent,
            "platform": platform.platform(),
            "uptime": uptime,
        }

    info = await asyncio.to_thread(_run)
    return {"success": True, "output": str(info), "data": info}
