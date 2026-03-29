from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from config import SidConfig


async def run_tool_in_subprocess(
    tool_name: str, kwargs: dict[str, Any], config: SidConfig
) -> dict[str, Any]:
    creationflags = 0
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        popen_kwargs["start_new_session"] = True

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "security.sandbox_runner",
        "--tool",
        tool_name,
        "--kwargs",
        json.dumps(kwargs),
        "--sid-dir",
        config.sid_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parents[1]),
        creationflags=creationflags,
        **popen_kwargs,
    )

    limit_state: dict[str, str] = {}
    monitor = asyncio.create_task(
        _monitor_process_limits(proc.pid, config, limit_state)
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=config.danger_tool_timeout_seconds,
        )
    except asyncio.TimeoutError:
        await _kill_process_tree(proc.pid)
        await proc.wait()
        monitor.cancel()
        return {
            "success": False,
            "output": f"Danger tool timed out after {config.danger_tool_timeout_seconds}s",
            "data": {"stdout": "", "stderr": "", "exit_code": -9},
        }

    monitor.cancel()
    raw_out = stdout.decode("utf-8", errors="ignore").strip()
    raw_err = stderr.decode("utf-8", errors="ignore").strip()

    if limit_state.get("reason"):
        return {
            "success": False,
            "output": limit_state["reason"],
            "data": {
                "stdout": raw_out[:2000],
                "stderr": raw_err[:2000],
                "exit_code": proc.returncode,
            },
        }

    try:
        result = json.loads(raw_out) if raw_out else {}
        if isinstance(result, dict):
            result.setdefault("data", {})
            if isinstance(result["data"], dict):
                result["data"].setdefault("sandbox_stdout", raw_out[:2000])
                result["data"].setdefault("sandbox_stderr", raw_err[:2000])
                result["data"].setdefault("sandbox_exit_code", proc.returncode)
            return result
    except json.JSONDecodeError:
        pass

    return {
        "success": False,
        "output": "Sandbox subprocess returned invalid output",
        "data": {
            "stdout": raw_out[:2000],
            "stderr": raw_err[:2000],
            "exit_code": proc.returncode,
        },
    }


async def _monitor_process_limits(
    pid: int, config: SidConfig, limit_state: dict[str, str]
) -> None:
    try:
        import psutil  # type: ignore
    except Exception:
        return

    try:
        proc = psutil.Process(pid)
    except Exception:
        return

    cpu_count = max(1, int(psutil.cpu_count() or 1))
    primed = False
    consecutive_violations = 0
    while True:
        if not proc.is_running():
            return
        try:
            family = [proc] + proc.children(recursive=True)
            rss_mb = sum((p.memory_info().rss for p in family), 0) / (1024 * 1024)
            if not primed:
                for child in family:
                    child.cpu_percent(interval=None)
                primed = True
                await asyncio.sleep(0.2)
                continue

            cpu = sum((p.cpu_percent(interval=None) for p in family), 0.0) / cpu_count
            over_limit = (
                rss_mb > config.sandbox_max_memory_mb
                or cpu > config.sandbox_max_cpu_percent
            )
            consecutive_violations = consecutive_violations + 1 if over_limit else 0
            if consecutive_violations >= 3:
                limit_state["reason"] = (
                    f"Danger tool exceeded sandbox limits "
                    f"(memory={rss_mb:.1f}MB cpu={cpu:.1f}%)."
                )
                await _kill_process_tree(pid)
                return
        except Exception:
            return
        await asyncio.sleep(0.2)


async def _kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        kill = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await kill.wait()
        return

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
