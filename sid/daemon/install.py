from __future__ import annotations

import getpass
import html
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path


def install_startup_daemon(binary_path: str) -> None:
    system = platform.system().lower()
    binary = str(Path(binary_path).expanduser().resolve())
    launch_args = _launch_args(binary)

    if system == "linux":
        service_path = Path.home() / ".config" / "systemd" / "user" / "sid.service"
        service_path.parent.mkdir(parents=True, exist_ok=True)
        exec_start = " ".join(shlex.quote(arg) for arg in launch_args)
        content = (
            "[Unit]\n"
            "Description=Sid AI Desktop Assistant\n"
            "After=network.target\n\n"
            "[Service]\n"
            f"ExecStart={exec_start}\n"
            "Restart=on-failure\n"
            "RestartSec=5\n\n"
            "[Install]\n"
            "WantedBy=default.target\n"
        )
        service_path.write_text(content, encoding="utf-8")
        subprocess.run(["systemctl", "--user", "enable", "sid"], check=False)
        subprocess.run(["systemctl", "--user", "start", "sid"], check=False)
        return

    if system == "darwin":
        plist = Path.home() / "Library" / "LaunchAgents" / "com.sid.agent.plist"
        plist.parent.mkdir(parents=True, exist_ok=True)
        program_arguments = "".join(
            f"<string>{html.escape(arg)}</string>" for arg in launch_args
        )
        content = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\"><dict>
<key>Label</key><string>com.sid.agent</string>
<key>ProgramArguments</key><array>{program_arguments}</array>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><true/>
</dict></plist>
"""
        plist.write_text(content, encoding="utf-8")
        subprocess.run(["launchctl", "load", str(plist)], check=False)
        return

    if system == "windows":
        result = subprocess.run(
            [
                "schtasks",
                "/create",
                "/tn",
                "Sid AI",
                "/tr",
                subprocess.list2cmdline(launch_args),
                "/sc",
                "onlogon",
                "/ru",
                _windows_task_user(),
                "/f",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error = (result.stderr or result.stdout).strip() or "schtasks /create failed"
            raise RuntimeError(error)
        return

    raise RuntimeError(f"Unsupported OS: {system}")


def _launch_args(binary_path: str) -> list[str]:
    binary = str(Path(binary_path).expanduser().resolve())
    if Path(binary).suffix.lower() == ".py":
        return [sys.executable, binary]
    return [binary]


def _windows_task_user() -> str:
    try:
        result = subprocess.run(
            ["whoami"], capture_output=True, text=True, check=False, timeout=5
        )
    except Exception:
        result = None
    if result and result.returncode == 0:
        user = result.stdout.strip()
        if user:
            return user

    username = os.getenv("USERNAME") or getpass.getuser()
    domain = os.getenv("USERDOMAIN") or os.getenv("COMPUTERNAME")
    if domain and username:
        return f"{domain}\\{username}"
    return username
