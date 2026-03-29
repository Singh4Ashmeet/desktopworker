from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from daemon.install import install_startup_daemon


def test_install_startup_daemon_windows_uses_python_and_current_user(
    tmp_path: Path, monkeypatch
):
    script_path = tmp_path / "main.py"
    script_path.write_text("print('sid')\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        del kwargs
        calls.append(args)
        if args == ["whoami"]:
            return subprocess.CompletedProcess(args, 0, "DESKTOP\\admin\n", "")
        if args[0] == "schtasks":
            return subprocess.CompletedProcess(args, 0, "SUCCESS", "")
        raise AssertionError(f"Unexpected subprocess call: {args}")

    monkeypatch.setattr("daemon.install.platform.system", lambda: "Windows")
    monkeypatch.setattr("daemon.install.subprocess.run", fake_run)

    install_startup_daemon(str(script_path))

    schtasks_args = calls[-1]
    task_command = schtasks_args[schtasks_args.index("/tr") + 1]
    task_user = schtasks_args[schtasks_args.index("/ru") + 1]

    assert task_user == "DESKTOP\\admin"
    assert task_command == subprocess.list2cmdline(
        [sys.executable, str(script_path.resolve())]
    )
