from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PermissionIssue:
    key: str
    title: str
    steps: list[str]
    deep_link: str | None
    critical: bool = True


class PermissionWizard:
    def __init__(self) -> None:
        self.issues: list[PermissionIssue] = []

    def check(self) -> list[PermissionIssue]:
        system = platform.system().lower()
        if system == "darwin":
            self.issues = self._check_macos()
        elif system == "windows":
            self.issues = self._check_windows()
        else:
            self.issues = self._check_linux()
        return list(self.issues)

    def has_critical_issues(self) -> bool:
        return any(issue.critical for issue in self.issues)

    def remediation_lines(self) -> list[str]:
        lines: list[str] = []
        for issue in self.issues:
            lines.append(issue.title)
            lines.extend(issue.steps[:3])
            if issue.deep_link:
                lines.append(f"Open: {issue.deep_link}")
        return lines[:8]

    def _check_windows(self) -> list[PermissionIssue]:
        issues: list[PermissionIssue] = []
        try:
            import ctypes

            is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            is_admin = False
        if not is_admin:
            issues.append(
                PermissionIssue(
                    key="uac",
                    title="Admin access is limited",
                    steps=[
                        "Open Windows Security settings.",
                        "Review UAC prompts before desktop automation tasks.",
                        "Relaunch Sid as administrator if app control fails.",
                    ],
                    deep_link="ms-settings:windowsdefender",
                    critical=False,
                )
            )

        try:
            import winreg

            key = winreg.CreateKey(
                winreg.HKEY_CURRENT_USER, r"Software\SidPermissionTest"
            )
            winreg.SetValueEx(key, "Probe", 0, winreg.REG_SZ, "ok")
            winreg.DeleteValue(key, "Probe")
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\SidPermissionTest")
        except Exception:
            issues.append(
                PermissionIssue(
                    key="registry-write",
                    title="Registry write access unavailable",
                    steps=[
                        "Open Settings > Privacy & security.",
                        "Check endpoint protection or policy restrictions.",
                        "Allow Sid to write under HKCU for startup integration.",
                    ],
                    deep_link="ms-settings:privacy",
                    critical=True,
                )
            )
        return issues

    def _check_macos(self) -> list[PermissionIssue]:
        issues: list[PermissionIssue] = []
        try:
            import ctypes

            app_services = ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
            )
            trusted = bool(app_services.AXIsProcessTrusted())
        except Exception:
            trusted = False
        if not trusted:
            issues.append(
                PermissionIssue(
                    key="accessibility",
                    title="Accessibility permission missing",
                    steps=[
                        "Open System Settings > Privacy & Security > Accessibility.",
                        "Enable Sid for UI automation.",
                        "Restart Sid after granting access.",
                    ],
                    deep_link="x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
                    critical=True,
                )
            )
        try:
            import Quartz  # type: ignore

            Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
            )
        except Exception:
            issues.append(
                PermissionIssue(
                    key="screen-recording",
                    title="Screen Recording permission missing",
                    steps=[
                        "Open System Settings > Privacy & Security > Screen Recording.",
                        "Enable Sid for screen monitoring.",
                        "Restart Sid so capture permissions refresh.",
                    ],
                    deep_link="x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
                    critical=True,
                )
            )
        return issues

    def _check_linux(self) -> list[PermissionIssue]:
        issues: list[PermissionIssue] = []
        session_type = os.getenv("XDG_SESSION_TYPE", "").lower()
        if not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
            issues.append(
                PermissionIssue(
                    key="display",
                    title="No desktop display session detected",
                    steps=[
                        "Log into a graphical desktop session.",
                        "Confirm DISPLAY or WAYLAND_DISPLAY is exported.",
                        "Relaunch Sid from that session.",
                    ],
                    deep_link=None,
                    critical=True,
                )
            )
        if session_type == "wayland":
            issues.append(
                PermissionIssue(
                    key="wayland-automation",
                    title="Wayland may limit UI automation",
                    steps=[
                        "Grant compositor automation permissions if supported.",
                        "Use XWayland or X11 if full desktop control is required.",
                        "Confirm screenshot and input synthesis permissions.",
                    ],
                    deep_link=None,
                    critical=False,
                )
            )
        if not Path("/dev/input").exists():
            issues.append(
                PermissionIssue(
                    key="dev-input",
                    title="Input device access unavailable",
                    steps=[
                        "Check your user group permissions for /dev/input.",
                        "Review udev rules for automation tools.",
                        "Restart Sid after permissions are fixed.",
                    ],
                    deep_link=None,
                    critical=True,
                )
            )
        return issues


__all__ = ["PermissionIssue", "PermissionWizard"]
