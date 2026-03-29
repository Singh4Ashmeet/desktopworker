from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta

from models import ToolResult
from tools.registry import ToolContext, tool

_CTX: ToolContext | None = None
_REMINDERS: dict[str, dict[str, str | int]] = {}


def init_context(ctx: ToolContext) -> None:
    global _CTX
    _CTX = ctx


def _notify(title: str, message: str, timeout: int) -> None:
    try:
        from plyer import notification  # type: ignore

        notification.notify(title=title, message=message, timeout=timeout)
    except Exception:
        pass


@tool(
    name="send_notification",
    description="Send a desktop notification.",
    parameters_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "message": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["title", "message"],
    },
)
async def send_notification(title: str, message: str, timeout: int = 5) -> ToolResult:
    await asyncio.to_thread(_notify, title, message, timeout)
    return {"success": True, "output": "Notification sent", "data": None}


@tool(
    name="set_reminder",
    description="Schedule a reminder in N minutes.",
    parameters_schema={
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "in_minutes": {"type": "integer"},
        },
        "required": ["message", "in_minutes"],
    },
)
async def set_reminder(message: str, in_minutes: int) -> ToolResult:
    rid = str(uuid.uuid4())
    run_at = datetime.now() + timedelta(minutes=in_minutes)
    _REMINDERS[rid] = {"id": rid, "message": message, "run_at": run_at.isoformat()}

    if _CTX and _CTX.scheduler is not None:
        await _CTX.scheduler.schedule_reminder(rid, message, run_at)
    else:

        async def delayed_notify() -> None:
            await asyncio.sleep(max(1, in_minutes * 60))
            _notify("Sid Reminder", message, 8)

        asyncio.create_task(delayed_notify())

    return {
        "success": True,
        "output": f"Reminder set for {in_minutes} minute(s)",
        "data": {"reminder_id": rid},
    }


@tool(
    name="list_reminders",
    description="List pending reminders.",
    parameters_schema={"type": "object", "properties": {}},
)
async def list_reminders() -> ToolResult:
    if _CTX and _CTX.scheduler is not None:
        try:
            reminders = await _CTX.scheduler.list_reminders()
        except Exception:
            reminders = list(_REMINDERS.values())
    else:
        reminders = list(_REMINDERS.values())
    preview = "\n".join(f"{r['id']}: {r['message']} @ {r['run_at']}" for r in reminders)
    return {
        "success": True,
        "output": preview or "No reminders",
        "data": {"reminders": reminders},
    }


@tool(
    name="cancel_reminder",
    description="Cancel a scheduled reminder.",
    parameters_schema={
        "type": "object",
        "properties": {"reminder_id": {"type": "string"}},
        "required": ["reminder_id"],
    },
)
async def cancel_reminder(reminder_id: str) -> ToolResult:
    existed = _REMINDERS.pop(reminder_id, None) is not None
    if _CTX and _CTX.scheduler is not None:
        await _CTX.scheduler.cancel_reminder(reminder_id)
    return {
        "success": existed,
        "output": "Reminder cancelled" if existed else "Reminder not found",
        "data": None,
    }
