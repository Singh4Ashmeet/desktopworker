from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


class TaskScheduler:
    """
    APScheduler wrapper for reminders and recurring tasks.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.path = Path(config.sid_dir).expanduser().resolve() / "scheduled_tasks.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._scheduler = None
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def run(self) -> None:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore
        except Exception:
            logging.warning("apscheduler unavailable; scheduler disabled")
            while True:
                await asyncio.sleep(60)

        self._scheduler = AsyncIOScheduler()
        self._scheduler.start()
        await self._restore_jobs()

        if self.config.enable_morning_briefing:
            await self._schedule_morning_briefing()

        while True:
            await asyncio.sleep(30)

    async def schedule_reminder(
        self, reminder_id: str, message: str, when: datetime
    ) -> None:
        async with self._lock:
            self._jobs[reminder_id] = {
                "id": reminder_id,
                "kind": "reminder",
                "message": message,
                "run_at": when.isoformat(),
            }
            if self._scheduler is not None:
                self._scheduler.add_job(
                    self._fire_reminder,
                    trigger="date",
                    id=reminder_id,
                    run_date=when,
                    args=[reminder_id, message],
                    replace_existing=True,
                )
            await self._persist_jobs_locked()

    async def schedule_daily(
        self, job_id: str, hour: int, minute: int, message: str
    ) -> None:
        async with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "kind": "daily",
                "hour": int(hour),
                "minute": int(minute),
                "message": message,
            }
            if self._scheduler is not None:
                self._scheduler.add_job(
                    self._fire_notification,
                    trigger="cron",
                    id=job_id,
                    hour=int(hour),
                    minute=int(minute),
                    args=["Sid", message],
                    replace_existing=True,
                )
            await self._persist_jobs_locked()

    async def schedule_interval(self, job_id: str, hours: int, message: str) -> None:
        async with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "kind": "interval",
                "hours": int(hours),
                "message": message,
            }
            if self._scheduler is not None:
                self._scheduler.add_job(
                    self._fire_notification,
                    trigger="interval",
                    id=job_id,
                    hours=max(1, int(hours)),
                    args=["Sid", message],
                    replace_existing=True,
                )
            await self._persist_jobs_locked()

    async def cancel_reminder(self, reminder_id: str) -> None:
        async with self._lock:
            self._jobs.pop(reminder_id, None)
            if self._scheduler is not None:
                try:
                    self._scheduler.remove_job(reminder_id)
                except Exception:
                    pass
            await self._persist_jobs_locked()

    async def list_reminders(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [
                v.copy() for v in self._jobs.values() if v.get("kind") == "reminder"
            ]

    async def _restore_jobs(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = await asyncio.to_thread(self.path.read_text, "utf-8")
            payload = json.loads(raw)
        except Exception:
            logging.exception("Failed restoring scheduled tasks")
            return

        jobs = payload.get("jobs", [])
        for job in jobs:
            kind = str(job.get("kind", ""))
            job_id = str(job.get("id", "")).strip()
            if not job_id:
                continue
            self._jobs[job_id] = job
            if self._scheduler is None:
                continue
            try:
                if kind == "reminder":
                    when = datetime.fromisoformat(str(job["run_at"]))
                    if when > datetime.now():
                        self._scheduler.add_job(
                            self._fire_reminder,
                            trigger="date",
                            id=job_id,
                            run_date=when,
                            args=[job_id, str(job.get("message", "Reminder"))],
                            replace_existing=True,
                        )
                elif kind == "daily":
                    self._scheduler.add_job(
                        self._fire_notification,
                        trigger="cron",
                        id=job_id,
                        hour=int(job.get("hour", 8)),
                        minute=int(job.get("minute", 30)),
                        args=["Sid", str(job.get("message", "Scheduled task"))],
                        replace_existing=True,
                    )
                elif kind == "interval":
                    self._scheduler.add_job(
                        self._fire_notification,
                        trigger="interval",
                        id=job_id,
                        hours=max(1, int(job.get("hours", 1))),
                        args=["Sid", str(job.get("message", "Scheduled task"))],
                        replace_existing=True,
                    )
            except Exception:
                logging.exception("Failed to restore scheduled job: %s", job_id)

    async def _persist_jobs_locked(self) -> None:
        payload = {"jobs": list(self._jobs.values())}
        await asyncio.to_thread(
            self.path.write_text, json.dumps(payload, indent=2), "utf-8"
        )

    async def _schedule_morning_briefing(self) -> None:
        if self._scheduler is None:
            return
        try:
            hh, mm = str(self.config.briefing_time).split(":", 1)
            hour = int(hh)
            minute = int(mm)
        except Exception:
            hour, minute = 8, 30

        self._jobs["morning-briefing"] = {
            "id": "morning-briefing",
            "kind": "daily",
            "hour": hour,
            "minute": minute,
            "message": "Morning briefing",
        }
        self._scheduler.add_job(
            self._fire_morning_briefing,
            trigger="cron",
            id="morning-briefing",
            hour=hour,
            minute=minute,
            replace_existing=True,
        )
        await self._persist_jobs_locked()

    def _fire_reminder(self, reminder_id: str, message: str) -> None:
        self._fire_notification("Sid Reminder", message)
        self._jobs.pop(reminder_id, None)

    def _fire_notification(self, title: str, message: str) -> None:
        try:
            from plyer import notification  # type: ignore

            notification.notify(title=title, message=message, timeout=8)
        except Exception:
            logging.info("%s: %s", title, message)

    def _fire_morning_briefing(self) -> None:
        self._fire_notification("Sid", "Good morning. Your briefing is ready.")


__all__ = ["TaskScheduler"]
