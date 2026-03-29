from __future__ import annotations

import asyncio
import hashlib
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProactiveTrigger:
    key: str
    type: str
    message: str


class TriggerEngine:
    def __init__(self, config) -> None:
        self.config = config
        self._last_fired_by_type: dict[str, datetime] = {}
        self._download_events: list[datetime] = []
        self._fired_hashes: deque[str] = deque(maxlen=20)

    def record_download_event(self) -> None:
        self._download_events.append(datetime.now())
        cutoff = datetime.now() - timedelta(minutes=10)
        self._download_events = [ts for ts in self._download_events if ts >= cutoff]

    async def check_all(
        self, world_model: dict[str, Any], current_state: str = "SLEEPING"
    ) -> ProactiveTrigger | None:
        checks = [
            self._error_detection(world_model),
            self._download_flood(world_model),
            self._long_idle(world_model, current_state),
            self._focus_break(world_model),
            self._desktop_accumulation(world_model),
        ]
        for trigger in checks:
            if trigger is None:
                continue
            if self._should_fire(trigger):
                return trigger
        return None

    def _error_detection(self, world_model: dict[str, Any]) -> ProactiveTrigger | None:
        if not world_model.get("error_detected"):
            return None
        return ProactiveTrigger(
            key=str(
                world_model.get("error_summary")
                or world_model.get("active_app")
                or "error"
            ),
            type="error_detection",
            message=f"I can see an error in {world_model.get('active_app') or 'the active app'}. Want me to take a look?",
        )

    def _download_flood(self, world_model: dict[str, Any]) -> ProactiveTrigger | None:
        if len(self._download_events) < 15:
            return None
        return ProactiveTrigger(
            key=str(len(self._download_events)),
            type="download_flood",
            message="Looks like a batch of files landed in your Downloads. Want me to sort them?",
        )

    def _long_idle(
        self, world_model: dict[str, Any], current_state: str
    ) -> ProactiveTrigger | None:
        if world_model.get("idle_minutes", 0) <= 45 or current_state != "ACTING":
            return None
        return ProactiveTrigger(
            type="long_idle",
            key="idle",
            message="Still working on that. Want a status update?",
        )

    def _focus_break(self, world_model: dict[str, Any]) -> ProactiveTrigger | None:
        if not self.config.enable_focus_breaks:
            return None
        if (
            world_model.get("focus_score", 0.0) <= 0.9
            or world_model.get("continuous_work_minutes", 0) <= 90
        ):
            return None
        return ProactiveTrigger(
            type="focus_break",
            key="focus",
            message="You've been at it for 90 minutes straight. Want me to summarize where you are?",
        )

    def _desktop_accumulation(
        self, world_model: dict[str, Any]
    ) -> ProactiveTrigger | None:
        desktop = Path.home() / "Desktop"
        if not desktop.exists():
            return None
        try:
            count = len(list(desktop.iterdir()))
        except Exception:
            count = 0
        if count <= 40:
            return None
        return ProactiveTrigger(
            type="desktop_accumulation",
            key=str(count),
            message="Your Desktop has over 40 items. Want me to clean it up?",
        )

    def _should_fire(self, trigger: ProactiveTrigger) -> bool:
        now = datetime.now()
        hash_value = hashlib.sha256(
            f"{trigger.type}|{trigger.key}".encode("utf-8")
        ).hexdigest()
        if hash_value in self._fired_hashes:
            return False
        cooldown = self._cooldown_for_type(trigger.type)
        last = self._last_fired_by_type.get(trigger.type)
        if last is not None and now - last < cooldown:
            return False
        self._last_fired_by_type[trigger.type] = now
        self._fired_hashes.append(hash_value)
        return True

    def _cooldown_for_type(self, trigger_type: str) -> timedelta:
        mapping = {
            "error_detection": timedelta(minutes=5),
            "download_flood": timedelta(minutes=60),
            "long_idle": timedelta(minutes=30),
            "focus_break": timedelta(minutes=120),
            "desktop_accumulation": timedelta(days=7),
        }
        return mapping.get(trigger_type, timedelta(minutes=30))

    async def monitor_loop(self, world_model_obj, orchestrator) -> None:
        while True:
            snapshot = world_model_obj.snapshot()
            current = orchestrator.state_manager.state.value
            trigger = await self.check_all(snapshot, current_state=current)
            if trigger is not None and self.config.enable_proactive:
                await orchestrator.on_proactive_trigger(trigger.message)
            await asyncio.sleep(5)


__all__ = ["TriggerEngine", "ProactiveTrigger"]
