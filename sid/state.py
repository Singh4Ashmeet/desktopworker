from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from config import SidConfig


class SidState(str, Enum):
    OFF = "OFF"
    SLEEPING = "SLEEPING"
    WAKING = "WAKING"
    LISTENING = "LISTENING"
    ACTING = "ACTING"
    COOLDOWN = "COOLDOWN"


VALID_TRANSITIONS: dict[SidState, set[SidState]] = {
    SidState.OFF: {SidState.SLEEPING},
    SidState.SLEEPING: {SidState.WAKING, SidState.OFF},
    SidState.WAKING: {SidState.LISTENING, SidState.COOLDOWN, SidState.OFF},
    SidState.LISTENING: {SidState.ACTING, SidState.COOLDOWN, SidState.OFF},
    SidState.ACTING: {SidState.COOLDOWN, SidState.OFF},
    SidState.COOLDOWN: {SidState.SLEEPING, SidState.OFF},
}


@dataclass(slots=True)
class TransitionEvents:
    wake_requested: asyncio.Event
    sleep_requested: asyncio.Event
    acting_requested: asyncio.Event
    shutdown_requested: asyncio.Event


class InvalidStateTransition(RuntimeError):
    pass


class StateManager:
    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self._state = SidState.OFF
        self._transition_lock = asyncio.Lock()
        self._trigger_lock = asyncio.Lock()
        self._pending_trigger_source: str | None = None

        self.wake_event = asyncio.Event()
        self.sleep_event = asyncio.Event()
        self.working_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()

        self.transition_events = TransitionEvents(
            wake_requested=self.wake_event,
            sleep_requested=self.sleep_event,
            acting_requested=self.working_event,
            shutdown_requested=self.shutdown_event,
        )

    @property
    def state(self) -> SidState:
        return self._state

    @property
    def pending_trigger_source(self) -> str | None:
        return self._pending_trigger_source

    async def load_persisted_state(self) -> SidState:
        if not self.config.state_file.exists():
            return SidState.OFF
        try:
            payload = json.loads(self.config.state_file.read_text(encoding="utf-8"))
            raw_state = SidState(str(payload.get("state", SidState.OFF.value)))
            state = raw_state
            # Recover any in-flight state back to sleeping so boot is safe.
            if state in {
                SidState.WAKING,
                SidState.LISTENING,
                SidState.ACTING,
                SidState.COOLDOWN,
            }:
                state = SidState.SLEEPING
            self._state = state
            if state != raw_state:
                self._persist_state("crash_recovery")
            return state
        except Exception:
            logging.exception("Failed to load persisted state")
            self._state = SidState.OFF
            return self._state

    async def transition(self, new_state: SidState, reason: str = "") -> SidState:
        async with self._transition_lock:
            old_state = self._state
            if old_state == new_state:
                raise InvalidStateTransition(
                    f"Invalid self-transition: {old_state.value} -> {new_state.value}"
                )
            if new_state not in VALID_TRANSITIONS.get(old_state, set()):
                raise InvalidStateTransition(
                    f"Invalid transition: {old_state.value} -> {new_state.value}"
                )

            self._state = new_state
            self._persist_state(reason)
            logging.info(
                "State transition: %s -> %s (%s)",
                old_state.value,
                new_state.value,
                reason or "no-reason",
            )
            return new_state

    async def request_wake(self, source: str) -> bool:
        async with self._trigger_lock:
            if (
                self._state not in {SidState.SLEEPING, SidState.COOLDOWN}
                or self.wake_event.is_set()
            ):
                return False
            self._pending_trigger_source = source
            self.wake_event.set()
            return True

    async def request_sleep(self, reason: str = "") -> None:
        self._pending_trigger_source = reason or self._pending_trigger_source
        self.sleep_event.set()

    async def request_act(self) -> None:
        self.working_event.set()

    async def request_shutdown(self) -> None:
        self.shutdown_event.set()

    def clear_wake_signal(self) -> None:
        self.wake_event.clear()

    def clear_sleep_signal(self) -> None:
        self.sleep_event.clear()

    def clear_act_signal(self) -> None:
        self.working_event.clear()

    def clear_shutdown_signal(self) -> None:
        self.shutdown_event.clear()

    def _persist_state(self, reason: str) -> None:
        payload = {
            "state": self._state.value,
            "reason": reason,
        }
        self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=str(self.config.state_file.parent)
        ) as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
            temp_path = Path(handle.name)
        temp_path.replace(self.config.state_file)


__all__ = [
    "SidState",
    "StateManager",
    "TransitionEvents",
    "InvalidStateTransition",
    "VALID_TRANSITIONS",
]
