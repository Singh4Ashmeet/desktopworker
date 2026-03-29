from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from state import InvalidStateTransition, SidState, StateManager


@dataclass(slots=True)
class Orchestrator:
    config: Any
    state_manager: StateManager
    stt: Any
    tts: Any
    hud: Any
    tray: Any
    world_model: Any
    ring_buffer: Any
    vector_store: Any
    fact_store: Any
    prompt_builder: Any
    react_loop: Any
    trigger_engine: Any
    action_log: Any
    permission_wizard: Any | None = None
    interaction_logger: Any | None = None
    startup_notices: list[str] = field(default_factory=list)

    _shutdown: bool = False
    _pending_proactive_message: str | None = None

    async def run(self) -> None:
        if self.state_manager.state == SidState.OFF:
            await self.state_manager.transition(SidState.SLEEPING, reason="startup")
        await self.hud.set_state(self.state_manager.state.value)

        if self.startup_notices:
            notice = " ".join(self.startup_notices)
            await self.hud.set_state("PROACTIVE")
            await self.hud.set_proactive_text(notice)
            await self.hud.set_detail_panel("Recovery", self.startup_notices)

        while not self._shutdown:
            event = await self._wait_for_core_event()
            if event == "shutdown":
                break
            if event == "wake":
                source = self.state_manager.pending_trigger_source or "voice"
                await self._handle_wake_cycle(source)

    async def _wait_for_core_event(self) -> str:
        wake_task = asyncio.create_task(self.state_manager.wake_event.wait())
        shutdown_task = asyncio.create_task(self.state_manager.shutdown_event.wait())
        done, pending = await asyncio.wait(
            {wake_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

        if shutdown_task in done and shutdown_task.result():
            self.state_manager.clear_shutdown_signal()
            return "shutdown"
        if wake_task in done and wake_task.result():
            self.state_manager.clear_wake_signal()
            return "wake"
        return "wake"

    async def _handle_wake_cycle(self, source: str) -> None:
        await self._transition(SidState.WAKING, f"wake:{source}")
        await self.hud.sync_to_world(self.world_model.snapshot())

        proactive_message = (
            self._pending_proactive_message if source == "proactive" else None
        )
        if proactive_message:
            await self.hud.set_state("PROACTIVE")
            await self.hud.set_proactive_text(proactive_message)
            await self.tts.speak(proactive_message)
        else:
            await self.hud.set_state("AWAKE")
            await self.tts.play_wake_chime()
            await self.tts.speak("Yes?")

        await self._transition(SidState.LISTENING, f"listen:{source}")
        timeout_seconds = 10 if proactive_message else 30
        stt_result = await self.stt.listen_once(
            timeout_seconds=timeout_seconds,
            on_partial=lambda txt: asyncio.create_task(self.hud.set_transcript(txt)),
        )
        command = (stt_result.text or "").strip()
        await self.hud.set_transcript(command)

        if not command:
            await self._cooldown("no_command")
            return

        if command.lower() in {"go to sleep", "sleep", "not now"}:
            await self._cooldown("user_sleep")
            return

        if (
            self.permission_wizard is not None
            and self.permission_wizard.has_critical_issues()
        ):
            lines = self.permission_wizard.remediation_lines()
            await self.hud.set_state("PROACTIVE")
            await self.hud.set_proactive_text(
                "A required desktop permission is still missing."
            )
            await self.hud.set_detail_panel("Permissions", lines)
            await self.tts.speak(
                "I need a permission fix before I can run desktop tasks."
            )
            await self._cooldown("permissions_blocked")
            return

        await self._execute_user_command(command)

    async def _execute_user_command(self, command: str) -> None:
        await self.ring_buffer.add("user", command)
        await self.world_model.set_last_command(command)

        world = self.world_model.snapshot()
        await self.hud.sync_to_world(world)
        await self.hud.set_state("WORKING")
        await self.hud.clear_detail_panel()

        await self._transition(SidState.ACTING, "execute_command")

        memories: list[str] = []
        if self.vector_store is not None:
            try:
                memories = await self.vector_store.query_relevant(command, n=6)
            except Exception:
                logging.exception("Vector memory query failed")

        history = [
            {"role": turn.role, "content": turn.content}
            for turn in await self.ring_buffer.get_all()
        ]
        system_prompt = self.prompt_builder.build_system_prompt(
            self.config.user_name, world, []
        )

        try:
            loop_result = await self.react_loop.run_react_loop(
                user_command=command,
                world_model=world,
                memory_context=memories,
                system_prompt=system_prompt,
                conversation_history=history[:-1],
                max_iterations=self.config.react_max_steps,
            )
            final_answer = loop_result["result"]
            if loop_result["status"] == "failed":
                await self.tts.play_error_chime()
            else:
                await self.tts.play_done_chime()
            await self.tts.speak(final_answer)
            await self.ring_buffer.add("assistant", final_answer)
            if self.interaction_logger is not None:
                self.interaction_logger.log_interaction(command, final_answer, "")
            if self.vector_store is not None:
                await self.vector_store.add_interaction(
                    command, final_answer, ["react_loop"]
                )
            self.tray.set_last_task(final_answer)
        except Exception:
            logging.exception("Task execution failed")
            await self.tts.play_error_chime()
            await self.tts.speak("I hit an error while doing that.")

        await self._cooldown("task_complete")

    async def confirm_dangerous_action(self, prompt: str) -> bool:
        await self.tts.speak("Please say confirm or yes to continue.")
        result = await self.stt.listen_once(timeout_seconds=8)
        answer = (result.text or "").strip().lower()
        return answer in {"yes", "confirm", "yes do it", "confirmed"}

    async def on_proactive_trigger(self, message: str) -> None:
        if self.state_manager.state not in {SidState.SLEEPING, SidState.COOLDOWN}:
            return
        self._pending_proactive_message = message
        await self.state_manager.request_wake("proactive")

    async def shutdown(self) -> None:
        self._shutdown = True
        await self.state_manager.request_shutdown()
        if self.state_manager.state != SidState.OFF:
            try:
                await self.state_manager.transition(SidState.OFF, reason="shutdown")
            except InvalidStateTransition:
                logging.debug("State already off or shutting down")

    async def _cooldown(self, reason: str) -> None:
        try:
            await self._transition(SidState.COOLDOWN, reason)
        except InvalidStateTransition:
            pass
        await asyncio.sleep(0.2)
        await self.hud.set_transcript("")
        await self.hud.set_state("SLEEP")
        await self.hud.clear_detail_panel()
        self._pending_proactive_message = None
        await self._transition(SidState.SLEEPING, f"{reason}:sleep")

    async def _transition(self, new_state: SidState, reason: str) -> None:
        try:
            await self.state_manager.transition(new_state, reason=reason)
        except InvalidStateTransition:
            logging.debug(
                "Ignored invalid transition to %s (%s)", new_state.value, reason
            )


__all__ = ["Orchestrator"]
