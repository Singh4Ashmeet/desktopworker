from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from config import SidConfig
from intelligence.context_budget import ContextBudgetManager
from intelligence.json_parser import ParsedAction
from models import ReactLoopResult, ToolUseBlock
from presentation.hud_summary import summarize_for_hud
from security.injection_firewall import sanitize_external_text
from security.permissions import PermissionChecker, PermissionTier
from tools.registry import ToolRegistry


ConfirmCallback = Callable[[str], Awaitable[bool]]


@dataclass(slots=True)
class ReactLoop:
    config: SidConfig
    llm_client: Any
    tool_registry: ToolRegistry
    permission_checker: PermissionChecker
    hud: Any
    tts: Any
    action_log: Any
    confirm_callback: ConfirmCallback | None = None

    async def run_react_loop(
        self,
        user_command: str,
        world_model: dict,
        memory_context: list[str],
        system_prompt: str,
        conversation_history: list[dict[str, str]] | None = None,
        max_iterations: int | None = None,
    ) -> ReactLoopResult:
        step_cap = max_iterations or self.config.react_max_steps
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for turn in conversation_history or []:
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            messages.append({"role": role, "content": content})
        if memory_context:
            messages.append(
                {
                    "role": "user",
                    "content": "Relevant memories:\n"
                    + "\n".join(f"- {item}" for item in memory_context[-8:]),
                }
            )
        messages.append({"role": "user", "content": user_command})
        memories = list(memory_context)
        repeated_calls: deque[str] = deque(maxlen=3)
        budget = ContextBudgetManager(
            self.config.current_context_limit,
            self.config.context_warn_ratio,
            self.config.context_drop_ratio,
        )

        for step in range(1, step_cap + 1):
            messages, memories = await self._enforce_context_budget(
                system_prompt, memories, messages, budget
            )
            tools = self.tool_registry.get_all_schemas()
            parsed, raw_output = await self.llm_client.react_step(
                system_prompt, messages[1:], tools, max_tokens=1024
            )
            if self.hud is not None and parsed.thought:
                await self.hud.add_task_line(parsed.thought)

            tool_calls = self._tool_calls_from_parsed(parsed, tools)
            if parsed.action and not tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "The previous tool call was invalid. Use a registered tool name and provide every required field in action_input.",
                    }
                )
                continue

            if not tool_calls:
                final_text = (parsed.final_answer or "").strip() or "Done."
                return {
                    "status": "completed",
                    "steps_taken": step,
                    "result": final_text,
                    "aborted_reason": None,
                }

            identical_abort = self._check_repeated_calls(tool_calls, repeated_calls)
            if identical_abort is not None:
                return {
                    "status": "aborted",
                    "steps_taken": step,
                    "result": identical_abort,
                    "aborted_reason": "identical_tool_calls",
                }

            if self.config.react_dry_run:
                plan_text = ", ".join(call.name for call in tool_calls)
                return {
                    "status": "aborted",
                    "steps_taken": step,
                    "result": f"Dry-run parallel execution plan: {plan_text}",
                    "aborted_reason": "dry_run",
                }

            results = await self._execute_tool_batch(tool_calls)

            for call, result in results:
                hud_line = summarize_for_hud(call.name, result)
                if self.hud is not None:
                    await self.hud.add_task_line(hud_line)
                    output = str(result.get("output", "") or "").strip()
                    if len(output) > 80:
                        await self.hud.set_detail_panel(
                            call.name, output.splitlines()[:6]
                        )

                sanitized = sanitize_external_text(
                    call.name, json.dumps(result, ensure_ascii=False)
                )
                if sanitized.changed and self.action_log is not None:
                    self.action_log.log_sanitization(
                        call.name, sanitized.original[:500], sanitized.sanitized[:500]
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "thought": parsed.thought,
                                "action": call.name,
                                "action_input": call.input,
                                "raw_output": raw_output,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool result for {call.name}:\n{sanitized.sanitized}",
                    }
                )

        partial = (
            "I reached the step cap and stopped. Here is the latest partial result."
        )
        return {
            "status": "aborted",
            "steps_taken": step_cap,
            "result": partial,
            "aborted_reason": "step_cap_reached",
        }

    async def _execute_tool_batch(
        self, tool_calls: list[ToolUseBlock]
    ) -> list[tuple[ToolUseBlock, dict[str, Any]]]:
        if len(tool_calls) == 1:
            call = tool_calls[0]
            result = await self._execute_tool_call(call)
            return [(call, result)]

        async def _run_one(call: ToolUseBlock) -> tuple[ToolUseBlock, dict[str, Any]]:
            return call, await self._execute_tool_call(call)

        gathered = await asyncio.gather(
            *[_run_one(call) for call in tool_calls], return_exceptions=False
        )
        return list(gathered)

    async def _execute_tool_call(self, call: ToolUseBlock) -> dict[str, Any]:
        tier = self.permission_checker.check(call.name, call.input)
        if tier == PermissionTier.BLOCKED:
            return {"success": False, "output": "Blocked by policy", "data": None}

        if tier == PermissionTier.DANGER:
            if self.tts is not None:
                await self.tts.speak("I need confirmation for that")
            approved = False
            if self.confirm_callback is not None:
                approved = await self.confirm_callback(f"confirm:{call.name}")
            if not approved:
                return {
                    "success": False,
                    "output": "Dangerous action cancelled",
                    "data": None,
                }

        if tier == PermissionTier.CAUTION and self.tts is not None:
            await self.tts.speak(f"I am about to run {call.name}")
            await asyncio.sleep(2)

        result = await self.tool_registry.execute(
            call.name, dict(call.input), self.permission_checker
        )
        if result.get("success"):
            return dict(result)

        await asyncio.sleep(0.4)
        retry = await self.tool_registry.execute(
            call.name, dict(call.input), self.permission_checker
        )
        return dict(retry)

    async def _enforce_context_budget(
        self,
        system_prompt: str,
        memories: list[str],
        messages: list[dict[str, Any]],
        budget: ContextBudgetManager,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        report = budget.measure(system_prompt, memories, messages)
        if report.total_tokens >= report.warn_threshold:
            messages, action = budget.summarize_messages(messages)
            if action and self.action_log is not None:
                self.action_log.log_budget_event(
                    action, report.total_tokens, report.limit
                )

        report = budget.measure(system_prompt, memories, messages)
        if report.total_tokens >= report.drop_threshold:
            memories, action = budget.drop_low_relevance_memories(memories)
            if action and self.action_log is not None:
                self.action_log.log_budget_event(
                    action, report.total_tokens, report.limit
                )
            messages = [
                msg
                for msg in messages
                if not (
                    msg.get("role") == "user"
                    and str(msg.get("content", "")).startswith("Relevant memories:\n")
                )
            ]
            if memories:
                messages.insert(
                    1,
                    {
                        "role": "user",
                        "content": "Relevant memories:\n"
                        + "\n".join(f"- {item}" for item in memories[-8:]),
                    },
                )
        return messages, memories

    def _check_repeated_calls(
        self, tool_calls: list[ToolUseBlock], repeated_calls: deque[str]
    ) -> str | None:
        for call in tool_calls:
            signature = f"{call.name}:{json.dumps(call.input, sort_keys=True)}"
            repeated_calls.append(signature)
        if len(repeated_calls) == 3 and len(set(repeated_calls)) == 1:
            return "I stopped because the last three tool calls were identical."
        return None

    def _tool_calls_from_parsed(
        self, parsed: ParsedAction, tools: list[dict[str, Any]]
    ) -> list[ToolUseBlock]:
        if not parsed.action:
            return []
        schema = next((item for item in tools if item["name"] == parsed.action), None)
        if schema is None:
            return []
        params = schema.get("parameters") or {}
        required = set(params.get("required") or [])
        supplied = set(parsed.action_input)
        if not required.issubset(supplied):
            return []
        properties = params.get("properties") or {}
        for key, value in parsed.action_input.items():
            expected = (properties.get(key) or {}).get("type")
            if expected == "string" and not isinstance(value, str):
                return []
            if expected == "integer" and not isinstance(value, int):
                return []
            if expected == "number" and not isinstance(value, (int, float)):
                return []
            if expected == "boolean" and not isinstance(value, bool):
                return []
            if expected == "array" and not isinstance(value, list):
                return []
            if expected == "object" and not isinstance(value, dict):
                return []
        unknown_keys = supplied - set(properties)
        if unknown_keys:
            return []
        return [
            ToolUseBlock(
                id=f"json-action-{parsed.action}",
                name=parsed.action,
                input=dict(parsed.action_input),
            )
        ]


async def run_react_loop(
    user_command: str,
    world_model: dict,
    memory_context: list[str],
    tool_registry: ToolRegistry,
    permission_checker: PermissionChecker,
    hud: Any,
    max_iterations: int = 20,
) -> ReactLoopResult:
    raise RuntimeError("Instantiate ReactLoop and call ReactLoop.run_react_loop")


__all__ = ["ReactLoop", "run_react_loop"]
