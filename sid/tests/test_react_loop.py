from __future__ import annotations

import pytest

from config import SidConfig
from intelligence.react_loop import ReactLoop
from intelligence.json_parser import ParsedAction
from security.permissions import PermissionTier


class FakeLLM:
    def __init__(self, sequence):
        self.sequence = sequence
        self.calls = 0

    async def react_step(self, _system_prompt, _messages, _tools, max_tokens=1024):
        del max_tokens
        idx = min(self.calls, len(self.sequence) - 1)
        self.calls += 1
        parsed = self.sequence[idx]
        return parsed, '{"ok": true}'


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def get_all_schemas(self):
        return [
            {
                "name": "tool_a",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "tool_b",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "delete_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "tool_x",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        ]

    async def execute(self, name, kwargs, _checker):
        self.calls.append((name, kwargs))
        return {"success": True, "output": f"ok:{name}", "data": None}


class FakePermission:
    def __init__(self, danger_for=None):
        self.danger_for = set(danger_for or [])

    def check(self, tool_name, _kwargs):
        return (
            PermissionTier.DANGER
            if tool_name in self.danger_for
            else PermissionTier.SAFE
        )


class FakeHUD:
    async def add_task_line(self, _line):
        return None

    async def set_detail_panel(self, _title, _lines):
        return None


class FakeTTS:
    async def speak(self, _msg):
        return None


def make_config(tmp_path):
    cfg = SidConfig(
        sid_dir=str(tmp_path / ".sid"),
        allowlist_paths=[str(tmp_path)],
        use_local_llm=True,
        use_offline_tts=True,
    )
    cfg.validate_startup()
    return cfg


@pytest.mark.asyncio
async def test_simple_command_completes_in_one_iteration(tmp_path):
    llm = FakeLLM(
        [
            ParsedAction(
                thought="done", action=None, action_input={}, final_answer="Done now."
            )
        ]
    )
    registry = FakeRegistry()
    loop = ReactLoop(
        make_config(tmp_path),
        llm,
        registry,
        FakePermission(),
        FakeHUD(),
        FakeTTS(),
        None,
    )

    out = await loop.run_react_loop("hi", {}, [], "sys", max_iterations=20)
    assert out["status"] == "completed"
    assert out["result"] == "Done now."


@pytest.mark.asyncio
async def test_multi_step_command_uses_multiple_tools(tmp_path):
    llm = FakeLLM(
        [
            ParsedAction(
                thought="step a", action="tool_a", action_input={}, final_answer=None
            ),
            ParsedAction(
                thought="step b", action="tool_b", action_input={}, final_answer=None
            ),
            ParsedAction(
                thought="done", action=None, action_input={}, final_answer="Finished."
            ),
        ]
    )
    registry = FakeRegistry()
    loop = ReactLoop(
        make_config(tmp_path),
        llm,
        registry,
        FakePermission(),
        FakeHUD(),
        FakeTTS(),
        None,
    )

    out = await loop.run_react_loop("do x", {}, [], "sys", max_iterations=20)
    assert out["result"] == "Finished."
    assert [call[0] for call in registry.calls] == ["tool_a", "tool_b"]


@pytest.mark.asyncio
async def test_loop_breaks_after_max_iterations(tmp_path):
    llm = FakeLLM(
        [
            ParsedAction(
                thought="repeat",
                action="tool_x",
                action_input={"x": 1},
                final_answer=None,
            )
        ]
        * 25
    )
    registry = FakeRegistry()
    loop = ReactLoop(
        make_config(tmp_path),
        llm,
        registry,
        FakePermission(),
        FakeHUD(),
        FakeTTS(),
        None,
    )

    out = await loop.run_react_loop("loop", {}, [], "sys", max_iterations=4)
    assert out["status"] == "aborted"
    assert out["aborted_reason"] in {"identical_tool_calls", "step_cap_reached"}


@pytest.mark.asyncio
async def test_danger_tool_triggers_confirmation(tmp_path):
    llm = FakeLLM(
        [
            ParsedAction(
                thought="danger",
                action="delete_file",
                action_input={"path": "x"},
                final_answer=None,
            )
        ]
    )
    registry = FakeRegistry()

    async def deny(_prompt: str) -> bool:
        return False

    loop = ReactLoop(
        make_config(tmp_path),
        llm,
        registry,
        FakePermission(danger_for={"delete_file"}),
        FakeHUD(),
        FakeTTS(),
        None,
        confirm_callback=deny,
    )

    out = await loop.run_react_loop("delete", {}, [], "sys", max_iterations=1)
    assert out["status"] == "aborted"
    assert registry.calls == []
