from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Awaitable, Callable, TypedDict

from config import SidConfig
from memory.fact_store import FactStore
from memory.undo_buffer import UndoBuffer
from memory.vector_store import VectorStore
from security.action_log import ActionLog
from security.permissions import PermissionChecker, PermissionTier
from security.sandboxing import run_tool_in_subprocess
from models import ToolResult
from tools.cache import ToolResultCache


class ToolSchema(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters_schema: dict[str, Any]
    func: Callable[..., Awaitable[ToolResult]]


@dataclass(slots=True)
class ToolContext:
    config: SidConfig
    undo_buffer: UndoBuffer
    action_log: ActionLog
    fact_store: FactStore | None = None
    vector_store: VectorStore | None = None
    scheduler: Any = None
    hud: Any = None
    vision_describer: Callable[[], Awaitable[str]] | None = None


_TOOL_DEFINITIONS: dict[str, ToolDefinition] = {}


def tool(
    name: str, description: str, parameters_schema: dict[str, Any]
) -> Callable[
    [Callable[..., Awaitable[ToolResult]]], Callable[..., Awaitable[ToolResult]]
]:
    def decorator(
        func: Callable[..., Awaitable[ToolResult]],
    ) -> Callable[..., Awaitable[ToolResult]]:
        _TOOL_DEFINITIONS[name] = ToolDefinition(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            func=func,
        )
        return func

    return decorator


class ToolRegistry:
    def __init__(
        self, config: SidConfig, undo_buffer: UndoBuffer, action_log: ActionLog
    ) -> None:
        self.context = ToolContext(
            config=config, undo_buffer=undo_buffer, action_log=action_log
        )
        self._tools: dict[str, ToolDefinition] = {}
        self._cache = ToolResultCache()
        self._load_default_tools()

    def set_memory_stores(
        self, fact_store: FactStore, vector_store: VectorStore
    ) -> None:
        self.context.fact_store = fact_store
        self.context.vector_store = vector_store

    def set_scheduler(self, scheduler: Any) -> None:
        self.context.scheduler = scheduler

    def set_hud(self, hud: Any) -> None:
        self.context.hud = hud

    def set_vision_describer(self, describer: Callable[[], Awaitable[str]]) -> None:
        self.context.vision_describer = describer

    def _load_default_tools(self) -> None:
        modules = [
            "tools.file_tools",
            "tools.shell_tools",
            "tools.app_tools",
            "tools.ui_tools",
            "tools.web_tools",
            "tools.clipboard_tools",
            "tools.notification_tools",
            "tools.memory_tools",
        ]

        for module_name in modules:
            module = importlib.import_module(module_name)
            self._inject_context(module)

        self._tools = dict(_TOOL_DEFINITIONS)

    def _inject_context(self, module: ModuleType) -> None:
        setter = getattr(module, "init_context", None)
        if callable(setter):
            setter(self.context)

    def get_all_schemas(self) -> list[ToolSchema]:
        schemas: list[ToolSchema] = []
        for td in self._tools.values():
            schemas.append(
                {
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters_schema,
                }
            )
        return schemas

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def cache_stats(self) -> dict[str, dict[str, int]]:
        tool_names = sorted(set(self._cache.hits) | set(self._cache.misses))
        return {
            name: {
                "hits": self._cache.hits.get(name, 0),
                "misses": self._cache.misses.get(name, 0),
            }
            for name in tool_names
        }

    async def execute(
        self, name: str, kwargs: dict[str, Any], permission_checker: PermissionChecker
    ) -> ToolResult:
        td = self._tools.get(name)
        if td is None:
            return {"success": False, "output": f"Unknown tool: {name}", "data": None}

        safe_kwargs = dict(kwargs)
        blocked, reason = permission_checker.is_blocked(name, safe_kwargs)
        if blocked:
            self.context.action_log.log_tool_call(
                state="WORKING",
                tool=name,
                args=safe_kwargs,
                tier=PermissionTier.BLOCKED.value,
                result="failure",
                output_preview=reason,
            )
            return {"success": False, "output": reason, "data": None}

        tier = permission_checker.check(name, safe_kwargs)
        default_timeout = (
            self.context.config.danger_tool_timeout_seconds
            if tier == PermissionTier.DANGER
            else self.context.config.tool_default_timeout_seconds
        )
        timeout = int(safe_kwargs.pop("_timeout", default_timeout))

        cached = self._maybe_get_cached(name, safe_kwargs)
        if cached is not None:
            return cached

        try:
            if tier == PermissionTier.DANGER and not bool(
                __import__("os").environ.get("SID_SANDBOX_CHILD")
            ):
                result = await run_tool_in_subprocess(
                    name, safe_kwargs, self.context.config
                )
            else:
                result = await asyncio.wait_for(td.func(**safe_kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            result = {
                "success": False,
                "output": f"Tool timeout after {timeout}s",
                "data": None,
            }
        except Exception as exc:
            logging.exception("Tool execution failed: %s", name)
            result = {"success": False, "output": f"Tool error: {exc}", "data": None}

        self.context.action_log.log_tool_call(
            state="WORKING",
            tool=name,
            args=safe_kwargs,
            tier=tier.value if isinstance(tier, PermissionTier) else str(tier),
            result="success" if result.get("success") else "failure",
            output_preview=result.get("output", ""),
        )
        self._maybe_store_cache(name, safe_kwargs, result)
        self._maybe_invalidate_cache(name)
        return result

    def _maybe_get_cached(self, name: str, kwargs: dict[str, Any]) -> ToolResult | None:
        if name not in {
            "list_dir",
            "get_system_info",
            "list_windows",
            "check_process",
            "list_processes",
        }:
            return None
        return self._cache.get(name, kwargs)

    def _maybe_store_cache(
        self, name: str, kwargs: dict[str, Any], result: ToolResult
    ) -> None:
        if not result.get("success"):
            return
        ttl_map = {
            "list_dir": self.context.config.cache_list_dir_ttl_seconds,
            "check_process": self.context.config.cache_process_list_ttl_seconds,
            "list_processes": self.context.config.cache_process_list_ttl_seconds,
            "get_system_info": self.context.config.cache_system_stats_ttl_seconds,
            "list_windows": self.context.config.cache_window_list_ttl_seconds,
        }
        ttl = ttl_map.get(name)
        if ttl is None:
            return
        self._cache.set(name, kwargs, dict(result), ttl)

    def _maybe_invalidate_cache(self, name: str) -> None:
        if name in {
            "write_file",
            "move_file",
            "delete_file",
            "create_folder",
            "copy_file",
        }:
            self._cache.invalidate_by_tool_prefix({"list_dir"})


__all__ = ["ToolRegistry", "ToolContext", "ToolDefinition", "ToolSchema", "tool"]
