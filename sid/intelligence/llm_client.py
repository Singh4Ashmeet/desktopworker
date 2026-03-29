from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from collections.abc import AsyncIterator
from typing import Any

import httpx

from config import SidConfig
from intelligence.json_parser import ParsedAction, parse_react_json
from network_guard import NetworkFirewall


class LLMClient:
    """Local-only Ollama client with JSON-mode ReAct helpers."""

    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self.firewall = NetworkFirewall(config)
        self._selected_model = config.ollama_model or ""
        self._health_lock = asyncio.Lock()

    @property
    def selected_model(self) -> str:
        return self._selected_model or self.config.ollama_model_candidates[0]

    async def ensure_runtime(self) -> str:
        async with self._health_lock:
            for attempt in range(3):
                try:
                    available = await self._list_models()
                    self._selected_model = self._pick_model(available)
                    return self.selected_model
                except Exception:
                    logging.warning(
                        "Ollama health check failed (attempt %s/3)", attempt + 1
                    )
                    await self._start_ollama()
                    # Exponential backoff: 1s, 2s, 4s
                    backoff_seconds = 2**attempt
                    logging.info("Waiting %s seconds before retry...", backoff_seconds)
                    await asyncio.sleep(backoff_seconds)
            raise RuntimeError("Ollama is not available after 3 attempts")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        max_tokens: int = 4096,
    ) -> AsyncIterator[dict[str, str]]:
        del tools
        model = await self.ensure_runtime()
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"num_predict": max_tokens},
        }

        async with self._client() as client:
            if stream:
                async with client.stream(
                    "POST", f"{self.config.ollama_host}/api/chat", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = str((chunk.get("message") or {}).get("content") or "")
                        if content:
                            yield {"type": "text", "text": content}
                return

            response = await client.post(
                f"{self.config.ollama_host}/api/chat", json=payload
            )
            response.raise_for_status()
            content = str((response.json().get("message") or {}).get("content") or "")
            if content:
                yield {"type": "text", "text": content}

    async def complete_text(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> str:
        chunks: list[str] = []
        all_messages = [{"role": "system", "content": system_prompt}, *messages]
        async for chunk in self.chat(all_messages, stream=True, max_tokens=max_tokens):
            chunks.append(chunk.get("text", ""))
        return "".join(chunks).strip()

    async def react_step(
        self,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> tuple[ParsedAction, str]:
        tool_text = json.dumps(tools, ensure_ascii=False, indent=2)
        json_prompt = (
            f"{system_prompt}\n\n"
            "You are in JSON-only ReAct mode.\n"
            "Available tools (OpenAI-style schema):\n"
            f"{tool_text}\n\n"
            "Respond with exactly one JSON object.\n"
            'If you want to use a tool: {"thought":"...","action":"tool_name","action_input":{...}}\n'
            'If you are done: {"thought":"...","final_answer":"..."}\n'
            "Do not wrap the JSON in prose."
        )
        raw = await self.complete_text(json_prompt, conversation, max_tokens=max_tokens)
        try:
            return parse_react_json(raw), raw
        except Exception:
            reprompt_messages = [
                *conversation,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Please output valid JSON only."},
            ]
            repaired = await self.complete_text(
                json_prompt, reprompt_messages, max_tokens=max_tokens
            )
            return parse_react_json(repaired), repaired

    async def _list_models(self) -> list[str]:
        async with self._client() as client:
            response = await client.get(f"{self.config.ollama_host}/api/tags")
            response.raise_for_status()
            payload = response.json()
        models = payload.get("models") or []
        return [
            str(item.get("name", "")).strip() for item in models if item.get("name")
        ]

    def _pick_model(self, available: list[str]) -> str:
        if self.config.ollama_model and self.config.ollama_model in available:
            return self.config.ollama_model
        for candidate in self.config.ollama_model_candidates:
            if candidate in available:
                return candidate
        if available:
            return available[0]
        return self.config.ollama_model_candidates[0]

    async def _start_ollama(self) -> None:
        command = self.config.ollama_start_command
        if not command:
            return
        if subprocess.list2cmdline([command]) == "":
            return
        creationflags = 0
        kwargs: dict[str, Any] = {}
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kwargs["start_new_session"] = True
        await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=creationflags,
            **kwargs,
        )

    def _client(self) -> httpx.AsyncClient:
        decision = self.firewall.check_url(self.config.ollama_host, purpose="runtime")
        if not decision.allowed:
            raise RuntimeError(decision.reason)
        return httpx.AsyncClient(timeout=self.config.web_request_timeout_seconds)


__all__ = ["LLMClient"]
