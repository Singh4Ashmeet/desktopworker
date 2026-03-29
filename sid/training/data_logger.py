from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import SidConfig


class InteractionLogger:
    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self.interactions_path = config.training_data_dir / "interactions.jsonl"
        self.dpo_path = config.training_data_dir / "dpo.jsonl"
        self.config.training_data_dir.mkdir(parents=True, exist_ok=True)

    def log_interaction(
        self, user_input: str, sid_response: str, user_feedback: str = ""
    ) -> None:
        self._append_jsonl(
            self.interactions_path,
            {
                "user_input": user_input,
                "sid_response": sid_response,
                "user_feedback": user_feedback,
            },
        )

    def log_dpo_pair(self, prompt: str, chosen: str, rejected: str) -> None:
        self._append_jsonl(
            self.dpo_path, {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        )

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["InteractionLogger"]
