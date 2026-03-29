from __future__ import annotations

from pathlib import Path

from config import SidConfig


TRAIN_SCRIPT_TEMPLATE = r"""#!/usr/bin/env python3
from pathlib import Path

print("Sid local fine-tuning entrypoint")
print("Use Unsloth + TRL against the interaction datasets in ~/.sid/training_data/")
print("Suggested base model:", "{base_model}")
print("LoRA output dir:", "{lora_dir}")
print("DPO dataset:", "{dpo_path}")
"""


def ensure_train_script(config: SidConfig) -> Path:
    path = config.sid_path / "train.py"
    if not path.exists():
        path.write_text(
            TRAIN_SCRIPT_TEMPLATE.format(
                base_model=config.ollama_model or config.ollama_model_candidates[0],
                lora_dir=str(config.models_dir / "lora"),
                dpo_path=str(config.training_data_dir / "dpo.jsonl"),
            ),
            encoding="utf-8",
        )
    return path


__all__ = ["ensure_train_script"]
