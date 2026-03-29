from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SidConfig  # noqa: E402
from memory.undo_buffer import UndoBuffer  # noqa: E402
from security.action_log import ActionLog  # noqa: E402
from security.permissions import PermissionChecker  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402


async def _run(
    tool_name: str, kwargs: dict[str, object], sid_dir: str | None = None
) -> dict[str, object]:
    os.environ["SID_SANDBOX_CHILD"] = "1"
    config = SidConfig.load(path=sid_dir)
    config.ensure_dirs()
    action_log = ActionLog(config)
    undo = UndoBuffer(config)
    registry = ToolRegistry(config, undo, action_log)
    checker = PermissionChecker(config)
    try:
        result = await registry.execute(tool_name, kwargs, checker)
        return dict(result)
    finally:
        await undo.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True)
    parser.add_argument("--kwargs", required=True)
    parser.add_argument("--sid-dir")
    args = parser.parse_args()

    kwargs = json.loads(args.kwargs)
    result = asyncio.run(_run(args.tool, kwargs, sid_dir=args.sid_dir))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
