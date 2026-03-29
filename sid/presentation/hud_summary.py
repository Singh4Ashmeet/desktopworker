from __future__ import annotations

from urllib.parse import urlparse


def summarize_for_hud(tool_name: str, result: dict) -> str:
    output = str(result.get("output", "")).strip()
    data = result.get("data") or {}

    if tool_name in {
        "move_file",
        "copy_file",
        "delete_file",
        "create_folder",
        "write_file",
    }:
        if tool_name == "move_file":
            return f"Moved {data.get('source', 'item')} to {data.get('destination', 'target')}"
        if tool_name == "copy_file":
            return f"Copied {data.get('source', 'item')} to {data.get('destination', 'target')}"
        if tool_name == "delete_file":
            return f"Deleted {data.get('path', 'item')}"
        if tool_name == "create_folder":
            return f"Created folder {data.get('path', '')}"
        if tool_name == "write_file":
            return f"Updated file {data.get('path', '')}"

    if tool_name in {"run_command", "run_script"}:
        first_line = str(
            (data.get("stdout") or output or "").splitlines()[0]
            if (data.get("stdout") or output)
            else output
        )
        return first_line[:60]

    if tool_name == "fetch_url":
        status = data.get("status")
        url = data.get("url") or ""
        domain = urlparse(url).netloc if url else "page"
        return f"{domain or 'page'} retrieved ({status})"

    if tool_name == "search_web":
        return "Search results retrieved"

    return output[:60] or tool_name
