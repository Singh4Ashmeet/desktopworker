from __future__ import annotations

import pytest

from intelligence.json_parser import extract_json_object, parse_react_json


def test_json_tool_call_parser_handles_markdown_fence():
    raw = """```json\n{"thought":"use tool","action":"search_files","action_input":{"query":"sid"}}\n```"""
    parsed = parse_react_json(raw)
    assert parsed.action == "search_files"
    assert parsed.action_input["query"] == "sid"


def test_json_tool_call_parser_handles_extra_text():
    raw = 'Model says:\n{"thought":"done","final_answer":"All set."}\nthanks'
    parsed = parse_react_json(raw)
    assert parsed.final_answer == "All set."


def test_json_tool_call_parser_rejects_invalid_json():
    with pytest.raises(ValueError):
        extract_json_object("not json at all")
