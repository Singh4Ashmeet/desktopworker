from __future__ import annotations

from bs4 import BeautifulSoup

from tools.shell_tools import _build_command_line
from tools.web_tools import _parse_duckduckgo_results


def test_build_command_line_uses_windows_safe_quoting(monkeypatch):
    monkeypatch.setattr("tools.shell_tools.os.name", "nt")
    cmd = _build_command_line(["python", r"C:\Temp Folder\sid test\hello.py"])
    assert cmd == 'python "C:\\Temp Folder\\sid test\\hello.py"'


def test_parse_duckduckgo_results_extracts_titles_urls_and_snippets():
    html = """
    <div class="result">
      <h2 class="result__title">
        <a href="https://example.com/a">Alpha Result</a>
      </h2>
      <a class="result__snippet">First snippet</a>
    </div>
    <div class="result">
      <h2 class="result__title">
        <a href="https://example.com/b">Beta Result</a>
      </h2>
      <div class="result__snippet">Second snippet</div>
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")

    results = _parse_duckduckgo_results(soup, 2)

    assert results == [
        {
            "title": "Alpha Result",
            "url": "https://example.com/a",
            "snippet": "First snippet",
        },
        {
            "title": "Beta Result",
            "url": "https://example.com/b",
            "snippet": "Second snippet",
        },
    ]
