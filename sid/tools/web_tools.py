from __future__ import annotations

from models import ToolResult
from network_guard import NetworkFirewall
from tools.registry import ToolContext, tool

_CTX: ToolContext | None = None


def init_context(ctx: ToolContext) -> None:
    global _CTX
    _CTX = ctx


def _ctx() -> ToolContext:
    if _CTX is None:
        raise RuntimeError("Tool context not initialized")
    return _CTX


@tool(
    name="fetch_url",
    description="Fetch a web page and extract readable text locally.",
    parameters_schema={
        "type": "object",
        "properties": {"url": {"type": "string"}, "extract_text": {"type": "boolean"}},
        "required": ["url"],
    },
)
async def fetch_url(url: str, extract_text: bool = True) -> ToolResult:
    import httpx
    from bs4 import BeautifulSoup
    try:
        from readability import Document
    except Exception:
        Document = None  # type: ignore[assignment]

    firewall = NetworkFirewall(_ctx().config)
    decision = firewall.check_url(url, purpose="explicit_web_fetch")
    if not decision.allowed:
        return {"success": False, "output": decision.reason, "data": None}

    verify_mode: bool = True
    try:
        async with httpx.AsyncClient(
            timeout=_ctx().config.web_request_timeout_seconds, follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text
    except httpx.HTTPError as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise
        verify_mode = False
        async with httpx.AsyncClient(
            timeout=_ctx().config.web_request_timeout_seconds,
            follow_redirects=True,
            verify=False,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text

    if not extract_text:
        return {
            "success": True,
            "output": html[:4000],
            "data": {
                "html": html,
                "status": response.status_code,
                "tls_verification_skipped": not verify_mode,
            },
        }

    title = ""
    if Document is not None:
        readable = Document(html)
        title = readable.title()
        soup = BeautifulSoup(readable.summary(), "html.parser")
    else:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = " ".join(title_tag.get_text(" ").split()) if title_tag else url
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    text = text[:4000]
    return {
        "success": True,
        "output": text,
        "data": {
            "text": text,
            "title": title,
            "status": response.status_code,
            "tls_verification_skipped": not verify_mode,
        },
    }


@tool(
    name="search_web",
    description="Search the web using local SearXNG or DuckDuckGo HTML fallback.",
    parameters_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}},
        "required": ["query"],
    },
)
async def search_web(query: str, num_results: int = 5) -> ToolResult:
    results = await _search_searxng(query, num_results)
    if not results:
        results = await _search_duckduckgo_html(query, num_results)
    preview = "\n".join(f"{item['title']} - {item['url']}" for item in results)
    return {"success": True, "output": preview, "data": {"results": results}}


@tool(
    name="get_media_metadata",
    description="Get metadata for a media URL without downloading the media.",
    parameters_schema={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
)
async def get_media_metadata(url: str) -> ToolResult:
    decision = NetworkFirewall(_ctx().config).check_url(
        url, purpose="explicit_web_fetch"
    )
    if not decision.allowed:
        return {"success": False, "output": decision.reason, "data": None}

    try:
        from yt_dlp import YoutubeDL  # type: ignore
    except Exception as exc:
        return {"success": False, "output": f"yt-dlp unavailable: {exc}", "data": None}

    def _extract() -> dict:
        with YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            return ydl.extract_info(url, download=False)

    info = await __import__("asyncio").to_thread(_extract)
    return {
        "success": True,
        "output": str(info.get("title") or url),
        "data": {
            "title": info.get("title"),
            "duration": info.get("duration"),
            "uploader": info.get("uploader"),
        },
    }


async def _search_searxng(query: str, num_results: int) -> list[dict[str, str]]:
    import httpx

    base_url = _ctx().config.searxng_url.rstrip("/")
    decision = NetworkFirewall(_ctx().config).check_url(base_url, purpose="runtime")
    if not decision.allowed:
        return []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{base_url}/search", params={"q": query, "format": "json"}
            )
            response.raise_for_status()
            payload = response.json()
        items = payload.get("results") or []
        return [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("content", "")),
            }
            for item in items[: max(1, num_results)]
        ]
    except Exception:
        return []


async def _search_duckduckgo_html(query: str, num_results: int) -> list[dict[str, str]]:
    import httpx
    from bs4 import BeautifulSoup

    url = "https://html.duckduckgo.com/html/"
    async with httpx.AsyncClient(
        timeout=10,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as client:
        response = await client.post(url, data={"q": query})
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return _parse_duckduckgo_results(soup, num_results)


def _parse_duckduckgo_results(soup, num_results: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for node in soup.select(".result"):
        anchor = node.select_one(".result__title a")
        if anchor is None:
            continue
        href = anchor.get("href") or ""
        title = " ".join(anchor.get_text(" ").split())
        if not href or not title:
            continue
        snippet_tag = node.select_one(".result__snippet")
        snippet = " ".join(snippet_tag.get_text(" ").split()) if snippet_tag else ""
        results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= max(1, num_results):
            break
    return results


__all__ = ["fetch_url", "search_web", "get_media_metadata"]
