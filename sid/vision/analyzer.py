from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any

from PIL import Image

from vision.ocr_cache import OCRCache


_OCR_CACHE = OCRCache(ttl_seconds=30)


async def analyze_screenshot(
    image_b64: str, vision_engine=None, phash: str | None = None
) -> dict[str, Any]:
    if vision_engine is not None and hasattr(vision_engine, "config"):
        _OCR_CACHE.ttl_seconds = int(
            getattr(vision_engine.config, "ocr_cache_ttl_seconds", 30)
        )

    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    ocr_text = await _ocr_text(image, phash)
    activity = ""
    if vision_engine is not None:
        try:
            activity = await vision_engine.describe(
                image_b64,
                "Describe the frontmost app, whether an error is visible, and what the user is doing in one sentence.",
            )
        except Exception:
            logging.exception("Moondream description failed")

    error_detected = bool(
        re.search(
            r"\b(error|exception|traceback|failed|fatal)\b",
            f"{ocr_text}\n{activity}",
            re.IGNORECASE,
        )
    )
    error_summary = _first_error_line(ocr_text or activity)
    return {
        "active_app": _guess_app_name(activity, ocr_text),
        "activity": (activity or _summarize_ocr(ocr_text) or "Screen visible").strip(),
        "visible_file": _guess_visible_file(ocr_text),
        "error_detected": error_detected,
        "error_summary": error_summary,
        "notable": _summarize_ocr(ocr_text, limit=120),
        "ocr_text": ocr_text[:1000],
    }


async def _ocr_text(image: Image.Image, phash: str | None) -> str:
    if phash:
        cached = _OCR_CACHE.get(phash, (0, 0, image.width, image.height))
        if cached:
            return cached
    try:
        import pytesseract  # type: ignore
    except Exception:
        return ""
    text = await asyncio_to_thread(pytesseract.image_to_string, image)
    cleaned = " ".join(text.split())
    if phash:
        _OCR_CACHE.set(phash, (0, 0, image.width, image.height), cleaned)
    return cleaned


async def asyncio_to_thread(func, *args, **kwargs):
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


def _guess_app_name(activity: str, ocr_text: str) -> str | None:
    haystack = f"{activity} {ocr_text}".lower()
    for candidate in [
        "chrome",
        "firefox",
        "edge",
        "code",
        "terminal",
        "powershell",
        "explorer",
        "finder",
    ]:
        if candidate in haystack:
            return candidate
    return None


def _guess_visible_file(text: str) -> str | None:
    match = re.search(r"([A-Za-z]:\\[^ ]+\.[A-Za-z0-9]+|/[^ ]+\.[A-Za-z0-9]+)", text)
    if match:
        return match.group(1)
    return None


def _first_error_line(text: str) -> str | None:
    for part in re.split(r"[.!?]", text):
        if re.search(
            r"\b(error|exception|traceback|failed|fatal)\b", part, re.IGNORECASE
        ):
            return part.strip()[:200]
    return None


def _summarize_ocr(text: str, limit: int = 180) -> str:
    return " ".join((text or "").split())[:limit]


def get_ocr_cache_stats() -> dict[str, float]:
    return {
        "hits": float(_OCR_CACHE.hits),
        "misses": float(_OCR_CACHE.misses),
        "hit_rate": _OCR_CACHE.hit_rate,
    }


__all__ = ["analyze_screenshot", "get_ocr_cache_stats"]
