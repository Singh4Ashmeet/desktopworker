from __future__ import annotations

import asyncio
import base64
import io
import logging

from PIL import Image

from state import SidState
from vision.analyzer import analyze_screenshot


class ScreenCaptureDependencyError(RuntimeError):
    pass


class ScreenCaptureLoop:
    def __init__(self, config, world_model, llm_client, state_manager=None) -> None:
        self.config = config
        self.world_model = world_model
        self.llm_client = llm_client
        self.state_manager = state_manager
        self._last_phash: str | None = None
        self._stable_ticks = 0

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._next_interval())
            try:
                image_b64, phash = await asyncio.to_thread(self._capture)
                if (
                    self._last_phash is not None
                    and _hamming_distance(self._last_phash, phash)
                    <= self.config.screen_phash_threshold
                ):
                    self._stable_ticks += 1
                    await self.world_model.update_screen_stability(self._stable_ticks)
                    continue

                self._stable_ticks = 0
                self._last_phash = phash
                await self.world_model.update_screen_stability(self._stable_ticks)
                analysis = await analyze_screenshot(
                    image_b64, self.llm_client, phash=phash
                )
                await self.world_model.update_from_screen(analysis)
            except ScreenCaptureDependencyError as exc:
                logging.warning("Screen capture disabled: %s", exc)
                return
            except Exception:
                logging.exception("Screen capture loop error")

    def _next_interval(self) -> float:
        state = (
            self.state_manager.state
            if self.state_manager is not None
            else SidState.SLEEPING
        )
        if state == SidState.ACTING:
            return 1
        if (
            state == SidState.SLEEPING
            and self.world_model.snapshot().get("idle_minutes", 0) > 30
        ):
            return 5
        if self._stable_ticks >= self.config.screen_stability_backoff_ticks:
            return 5
        return max(0.5, float(self.config.screen_capture_interval))

    def _capture(self) -> tuple[str, str]:
        try:
            import mss  # type: ignore
        except Exception as exc:
            raise ScreenCaptureDependencyError(
                "mss is required for screen capture"
            ) from exc

        with mss.mss() as sct:
            mon = sct.monitors[1]
            grab = sct.grab(mon)
            image = Image.frombytes("RGB", grab.size, grab.rgb)
            image = _resize_keep_aspect(image, 1280)
            phash = _perceptual_hash(image)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii"), phash


def _resize_keep_aspect(image: Image.Image, width: int) -> Image.Image:
    if image.width <= width:
        return image
    ratio = width / float(image.width)
    return image.resize((width, int(image.height * ratio)), Image.Resampling.LANCZOS)


def _perceptual_hash(image: Image.Image) -> str:
    small = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    bits = ["1" if px >= avg else "0" for px in pixels]
    return "".join(bits)


def _hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for left, right in zip(a, b) if left != right)
