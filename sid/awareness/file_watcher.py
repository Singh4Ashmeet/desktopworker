from __future__ import annotations

import asyncio
import logging
from pathlib import Path


class FileWatcher:
    def __init__(self, config, world_model, trigger_engine) -> None:
        self.config = config
        self.world_model = world_model
        self.trigger_engine = trigger_engine
        self.downloads = Path.home() / "Downloads"

    async def run(self) -> None:
        if not self.downloads.exists():
            while True:
                await asyncio.sleep(60)

        try:
            from watchdog.events import FileSystemEventHandler  # type: ignore
            from watchdog.observers import Observer  # type: ignore
        except Exception:
            logging.warning(
                "watchdog unavailable; file watcher running polling fallback"
            )
            await self._polling_loop()
            return

        trigger_engine = self.trigger_engine

        class Handler(FileSystemEventHandler):
            def on_created(self, event):
                if event.is_directory:
                    return
                trigger_engine.record_download_event()

        observer = Observer()
        handler = Handler()
        observer.schedule(handler, str(self.downloads), recursive=False)
        observer.start()
        try:
            while True:
                await asyncio.sleep(5)
        finally:
            observer.stop()
            observer.join(timeout=2)

    async def _polling_loop(self) -> None:
        seen: set[str] = set()
        while True:
            try:
                for p in self.downloads.glob("*"):
                    key = str(p.resolve())
                    if key in seen or p.is_dir():
                        continue
                    seen.add(key)
                    self.trigger_engine.record_download_event()
            except Exception:
                logging.exception("Polling watcher failed")
            await asyncio.sleep(30)
