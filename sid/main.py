#!/usr/bin/env python3
"""Sid - fully local desktop assistant entrypoint."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import platform
import signal
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from awareness.file_watcher import FileWatcher
from awareness.triggers import TriggerEngine
from awareness.world_model import WorldModel
from audio.stt import STTEngine
from audio.tts import TTSEngine
from audio.wake_word import WakeWordDetector
from config import SidConfig
from daemon.install import install_startup_daemon
from intelligence.llm_client import LLMClient
from intelligence.orchestrator import Orchestrator
from intelligence.prompt_builder import PromptBuilder
from intelligence.react_loop import ReactLoop
from memory.fact_store import FactStore
from memory.ring_buffer import RingBuffer
from memory.undo_buffer import UndoBuffer
from memory.vector_store import VectorStore
from presentation.hud import HUD
from presentation.permission_wizard import PermissionWizard
from presentation.tray import SystemTray
from scheduler.task_scheduler import TaskScheduler
from security.action_log import ActionLog
from security.permissions import PermissionChecker
from state import StateManager
from tools.registry import ToolRegistry
from training.data_logger import InteractionLogger
from training.local_trainer import ensure_train_script
from vision.analyzer import get_ocr_cache_stats
from vision.capture import ScreenCaptureLoop
from vision.moondream import MoondreamVisionEngine

VERSION = "2.0.0-offline"


def setup_logging(config: SidConfig) -> None:
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, str(config.log_level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        config.logs_dir / "sid.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


async def first_run_setup(
    config: SidConfig,
    fact_store: FactStore,
    tts: TTSEngine,
    llm_client: LLMClient,
    tts_health=None,
) -> None:
    if config.initialized_flag.exists():
        return

    print("Setting up Sid for the first time...")
    voice_ready = bool(getattr(tts_health, "available", False))
    if voice_ready:
        await tts.speak("What should I call you?")
    user_name = (
        await asyncio.to_thread(input, "What should I call you? ")
    ).strip() or config.user_name
    await fact_store.set_fact("user_name", user_name, "personal")
    await fact_store.set_fact("os", platform.system(), "general")
    await fact_store.set_fact("home_dir", str(Path.home().resolve()), "general")
    await fact_store.set_fact("projects_dir", config.projects_dir, "project")

    consent = (
        (
            await asyncio.to_thread(
                input,
                "Allow first-run model downloads from Hugging Face/GitHub? (y/N) ",
            )
        )
        .strip()
        .lower()
    )
    if consent.startswith("y"):
        config.model_download_consent = True
        config.save()

    try:
        selected = await llm_client.ensure_runtime()
        logging.info("Selected local Ollama model: %s", selected)
    except Exception:
        logging.exception("Ollama was not ready during first-run setup")

    daemon_answer = (
        (await asyncio.to_thread(input, "Install startup daemon now? (y/N) "))
        .strip()
        .lower()
    )
    if daemon_answer.startswith("y"):
        try:
            install_startup_daemon(str(Path(sys.argv[0]).resolve()))
        except Exception:
            logging.exception("Failed to install startup daemon")

    try:
        await tts.play_wake_chime()
    except Exception:
        logging.exception("Failed to play wake chime")
    if voice_ready:
        await tts.speak("I'm Sid. Local, offline, and ready.")
    config.initialized_flag.write_text("initialized", encoding="utf-8")


async def _hotkey_listener(
    config: SidConfig, state_manager: StateManager, loop: asyncio.AbstractEventLoop
) -> None:
    try:
        import keyboard  # type: ignore
    except Exception:
        logging.info(
            "Global hotkey listener unavailable; install 'keyboard' for %s",
            config.hotkey,
        )
        return

    def _wake() -> None:
        loop.call_soon_threadsafe(
            asyncio.create_task, state_manager.request_wake("hotkey")
        )

    keyboard.add_hotkey(config.hotkey, _wake)
    while True:
        await asyncio.sleep(60)


async def _describe_screen_with_local_vision(
    vision_engine: MoondreamVisionEngine,
) -> str:
    try:
        import mss  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        return "Vision dependencies unavailable"

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        grab = sct.grab(monitor)
        image = Image.frombytes("RGB", grab.size, grab.rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return await vision_engine.describe(
        image_b64, "Describe the screen and the frontmost application."
    )


async def main() -> None:
    boot_started = time.perf_counter()
    startup_timings: dict[str, float] = {}

    def mark(label: str, started_at: float) -> None:
        startup_timings[label] = round((time.perf_counter() - started_at) * 1000.0, 2)

    t0 = time.perf_counter()
    config = SidConfig.load()
    config.validate_startup()
    ensure_train_script(config)
    mark("config_load", t0)

    setup_logging(config)
    logging.info("Sid v%s starting in offline mode", VERSION)
    logging.info(
        "System dependency hints: linux='apt install tesseract-ocr espeak-ng', macos='brew install tesseract espeak', windows='winget install UB-Mannheim.TesseractOCR && winget install eSpeak.eSpeak'"
    )

    action_log = ActionLog(config)
    action_log.log_startup(VERSION)

    t1 = time.perf_counter()
    state_manager = StateManager(config)
    await state_manager.load_persisted_state()
    world_model = WorldModel(config)
    fact_store = FactStore(config)
    undo_buffer = UndoBuffer(config)
    mark("state_and_db_init", t1)
    action_log.log_session_stats(
        "sqlite_tuning", {"facts": fact_store.tuning, "undo": undo_buffer.tuning}
    )

    t2 = time.perf_counter()
    llm_client = LLMClient(config)
    vision_engine = MoondreamVisionEngine(config)
    tts = TTSEngine(config)
    stt = STTEngine(config)
    hud = HUD(config)
    tray = SystemTray(config, state_manager)
    permission_wizard = PermissionWizard()
    permission_issues = permission_wizard.check()
    tts_health = await tts.verify_offline_tts()
    selected_model = ""
    try:
        selected_model = await llm_client.ensure_runtime()
    except Exception:
        logging.exception("Ollama runtime unavailable at startup")
    mark("local_models_health", t2)

    hud_task = asyncio.create_task(hud.run())
    tray_task = asyncio.create_task(tray.run())
    await asyncio.sleep(0.05)
    startup_timings["tray_visible"] = round(
        (time.perf_counter() - boot_started) * 1000.0, 2
    )

    t3 = time.perf_counter()
    vector_store = VectorStore(config)
    ring_buffer = RingBuffer(max_turns=20)
    interaction_logger = InteractionLogger(config)
    scheduler = TaskScheduler(config)
    trigger_engine = TriggerEngine(config)
    permission_checker = PermissionChecker(config)
    prompt_builder = PromptBuilder(config)
    tool_registry = ToolRegistry(config, undo_buffer, action_log)
    tool_registry.set_memory_stores(fact_store, vector_store)
    tool_registry.set_scheduler(scheduler)
    tool_registry.set_hud(hud)
    tool_registry.set_vision_describer(
        lambda: _describe_screen_with_local_vision(vision_engine)
    )
    react_loop = ReactLoop(
        config, llm_client, tool_registry, permission_checker, hud, tts, action_log
    )
    mark("memory_tools_init", t3)

    pending_undo_ops = await undo_buffer.list_pending_crash_recovery()
    notices: list[str] = []
    if pending_undo_ops:
        notices.append(
            f"I found {len(pending_undo_ops)} pending undo record(s) from the previous session."
        )
    if config.use_offline_tts and not tts_health.available:
        notices.append(f"Voice output unavailable. {tts_health.message}")
    if permission_issues:
        notices.append(
            f"{len(permission_issues)} OS permission issue(s) need attention before full automation works."
        )
    if selected_model:
        notices.append(f"Using local model {selected_model}.")

    orchestrator = Orchestrator(
        config=config,
        state_manager=state_manager,
        stt=stt,
        tts=tts,
        hud=hud,
        tray=tray,
        world_model=world_model,
        ring_buffer=ring_buffer,
        vector_store=vector_store,
        fact_store=fact_store,
        prompt_builder=prompt_builder,
        react_loop=react_loop,
        trigger_engine=trigger_engine,
        action_log=action_log,
        permission_wizard=permission_wizard,
        interaction_logger=interaction_logger,
        startup_notices=notices,
    )
    react_loop.confirm_callback = orchestrator.confirm_dangerous_action

    await first_run_setup(config, fact_store, tts, llm_client, tts_health=tts_health)

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def handle_signal() -> None:
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            signal.signal(sig, lambda *_args: shutdown_event.set())

    def _trigger_wake(source: str) -> None:
        asyncio.create_task(state_manager.request_wake(source))

    wake_detector = WakeWordDetector(config, loop, _trigger_wake)
    wake_detector.start()

    background_tasks = [
        hud_task,
        tray_task,
        asyncio.create_task(orchestrator.run()),
        asyncio.create_task(world_model.update_loop()),
        asyncio.create_task(scheduler.run()),
        asyncio.create_task(_hotkey_listener(config, state_manager, loop)),
        asyncio.create_task(FileWatcher(config, world_model, trigger_engine).run()),
        asyncio.create_task(trigger_engine.monitor_loop(world_model, orchestrator)),
    ]
    if config.enable_vision:
        background_tasks.append(
            asyncio.create_task(
                ScreenCaptureLoop(
                    config, world_model, vision_engine, state_manager=state_manager
                ).run()
            )
        )

    action_log.log_startup_complete(startup_timings)
    await shutdown_event.wait()

    await orchestrator.shutdown()
    wake_detector.stop()
    tray.stop()
    hud.close()
    for task in background_tasks:
        task.cancel()
    await asyncio.gather(*background_tasks, return_exceptions=True)
    await vector_store.flush()
    await fact_store.close()
    await undo_buffer.close()
    action_log.log_session_stats("tool_cache", tool_registry.cache_stats())
    action_log.log_session_stats("ocr_cache", get_ocr_cache_stats())
    tts.close()
    action_log.log_shutdown(int(time.perf_counter() - boot_started))


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
