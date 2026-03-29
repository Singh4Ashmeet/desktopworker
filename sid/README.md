# Sid

Sid is an autonomous desktop AI assistant with wake-word activation, tool execution, HUD/tray UX, memory, proactive awareness, and startup daemon support.

## Quickstart

1. Create virtual env and install:
   - `python -m venv .venv`
   - `.venv\\Scripts\\activate`
   - `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill API keys.
3. (Optional) store keys in keyring.
4. Run `python main.py`.

## Architecture

- `audio/`: wake word + STT + TTS
- `presentation/`: HUD overlay + tray icon
- `intelligence/`: prompting, LLM client, orchestrator, ReAct loop
- `tools/`: async tool registry and implementations
- `memory/`: ring buffer, facts, vectors, undo ops
- `security/`: permission tiers, sandbox, action logging
- `vision/` + `awareness/`: screen context and proactive triggers
- `scheduler/`: reminders and recurring jobs
- `daemon/`: startup install assets/scripts

## Testing

Run:

`pytest -q`

## Packaging

`pyinstaller --onefile main.py --name sid`
