"""One-command launcher for deploy presets. stdlib-only — runs in any python.

    python3 -m deploy.launch pi05 --task="Stack the cup on top of the bowl."

Loads deploy/presets/<name>.json, then on the preset's port: REUSES a warm
server already serving the same checkpoint (skips pi05's ~15-20s compile cold
start), REFUSES the port if a different policy is on it (never kills someone
else's server), or SPAWNS the server detached with logs under deploy/logs/.
Finally runs deploy.client in the foreground; Ctrl-C stops only the client,
the server stays warm for the next run.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def load_preset(name_or_path: str, presets_dir: Path = PRESETS_DIR) -> dict:
    path = Path(name_or_path)
    if path.suffix != ".json":
        path = presets_dir / f"{name_or_path}.json"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in presets_dir.glob("*.json")))
        raise SystemExit(f"unknown preset '{name_or_path}'. Available: {available}")
    preset = json.loads(path.read_text())
    server = preset.get("server")
    if not isinstance(server, dict):
        raise SystemExit(f"{path}: preset needs a 'server' section")
    missing = [key for key in ("python", "adapter", "port") if key not in server]
    if missing:
        raise SystemExit(f"{path}: server section is missing {', '.join(missing)}")
    return preset


def resolve_path(value: str) -> str:
    """Resolve preset paths against the repo root; leave hub ids untouched."""
    candidate = REPO_ROOT / value
    return str(candidate.resolve()) if candidate.exists() else value


def decide(info: dict | None, want_checkpoint: str | None) -> str:
    """Reuse a warm matching server, refuse a busy port, spawn if free.

    info is /info of whatever answers on the port (None = nothing listening).
    A response without a 'checkpoint' key is not a deploy server -> refuse.
    """
    if info is None:
        return "spawn"
    if "checkpoint" in info and info["checkpoint"] == want_checkpoint:
        return "reuse"
    return "refuse"
