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
import os
import subprocess
import time
import urllib.error
import urllib.request
from argparse import ArgumentParser
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


LOGS_DIR = REPO_ROOT / "deploy" / "logs"

# client preset keys that build_client_cmd handles specially; everything else
# is forwarded verbatim as --key=value (e.g. first_predict_timeout_s)
CLIENT_STRUCTURAL_KEYS = {"python", "pythonpath", "robot_type", "robot_id", "cameras", "camera_map"}


def server_info(port: int, timeout: float = 3.0) -> dict | None:
    """/info of whatever listens on the port. None = nothing listening;
    {} = something answered but not like a deploy server."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/info", timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.HTTPError, json.JSONDecodeError):
        return {}
    except OSError:
        return None


def _log_path(name: str) -> Path:
    return LOGS_DIR / f"server-{name}.log"


def _log_tail(name: str, lines: int = 15) -> str:
    try:
        return "\n".join(_log_path(name).read_text().splitlines()[-lines:])
    except OSError:
        return ""


def build_server_cmd(preset: dict) -> tuple[list[str], dict]:
    server = preset["server"]
    cmd = [
        server["python"], "-m", "deploy.server",
        f"--adapter={server['adapter']}",
        f"--port={server['port']}",
    ]
    for key, value in server.get("args", {}).items():
        if key == "checkpoint":
            value = resolve_path(str(value))
        cmd.append(f"--{key}={value}")
    env = dict(os.environ)
    env.update(server.get("env", {}))
    env["PYTHONPATH"] = os.pathsep.join(resolve_path(p) for p in server.get("pythonpath", ["."]))
    return cmd, env


def spawn_server(preset: dict, name: str) -> subprocess.Popen:
    cmd, env = build_server_cmd(preset)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = open(_log_path(name), "ab")
    print(f"[deploy.launch] starting server: {' '.join(cmd)}")
    print(f"[deploy.launch] server log: {_log_path(name)}")
    # start_new_session: Ctrl-C hits the terminal's foreground process group
    # (launcher + client); the server must NOT be in it so it stays warm.
    return subprocess.Popen(
        cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_ready(port: int, proc: subprocess.Popen, name: str, timeout_s: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"server exited (code {proc.returncode}) — see {_log_path(name)}\n{_log_tail(name)}"
            )
        info = server_info(port)
        if info:
            return info
        time.sleep(0.5)
    raise SystemExit(
        f"server not ready after {timeout_s:.0f}s — left running as pid {proc.pid} "
        f"(may just be a slow model load). Watch {_log_path(name)}; rerun this command "
        f"to reuse it once ready, or `kill {proc.pid}` to abort it."
    )


def build_client_cmd(preset: dict, task: str, extra_flags: list[str]) -> tuple[list[str], dict]:
    client = preset["client"]
    cmd = [
        client["python"], "-m", "deploy.client",
        f"--robot.type={client['robot_type']}",
        f"--robot.id={client['robot_id']}",
        f"--robot.cameras={json.dumps(client['cameras'])}",
        f"--server=http://127.0.0.1:{preset['server']['port']}",
        f"--task={task}",
        f"--camera_map={json.dumps(client.get('camera_map', {}))}",
    ]
    for key, value in client.items():
        if key not in CLIENT_STRUCTURAL_KEYS:
            cmd.append(f"--{key}={value}")
    cmd += extra_flags  # last value wins in draccus, so CLI overrides preset
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(resolve_path(p) for p in client.get("pythonpath", ["src", "."]))
    return cmd, env


def main(argv: list[str] | None = None) -> subprocess.Popen | None:
    parser = ArgumentParser(description="One-command Piper deploy: warm-reused policy server + robot client")
    parser.add_argument("preset", help="preset name under deploy/presets/, or a path to a preset .json")
    parser.add_argument("--task", default="", help="natural-language task for the policy")
    parser.add_argument("--duration_s", default=None, help="episode length (client default: 60)")
    parser.add_argument("--port", type=int, default=None, help="override the preset's server port")
    parser.add_argument("--checkpoint", default=None, help="override the preset's checkpoint")
    parser.add_argument("--fps", default=None, help="override the preset's fps")
    args, extra = parser.parse_known_args(argv)
    for flag in extra:
        if not (flag.startswith("--") and "=" in flag):
            parser.error(f"extra client flags must look like --key=value, got: {flag}")

    preset = load_preset(args.preset)
    name = Path(args.preset).stem
    if args.port is not None:
        preset["server"]["port"] = args.port
    for key in ("checkpoint", "fps"):
        value = getattr(args, key)
        if value is not None:
            preset["server"].setdefault("args", {})[key] = value

    port = preset["server"]["port"]
    want = preset["server"].get("args", {}).get("checkpoint")
    want = resolve_path(str(want)) if want is not None else None

    proc = None
    info = server_info(port)
    action = decide(info, want)
    if action == "reuse":
        print(f"[deploy.launch] reusing warm server on port {port}: {info['name']}")
    elif action == "refuse":
        raise SystemExit(
            f"port {port} is busy with a different policy "
            f"(serving checkpoint={info.get('checkpoint')!r}, preset wants {want!r}).\n"
            f"Not killing it — someone may be using it. Rerun with --port=<free port>."
        )
    else:
        proc = spawn_server(preset, name)
        info = wait_ready(port, proc, name)
        print(f"[deploy.launch] server ready: {info['name']}")

    if "client" not in preset:
        print(f"[deploy.launch] preset '{name}' is server-only — not driving the arms.")
        print(f"[deploy.launch] poke it:  curl http://127.0.0.1:{port}/info")
        return proc

    if not args.task:
        parser.error("--task is required to drive the robot")
    if args.duration_s is not None:
        extra.append(f"--duration_s={args.duration_s}")
    cmd, env = build_client_cmd(preset, args.task, extra)
    print(f"[deploy.launch] starting client (Ctrl-C stops the client; the server stays warm)")
    client_proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env)
    try:
        client_proc.wait()
    except KeyboardInterrupt:
        client_proc.wait()  # client got the same SIGINT; let it disconnect the arms cleanly
    if client_proc.returncode:
        raise SystemExit(client_proc.returncode)
    return proc


if __name__ == "__main__":
    main()
