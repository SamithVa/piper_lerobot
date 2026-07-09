# Deploy Launcher + Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One-command policy deployment on the Piper arms (`python3 -m deploy.launch pi05 --task="..."`) with warm-server reuse, JSON presets replacing the four shell scripts, and a runnable no-GPU sample for people adding their own VLA.

**Architecture:** A stdlib-only `deploy/launch.py` loads `deploy/presets/<name>.json`, probes `GET /info` on the preset's port, then reuses / refuses / spawns the policy server (detached, logged) and runs `deploy.client` as a foreground subprocess. `/info` gains a `checkpoint` field for the reuse check. `client.py`'s hand-rolled thread+dict async machinery becomes a `concurrent.futures` Future.

**Tech Stack:** Python stdlib only for launcher/server/protocol (`json`, `subprocess`, `urllib`, `concurrent.futures`); numpy for the wire format; pytest for tests.

**Spec:** `docs/superpowers/specs/2026-07-09-deploy-launcher-presets-design.md`

## Global Constraints

- `deploy/launch.py`, `deploy/server.py`, `deploy/protocol.py` must import **stdlib (+numpy for protocol/server) only** — no torch, no lerobot, no yaml. Presets are JSON for this reason.
- Never kill an existing server: a busy port with a different checkpoint → refuse with a clear message.
- Spawned servers must survive the client's Ctrl-C (`start_new_session=True`) so they stay warm.
- A preset without a `client` section is **server-only**: the launcher must never send dummy/test actions to real arms.
- Test command (run from repo root): `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v`
- Behavior of the client control loop (async overlap, hold-position recovery, first-predict timeout) must not change.
- Repo root on this machine: `/data/wanshan/VLAs/piper_lerobot`. Run all commands from there.

---

### Task 0: Commit the pending pi05 working-tree changes

The working tree already contains finished pi05 deploy work that later tasks
modify or delete; commit it first so history keeps the recipe. **Do not** touch
`utils/find_all_can_port.sh` or `utils/test.py` (unrelated, not ours).

**Files:**
- Commit (already modified): `deploy/README.md`, `deploy/client.py`
- Commit (untracked): `deploy/serve_pi05.sh`, `deploy/run_client_pi05.sh`

- [ ] **Step 1: Verify the tests pass before starting**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v`
Expected: all PASS (integration tests skip without `DEPLOY_IT=1`).

- [ ] **Step 2: Commit**

```bash
git add deploy/README.md deploy/client.py deploy/serve_pi05.sh deploy/run_client_pi05.sh
git commit -m "feat(deploy): pi05 serving recipe + first-predict timeout"
```

---

### Task 1: `/info` reports the served checkpoint

**Files:**
- Modify: `deploy/adapters/base.py` (info() docstring)
- Modify: `deploy/adapters/dummy.py:32-40` (info())
- Modify: `deploy/adapters/lerobot.py:78-86` (info())
- Test: `deploy/tests/test_adapters.py`

**Interfaces:**
- Produces: `adapter.info()["checkpoint"]` — `str` (whatever `--checkpoint` value the adapter was constructed with) or `None` for adapters without one. Task 4's launcher compares this against the preset's resolved checkpoint.

- [ ] **Step 1: Write the failing test**

Append to `deploy/tests/test_adapters.py`:

```python
def test_dummy_info_reports_null_checkpoint():
    assert DummyAdapter().info()["checkpoint"] is None
```

(If the file imports `DummyAdapter` already, reuse the import; otherwise add
`from deploy.adapters.dummy import DummyAdapter`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_adapters.py -v -k checkpoint`
Expected: FAIL with `KeyError: 'checkpoint'`

- [ ] **Step 3: Implement**

In `deploy/adapters/dummy.py`, add to the dict returned by `info()`:

```python
            "checkpoint": None,
```

In `deploy/adapters/lerobot.py`, add to the dict returned by `info()`:

```python
            "checkpoint": self.checkpoint,
```

In `deploy/adapters/base.py`, update the `info()` docstring to:

```python
        """Static metadata:
        {"name": str, "image_keys": list[str], "state_dim": int,
         "action_dim": int, "chunk_size": int, "fps": float,
         "checkpoint": str | None}   # exact value the adapter was given, or None"""
```

- [ ] **Step 4: Run the full deploy suite**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v`
Expected: all PASS (`test_server.py::test_info` compares against `adapter.info()` so it picks the field up automatically).

- [ ] **Step 5: Commit**

```bash
git add deploy/adapters/base.py deploy/adapters/dummy.py deploy/adapters/lerobot.py deploy/tests/test_adapters.py
git commit -m "feat(deploy): /info reports served checkpoint for launcher reuse check"
```

---

### Task 2: client.py — Future-based overlap + `check_dims` helper

Replace the hand-rolled `pending` dict + daemon `threading.Thread` per request
with one `ThreadPoolExecutor(max_workers=1)` Future, and extract the dim
validation into a pure helper. **No behavior change.**

**Files:**
- Modify: `deploy/client.py`
- Test: `deploy/tests/test_client_helpers.py`

**Interfaces:**
- Produces: `check_dims(info: dict, motor_keys: list[str]) -> None` in `deploy/client.py` — raises `SystemExit` on `state_dim`/`action_dim` mismatch.
- Consumes: `ChunkExecutor` API unchanged (`next_action`, `should_request`, `mark_requested`, `on_chunk`, `on_request_failed`).

- [ ] **Step 1: Write the failing tests**

Append to `deploy/tests/test_client_helpers.py`:

```python
def test_check_dims_passes_on_match():
    from deploy.client import check_dims

    check_dims({"state_dim": 2, "action_dim": 2}, ["a.pos", "b.pos"])  # no raise


def test_check_dims_rejects_mismatch():
    from deploy.client import check_dims

    with pytest.raises(SystemExit, match="action_dim"):
        check_dims({"state_dim": 2, "action_dim": 3}, ["a.pos", "b.pos"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_client_helpers.py -v -k check_dims`
Expected: FAIL with `ImportError: cannot import name 'check_dims'`

- [ ] **Step 3: Implement**

In `deploy/client.py`:

1. Replace `import threading` with `import concurrent.futures` in the imports.

2. Add after `resolve_camera_map` (in the "pure helpers" section):

```python
def check_dims(info: dict, motor_keys: list[str]) -> None:
    """Fail fast — before touching the arms — if policy and robot disagree."""
    for key in ("state_dim", "action_dim"):
        if info[key] != len(motor_keys):
            raise SystemExit(
                f"{key} mismatch: policy expects {info[key]}, robot has {len(motor_keys)} motors"
            )
```

3. Replace the whole body of `main()` from `robot = make_robot_from_config(...)`
   down to the end of the function with:

```python
    robot = make_robot_from_config(cfg.robot)
    motor_keys = list(robot.action_features)
    camera_map = resolve_camera_map(
        cfg.camera_map, info["image_keys"], list(robot.cameras)
    )
    check_dims(info, motor_keys)

    robot.connect()
    executor = ChunkExecutor(cfg.chunk_threshold)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    inflight: concurrent.futures.Future | None = None  # at most one /predict in flight

    try:
        http_post(cfg.server + "/reset")

        # Blocking first chunk so the loop starts with actions in hand. Uses the
        # generous first_predict_timeout_s because the server pays a one-time
        # cold-start (compile + CUDA-graph capture) on its very first predict.
        logging.info("requesting first chunk (server cold start may take ~15-20s)...")
        payload = capture_payload(robot, camera_map, motor_keys, cfg.task)
        executor.mark_requested()
        executor.on_chunk(
            protocol.decode_chunk(http_post(cfg.server + "/predict", payload, cfg.first_predict_timeout_s))
        )
        logging.info(f"running at {fps:.0f} fps for {cfg.duration_s:.0f}s — Ctrl-C to stop")

        last_action: dict | None = None
        dry_ticks = 0
        t_end = time.perf_counter() + cfg.duration_s
        next_t = time.perf_counter()
        while time.perf_counter() < t_end:
            if inflight is not None and inflight.done():
                try:
                    usable = executor.on_chunk(protocol.decode_chunk(inflight.result()))
                    if usable == 0:
                        logging.warning(
                            "chunk arrived fully stale (inference slower than execution) — re-requesting"
                        )
                except Exception as exc:  # noqa: BLE001 — a failed predict must not kill the loop
                    logging.warning(f"/predict failed: {exc}")
                    executor.on_request_failed()
                inflight = None

            row = executor.next_action()
            if row is not None:
                last_action = action_to_dict(row, motor_keys)
                robot.send_action(last_action)
                dry_ticks = 0
            elif last_action is not None:
                robot.send_action(last_action)  # hold position while recovering
                dry_ticks += 1
                if dry_ticks % max(int(fps), 1) == 1:
                    logging.warning("action queue dry — holding position")

            if executor.should_request():
                payload = capture_payload(robot, camera_map, motor_keys, cfg.task)
                executor.mark_requested()
                inflight = pool.submit(
                    http_post, cfg.server + "/predict", payload, cfg.predict_timeout_s
                )

            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        logging.info("interrupted — stopping")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        robot.disconnect()
        logging.info("robot disconnected")
```

4. Delete the now-unused `post_predict_async` closure and the `pending` dict
   (they were inside the old `main()` body). The module must no longer
   reference `threading`.

Note `protocol.decode_chunk` moved from the worker thread into the loop thread
(the Future returns raw bytes); decoding is a cheap `np.load`, and the
`except Exception` still covers both HTTP and decode failures exactly as the
old worker's try/except did.

- [ ] **Step 4: Run the full deploy suite**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v`
Expected: all PASS

- [ ] **Step 5: Verify no stray references**

Run: `grep -n "threading\|pending\|post_predict_async" deploy/client.py`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add deploy/client.py deploy/tests/test_client_helpers.py
git commit -m "refactor(deploy): client async overlap via Future; extract check_dims"
```

---

### Task 3: Preset files + `load_preset` / `resolve_path` / `decide`

The pure half of the launcher: preset loading/validation and the
reuse/refuse/spawn decision, plus the three shipped presets.

**Files:**
- Create: `deploy/launch.py`
- Create: `deploy/presets/pi05.json`
- Create: `deploy/presets/smolvla.json`
- Create: `deploy/presets/example.json`
- Test: `deploy/tests/test_launch.py`
- Modify: `.gitignore` (add `deploy/logs/`)

**Interfaces:**
- Produces (in `deploy/launch.py`, all consumed by Task 4):
  - `REPO_ROOT: Path`, `PRESETS_DIR: Path`
  - `load_preset(name_or_path: str, presets_dir: Path = PRESETS_DIR) -> dict` — bare name → `deploy/presets/<name>.json`; anything ending in `.json` is a path; unknown → `SystemExit` listing available presets; validates `server` section has `python`, `adapter`, `port`.
  - `resolve_path(value: str) -> str` — resolve against `REPO_ROOT` if it exists on disk, else return unchanged (hub ids).
  - `decide(info: dict | None, want_checkpoint: str | None) -> str` — `"spawn"` | `"reuse"` | `"refuse"`.

- [ ] **Step 1: Write the failing tests**

Create `deploy/tests/test_launch.py`:

```python
import json

import pytest

from deploy import launch


def write_preset(tmp_path, name, body):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(body))
    return path


MINIMAL = {"server": {"python": "python3", "adapter": "dummy", "port": 8090}}


def test_load_preset_by_name(tmp_path):
    write_preset(tmp_path, "mini", MINIMAL)
    assert launch.load_preset("mini", presets_dir=tmp_path)["server"]["adapter"] == "dummy"


def test_load_preset_by_path(tmp_path):
    path = write_preset(tmp_path, "mini", MINIMAL)
    assert launch.load_preset(str(path))["server"]["port"] == 8090


def test_load_preset_unknown_lists_available(tmp_path):
    write_preset(tmp_path, "mini", MINIMAL)
    with pytest.raises(SystemExit, match="mini"):
        launch.load_preset("nope", presets_dir=tmp_path)


def test_load_preset_rejects_missing_server_field(tmp_path):
    write_preset(tmp_path, "bad", {"server": {"python": "python3"}})
    with pytest.raises(SystemExit, match="adapter"):
        launch.load_preset("bad", presets_dir=tmp_path)


def test_decide_spawn_when_nothing_listening():
    assert launch.decide(None, "/ckpt") == "spawn"


def test_decide_reuse_on_matching_checkpoint():
    assert launch.decide({"checkpoint": "/ckpt"}, "/ckpt") == "reuse"
    assert launch.decide({"checkpoint": None}, None) == "reuse"  # dummy adapter


def test_decide_refuse_on_mismatch_or_foreign_server():
    assert launch.decide({"checkpoint": "/other"}, "/ckpt") == "refuse"
    assert launch.decide({}, None) == "refuse"  # no checkpoint key: not ours


def test_resolve_path_repo_relative_and_hub_id():
    assert launch.resolve_path(".") == str(launch.REPO_ROOT)
    assert launch.resolve_path("samithva/pi05_stack_cup_bowl") == "samithva/pi05_stack_cup_bowl"


def test_shipped_presets_are_valid():
    for name in ("pi05", "smolvla", "example"):
        preset = launch.load_preset(name)
        assert "server" in preset
    assert "client" not in launch.load_preset("example")  # server-only sample
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_launch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deploy.launch'` (collection error).

- [ ] **Step 3: Create `deploy/launch.py`**

```python
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
```

- [ ] **Step 4: Create the three presets**

Create `deploy/presets/pi05.json` (values verbatim from the deleted-in-Task-5
`serve_pi05.sh` / `run_client_pi05.sh`):

```json
{
  "_notes": [
    "Checkpoint outputs/pi05 (samithva/pi05_stack_cup_bowl) was trained with lerobot v0.5.1 + transformers 5.3.0; BOTH must match exactly.",
    "transformers 5.5.x loads and runs but SILENTLY ignores vision/state inputs (nMAE 0.96 vs 0.03) — must be 5.3.0.",
    "lerobot_pi05 env = clone of the lerobot env with transformers pinned to 5.3.0; pythonpath puts the v0.5.1 lerobot source worktree first.",
    "HF_HUB_OFFLINE: the box's proxy is SOCKS (python can't use it); the gated paligemma tokenizer loads from the HF cache.",
    "pi05 image keys are the raw camera names, hence the identity camera_map.",
    "See README 'This machine's setup notes' for the full story."
  ],
  "server": {
    "python": "/home/embodied/miniconda3/envs/lerobot_pi05/bin/python",
    "pythonpath": ["/data/wanshan/VLAs/lerobot-pi05-serve/src", "."],
    "env": {
      "CUDA_VISIBLE_DEVICES": "0",
      "HF_HOME": "/data/.cache/huggingface",
      "HF_HUB_OFFLINE": "1",
      "TRANSFORMERS_OFFLINE": "1"
    },
    "adapter": "lerobot",
    "args": {"checkpoint": "outputs/pi05", "device": "cuda", "fps": "30"},
    "port": 8080
  },
  "client": {
    "python": "/home/embodied/miniconda3/bin/python",
    "pythonpath": ["src", "."],
    "robot_type": "bi_piper_follower",
    "robot_id": "bi_piper",
    "cameras": {
      "l_wrist": {"type": "opencv", "index_or_path": "/dev/l_wrist", "width": 480, "height": 640, "fps": 120, "rotation": -90, "fourcc": "MJPG"},
      "top":     {"type": "opencv", "index_or_path": "/dev/top",     "width": 640, "height": 480, "fps": 120, "rotation": 0,   "fourcc": "MJPG"},
      "r_wrist": {"type": "opencv", "index_or_path": "/dev/r_wrist", "width": 480, "height": 640, "fps": 120, "rotation": 90,  "fourcc": "MJPG"}
    },
    "camera_map": {"top": "top", "l_wrist": "l_wrist", "r_wrist": "r_wrist"},
    "first_predict_timeout_s": 90
  }
}
```

Create `deploy/presets/smolvla.json` (values verbatim from `serve_smolvla.sh` /
`run_client.sh`):

```json
{
  "_notes": [
    "Trained by lerobot-train with the lerobot env's installed lerobot (0.5.x), so server pythonpath is repo root only — NOT src (this checkout's older lerobot can't parse the saved config).",
    "smolvla was trained with renamed cameras, hence the camera1/2/3 camera_map."
  ],
  "server": {
    "python": "/home/embodied/miniconda3/envs/lerobot/bin/python",
    "pythonpath": ["."],
    "env": {"CUDA_VISIBLE_DEVICES": "0"},
    "adapter": "lerobot",
    "args": {
      "checkpoint": "outputs/train/smolvla_bimanual_stack_cup_bowl/checkpoints/last/pretrained_model",
      "device": "cuda",
      "fps": "30"
    },
    "port": 8080
  },
  "client": {
    "python": "/home/embodied/miniconda3/bin/python",
    "pythonpath": ["src", "."],
    "robot_type": "bi_piper_follower",
    "robot_id": "bi_piper",
    "cameras": {
      "l_wrist": {"type": "opencv", "index_or_path": "/dev/l_wrist", "width": 480, "height": 640, "fps": 120, "rotation": -90, "fourcc": "MJPG"},
      "top":     {"type": "opencv", "index_or_path": "/dev/top",     "width": 640, "height": 480, "fps": 120, "rotation": 0,   "fourcc": "MJPG"},
      "r_wrist": {"type": "opencv", "index_or_path": "/dev/r_wrist", "width": 480, "height": 640, "fps": 120, "rotation": 90,  "fourcc": "MJPG"}
    },
    "camera_map": {"top": "camera1", "l_wrist": "camera2", "r_wrist": "camera3"}
  }
}
```

Create `deploy/presets/example.json` (server-only sample — **no `client`
section**, so the launcher never drives real arms with dummy actions):

```json
{
  "_notes": [
    "Runnable sample for 'deploy your own VLA': dummy adapter, no GPU, no checkpoint.",
    "Server-only (no 'client' section): launch starts the server and checks /info, nothing touches the arms.",
    "Copy this + deploy/adapters/dummy.py as the starting point for a new policy; any python with numpy works.",
    "Port 8090 so it never collides with a real policy on 8080."
  ],
  "server": {
    "python": "python3",
    "pythonpath": ["."],
    "adapter": "dummy",
    "args": {"state_dim": "14", "action_dim": "14", "chunk_size": "10", "fps": "30"},
    "port": 8090
  }
}
```

Append to `.gitignore`:

```
deploy/logs/
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_launch.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add deploy/launch.py deploy/presets/ deploy/tests/test_launch.py .gitignore
git commit -m "feat(deploy): presets + launcher preset loading and reuse/refuse/spawn decision"
```

---

### Task 4: Launcher process management + CLI + `run.sh`

The impure half: probe the port, spawn the server detached with logging, wait
for readiness, build and run the client subprocess.

**Files:**
- Modify: `deploy/launch.py`
- Create: `deploy/run.sh`
- Test: `deploy/tests/test_launch.py`

**Interfaces:**
- Consumes: Task 3's `load_preset`, `resolve_path`, `decide`, `REPO_ROOT`; Task 1's `/info.checkpoint`.
- Produces:
  - `server_info(port: int, timeout: float = 3.0) -> dict | None` — `None` = nothing listening; `{}` = something non-deploy answered.
  - `spawn_server(preset: dict, name: str) -> subprocess.Popen` — detached (`start_new_session=True`), log at `deploy/logs/server-<name>.log`.
  - `wait_ready(port: int, proc: subprocess.Popen, name: str, timeout_s: float = 120.0) -> dict` — polls `/info`; `SystemExit` with log tail if the process dies or times out.
  - `build_client_cmd(preset: dict, task: str, extra_flags: list[str]) -> tuple[list[str], dict]` — argv + env for the client subprocess.
  - `main(argv: list[str] | None = None) -> subprocess.Popen | None` — the Popen if it spawned a server, `None` if it reused one (tests use this for teardown).

- [ ] **Step 1: Write the failing tests**

Append to `deploy/tests/test_launch.py`:

```python
import socket
import sys


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def dummy_preset(port):
    return {
        "server": {
            "python": sys.executable,
            "pythonpath": ["."],
            "adapter": "dummy",
            "args": {"fps": "30"},
            "port": port,
        }
    }


def test_build_client_cmd_maps_preset_to_flags():
    preset = {
        "server": {"python": "p", "adapter": "lerobot", "port": 8080},
        "client": {
            "python": "/some/python",
            "pythonpath": ["src", "."],
            "robot_type": "bi_piper_follower",
            "robot_id": "bi_piper",
            "cameras": {"top": {"type": "opencv"}},
            "camera_map": {"top": "camera1"},
            "first_predict_timeout_s": 90,
        },
    }
    cmd, env = launch.build_client_cmd(preset, "stack", ["--fps=15"])
    assert cmd[0] == "/some/python"
    assert cmd[1:3] == ["-m", "deploy.client"]
    assert "--robot.type=bi_piper_follower" in cmd
    assert "--robot.id=bi_piper" in cmd
    assert '--robot.cameras={"top": {"type": "opencv"}}' in cmd
    assert "--server=http://127.0.0.1:8080" in cmd
    assert "--task=stack" in cmd
    assert '--camera_map={"top": "camera1"}' in cmd
    assert "--first_predict_timeout_s=90" in cmd  # non-structural keys forwarded
    assert cmd[-1] == "--fps=15"  # CLI extras win (draccus takes the last value)
    assert str(launch.REPO_ROOT / "src") in env["PYTHONPATH"]


def test_launch_spawns_then_reuses_then_refuses(tmp_path):
    port = free_port()
    path = write_preset(tmp_path, "it", dummy_preset(port))
    proc = launch.main([str(path)])
    try:
        assert proc is not None, "first launch should spawn a server"
        info = launch.server_info(port)
        assert info["name"] == "dummy"
        assert info["checkpoint"] is None
        assert launch.main([str(path)]) is None, "second launch should reuse it"

        other = dummy_preset(port)
        other["server"]["args"]["checkpoint"] = "outputs/does_not_exist"
        other_path = write_preset(tmp_path, "other", other)
        with pytest.raises(SystemExit, match="busy"):
            launch.main([str(other_path)])
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=10)


def test_wait_ready_reports_dead_server(tmp_path):
    port = free_port()
    preset = dummy_preset(port)
    preset["server"]["args"] = {"no_such_flag": "boom"}  # DummyAdapter() raises TypeError
    path = write_preset(tmp_path, "dead", preset)
    with pytest.raises(SystemExit, match="server-dead.log"):
        launch.main([str(path)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_launch.py -v -k "build_client or spawns or dead"`
Expected: FAIL with `AttributeError: module 'deploy.launch' has no attribute ...`

- [ ] **Step 3: Implement the process-management half**

Add to `deploy/launch.py` imports:

```python
import os
import subprocess
import time
import urllib.error
import urllib.request
from argparse import ArgumentParser
```

Add below `decide()`:

```python
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
    raise SystemExit(f"server not ready after {timeout_s:.0f}s — see {_log_path(name)}")


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests/test_launch.py -v`
Expected: all PASS (the integration tests spawn a real dummy server with `sys.executable`; no GPU needed).

- [ ] **Step 5: Create `deploy/run.sh`**

```bash
#!/bin/bash
# One-command deploy. Usage: bash deploy/run.sh <preset> ["task"] [duration_s]
# e.g.: bash deploy/run.sh pi05 "Stack the cup on top of the bowl." 60
set -eu
cd "$(dirname "$0")/.."
ARGS=("${1:?usage: run.sh <preset> [task] [duration_s]}")
# note: if-statements, not `[ ... ] && ...` — a failing && list trips set -e
if [ -n "${2-}" ]; then ARGS+=(--task="$2"); fi
if [ -n "${3-}" ]; then ARGS+=(--duration_s="$3"); fi
exec python3 -m deploy.launch "${ARGS[@]}"
```

Run: `chmod +x deploy/run.sh`

- [ ] **Step 6: End-to-end smoke test with the example preset**

```bash
python3 -m deploy.launch example
curl -s http://127.0.0.1:8090/info
python3 -m deploy.launch example   # second run must print "reusing warm server"
pkill -f "deploy.server --adapter=dummy" || true   # clean up the detached demo server
```

Expected: first run prints `server ready: dummy` then the server-only message;
curl returns JSON with `"checkpoint": null`; second run prints
`reusing warm server on port 8090: dummy`. The pkill cleans up the demo server.

- [ ] **Step 7: Run the full deploy suite**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add deploy/launch.py deploy/run.sh deploy/tests/test_launch.py
git commit -m "feat(deploy): one-command launcher with warm-server reuse + run.sh"
```

---

### Task 5: Delete shell scripts, expand dummy docstring, rewrite README

**Files:**
- Delete: `deploy/serve_pi05.sh`, `deploy/run_client_pi05.sh`, `deploy/serve_smolvla.sh`, `deploy/run_client.sh`
- Modify: `deploy/adapters/dummy.py:1-6` (docstring only)
- Rewrite: `deploy/README.md`

**Interfaces:**
- Consumes: everything shipped in Tasks 1-4 (commands in the README must match the actual CLI).

- [ ] **Step 1: Delete the four shell scripts**

```bash
git rm deploy/serve_pi05.sh deploy/run_client_pi05.sh deploy/serve_smolvla.sh deploy/run_client.sh
```

(Their env/PYTHONPATH knowledge now lives in the presets; the pi05 "why"
commentary lands in the README section below.)

- [ ] **Step 2: Expand the dummy adapter docstring**

Replace `deploy/adapters/dummy.py`'s module docstring (lines 1-6) with:

```python
"""Deterministic stand-in policy: no torch, instant. Used by the deploy test
suite, by presets/example.json, and as the reference for writing new adapters.

The full adapter contract (see also base.py):
  info() -> {"name": str, "image_keys": list[str], "state_dim": int,
             "action_dim": int, "chunk_size": int, "fps": float,
             "checkpoint": str | None}
  predict_chunk(images, state, task) -> np.ndarray
      images: {image_key: HWC uint8 RGB array} — exactly the keys in
              info()["image_keys"]; state: float32 (state_dim,); task: str.
      Returns float32 (chunk_size, action_dim): absolute motor targets in the
      same units/order the robot's action_features use.
  reset() -> None — clear per-episode state (action queues, KV caches, ...).

Constructor kwargs arrive as STRINGS (forwarded from --key=value server
flags), so coerce types yourself, as done below.

Action row t here is the constant vector [t, t, ..., t], so tests can tell
which step of a chunk got executed.
"""
```

- [ ] **Step 3: Rewrite `deploy/README.md`**

Replace the whole file with:

````markdown
# deploy — run a policy on the Piper arms

Client-server split: the **server** hosts a policy inside whatever conda env
that policy needs; the **client** drives the arms + cameras (base python) and
streams action chunks over localhost HTTP with async overlap, so the arms
never pause for inference.

Design specs: `docs/superpowers/specs/2026-07-04-piper-deploy-client-server-design.md`,
`docs/superpowers/specs/2026-07-09-deploy-launcher-presets-design.md`

## 1. Run a policy (one command)

```bash
python3 -m deploy.launch pi05 --task="Stack the cup on top of the bowl." --duration_s=60
# same thing:
bash deploy/run.sh pi05 "Stack the cup on top of the bowl." 60
```

A **preset** (`deploy/presets/<name>.json`) holds everything you'd otherwise
have to know: which conda env serves the policy, checkpoint, port, camera
configs, camera→image-key map.

| preset    | policy                                             | notes                        |
|-----------|----------------------------------------------------|------------------------------|
| `pi05`    | `outputs/pi05` (samithva/pi05_stack_cup_bowl)      | pinned env — see §4          |
| `smolvla` | smolvla_bimanual_stack_cup_bowl (last checkpoint)  | lerobot env                  |
| `example` | dummy adapter — no GPU, no checkpoint, no arms     | server-only teaching sample  |

What the launcher does with the preset's port:

- **reuse** — a server is already up with the same checkpoint → skip straight
  to the client. This matters for pi05: its first `/predict` costs ~15-20s of
  one-time `torch.compile` + CUDA-graph capture, and a warm server skips it.
- **refuse** — the port serves a *different* policy → clear error, nothing is
  killed (someone may be using it). Rerun with `--port=<free port>`.
- **spawn** — nothing listening → start the server detached
  (log: `deploy/logs/server-<preset>.log`), wait until `/info` answers, go.

Ctrl-C stops the **client only**; the server stays warm for the next run.
Overrides: `--checkpoint=`, `--port=`, `--fps=`, plus any client flag as
`--key=value` (e.g. `--chunk_threshold=0.7`).

## 2. Deploy your own VLA (openpi, GR00T, ...) in 3 steps

First see the whole pipeline work with zero setup — no GPU, checkpoint, or
robot involved:

```bash
python3 -m deploy.launch example      # dummy policy server + /info check
curl -s http://127.0.0.1:8090/info
```

Then:

1. **Adapter** — copy `deploy/adapters/dummy.py` → `deploy/adapters/<name>.py`
   and implement `info()`, `predict_chunk(images, state, task)`, `reset()`
   (the contract is spelled out in `dummy.py`'s docstring and
   `adapters/base.py`). Register it in `deploy/adapters/__init__.py::make_adapter`.
2. **Preset** — copy `deploy/presets/example.json` → `presets/<name>.json`;
   point `server.python` at whatever env your model needs and add a `client`
   section (copy `pi05.json`'s). The server imports stdlib+numpy only, so it
   runs inside *any* env.
3. **Run** — `python3 -m deploy.launch <name> --task="..."`.

The client, protocol, and chunking never change — the adapter (plus a preset)
is the whole integration surface.

## 3. How it works

```
launch.py ──spawns/reuses──▶ server.py (policy env)      adapters/<name>.py
    │                            ▲   /info /predict /reset    ▲
    └──runs──▶ client.py ────────┘ npz obs → npy chunk        │ policy code
               (base python: arms + cameras)             torch etc. live here
```

- **Protocol** (`protocol.py`): `GET /info` →
  `{"name", "image_keys", "state_dim", "action_dim", "chunk_size", "fps", "checkpoint"}`;
  `POST /predict` — body: npz of images (`img_<key>`, HWC uint8 RGB) + `state`
  (float32) + `task` (str), reply: `.npy` bytes `(chunk_size, action_dim)`;
  `POST /reset` clears episode state. Errors → HTTP 500 with the traceback.
- **Async overlap** (`chunking.py` + `client.py`): the client executes the
  current chunk at the control fps and fires the next `/predict` when the
  chunk is half consumed (`chunk_threshold`); rows the arm already executed
  while the request was in flight are skipped from the fresh chunk. If the
  queue runs dry (slow inference), the client holds position and re-requests.
- **Files**: `launch.py` (one-command entry), `server.py` (stdlib HTTP host),
  `client.py` (robot loop), `protocol.py` (wire format), `chunking.py`
  (overlap math), `adapters/` (one per policy family), `presets/` (one per
  deployable policy).

## 4. This machine's setup notes (pi05)

- **Version pinning**: `outputs/pi05` was trained with **lerobot v0.5.1 +
  transformers 5.3.0**; both must match exactly. transformers 5.5.x loads and
  runs but *silently breaks* vision+state→action conditioning — the policy
  ignores inputs and emits a near-constant chunk (sensitivity probe: nMAE 0.03
  with tf 5.3.0 vs 0.96 with 5.5.x). lerobot 0.5.2 renamed
  `delta_actions_processor` → `relative_actions_processor` and can't parse the
  v0.5.1 config.
- **How the preset satisfies that**: conda env `lerobot_pi05` (clone of
  `lerobot` with `transformers==5.3.0`) + lerobot **source** worktree at tag
  v0.5.1 (`/data/wanshan/VLAs/lerobot-pi05-serve`) first on `PYTHONPATH`.
- **HF offline**: the box's proxy is SOCKS (python can't use it), so the gated
  paligemma tokenizer loads from the HF cache (`HF_HOME=/data/.cache/huggingface`,
  `HF_HUB_OFFLINE=1`); the once-missing `config.json` was added to the cache.
- **Cold start**: first `/predict` after a server start takes ~15-20s
  (`torch.compile` + CUDA-graph capture). The client's first blocking request
  uses `first_predict_timeout_s` (90s) to absorb it. Do **not** add a
  server-side warmup predict — a throwaway forward pass poisons pi05's
  CUDA-graph state (`Offset increment outside graph capture` on every real
  predict after).
- **Two-env split**: client always runs in base python (pyAgxArm + cameras);
  servers run in per-policy envs. Install deps into the right one.

## Tests

```bash
PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v
# integration (downloads smolvla_base, needs GPU):
DEPLOY_IT=1 PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python \
    -m pytest deploy/tests/test_lerobot_adapter.py -v
```
````

- [ ] **Step 4: Check nothing still references the deleted scripts**

Run: `grep -rn "serve_pi05\|serve_smolvla\|run_client.sh\|run_client_pi05" --include="*.py" --include="*.sh" --include="*.md" . | grep -v docs/superpowers | grep -v .git`
Expected: no output (spec/plan history in `docs/superpowers/` may mention them; that's fine).

- [ ] **Step 5: Run the full deploy suite**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add -A deploy/README.md deploy/adapters/dummy.py
git commit -m "docs(deploy): presets replace shell scripts; newcomer-first README + adapter contract"
```

---

### Task 6: Final verification

- [ ] **Step 1: Full suite**

Run: `PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v`
Expected: all PASS

- [ ] **Step 2: Real pi05 smoke test (requires arms powered + CAN up; skip if hardware unavailable)**

```bash
python3 -m deploy.launch pi05 --task="Stack the cup on top of the bowl." --duration_s=15
```

Expected: server spawns (or reuses), `server ready: lerobot:pi05:...` printed,
client connects, arms run for 15s, Ctrl-C-free exit. Then rerun the same
command: it must print `reusing warm server` and start within a few seconds
with **no** compile cold-start (first chunk < 1s).

- [ ] **Step 3: Update the spec status line**

In `docs/superpowers/specs/2026-07-09-deploy-launcher-presets-design.md`,
change `**Status:** approved` to `**Status:** implemented`.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-07-09-deploy-launcher-presets-design.md
git commit -m "docs(deploy): mark launcher+presets spec implemented"
```
