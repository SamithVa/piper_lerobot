# Deploy launcher + presets design

**Date:** 2026-07-09
**Status:** approved
**Builds on:** `2026-07-04-piper-deploy-client-server-design.md` (the working
client/server in `deploy/`)

## Goal

Let labmates on this machine run any deployed policy on the Piper arms with
**one command**, without knowing which conda env, PYTHONPATH, camera config,
or checkpoint flags to use — and give people adding a new VLA (openpi, GR00T,
…) a runnable sample showing exactly what to implement. Also simplify the
existing deploy code where a newcomer actually struggles.

Audience: labmates on this same machine (not packaging / remote deployment).

## Pain points addressed

1. **Two-terminal dance** — start server, wait, start client with matching port.
2. **Tribal knowledge** — env paths, PYTHONPATH ordering, camera maps live in
   shell script bodies and people's heads.
3. **pi05 cold start** — first `/predict` after a server start costs ~15-20s
   (torch.compile + CUDA-graph capture), so the server should stay warm and be
   reused across runs.

## Design

### 1. Preset files — `deploy/presets/<name>.json`

One JSON file per policy capturing everything currently smeared across shell
scripts. JSON (not YAML) so the base-python launcher stays stdlib-only.

```json
{
  "_notes": "why this env/pins — free text, ignored by the launcher",
  "server": {
    "python": "/home/embodied/miniconda3/envs/lerobot_pi05/bin/python",
    "pythonpath": ["/data/wanshan/VLAs/lerobot-pi05-serve/src", "."],
    "env": {
      "HF_HOME": "/data/.cache/huggingface",
      "HF_HUB_OFFLINE": "1",
      "TRANSFORMERS_OFFLINE": "1",
      "CUDA_VISIBLE_DEVICES": "0"
    },
    "adapter": "lerobot",
    "args": {"checkpoint": "outputs/pi05", "device": "cuda", "fps": "30"},
    "port": 8080
  },
  "client": {
    "robot_type": "bi_piper_follower",
    "robot_id": "bi_piper",
    "cameras": {"top": {"type": "opencv", "index_or_path": "/dev/top", "...": "..."}},
    "camera_map": {"top": "top", "l_wrist": "l_wrist", "r_wrist": "r_wrist"},
    "first_predict_timeout_s": 90
  }
}
```

Ships with three presets:

- `pi05.json` — lerobot adapter, `lerobot_pi05` env + v0.5.1 worktree
  PYTHONPATH, identity camera_map (current `serve_pi05.sh` +
  `run_client_pi05.sh` knowledge).
- `smolvla.json` — lerobot adapter, `lerobot` env, camera1/2/3 rename map
  (current `serve_smolvla.sh` + `run_client.sh` knowledge).
- `example.json` — dummy adapter, base python, no GPU/checkpoint needed;
  the runnable sample for people adding their own VLA.

Relative paths in a preset (e.g. `checkpoint`, `"."` in pythonpath) resolve
against the repo root.

### 2. Launcher — `deploy/launch.py` (stdlib-only, runs in base python)

```bash
python -m deploy.launch pi05 --task="Stack the cup on top of the bowl." --duration_s=60
# or: bash deploy/run.sh pi05 "Stack the cup on top of the bowl." 60
```

Flow:

1. Load and validate the preset (unknown preset name → list available ones).
2. `GET /info` on the preset's port.
   - Server up **and** `info["checkpoint"]` matches the preset → **reuse
     warm** (skips model load + compile cold start).
   - Server up with a *different* checkpoint → **refuse** with a clear
     message (someone else's policy is on that port; suggest `--port=...`).
     Never kill an existing server.
   - Nothing listening → **spawn**: `subprocess.Popen` of
     `<preset python> -m deploy.server --adapter=... --<args>` with the
     preset's env + PYTHONPATH, stdout/stderr → `deploy/logs/server-<preset>.log`,
     then poll `/info` until ready (timeout 120s for model load; on timeout,
     print the log path and tail).
3. Run the client **in the foreground** (same process via `deploy.client`
   entry, constructed from the preset's robot/camera config + CLI args).
   If the preset has no `client` section (e.g. `example.json`), stop after
   the server is confirmed healthy and print next steps instead.
4. Ctrl-C stops the client only; the spawned server keeps running (detached)
   so the next run reuses it warm.

CLI overrides for preset values: `--port`, `--checkpoint`, `--fps`,
`--duration_s`, `--task`, plus any client config field. The reuse/refuse/spawn
decision is a pure helper function (testable without sockets).

### 3. Protocol addition

`GET /info` gains a `"checkpoint"` field so the launcher can verify what a
running server serves. Adapter-reported; adapters without a checkpoint (dummy)
report `null`. Backward-compatible addition.

### 4. Sample for other VLAs — "deploy your own model in 3 steps"

- `deploy/adapters/dummy.py` stays the reference adapter; its docstring is
  expanded to state the full contract explicitly (image dict shapes/dtype,
  state shape, return shape/dtype, what `reset()` must clear).
- `presets/example.json` makes the server side runnable with no GPU or
  checkpoint. It has **no `client` section**: a preset without one is
  server-only, so `python -m deploy.launch example` starts (or reuses) the
  dummy server, verifies `/info`, and prints what to do next instead of
  driving the arms — dummy actions must never reach a real robot. This is
  also the launcher behavior rule for any future server-only preset.
- README section **“Deploy your own VLA in 3 steps”**:
  1. Copy `deploy/adapters/dummy.py` → `deploy/adapters/<name>.py`; implement
     `info()`, `predict_chunk(images, state, task)`, `reset()`; register in
     `deploy/adapters/__init__.py::make_adapter`.
  2. Write `deploy/presets/<name>.json` pointing at whatever python/env the
     model needs.
  3. `python -m deploy.launch <name> --task="..."`.
  The client, protocol, and chunking are the API boundary — untouched.

### 5. Code simplifications (readability for newcomers)

- **`deploy/client.py`** — replace the hand-rolled `pending` dict + daemon
  `threading.Thread` per request with a
  `concurrent.futures.ThreadPoolExecutor(max_workers=1)`; the in-flight
  request becomes a `Future` checked with `.done()` / `.result()`. Extract
  the handshake/validation block (`/info`, camera_map resolution, dim checks)
  into a helper so `main()`'s control loop fits on one screen. Behavior
  unchanged (async overlap, hold-position recovery, first-predict timeout).
- **Delete the four shell scripts** — `serve_pi05.sh`, `run_client_pi05.sh`,
  `serve_smolvla.sh`, `run_client.sh` — replaced by presets + one
  `deploy/run.sh` (3-line wrapper around `python -m deploy.launch`). The
  "why this env" commentary from `serve_pi05.sh`'s header moves to the README
  machine-notes section.
- **README restructure**, ordered for a reader who knows nothing:
  1. *Run a policy* — one command, table of available presets.
  2. *Deploy your own VLA* — the 3 steps above + example preset.
  3. *How it works* — protocol endpoints, async-overlap chunking, one
     paragraph + small diagram each.
  4. *This machine's setup notes* — pi05 env archaeology (lerobot v0.5.1 +
     transformers 5.3.0 pinning story, HF offline cache), CAN/camera udev
     notes.
- **Not touched**: `protocol.py`, `server.py`, `chunking.py`, adapter
  interface — already small, pure, and tested; churn would hurt.

### 6. Testing

- Unit: preset loading/validation (missing fields, unknown preset, relative
  path resolution); reuse/refuse/spawn decision helper (mocked `/info`
  responses); client Future-based overlap logic (existing chunking tests
  already cover the queue math).
- Integration (no GPU): launcher spawns the `example` preset's dummy server,
  polls `/info`, asserts `checkpoint: null` and reuse-on-second-launch;
  teardown kills the spawned process.
- Existing `deploy/tests` must keep passing; client refactor must not change
  observable loop behavior (tests in `test_client_helpers.py` extended for
  the extracted handshake helper).

## Out of scope

- Remote (non-localhost) deployment, auth, packaging/pip install, Docker.
- `deploy up/down/status` verbs (launcher can grow a `--stop-server` later).
- Any change to the wire protocol beyond the `/info.checkpoint` field.
