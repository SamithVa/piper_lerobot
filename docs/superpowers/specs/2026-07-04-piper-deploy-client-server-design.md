# Piper policy deployment client-server — design

Date: 2026-07-04
Status: approved by user (conversation), pending spec review

## Problem

Each policy family needs a different conda env on this box (SmolVLA → `lerobot`
env with transformers 5.x; Pi0.5 → patched `transformers@fix/lerobot_openpi`
fork; GR00T conflicts with both), while robot control (piper_sdk + cameras)
only works reliably in base python. Deploying a new policy today means hand-
wiring a bespoke eval script per policy/env combination.

## Goal

One reusable client-server pair on this single machine (RTX 4090 + Piper arms):

- **Server** runs in the policy's own conda env and exposes a minimal
  policy-agnostic inference API.
- **Client** always runs in base python (`PYTHONPATH=src`), drives cameras +
  CAN, and executes action chunks smoothly.
- Deploying a future policy = implement one adapter class (or reuse the
  generic lerobot adapter) + a launcher script. No client changes.

Non-goals: remote/multi-machine deployment (localhost only, though nothing
prevents pointing the client at another host), authentication, multi-client
serving, dataset recording during eval.

## Architecture

```
robot box (single machine)
┌─ client: base python, PYTHONPATH=src ─────────────┐
│ bi_piper_follower / piper_follower + cameras      │
│ async-overlap chunk executor @ policy fps         │
└──────────────┬────────────────────────────────────┘
               │ HTTP, localhost:8080 (npz payloads)
┌─ server: policy's conda env ──────────────────────┐
│ stdlib http.server → PolicyAdapter                │
│ adapters: lerobot (smolvla/act/pi05), future ones │
└───────────────────────────────────────────────────┘
```

New top-level `deploy/` package (not under `src/lerobot` — the server side
must be importable in envs that cannot import this checkout's lerobot):

```
deploy/
├── server.py            # stdlib-only HTTP server; --adapter, --port, flags forwarded to adapter
├── adapters/
│   ├── __init__.py      # adapter registry (name → class)
│   ├── base.py          # PolicyAdapter ABC
│   └── lerobot.py       # generic lerobot-policy adapter
├── client.py            # robot loop (imports this repo's lerobot for robots/cameras)
├── serve_smolvla.sh     # launcher: lerobot env + smolvla checkpoint
└── run_client.sh        # launcher: base python + bimanual robot config
```

## Components

### PolicyAdapter (deploy/adapters/base.py)

```python
class PolicyAdapter(ABC):
    def info(self) -> dict:
        # {"image_keys": [...], "state_dim": int, "action_dim": int,
        #  "chunk_size": int, "fps": float, "name": str}
    def predict_chunk(self, images: dict[str, np.ndarray],  # HWC uint8 RGB
                      state: np.ndarray, task: str) -> np.ndarray:
        # returns (chunk_size, action_dim) float32
    def reset(self) -> None: ...
```

Adapters may accept arbitrary CLI kwargs (checkpoint path, device, dtype);
`server.py` forwards unrecognized `--key=value` flags to the adapter ctor.

### Generic lerobot adapter (deploy/adapters/lerobot.py)

Loads any lerobot checkpoint via `policy.path` using the policy factory +
pre/post processors (same path `lerobot_record` uses when evaluating).
Covers SmolVLA, ACT, Pi0.5 with zero policy-specific code. It reads
image keys / dims / chunk size from the checkpoint config. Runs inside
whatever env the launcher activates.

### Server (deploy/server.py)

- Stdlib `http.server.ThreadingHTTPServer` + numpy only. No third-party web
  framework, so it imports in every conda env.
- Endpoints:
  - `GET /info` → JSON from `adapter.info()`.
  - `POST /predict` → body is `np.savez_compressed` bytes containing
    `img_<key>` arrays, `state`, and `task` (0-d unicode array); response is
    raw `np.save` bytes of the chunk.
  - `POST /reset` → clears adapter state, 200.
- Single-threaded inference (a lock around `predict_chunk`); requests queue.
- Errors: HTTP 500 with the traceback as text body.

### Client (deploy/client.py)

- Builds the robot from `--robot.type` + `--robot.cameras` JSON, identical
  config format to `record_bimanual.sh`.
- `--server=http://127.0.0.1:8080`, `--task`, `--duration_s`, `--fps`
  (default: server's fps), `--camera_map` JSON mapping robot camera names →
  checkpoint image keys (e.g. `{"top": "camera1", "l_wrist": "camera2",
  "r_wrist": "camera3"}` to match the training rename map).
- On start: `GET /info`, validate state/action dims against the robot,
  validate camera_map covers all `image_keys`, `POST /reset`.
- **Async-overlap loop** at fps:
  1. Execute actions from the current chunk queue, one per tick.
  2. When consumed fraction ≥ `--chunk_threshold` (default 0.5) and no
     request is in flight, capture a fresh observation and `POST /predict`
     in a background thread; record the queue index at capture time.
  3. When the reply arrives, drop the rows already executed from the old
     queue during flight (held/dry ticks consume none) and replace the
     remaining queue with the rest; a fully-stale chunk logs a warning.
- Grippers/joints: actions are sent via `robot.send_action` with the same
  ordering as `observation.state` (the lerobot piper convention).

## Error handling

- Server: any adapter exception → 500 + traceback; server stays up.
- Client: failed/timeout `/predict` → log, keep executing the remaining
  queue, retry with a fresh observation; if the queue runs dry, hold the
  last position (resend last action) while retrying.
- Ctrl-C / `--duration_s` reached → stop sending, `robot.disconnect()`.
- Dim mismatch or unmapped camera at startup → abort with a clear message
  before touching the arms.

## Testing

1. **Adapter unit test** (no robot, no server): load the SmolVLA bimanual
   checkpoint, feed random HWC uint8 images + random 14-D state, assert
   chunk shape `(chunk_size, 14)` and finite values.
2. **End-to-end smoke** (no robot): start `server.py` with the lerobot
   adapter, run a fake client that posts random observations; assert
   `/info` contents, chunk shape, and round-trip latency.
3. **Real robot**: `serve_smolvla.sh` + `run_client.sh` on the bimanual
   stack-cup task — manual verification.

## Future policies (how this pays off)

openpi / GR00T / anything else: write `deploy/adapters/<name>.py`
implementing the 3-method interface, create `serve_<name>.sh` activating its
env. The client, transport, and control loop are untouched.
