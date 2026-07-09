**English** | [中文](README.zh.md)

# deploy — run any policy on the Piper arms

The **server** hosts a policy inside whatever env it needs; the **client**
drives the arms + cameras and streams action chunks over localhost HTTP with
async overlap, so the arms never pause for inference. Any model works — openpi,
GR00T, lerobot, anything: you write one small adapter, everything else stays
untouched.

## Deploy your own policy in 3 steps

See the pipeline work first with zero setup (no GPU, checkpoint, or arms):

```bash
python -m deploy.launch example        # dummy policy server
curl -s http://127.0.0.1:8090/info
```

1. **Adapter** — copy `deploy/adapters/dummy.py` → `deploy/adapters/<name>.py`
   and implement its three methods: `info()`, `predict_chunk(...)`, `reset()`.
   The full contract (shapes, dtypes, kwargs) is in `dummy.py`'s docstring and
   `adapters/base.py`. Register the adapter in
   `deploy/adapters/__init__.py::make_adapter`.
2. **Preset** — copy `deploy/presets/example.json` → `presets/<name>.json`;
   point `server.python` at your model's env (the server imports stdlib+numpy
   only, so it runs in *any* env) and add a `client` section (copy `pi05.json`'s
   cameras/camera_map).
3. **Run** — `python3 -m deploy.launch <name> --task="..."`.

## Run an existing policy

```bash
python3 -m deploy.launch pi05 --task="Stack the cup on top of the bowl." --duration_s=60
# same thing:
bash deploy/run.sh pi05 "Stack the cup on top of the bowl." 60
```

| preset    | policy                                            | notes                        |
|-----------|---------------------------------------------------|------------------------------|
| `pi05`    | `outputs/pi05` (samithva/pi05_stack_cup_bowl)     | pinned env — see notes below |
| `smolvla` | smolvla_bimanual_stack_cup_bowl (last checkpoint) | lerobot env                  |
| `example` | dummy adapter — no GPU, no checkpoint, no arms    | server-only teaching sample  |

A preset holds everything you'd otherwise have to know: serving env,
checkpoint, port, cameras, camera→image-key map. The launcher **reuses** a warm
server already serving the preset's checkpoint (skips pi05's ~15-20s compile
cold start), **refuses** a port busy with a different policy (never kills), or
**spawns** one detached (log: `deploy/logs/server-<preset>.log`). Ctrl-C stops
the client only — the server stays warm for the next run. Overrides:
`--checkpoint=`, `--port=`, `--fps=`, plus any client flag as `--key=value`.

## How it works

```
launch.py ──spawns/reuses──▶ server.py (policy env) ◀── adapters/<name>.py (your code)
    │                            ▲   /info /predict /reset
    └──runs──▶ client.py ────────┘   npz obs → npy chunk
               (base python: arms + cameras)
```

- **Protocol** (`protocol.py`): `GET /info` → `{name, image_keys, state_dim,
  action_dim, chunk_size, fps, checkpoint}`; `POST /predict` — npz of images
  (`img_<key>`, HWC uint8 RGB) + `state` (float32) + `task` → `.npy` bytes
  `(chunk_size, action_dim)`; `POST /reset` clears episode state. Errors →
  HTTP 500 with the traceback.
- **Async overlap** (`chunking.py`): the next `/predict` fires when the current
  chunk is half consumed; rows already executed while the request was in flight
  are skipped from the fresh chunk; if the queue runs dry, the client holds
  position and re-requests.

## Machine notes (pi05)

- Serving env must match training **exactly** (here: lerobot v0.5.1 +
  transformers 5.3.0 — newer transformers runs but silently ignores inputs).
  Full story, incl. the HF-offline setup: `_notes` in `deploy/presets/pi05.json`.
- First `/predict` after a spawn takes ~15-20s (`torch.compile` + CUDA-graph
  capture); the client absorbs it with `first_predict_timeout_s` (90s). Do
  **not** add a server-side warmup predict — it poisons pi05's CUDA-graph state.
- Two-env split: the client always runs in base python (pyAgxArm + cameras);
  each policy serves from its own env.

### RTC (smooth chunk transitions)

pi05 chunks are sampled independently by flow matching, so each chunk swap
commanded a small trajectory jump — visible jerk roughly every half chunk
(~0.8s at 30fps). RTC ([lerobot's real-time chunking](https://huggingface.co/docs/lerobot/rtc))
guides the first `rtc_execution_horizon` rows of each new chunk to continue
the previous chunk's leftover instead of resampling from scratch.

How it flows here: the client sends `consumed`/`delay_ticks` with each
observation; server-side `LerobotAdapter` keeps the previous raw (normalized)
chunk and passes `prev_chunk_left_over` + `inference_delay` into
`predict_action_chunk`; the client warms the guidance path with a second
blocking predict before the arm moves.

Knobs (preset `server.args`): `rtc` (`"1"`/`"0"`), `rtc_schedule` (`exp`),
`rtc_execution_horizon` (default 10), `rtc_guidance` (default 10.0), `compile`
(`"0"` shipped — see below). Nothing new on the client side; `chunk_threshold`
stays 0.5.

Diagnostic (no robot needed):

```bash
PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot_pi05/bin/python \
    -m deploy.probe_rtc --server=http://127.0.0.1:8080
```

reports per-predict latency and chunk-boundary continuity.

**Verification (2026-07-09, RTX-class 49GB GPU, `outputs/pi05`).** Compiled
(`max-autotune`) + RTC crashed on the 2nd predict — cudagraph thread-local
state vs per-request threads — fixed by pinning inference to one worker
thread in `deploy/server.py` (commit `2da2392`), then measured 3.3-3.8s/predict
(autograd guidance defeats the compiled graph). Eager (`compile=0`): 0.19s
steady-state predict, 0.62s first predict (no compile cold start); continuity
mean|Δ| over the 10-row guidance window 0.0035 vs 0.0095 without RTC — frozen
prefix rows ~0.0008 (12x tighter), blend rows releasing 0.0036→0.0075. Per the
plan's decision matrix (ship if L < 0.7s): **ship RTC with `compile=0`**.

Operational note: the launcher refuses to reuse a warm server whose RTC mode
differs from the preset (old servers report no `rtc` field → treated as
RTC-off) — stop the old server on the port first.

## Tests

```bash
PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v
# integration (downloads smolvla_base, needs GPU):
DEPLOY_IT=1 PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python \
    -m pytest deploy/tests/test_lerobot_adapter.py -v
```
