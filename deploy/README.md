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

## Tests

```bash
PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v
# integration (downloads smolvla_base, needs GPU):
DEPLOY_IT=1 PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python \
    -m pytest deploy/tests/test_lerobot_adapter.py -v
```
