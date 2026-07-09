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
