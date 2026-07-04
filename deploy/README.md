# deploy — policy client-server for the Piper arms

Run any trained policy on the Piper robot. The **server** hosts the policy in
whatever conda env it needs; the **client** always runs in base python
(piper_sdk + cameras) and streams action chunks over localhost HTTP with no
pauses between chunks (async overlap).

Design spec: `docs/superpowers/specs/2026-07-04-piper-deploy-client-server-design.md`

## Quick start (SmolVLA, bimanual)

```bash
# terminal 1 — policy server (lerobot env)
bash deploy/serve_smolvla.sh                       # or: serve_smolvla.sh <checkpoint> <port>

# terminal 2 — robot client (base python)
bash deploy/run_client.sh "Stack the cup on top of the bowl." 60
```

`run_client.sh` maps cameras per the training rename map
(top→camera1, l_wrist→camera2, r_wrist→camera3).

## Any lerobot checkpoint

The `lerobot` adapter serves any lerobot policy (smolvla / act / pi05):

```bash
PYTHONPATH=. <env-python> -m deploy.server --adapter=lerobot \
    --checkpoint=<hub-id-or-local-path> --device=cuda --fps=30
```

Pick `<env-python>` = an env whose **lerobot version matches the one that
trained the checkpoint** (a newer lerobot writes config fields an older one
refuses to parse). Checkpoints from `train_smolvla_bimanual.sh` / bare
`lerobot-train` were trained by the lerobot env's installed lerobot, so
`PYTHONPATH=.` (repo root only). Add `src` in front (`PYTHONPATH=src:.`) only
for checkpoints trained with this checkout's lerobot (e.g. pi05 via
`python src/lerobot/scripts/lerobot_train.py`).

## Adding a NON-lerobot policy (openpi, GR00T, ...)

1. Copy `deploy/adapters/dummy.py` to `deploy/adapters/<name>.py`; implement
   `info()`, `predict_chunk(images, state, task)`, `reset()` (see
   `deploy/adapters/base.py` for the contract).
2. Register it in `deploy/adapters/__init__.py` (`make_adapter`).
3. Launch `deploy.server --adapter=<name> ...` from the policy's env.

The client, protocol, and control loop stay untouched.

## Protocol

- `GET /info` → `{"name", "image_keys", "state_dim", "action_dim", "chunk_size", "fps"}`
- `POST /predict` — body: npz of images (`img_<key>`, HWC uint8 RGB) + `state`
  (float32) + `task` (str); reply: `.npy` bytes, shape `(chunk_size, action_dim)`
- `POST /reset` — clear episode state
- Errors → HTTP 500 with the Python traceback as the body

## Tests

```bash
PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v
# integration (downloads smolvla_base, needs GPU):
DEPLOY_IT=1 PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python \
    -m pytest deploy/tests/test_lerobot_adapter.py -v
```
