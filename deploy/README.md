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
for checkpoints trained with this checkout's lerobot.

## pi05 (`outputs/pi05`, from `samithva/pi05_stack_cup_bowl`)

```bash
# terminal 1 — server (dedicated env + training-matching lerobot)
bash deploy/serve_pi05.sh                 # or: serve_pi05.sh <checkpoint> <port>
# terminal 2 — client (base python), IDENTITY camera_map
bash deploy/run_client_pi05.sh "Stack the cup on top of the bowl." 60
```

The server's **first** `/predict` takes ~15-20s (one-time `torch.compile` + CUDA-graph
capture); every predict after is <0.5s. The client's first blocking request uses
`first_predict_timeout_s` (90s) to absorb this, while in-loop predicts keep the tight
`predict_timeout_s` (15s). Don't try to hide the cold start with a server-side warmup
predict — a throwaway forward pass poisons pi05's CUDA-graph state and every real
predict then fails with `Offset increment outside graph capture`.

This checkpoint needs a different env than smolvla — matching **both** lerobot
*and* transformers to what trained it (**lerobot v0.5.1 + transformers 5.3.0**).
`serve_pi05.sh` encodes the whole recipe; its header explains why. v0.5.1 predates
the `delta_actions_processor` → `relative_actions_processor` rename and pins
`transformers==5.3.0`, so we serve from:

- lerobot source at tag `v0.5.1` — git worktree at `/data/wanshan/VLAs/lerobot-pi05-serve`
  (`GIT_LFS_SKIP_SMUDGE=1 git -C /data/wanshan/VLAs/lab_challenge/lerobot worktree add --detach <path> v0.5.1`);
- conda env `lerobot_pi05` — a clone of `lerobot` with `transformers` pinned to 5.3.0
  (`conda create --clone lerobot -n lerobot_pi05`; `pip install "transformers==5.3.0"`).

**Both versions must match training exactly.** transformers 5.5.x loads and runs
but *silently breaks* vision+state→action conditioning — the policy ignores its
inputs and emits a near-constant chunk. Verified with a sensitivity probe: on the
training dataset, v0.5.1+tf5.3.0 reproduces recorded actions to nMAE ≈ 0.03 (vs
0.96 — worse than a mean baseline — under tf 5.5.x).

The pi05 image keys are the raw camera names (`l_wrist`/`top`/`r_wrist`), so the
client uses an **identity** `--camera_map`, not the smolvla `camera1/2/3` rename.
The gated paligemma tokenizer loads from the HF cache offline (the box's SOCKS
proxy isn't usable by python); its one missing cache file, `config.json`, has
been added under `$HF_HOME`.

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
