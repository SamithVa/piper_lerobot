#!/bin/bash
# Serve a pi05 checkpoint for Piper deployment.
# Usage: bash deploy/serve_pi05.sh [checkpoint] [port]
#
# WHY this needs its own env + lerobot source (not serve_smolvla.sh's):
# The samithva/pi05_stack_cup_bowl checkpoint was trained with lerobot v0.5.1,
# which:
#   * registers the preprocessor step as `delta_actions_processor` (renamed to
#     `relative_actions_processor` after 0.5.1 -> the installed lerobot 0.5.2
#     can't build the processor pipeline);
#   * pins transformers==5.3.0 and calls create_causal_mask(cache_position=...)
#     (the lerobot env's transformers 5.13 dropped that kwarg).
# BOTH versions must match training exactly. transformers 5.5.x LOADS and RUNS
# but SILENTLY BREAKS vision+state->action conditioning (predictions become
# input-blind; verified via deploy sensitivity probe). Must be 5.3.0.
# So we serve from:
#   * lerobot source at tag v0.5.1 (git worktree, PYTHONPATH ahead), and
#   * conda env `lerobot_pi05` = clone of `lerobot` with transformers pinned to 5.3.0.
# pi05's paligemma tokenizer (google/paligemma-3b-pt-224, gated) is loaded from the
# HF cache; HF_HUB_OFFLINE avoids re-fetching (the box's proxy is SOCKS, which
# python httpx can't use). The one missing cache file (config.json) has been added.
set -e
REPO=/data/wanshan/VLAs/piper_lerobot
LEROBOT_SRC=/data/wanshan/VLAs/lerobot-pi05-serve/src   # worktree @ tag v0.5.1
PY=/home/embodied/miniconda3/envs/lerobot_pi05/bin/python

CHECKPOINT="${1:-$REPO/outputs/pi05}"
PORT="${2:-8080}"

# LEROBOT_SRC ahead of $REPO so the training-matching lerobot wins over the
# env's editable install; $REPO is on the path only for the `deploy` package.
CUDA_VISIBLE_DEVICES=0 \
  HF_HOME=/data/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="$LEROBOT_SRC:$REPO" "$PY" -m deploy.server \
  --adapter=lerobot \
  --port="$PORT" \
  --checkpoint="$CHECKPOINT" \
  --device=cuda \
  --fps=30
