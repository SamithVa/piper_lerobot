#!/bin/bash
# Serve a SmolVLA checkpoint for Piper deployment (lerobot conda env).
# Usage: bash deploy/serve_smolvla.sh [checkpoint] [port]
set -e
REPO=/data/wanshan/VLAs/piper_lerobot
PY=/home/embodied/miniconda3/envs/lerobot/bin/python

CHECKPOINT="${1:-$REPO/outputs/train/smolvla_bimanual_stack_cup_bowl/checkpoints/last/pretrained_model}"
PORT="${2:-8080}"

# PYTHONPATH has only the repo root (for `deploy`), NOT $REPO/src: checkpoints
# trained with `lerobot-train` used the lerobot env's installed lerobot (0.5.x),
# whose saved configs this checkout's older lerobot cannot parse.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" "$PY" -m deploy.server \
  --adapter=lerobot \
  --port="$PORT" \
  --checkpoint="$CHECKPOINT" \
  --device=cuda \
  --fps=30
