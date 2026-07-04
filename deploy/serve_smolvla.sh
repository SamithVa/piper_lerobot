#!/bin/bash
# Serve a SmolVLA checkpoint for Piper deployment (lerobot conda env).
# Usage: bash deploy/serve_smolvla.sh [checkpoint] [port]
set -e
REPO=/data/wanshan/VLAs/piper_lerobot
PY=/home/embodied/miniconda3/envs/lerobot/bin/python

CHECKPOINT="${1:-$REPO/outputs/train/smolvla_bimanual_stack_cup_bowl/checkpoints/last/pretrained_model}"
PORT="${2:-8080}"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO/src:$REPO" "$PY" -m deploy.server \
  --adapter=lerobot \
  --port="$PORT" \
  --checkpoint="$CHECKPOINT" \
  --device=cuda \
  --fps=30
