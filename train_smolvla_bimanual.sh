#!/bin/bash
# Fine-tune SmolVLA (lerobot/smolvla_base, 450M) on a bimanual Piper dataset.
# Ref: https://huggingface.co/docs/lerobot/smolvla
#
# Uses the `lerobot` conda env (transformers 5.x) but forces import of THIS
# checkout's src, same pattern as record_bimanual.sh.
#
# Usage: bash train_smolvla_bimanual.sh [dataset_repo_id] [job_name]
set -e
REPO=/data/wanshan/VLAs/piper_lerobot
# PY=/home/embodied/miniconda3/envs/lerobot/bin/python

REPO_ID="${1:-samithva/bimanual_stack_cup_bowl}"
JOB="${2:-smolvla_bimanual_stack_cup_bowl}"

# smolvla_base expects standardized camera keys camera1/2/3; map our cameras onto them.
RENAME_MAP='{
  "observation.images.top":     "observation.images.camera1",
  "observation.images.l_wrist": "observation.images.camera2",
  "observation.images.r_wrist": "observation.images.camera3"
}'

CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="$REPO_ID" \
  --rename_map="$RENAME_MAP" \
  --output_dir="$REPO/outputs/train/$JOB" \
  --job_name="$JOB" \
  --policy.device=cuda \
  --policy.repo_id="samithva/$JOB" \
  --wandb.enable=true \
  --batch_size=64 \
  --steps=20000 \
  --num_workers=16 \
  --log_freq=50 \
  --save_freq=5000
