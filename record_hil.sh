#!/bin/bash
# Human-in-the-loop (HIL) data collection: the policy drives the follower arms,
# press SPACE to take over with the leader arms, SPACE again to hand back.
# Each frame is saved with an `intervention` flag (1 = human, 0 = policy).
#
# Prerequisites:
#   1. CAN up + camera symlinks (same as record_bimanual.sh)
#   2. Policy server running, e.g.:  bash deploy_piper/deploy/run.sh server pi05
#
# Keys: SPACE start episode / toggle intervention, → save episode,
#       ← re-record episode, ESC stop session.
#
# Usage: bash record_hil.sh [repo_id] [task] [num_episodes] [config] [extra flags...]
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/embodied/miniconda3/bin/python

REPO_ID="${1:-samithva/stack_cup_hil}"
TASK="${2:-Stack the cup on top of the bowl.}"
NUM_EP="${3:-10}"
CONFIG="${4:-pi05}"
shift $(( $# < 4 ? $# : 4 ))

cd "$REPO/deploy_piper"
PYTHONPATH="$REPO/src:$REPO/deploy_piper" "$PY" -m deploy.hil_record \
  --config "$CONFIG" \
  --task="$TASK" \
  --teleop.type=bi_piper_leader \
  --teleop.id=bi_piper_leader \
  --teleop.ema_alpha=0.4 \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$REPO/dataset/$REPO_ID" \
  --dataset.num_episodes="$NUM_EP" \
  --dataset.episode_time_s=300 \
  --dataset.num_image_writer_processes=16 \
  --dataset.video_encoding_batch_size=10 \
  --dataset.async_video_encoding=true \
  "$@"
