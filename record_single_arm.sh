#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/embodied/miniconda3/bin/python
ARM="${1:-left}"
DATASET="${2:-samithva/single_arm_test}"
TASK="${3:-Pick up the object.}"
EPISODES="${4:-1}"

case "$ARM" in
  left)
    FOLLOWER=left_follower
    LEADER=left_leader
    CAMERAS='{
      "wrist": {"type": "opencv", "index_or_path": "/dev/l_wrist", "width": 480, "height": 640, "fps": 30, "rotation": -90, "fourcc": "MJPG"},
      "top":   {"type": "opencv", "index_or_path": "/dev/top",     "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}
    }'
    ;;
  right)
    FOLLOWER=right_follower
    LEADER=right_leader
    CAMERAS='{
      "wrist": {"type": "opencv", "index_or_path": "/dev/r_wrist", "width": 480, "height": 640, "fps": 30, "rotation": 90, "fourcc": "MJPG"},
      "top":   {"type": "opencv", "index_or_path": "/dev/top",     "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}
    }'
    ;;
  *)
    echo "usage: $0 [left|right] [repo_id] [task] [num_episodes]" >&2
    exit 2
    ;;
esac

PYTHONPATH="$ROOT/src" "$PY" -m lerobot.scripts.lerobot_record \
  --robot.type=piper_follower \
  --robot.id="${ARM}_follower" \
  --robot.can_name="$FOLLOWER" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=piper_leader \
  --teleop.id="${ARM}_leader" \
  --teleop.can_name="$LEADER" \
  --display_data=true \
  --dataset.repo_id="$DATASET" \
  --dataset.root="$ROOT/dataset/$DATASET" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$EPISODES" \
  --dataset.num_image_writer_processes=4 \
  --dataset.video_encoding_batch_size=10 \
  --dataset.async_video_encoding=true \
  --dataset.reset_time_s=0 \
  --dataset.episode_time_s=60 \
  --dataset.push_to_hub=false
