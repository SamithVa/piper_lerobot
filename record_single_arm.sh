#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/embodied/miniconda3/bin/python
ARM="${1:-left}"
DATASET="${2:-samithva/pear_orange}"
TASK="${3:-Place pear in the green bowl, place orange in the red bowl.}"
EPISODES="${4:-1}"
DISPLAY_DATA="${5:-true}"
DISPLAY_DATA="${DISPLAY_DATA#--display_data=}"

case "$ARM" in
  left)
    FOLLOWER=left_follower
    LEADER=left_leader
    CAMERAS='{
      "wrist": {"type": "intelrealsense", "serial_number_or_name": "263222075317", "width": 640, "height": 480, "fps": 30},
      "top":   {"type": "opencv", "index_or_path": "/dev/top",     "width": 640, "height": 480, "fps": 120, "fourcc": "MJPG"}
    }'
    ;;
  right)
    FOLLOWER=right_follower
    LEADER=right_leader
    CAMERAS='{
      "wrist": {"type": "opencv", "index_or_path": "/dev/r_wrist", "width": 640, "height": 480, "fps": 120, "fourcc": "MJPG"},
      "top":   {"type": "opencv", "index_or_path": "/dev/top",     "width": 640, "height": 480, "fps": 120, "fourcc": "MJPG"}
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
  --display_data="$DISPLAY_DATA" \
  --dataset.repo_id="$DATASET" \
  --dataset.root="$ROOT/dataset/$DATASET" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$EPISODES" \
  --dataset.num_image_writer_processes=16 \
  --dataset.video_encoding_batch_size=10 \
  --dataset.async_video_encoding=true \
  --dataset.reset_time_s=0 \
  --dataset.episode_time_s=300 \
  --dataset.push_to_hub=false
