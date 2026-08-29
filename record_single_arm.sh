#!/usr/bin/env bash
# Record a single-arm Piper dataset through teleoperation.
#
# Usage:
#   bash record_single_arm.sh [left|right] [repo_id] [task] [num_episodes]
#
# Prerequisites:
#   1. Activate the matching follower and leader CAN interfaces.
#   2. Create the camera symlinks used below (/dev/l_wrist or /dev/r_wrist,
#      and /dev/top).
#
# The base Python environment is used for the Piper SDK dependencies while
# this checkout's LeRobot source is forced onto PYTHONPATH.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/home/embodied/miniconda3/bin/python"

ARM="${1:-left}"
REPO_ID="${2:-samithva/single_arm_test}"
TASK="${3:-Pick up the object.}"
NUM_EP="${4:-1}"

case "$ARM" in
  left)
    FOLLOWER_CAN="left_follower"
    LEADER_CAN="left_leader"
    WRIST_CAMERA="/dev/l_wrist"
    WRIST_ROTATION="-90"
    ;;
  right)
    FOLLOWER_CAN="right_follower"
    LEADER_CAN="right_leader"
    WRIST_CAMERA="/dev/r_wrist"
    WRIST_ROTATION="90"
    ;;
  *)
    echo "usage: $0 [left|right] [repo_id] [task] [num_episodes]" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PY" ]]; then
  echo "Python executable not found: $PY" >&2
  exit 1
fi

DATASET_ROOT="$REPO/dataset/$REPO_ID"
CAMERAS=$(cat <<JSON
{
  "wrist": {
    "type": "opencv",
    "index_or_path": "$WRIST_CAMERA",
    "width": 480,
    "height": 640,
    "fps": 30,
    "rotation": $WRIST_ROTATION,
    "fourcc": "MJPG"
  },
  "top": {
    "type": "opencv",
    "index_or_path": "/dev/top",
    "width": 640,
    "height": 480,
    "fps": 30,
    "rotation": 0,
    "fourcc": "MJPG"
  }
}
JSON
)

echo "Recording $ARM single-arm dataset"
echo "  follower CAN: $FOLLOWER_CAN"
echo "  leader CAN:   $LEADER_CAN"
echo "  dataset:      $REPO_ID"
echo "  episodes:     $NUM_EP"

PYTHONPATH="$REPO/src" "$PY" -m lerobot.scripts.lerobot_record \
  --robot.type=piper_follower \
  --robot.id="${ARM}_follower" \
  --robot.can_name="$FOLLOWER_CAN" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=piper_leader \
  --teleop.id="${ARM}_leader" \
  --teleop.can_name="$LEADER_CAN" \
  --display_data=true \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM_EP" \
  --dataset.num_image_writer_processes=4 \
  --dataset.video_encoding_batch_size=10 \
  --dataset.async_video_encoding=true \
  --dataset.reset_time_s=0 \
  --dataset.episode_time_s=60 \
  --dataset.push_to_hub=false
