#!/bin/bash
# Run the Piper deployment client (pyAgxArm + cameras).
# Usage: bash deploy/run_client.sh ["task"] [duration_s] [server]
set -e
REPO=/data/wanshan/VLAs/piper_lerobot
PY=/home/embodied/miniconda3/bin/python

TASK="${1:-Stack the cup on top of the bowl.}"
DURATION="${2:-60}"
SERVER="${3:-http://127.0.0.1:8080}"

CAMERAS='{
  "l_wrist": {"type": "opencv", "index_or_path": "/dev/l_wrist", "width": 480, "height": 640, "fps": 120, "rotation": -90, "fourcc": "MJPG"},
  "top":     {"type": "opencv", "index_or_path": "/dev/top",     "width": 640, "height": 480, "fps": 120, "rotation": 0,   "fourcc": "MJPG"},
  "r_wrist": {"type": "opencv", "index_or_path": "/dev/r_wrist", "width": 480, "height": 640, "fps": 120, "rotation": 90,  "fourcc": "MJPG"}
}'

CAMERA_MAP='{"top": "camera1", "l_wrist": "camera2", "r_wrist": "camera3"}'

PYTHONPATH="$REPO/src:$REPO" "$PY" -m deploy.client \
  --robot.type=bi_piper_follower \
  --robot.id=bi_piper \
  --robot.cameras="$CAMERAS" \
  --server="$SERVER" \
  --task="$TASK" \
  --camera_map="$CAMERA_MAP" \
  --duration_s="$DURATION"
