#!/bin/bash
# Run the Piper deployment client against a pi05 server (base python: piper_sdk + cameras).
# Usage: bash deploy/run_client_pi05.sh ["task"] [duration_s] [server]
#
# Same as run_client.sh EXCEPT the camera_map is identity: the pi05 checkpoint's
# image keys are the raw camera names (l_wrist/top/r_wrist), not the camera1/2/3
# rename the smolvla checkpoint was trained with.
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

CAMERA_MAP='{"top": "top", "l_wrist": "l_wrist", "r_wrist": "r_wrist"}'

PYTHONPATH="$REPO/src:$REPO" "$PY" -m deploy.client \
  --robot.type=bi_piper_follower \
  --robot.id=bi_piper \
  --robot.cameras="$CAMERAS" \
  --server="$SERVER" \
  --task="$TASK" \
  --camera_map="$CAMERA_MAP" \
  --duration_s="$DURATION"
