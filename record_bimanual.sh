#!/bin/bash
# Record a bimanual (dual-arm) Piper dataset via teleoperation.
#
# Prerequisites (run once):
#   1. CAN interfaces up with role names:   bash utils/activate_all_can.sh
#      (left_leader / left_follower / right_leader / right_follower)
#   2. Camera symlinks:                     sudo bash utils/setup_camera_symlinks.sh
#      (creates /dev/l_wrist, /dev/top, /dev/r_wrist)
#
# Uses base python (has piper_sdk + deps) but forces import of THIS checkout's src.
set -e
REPO=/data/wanshan/VLAs/piper_lerobot
PY=/home/embodied/miniconda3/bin/python

REPO_ID="${1:-samithva/test_3}"
TASK="${2:-Stack the cup on top of the bowl.}"
NUM_EP="${3:-1}"

# Save datasets under this repo's ./dataset/<repo_id> instead of ~/.cache/huggingface/lerobot.
DATASET_ROOT="$REPO/dataset/$REPO_ID"

# Cameras: MJPG (compressed = the bandwidth-efficient method, like lerobot uses for
# USB cams) at 640x480. MJPG@640x480 on these cameras only offers 120fps; the record
# loop still samples at 30fps. Compressed ~2 MB/s each -> all 3 fit the shared bus.
CAMERAS='{
  "l_wrist": {"type": "opencv", "index_or_path": "/dev/l_wrist", "width": 480, "height": 640, "fps": 30, "rotation": -90, "fourcc": "MJPG"},
  "top":     {"type": "opencv", "index_or_path": "/dev/top",     "width": 640, "height": 480, "fps": 30, "rotation": 0,   "fourcc": "MJPG"},
  "r_wrist": {"type": "opencv", "index_or_path": "/dev/r_wrist", "width": 480, "height": 640, "fps": 30, "rotation": 90,  "fourcc": "MJPG"}
}'

PYTHONPATH="$REPO/src" "$PY" -m lerobot.scripts.lerobot_record \
  --robot.type=bi_piper_follower \
  --robot.id=bi_piper \
  --robot.cameras="$CAMERAS" \
  --teleop.type=bi_piper_leader \
  --teleop.id=bi_piper_leader \
  --teleop.ema_alpha=0.4 \
  --display_data=true \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM_EP" \
  --dataset.num_image_writer_processes=32 \
  --dataset.video_encoding_batch_size=30 \
  --dataset.async_video_encoding=true \
  --dataset.reset_time_s=0 \
  --dataset.episode_time_s=6000 \
  --dataset.push_to_hub=false
 