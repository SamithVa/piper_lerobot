# Put A into the brown box.
````
lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video0",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90,
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/video6",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0,
    }
  }' \
  --teleop.type=piper_leader \
  --display_data=true \
  --dataset.reset_time_s=5 \
  --dataset.repo_id=jokeru/record1 \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=10 \
  --dataset.single_task="Put round yellow tape into the brown box."
````

# 清除记录

‵‵‵‵
rm -r ~/.cache/huggingface/lerobot/jokeru/record1
````

# 查看结果

````
vlc observation.images.ground/chunk-000/file-000.mp4
vlc observation.images.wrist/chunk-000/file-000.mp4
````