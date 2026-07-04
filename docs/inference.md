# 推理与真机评测

## ACT 真机评测

[lerobot huggingface 真机文档](https://huggingface.co/docs/lerobot/il_robots)

````
lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/l_wrist",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": -90
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/top",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    }
  }' \
  --display_data=true \
  --dataset.repo_id=jokeru/eval_act_pick_and_place \
  --dataset.num_episodes=3 \
  --dataset.episode_time_s=120 \
  --dataset.push_to_hub=false \
  --dataset.single_task="Pick up it and put it into the basket." \
  --policy.path=jokeru/act_pick_and_place
````

## RTC 本地推理（pi05）

预训练模型：

````
python examples/rtc/eval_with_real_robot.py \
  --policy.path=lerobot/pi05_base \
  --robot.type=piper_follower \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/l_wrist",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": -90
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/top",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    }
  }' \
  --task="Pick up it and put it into the basket." \
  --duration=120 \
  --action_queue_size_to_get_new_actions=30 \
  --fps=50 \
  --rtc.execution_horizon=5 \
  --display_data=true \
  --device=cuda
````

微调后的模型把 `--policy.path` 换成自己的，如 `jokeru/pi05_pick_and_place`（其余参数相同，可去掉 `--display_data`）。

## 异步推理（本地显存不够时）

### 安装

````
pip install -e ".[async]"
````

### 启动远程推理服务器

用 CUDA_VISIBLE_DEVICES 设置用空闲的 GPU 推理，否则会默认用 GPU0：

````
CUDA_VISIBLE_DEVICES=1 python -m src.lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=1
````

### 若端口未开放需建立转发端口

在客户端建立端口转发，通过 SSH 把本地电脑的 8080 端口转发到远程服务器的 8080 端口，从而访问服务器上运行的服务：

````
ssh -L 8080:127.0.0.1:8080 服务器用户名@服务器地址 -N
````

验证端口转发建立成功：

````
nc -zv 127.0.0.1 8080
````

### 客户端接入

````
python -m src.lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=piper_follower \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": "/dev/video6", "width": 480, "height": 640, "fps": 30, "rotation": 90}, "ground": {"type": "opencv", "index_or_path": "/dev/video0", "width": 480, "height": 640, "fps": 30, "rotation": -90}}' \
    --task="Pick up the apple and put it into the basket." \
    --policy_type=pi05 \
    --pretrained_name_or_path=jokeru/pi05_apple \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
````
