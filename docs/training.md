# 训练

## ACT

num_workers、batch_size、steps 等训练参数参照自己的设备：

````
lerobot-train \
  --dataset.repo_id=jokeru/record2 \
  --policy.type=act \
  --output_dir=outputs/train/record2 \
  --job_name=act_finetune_pick_apple \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=jokeru/act_pick_apple \
  --batch_size=128 \
  --steps=12_000 \
  --num_workers=128
````

### 仿真环境中评测

````
lerobot-eval \
    --policy.path=jokeru/act_pick_apple \
    --env.type=your_env \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
````

真机评测见[推理](inference.md)。

## openpi（pi05）

### 环境安装

安装 lerobot 的 pi 相关依赖：

````
pip install -e ".[pi]"
````

### 单卡训练

````
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=jokeru/record2 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.repo_id=jokeru/pi05 \
    --policy.pretrained_path=lerobot/pi05_libero \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32
````

pi05_base 或 pi05_libero 会下载在如 `~/.cache/huggingface/hub/models--lerobot--pi05_base`。

### 多卡训练

可用 `tests/training/test_multi_gpu.py` 测试（需先 `pip install pytest`）：

````
nohup accelerate launch --num_processes=8 \
  src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=jokeru/record2 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.repo_id=jokeru/pi05 \
    --policy.pretrained_path=lerobot/pi05_libero \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32 > outputs/pi05_training.log 2>&1 &
````
