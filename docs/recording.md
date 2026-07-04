# 数据采集

## 双臂采集（推荐入口）

前置条件（各执行一次）：

````
bash utils/activate_all_can.sh              # 激活 4 条 CAN（角色命名）
sudo bash utils/setup_camera_symlinks.sh    # 创建 /dev/l_wrist /dev/top /dev/r_wrist
````

采集：

````
bash record_bimanual.sh [repo_id] [task] [num_episodes]
# 例：bash record_bimanual.sh samithva/bimanual_stack_cup 'Stack the cup on top of the bowl.' 30
````

数据保存在 `./dataset/<repo_id>`，结束后自动推送到 Hugging Face Hub。
每条 episode 按空格开始录制，方便操作者就位。

### 录制管线（脚本内已配置的关键参数）

| 参数 | 作用 |
| --- | --- |
| `--dataset.video_encoding_batch_size=N` | 攒 N 条 episode 再做视频编码合并（设为总条数 → 只在最后合并一次） |
| `--dataset.async_video_encoding=true` | **后台编码**：录下一条时，上一条的 PNG→视频编码已在后台进行；结束时只需等最后一条编完 + 快速拼接，无需长时间等待 |
| `--teleop.ema_alpha=0.4` | 主臂动作 EMA 平滑，见[遥操作](teleop.md) |
| `--dataset.num_image_writer_processes` | PNG 异步写盘的进程数 |

后台编码说明：

- 每条 episode 保存后进入编码队列，后台线程逐条（相机间也串行）编码到
  `dataset/<repo_id>/videos_tmp/`，编码成功后立即删除该条的 PNG，磁盘占用
  始终只有 1~2 条 episode 的图片。
- 会话结束（含 ESC、Ctrl-C、异常退出）会先等后台队列清空再做合并，
  日志显示 `Waiting for background video encoding to finish...`。
- 某条编码失败不会中断录制：保留 PNG，最后的合并阶段自动重编。

## 单臂采集（lerobot-record）

/dev/l_wrist 等参数改为自己对应的端口：

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
      "rotation": 0,
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/top",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0,
    }
  }' \
  --teleop.type=piper_leader \
  --display_data=true \
  --dataset.reset_time_s=5 \
  --dataset.repo_id=jokeru/record2 \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick up the apple and put it into the basket."
````

### 其他可选参数

````
--dataset.episode_time_s=60   # 每条 episode 的持续时间（默认 60 秒），可提前结束
--dataset.reset_time_s=60     # 每条 episode 之后重置环境的时长（默认 60 秒）
--dataset.num_episodes=50     # 记录的总 episode 数（默认 50）
````

不指定 `--dataset.root` 时数据保存到 `~/.cache/huggingface/lerobot/<repo_id>`。

## 键盘快捷键

- 空格：开始录制当前 episode（record_bimanual.sh 的逐条门控）
- 右箭头（→）：提前结束当前 episode 或重置阶段，切换到下一条
- 左箭头（←）：取消当前 episode 并重新录制
- ESC：立即停止会话，完成编码并上传数据集

## 合并数据集

````
# 合并多个数据集（要求所有数据集特征完全一致）
lerobot-edit-dataset \
  --repo_id jokeru/pick_and_place \
  --operation.type merge \
  --operation.repo_ids "['jokeru/record_apple', 'jokeru/record_banana','jokeru/record_watermelon','jokeru/record_tape']" \
  --push_to_hub true
````

## 可视化数据集

````
python src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id jokeru/record1 \
    --episode-index 0
````

这种方法只能看一条 episode。也可以用 vlc 直接看 mp4 文件：

````
vlc *.mp4
````
