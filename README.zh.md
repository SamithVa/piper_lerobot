[English](README.md) | [**中文**](README.zh.md)

# Piper LeRobot

面向 AgileX Piper 机械臂的 LeRobot 集成，支持单臂/双臂遥操作、数据采集、
ACT/pi05 训练和真机部署。

## 主要功能

- `piper_follower` 和 `bi_piper_follower` 机器人配置
- 双臂主从控制与可选 EMA 平滑
- 数据采集期间异步编码视频
- CAN 设置、相机设置、夹爪校准和安全失能工具
- 本地、RTC 和异步策略推理

## 安装

建议使用独立 Conda 环境，并将本仓库一次性以 editable 模式安装。安装后，
`lerobot` 会直接加载 `src/lerobot`，无需每次设置 `PYTHONPATH`。

```bash
conda create -n piper-lerobot python=3.10 -y
conda activate piper-lerobot

cd /path/to/piper_lerobot
python -m pip install -e .
python -c "import lerobot; print(lerobot.__file__)"
```

请按照 AgileX Piper SDK 的说明，在同一环境中安装 `pyAgxArm`。

## 双臂数据采集

```bash
# 准备四条 CAN 接口和稳定的相机设备名。
bash utils/activate_all_can.sh
sudo bash utils/setup_camera_symlinks.sh

# 采集并上传数据集。
bash record_bimanual.sh <repo_id> <task> <num_episodes>
```

录制按键：`空格` 开始一条 episode，`右方向键` 提前结束，`左方向键` 取消重录，
`Esc` 结束整个会话。

## 人在回路（HIL）数据采集

已部署的策略驱动从臂，主臂同步跟随；按 `空格` 用主臂接管，再按 `空格` 交还给策略。
每条 episode 都带成功/失败标签和逐帧控制标志，用于 Recap 式训练。采集脚本在
[`deploy_piper`](https://github.com/thusigs-eilab/deploy_piper/tree/hil) 仓库（`hil` 分支，
本地检出为 `deploy_piper_hil/`）：

```bash
cd deploy_piper_hil
python -m deploy.server --config pi05_long_horizon_plate --checkpoint=<ckpt>   # 环境 lerobot_pi05
python -m deploy.hil_record --config pi05_long_horizon_plate --task="..." \
  --teleop.type=piper_leader --teleop.id=left_leader --teleop.can_name=left_leader \
  --dataset.repo_id=<repo_id> --dataset.root=../dataset/<repo_id>              # 环境 piper_sdk
```

详见 [HIL 快速上手](docs/hil.md) 和
[完整说明](https://github.com/thusigs-eilab/deploy_piper/blob/hil/docs/hil_collection.md)
（参数、按键、保存内容、故障排查）。

## 安全

机械臂抬起时不要直接失能或断电，否则机械臂可能砸落。请使用软失能工具：

```bash
python utils/gentle_disable_arm.py
```

高带宽相机应连接到不同的 USB 总线。可运行
`python utils/capture_cameras.py` 检查已配置的相机画面。

## 工具脚本

- `activate_all_can.sh` — 激活所有已配置的 Piper CAN 接口
- `find_all_can_port.sh` — 查找已连接的 CAN 适配器
- `can_activate.sh` — 激活单个 CAN 接口
- `can_health.sh` — 检查 CAN 接口状态
- `setup_camera_symlinks.sh` — 使用 `99-camera-symlinks.rules` 安装稳定的相机设备名
- `capture_cameras.py` — 抓取相机测试画面
- `gentle_disable_arm.py` — 缓慢放下并安全失能 Piper 机械臂
- `gripper_zero_calibration.py` — 校准夹爪闭合零点
- `push_dataset.py` — 将本地数据集上传至 Hugging Face Hub

## 文档

- [环境与硬件安装](docs/setup.md)
- [遥操作](docs/teleop.md)
- [数据采集](docs/recording.md)
- [HIL 数据采集](docs/hil.md) — 策略驱动 + 主臂接管的采集（[完整说明](https://github.com/thusigs-eilab/deploy_piper/blob/hil/docs/hil_collection.md)）
- [Hugging Face 上传](docs/huggingface.md)
- [训练](docs/training.md)
- [推理与部署](docs/inference.md)
- [机械臂工具](docs/arm_utils.md)
