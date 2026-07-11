[English](README.md) | [**中文**](README.zh.md)

# Piper LeRobot

面向 AgileX Piper 机械臂的 LeRobot 集成，支持单臂/双臂遥操作、数据采集、
ACT/pi05 训练和真机部署。

## 主要功能

- `piper_follower` 和 `bi_piper_follower` 机器人配置
- 双臂主从控制与可选 EMA 平滑
- 数据采集期间异步编码视频
- CAN、相机命名、回零、夹爪和安全失能工具
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

## 安全

机械臂抬起时不要直接失能或断电，否则机械臂可能砸落。请使用软失能工具：

```bash
python utils/gentle_disable_arm.py
```

高带宽相机应连接到不同的 USB 总线。相机异常时可运行
`bash utils/which_usb_bus.sh` 和 `python utils/bandwidth_debug.py` 排查。

## 文档

- [环境与硬件安装](docs/setup.md)
- [遥操作](docs/teleop.md)
- [数据采集](docs/recording.md)
- [Hugging Face 上传](docs/huggingface.md)
- [训练](docs/training.md)
- [推理与部署](docs/inference.md)
- [机械臂工具](docs/arm_utils.md)
- [相机与 CAN 对照](utils/cameras.md)
