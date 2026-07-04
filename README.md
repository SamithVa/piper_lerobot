# piper_lerobot

基于 [LeRobot](http://huggingface.co/docs/lerobot)（数据集 v3 格式）改造的 AgileX Piper 机械臂
数据采集与训练仓库：支持**单臂/双臂遥操作采集**，数据推送 Hugging Face Hub，
用 ACT 或 pi05（openpi）训练，并支持真机评测、RTC 与异步推理。

## 概览

**硬件**：2 主臂 + 2 从臂（AgileX Piper，4 条 CAN，按 left/right × leader/follower 角色命名）+
3 个 USB 相机（`/dev/l_wrist`、`/dev/top`、`/dev/r_wrist`，MJPG 压缩 640×480@30fps 采样）。

**相对上游 LeRobot 的主要改动**：

- `bi_piper_follower` / `bi_piper_leader`：双臂机器人与双臂主臂类型，动作键带 `left_` / `right_` 前缀
- 主臂动作 **EMA 平滑**（`--teleop.ema_alpha`，只平滑关节、夹爪直通），从臂运动更顺滑且录制动作与执行一致
- **后台视频编码**（`--dataset.async_video_encoding`）：录下一条 episode 时上一条已在后台 PNG→AV1 编码，
  会话结束只需秒级收尾，不再等几分钟；配合 `--dataset.video_encoding_batch_size` 批量合并
- Piper CAN 总线连接的冷启动竞态修复（首次 record 遥操作失败问题）、使能超时重试
- `utils/` 下一套实用脚本：CAN 批量激活/健康检查、相机符号链接、软失能（避免断电砸落）、夹爪工具等

## 快速开始：双臂数据采集

````
# 前置（各执行一次）
bash utils/activate_all_can.sh              # 激活 4 条 CAN（left/right × leader/follower）
sudo bash utils/setup_camera_symlinks.sh    # 创建 /dev/l_wrist /dev/top /dev/r_wrist

# 采集（数据存 ./dataset/<repo_id>，结束自动推送 HF Hub）
bash record_bimanual.sh [repo_id] [task] [num_episodes]
````

录制控制：每条 episode 按**空格**开始（操作者就位后再录），**→** 提前结束当前条，
**←** 取消重录，**ESC** 结束会话（会先等后台编码完成再合并上传）。

## 重要注意事项

- **两个相机不能接同一个扩展坞/USB 总线**，否则带宽不足读取出错；排查用
  `bash utils/which_usb_bus.sh`、`python utils/bandwidth_debug.py`。
- 出现 `SendCanMessage 100017`（ENOBUFS）是 CAN 发送队列溢出，不是总线故障；
  `utils/activate_all_can.sh` 已把 txqueuelen 调大到 1000。
- 机械臂**不要直接断电失能**（会砸落），用 `python utils/gentle_disable_arm.py` 软失能。
- 本仓库脚本用 `PYTHONPATH=src` 强制加载本仓库代码（见 record_bimanual.sh），
  避免 import 到环境里安装的旧版 lerobot。

## 文档

| 主题 | 内容 |
| --- | --- |
| [环境安装](docs/setup.md) | conda 环境、piper 依赖、CAN 激活、相机符号链接与测试 |
| [遥操作](docs/teleop.md) | 单臂/双臂遥操作、EMA 动作平滑参数调节 |
| [数据采集](docs/recording.md) | record_bimanual.sh、后台视频编码、lerobot-record、合并与可视化 |
| [Hugging Face](docs/huggingface.md) | 登录、国内镜像、上传数据集/模型 |
| [训练](docs/training.md) | ACT、pi05/openpi、多卡训练 |
| [推理](docs/inference.md) | ACT 真机评测、RTC 本地推理、异步推理 |
| [机械臂工具](docs/arm_utils.md) | 软失能、回 home、夹爪与 CAN 工具 |

相机接线与 CAN 角色对照：[utils/cameras.md](utils/cameras.md)
