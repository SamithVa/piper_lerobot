# 遥操作

## 单臂

````
lerobot-teleoperate \
    --robot.type=piper_follower \
    --robot.id=my_follower_arm \
    --teleop.type=piper_leader \
    --teleop.id=my_leader_arm \
    --display_data=true
````

## 双臂

需要先激活 4 条 CAN（见[环境安装](setup.md)）。使用本仓库 src 下的代码：

````
PYTHONPATH=src python -m lerobot.scripts.lerobot_teleoperate \
    --robot.type=bi_piper_follower \
    --robot.id=bi_piper \
    --teleop.type=bi_piper_leader \
    --teleop.id=bi_piper_leader \
    --display_data=true
````

## EMA 动作平滑

主臂读数带抖动时，可用指数移动平均（EMA）平滑关节动作，让从臂运动更顺滑：

````
--teleop.ema_alpha=0.4
````

- 只平滑 joint_1~joint_6，**夹爪不平滑**（避免开合延迟影响抓取时机）。
- 平滑发生在主臂 `get_action()` 内，因此**录进数据集的动作与从臂实际执行的一致**。
- 取值范围 (0, 1]，不传则关闭。30fps 下延迟约为 `(1-α)/α` 帧：
  0.5 → 约 33ms，0.4 → 约 50ms（推荐起点），0.2 → 约 130ms。
  仍觉得抖就调小，觉得跟手性差就调大。

## 调试工具

````
python utils/watch_leader_grippers.py   # 实时查看主臂夹爪读数
python utils/test_gripper.py            # 测试夹爪
python utils/zero_gripper.py            # 夹爪零位
````
