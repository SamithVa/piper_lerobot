# 机械臂工具（utils/）

## 软失能（避免机械臂掉落）

直接失能会瞬间断电，机械臂仍在对抗重力，断电瞬间会硬砸下去。
软失能流程：先用位置控制缓慢回到 home 姿态，再用 MIT 力控保持当前关节角，
把位置增益 kp 缓慢降到 0（保留阻尼 kd），机械臂被阻尼缓慢放下而不是自由落体，最后再真正失能。

````
python utils/gentle_disable_arm.py                        # 默认软失能全部 4 个机械臂
python utils/gentle_disable_arm.py left_follower          # 失能单个机械臂
python utils/gentle_disable_arm.py left_follower right_follower   # 失能多个机械臂
python utils/gentle_disable_arm.py left_follower --duration 3.0 --kp 12 --kd 1.2
python utils/gentle_disable_arm.py left_follower --no-home # 跳过回 home，从当前姿态软放下
````

can 名称参照 [utils/cameras.md](../utils/cameras.md)。

kp/kd/duration 需在真机上微调：kd 越大放下越慢越软（过大会抖动），duration 为 kp 降到 0 的时长，
--home-speed 为回 home 的速度百分比（默认 15，较慢）。

## 回 home

````
python utils/home_each_arm.py
````

## 夹爪

````
python utils/test_gripper.py            # 测试夹爪
python utils/zero_gripper.py            # 夹爪零位
python utils/watch_leader_grippers.py   # 实时查看主臂夹爪读数
````

## CAN

````
bash utils/activate_all_can.sh          # 按角色激活全部 4 条 CAN
bash utils/can_health.sh                # CAN 健康检查
bash utils/find_all_can_port.sh         # 列出所有 CAN 端口
bash utils/reactivate_can_master2.sh    # 重新激活指定 CAN
````
