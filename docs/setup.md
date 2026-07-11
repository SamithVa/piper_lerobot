# 环境安装与硬件连接

## 1. 环境创建

### 安装 lerobot 依赖

````
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install -c conda-forge ffmpeg=7.1.1 -y
pip install transformers --upgrade
git clone https://github.com/jokeru8/piper_lerobot.git
cd piper_lerobot
pip install -e .
````

### 安装 piper 依赖 (pyAgxArm SDK)

````
pip install python-can
pip install -e third_party/pyAgxArm
````

## 2. 连接机械臂（CAN）

### 双臂（4 条 CAN，按角色命名）

一次性把 left_leader / left_follower / right_leader / right_follower 全部激活
（串口号 → 角色的映射在脚本里配置，txqueuelen 已调大避免发送队列溢出）：

````
bash utils/activate_all_can.sh
````

检查 CAN 状态：

````
bash utils/can_health.sh
````

### 单臂（手动激活）

"3-7.1:1.0" 根据输出的 can 端口号改为自己的：

````
conda activate lerobot
bash utils/find_all_can_port.sh
bash utils/can_activate.sh can_master 1000000 "1-8.2:1.0"
bash utils/can_activate.sh can_follower 1000000 "1-8.3:1.0"
````

## 3. 相机

### 符号链接（持久化设备名）

创建 /dev/l_wrist、/dev/top、/dev/r_wrist（重启后仍有效）：

````
sudo bash utils/setup_camera_symlinks.sh
````

### 测试相机

注意两个相机不能从同一个扩展坞连接电脑，否则可能读取会出问题。

````
sudo apt install guvcview       # 安装 Guvcview
guvcview --device=/dev/l_wrist  # 测试 l_wrist 相机
guvcview --device=/dev/top      # 测试 top 相机
guvcview --device=/dev/r_wrist  # 测试 r_wrist 相机
````

