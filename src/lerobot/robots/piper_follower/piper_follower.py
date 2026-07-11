#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.piper.piper import PiperMotorsBus, PiperMotorsBusConfig

from ..robot import Robot
from .config_piper_follower import PIPERFollowerConfig

logger = logging.getLogger(__name__)

# How many enable+home attempts before giving up on a follower arm. A single miss
# is usually a transient CAN/USB stall on the shared 2.0 hub; retrying in-process
# clears it without crashing the whole record.
ARM_READY_ATTEMPTS = 3

def get_motor_names(arm: dict[str, Any]) -> list[str]:
    return [motor for arm_key, bus in arm.items() for motor in bus.motors]

class PIPERFollower(Robot):
    config_class = PIPERFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PIPERFollowerConfig):
        super().__init__(config)
        self.config = config
        self.bus = PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name=config.can_name,
                motors={
                    "joint_1": (1, "agilex_piper"),
                    "joint_2": (2, "agilex_piper"),
                    "joint_3": (3, "agilex_piper"),
                    "joint_4": (4, "agilex_piper"),
                    "joint_5": (5, "agilex_piper"),
                    "joint_6": (6, "agilex_piper"),
                    "gripper": (7, "agilex_piper"),
                }
            )
        )
        self.logs = {}
        self._is_connected = False
        self._is_calibrated = False
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,   
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        arm_dict = {"follower": self.bus}
        action_names = get_motor_names(arm_dict)
        state_names = get_motor_names(arm_dict)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def _motors_ft(self) -> dict[str, type]:
        """用于 record/replay 的电机动作描述"""
        arm_dict = {"follower": self.bus}
        motor_names = get_motor_names(arm_dict)
        return {f"{name}.pos": float for name in motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """用于 record/replay 的相机图像描述"""
        #return {f"observation.images.{cam_key}": (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}
        return {cam_key: (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def configure(self, **kwargs):
        # 不做任何事,过抽象类用
        pass

    @property
    def is_connected(self) -> bool:
        """机器人和所有相机是否都已连接"""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """机器人是否已完成标定"""
        return self.bus.is_calibrated

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self) -> None:
        """Connect piper and cameras"""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "Piper is already connected. Do not run robot.connect() twice."
            )

        # Enable the arm AND verify it actually reaches home -- together, with
        # retries. bus.connect() confirms all motors enabled + feedback live;
        # move_to_home() proves the arm accepts motion (reaches home). On the
        # shared USB-2.0 CAN hub a transient stall can make one attempt miss
        # (enable not sticking, or move commands not landing); a fresh enable+home
        # almost always clears it. Retrying in-process automates the old "just
        # rerun the script" instead of crashing on the first miss.
        self._is_connected = True  # move_to_home()/read() require this; cleared on total failure
        for attempt in range(ARM_READY_ATTEMPTS):
            if self.bus.connect(enable=True) and self.bus.move_to_home():
                self._is_calibrated = True
                break
            print(f"piper follower not ready (enable/home) — retry {attempt + 1}/{ARM_READY_ATTEMPTS} ...")
        else:
            self._is_connected = False
            last = {k: round(v) for k, v in self.bus.read().items()}
            raise DeviceNotConnectedError(
                f"{self} did not reach home after {ARM_READY_ATTEMPTS} enable+home attempts "
                f"(last joints={last}). Likely a CAN/USB stall on the shared 2.0 hub or an "
                "unpowered/faulted arm — re-run utils/activate_all_can.sh and check power/CAN."
            )
        print("piper follower connected")

        # connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            print(f"camera {name} connected")

        print("All connected")

    def disconnect(self) -> None:
        """回 home 后软失能(阻尼缓慢放下, 避免自由落体), 再断开相机。"""
        print("piper gently disabling (home + soft release)...")
        self.bus.gentle_disable()
        print("piper gently disabled")
        self.bus.disconnect()

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self._is_connected = False

    def calibrate(self):
        """move piper to the home position (polled + verified, with retries)"""
        if not self._is_connected:
            raise ConnectionError()

        # move_to_home() drives to home and returns whether it was reached. Retry
        # a few times: a single miss is usually a transient CAN/USB stall, not a
        # dead arm. Only fail loudly (teleop would be broken) after exhausting.
        for attempt in range(ARM_READY_ATTEMPTS):
            if self.bus.move_to_home():
                self._is_calibrated = True
                return
            print(f"piper follower re-home — retry {attempt + 1}/{ARM_READY_ATTEMPTS} ...")
        raise DeviceNotConnectedError(
            f"{self} did not reach home after {ARM_READY_ATTEMPTS} attempts -- the arm is "
            "not accepting motion commands. Check power/CAN and retry."
        )

    def get_observation(self) -> dict:
        """Capture current joint positions and camera images"""
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected. Run `robot.connect()` first.")

        # 读取关节状态
        state = self.bus.read()  # e.g., {'joint_1': 0.1, ..., 'gripper': 0.0}
        obs_dict = {f"{joint}.pos": float(val) for joint, val in state.items()}

        # 读取图像
        for name, cam in self.cameras.items():
            #obs_dict[f"observation.images.{name}"] = cam.async_read()
            obs_dict[name] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Receive action dict from record() and send to motor"""
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")

        motor_order = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        target_joints = [action[f"{motor}.pos"] for motor in motor_order]

        self.bus.write(target_joints)
        return action
