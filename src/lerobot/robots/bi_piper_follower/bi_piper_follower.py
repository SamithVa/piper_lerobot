#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs

from ..piper_follower import PIPERFollower
from ..piper_follower.config_piper_follower import PIPERFollowerConfig
from ..robot import Robot
from .config_bi_piper_follower import BiPiperFollowerConfig

logger = logging.getLogger(__name__)


class BiPiperFollower(Robot):
    """
    Bimanual AgileX Piper follower: wraps a left and a right ``PIPERFollower``.

    Joint keys are prefixed with ``left_`` / ``right_`` so the two arms produce
    distinct action/state dimensions. Cameras are shared and owned here (the
    wrapped single-arm followers are created with no cameras).
    """

    config_class = BiPiperFollowerConfig
    name = "bi_piper_follower"

    def __init__(self, config: BiPiperFollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = PIPERFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            can_name=config.left_arm_can,
            cameras={},
        )
        right_arm_config = PIPERFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            can_name=config.right_arm_can,
            cameras={},
        )

        self.left_arm = PIPERFollower(left_arm_config)
        self.right_arm = PIPERFollower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam_key: (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def connect(self) -> None:
        # Each single-arm connect() enables the arm, homes it, and marks connected.
        self.left_arm.connect()
        self.right_arm.connect()

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self, **kwargs) -> None:
        self.left_arm.configure(**kwargs)
        self.right_arm.configure(**kwargs)

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)

        prefixed_left = {f"left_{key}": value for key, value in sent_left.items()}
        prefixed_right = {f"right_{key}": value for key, value in sent_right.items()}

        return {**prefixed_left, **prefixed_right}

    def disconnect(self) -> None:
        # Each single-arm disconnect() homes the arm then soft-releases it (MIT
        # damping ramp) so it droops instead of free-falling. The two arms are on
        # separate CAN buses, so run them concurrently -> ~half the wall time.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(self.left_arm.disconnect), ex.submit(self.right_arm.disconnect)]
            for f in futures:
                f.result()

        for cam in self.cameras.values():
            cam.disconnect()
