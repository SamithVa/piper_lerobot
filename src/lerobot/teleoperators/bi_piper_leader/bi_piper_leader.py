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
from functools import cached_property

from ..piper_leader import PIPERLeader
from ..piper_leader.config_piper_leader import PIPERLeaderConfig
from ..teleoperator import Teleoperator
from .config_bi_piper_leader import BiPiperLeaderConfig

logger = logging.getLogger(__name__)


class BiPiperLeader(Teleoperator):
    """
    Bimanual AgileX Piper leader: wraps a left and a right ``PIPERLeader``.

    Action keys are prefixed with ``left_`` / ``right_`` to match ``BiPiperFollower``.
    """

    config_class = BiPiperLeaderConfig
    name = "bi_piper_leader"

    def __init__(self, config: BiPiperLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = PIPERLeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            can_name=config.left_arm_can,
            ema_alpha=config.ema_alpha,
        )
        right_arm_config = PIPERLeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            can_name=config.right_arm_can,
            ema_alpha=config.ema_alpha,
        )

        self.left_arm = PIPERLeader(left_arm_config)
        self.right_arm = PIPERLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{key}": float for key in self.left_arm.action_features} | {
            f"right_{key}": float for key in self.right_arm.action_features
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self, **kwargs) -> None:
        self.left_arm.configure(**kwargs)
        self.right_arm.configure(**kwargs)

    def get_action(self) -> dict[str, float]:
        action_dict = {}

        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Leader arms have no feedback channel; nothing to do.
        pass

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
