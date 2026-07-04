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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PIPERLeaderConfig(TeleoperatorConfig):
    # CAN interface name for this leader arm (see 99-piper-can.rules)
    can_name: str = "left_leader"

    # EMA smoothing factor for joint actions read in get_action(). None disables
    # smoothing. In (0, 1]: smaller = smoother but laggier. At a 30 fps teleop
    # loop, lag ≈ (1-α)/α frames: 0.5 → ~33 ms, 0.4 → ~50 ms, 0.2 → ~130 ms.
    # The gripper channel is never smoothed (would delay open/close timing).
    ema_alpha: float | None = None

    def validate_ema_alpha(self) -> None:
        if self.ema_alpha is not None and not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1] or None, got {self.ema_alpha}")