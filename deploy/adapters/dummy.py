"""Deterministic stand-in policy: no torch, instant. Used by the deploy test
suite, by presets/example.json, and as the reference for writing new adapters.

The full adapter contract (see also base.py):
  info() -> {"name": str, "image_keys": list[str], "state_dim": int,
             "action_dim": int, "chunk_size": int, "fps": float,
             "checkpoint": str | None}
  predict_chunk(images, state, task, consumed=-1, delay_ticks=0) -> np.ndarray
      images: {image_key: HWC uint8 RGB array} — exactly the keys in
              info()["image_keys"]; state: float32 (state_dim,); task: str;
              consumed/delay_ticks: RTC hints from the client (see base.py).
      Returns float32 (chunk_size, action_dim): absolute motor targets in the
      same units/order the robot's action_features use.
  reset() -> None — clear per-episode state (action queues, KV caches, ...).

Constructor kwargs arrive as STRINGS (forwarded from --key=value server
flags), so coerce types yourself, as done below.

Action row t here is the constant vector [t, t, ..., t], so tests can tell
which step of a chunk got executed.
"""
from __future__ import annotations

import numpy as np

from .base import PolicyAdapter


class DummyAdapter(PolicyAdapter):
    def __init__(
        self,
        state_dim=14,
        action_dim=14,
        chunk_size=10,
        fps=30.0,
        image_keys="camera1,camera2",
        fail=False,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.fps = float(fps)
        self.image_keys = [key for key in str(image_keys).split(",") if key]
        self.fail = bool(int(fail)) if isinstance(fail, str) else bool(fail)
        self.reset_count = 0
        self.last_meta = None

    def info(self) -> dict:
        return {
            "name": "dummy",
            "image_keys": self.image_keys,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "fps": self.fps,
            "checkpoint": None,
        }

    def predict_chunk(self, images, state, task, consumed=-1, delay_ticks=0) -> np.ndarray:
        self.last_meta = {"consumed": consumed, "delay_ticks": delay_ticks}
        if self.fail:
            raise RuntimeError("dummy failure requested")
        steps = np.arange(self.chunk_size, dtype=np.float32)[:, None]
        return np.repeat(steps, self.action_dim, axis=1)

    def reset(self) -> None:
        self.reset_count += 1
