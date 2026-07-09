"""Deterministic stand-in policy: no torch, instant. Used by the deploy test
suite and as the smallest reference for writing new adapters.

Action row t is the constant vector [t, t, ..., t], so tests can tell which
step of a chunk got executed.
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

    def predict_chunk(self, images, state, task) -> np.ndarray:
        if self.fail:
            raise RuntimeError("dummy failure requested")
        steps = np.arange(self.chunk_size, dtype=np.float32)[:, None]
        return np.repeat(steps, self.action_dim, axis=1)

    def reset(self) -> None:
        self.reset_count += 1
