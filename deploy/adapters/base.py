"""The one interface a policy must implement to be deployable on the Piper."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PolicyAdapter(ABC):
    """Serve one policy. Implement these methods in whatever conda env the
    policy needs; the server, transport, and robot client never change."""

    @abstractmethod
    def info(self) -> dict:
        """Static metadata:
        {"name": str, "image_keys": list[str], "state_dim": int,
         "action_dim": int, "chunk_size": int, "fps": float}"""

    @abstractmethod
    def predict_chunk(
        self, images: dict[str, np.ndarray], state: np.ndarray, task: str
    ) -> np.ndarray:
        """images: HWC uint8 RGB keyed by the policy's image keys;
        state: (state_dim,). Returns (chunk_size, action_dim) float32."""

    def reset(self) -> None:
        """Clear per-episode state (action queues etc.). Optional."""
