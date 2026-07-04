"""Pure queue logic for the async-overlap control loop. No I/O, no threads —
the client calls everything from its single control-loop thread.

Per control tick:
    action = executor.next_action()          # None -> queue dry, hold position
    if executor.should_request():
        # capture obs NOW, then:
        executor.mark_requested()
        # ... send obs to the server on a background thread ...
    # when the reply lands (any later tick):
    executor.on_chunk(chunk)                  # or executor.on_request_failed()
"""
from __future__ import annotations

from collections import deque

import numpy as np


class ChunkExecutor:
    def __init__(self, chunk_threshold: float = 0.5):
        if not 0.0 < chunk_threshold <= 1.0:
            raise ValueError(f"chunk_threshold must be in (0, 1], got {chunk_threshold}")
        self.chunk_threshold = chunk_threshold
        self._queue: deque = deque()
        self._chunk_len = 0  # length of the chunk the current queue came from
        self._tick = 0
        self._request_tick: int | None = None

    @property
    def in_flight(self) -> bool:
        return self._request_tick is not None

    def next_action(self):
        """Advance one control tick; return the next action row or None if dry."""
        self._tick += 1
        if self._queue:
            return self._queue.popleft()
        return None

    def should_request(self) -> bool:
        if self.in_flight:
            return False
        if self._chunk_len == 0 or not self._queue:
            return True
        consumed = 1.0 - len(self._queue) / self._chunk_len
        return consumed >= self.chunk_threshold

    def mark_requested(self) -> None:
        """Call at the tick the observation was captured."""
        self._request_tick = self._tick

    def on_chunk(self, chunk: np.ndarray) -> None:
        """Install a fresh chunk, skipping the rows whose time already passed.

        Row 0 of the chunk is the action for the observation's tick; if k ticks
        elapsed between capture and arrival, rows [0, k) are stale.
        """
        elapsed = 0 if self._request_tick is None else self._tick - self._request_tick
        self._request_tick = None
        rows = list(np.asarray(chunk))
        usable = rows[elapsed:]
        self._queue = deque(usable)
        self._chunk_len = len(usable)

    def on_request_failed(self) -> None:
        self._request_tick = None
