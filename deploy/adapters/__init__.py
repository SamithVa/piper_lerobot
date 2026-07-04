"""Adapter registry. Imports are lazy so listing/creating the dummy adapter
never pulls torch, and the lerobot adapter only imports in the server env."""
from __future__ import annotations

from .base import PolicyAdapter as PolicyAdapter

_ADAPTERS = ("dummy", "lerobot")


def make_adapter(name: str, **kwargs) -> PolicyAdapter:
    if name == "dummy":
        from .dummy import DummyAdapter

        return DummyAdapter(**kwargs)
    if name == "lerobot":
        from .lerobot import LerobotAdapter

        return LerobotAdapter(**kwargs)
    raise ValueError(f"Unknown adapter '{name}'. Available: {', '.join(_ADAPTERS)}")
