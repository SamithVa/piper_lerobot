"""Pi0.5 deployment package — public interface only.

The implementation lives in the compiled ``_core`` C-extension; its source is
not shipped. Only ``Pi05Deployer`` and ``DEFAULT_CHECKPOINT`` are exposed.
"""
from ._core import DEFAULT_CHECKPOINT, Pi05Deployer

__all__ = ["Pi05Deployer", "DEFAULT_CHECKPOINT"]
__version__ = "0.1.0"
