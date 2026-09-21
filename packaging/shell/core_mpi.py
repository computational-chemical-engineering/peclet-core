"""peclet.core.mpi — the pre-1.2 spelling of peclet.halo; `from peclet import halo` is canonical.

Re-exports peclet.halo unchanged: the objects here ARE the objects there. Works unchanged in
peclet 1.2.0, warns in 1.3.0, removed in 2.0.0 (suite/docs/CORE_BOUNDARY.md).
"""
from peclet.halo import *  # noqa: F401,F403
from peclet import halo as _canonical

__all__ = [n for n in dir(_canonical) if not n.startswith("_")]
