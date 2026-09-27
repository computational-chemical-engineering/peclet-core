"""peclet.core.mpi — the pre-1.2 spelling of peclet.halo; `from peclet import halo` is canonical.

Re-exports peclet.halo unchanged: the objects here ARE the objects there. Works unchanged in
peclet 1.2.0, warns from peclet-core 1.3.1 (family 1.3.0), removed in 2.0.0
(suite/docs/CORE_BOUNDARY.md).
"""
import warnings

warnings.warn(
    "peclet.core.mpi is peclet.halo since peclet 1.2.0 — use `from peclet import halo`. "
    "peclet.core is removed in peclet 2.0.0.", DeprecationWarning, stacklevel=2)

from peclet.halo import *  # noqa: F401,F403,E402
from peclet import halo as _canonical  # noqa: E402

__all__ = [n for n in dir(_canonical) if not n.startswith("_")]
