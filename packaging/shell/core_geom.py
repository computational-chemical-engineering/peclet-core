"""peclet.core.geom — the pre-1.2 spelling of peclet.geom; `from peclet import geom` is canonical.

Re-exports peclet.geom unchanged: the objects here ARE the objects there. Works unchanged in
peclet 1.2.0, warns from peclet-core 1.3.1 (family 1.3.0), removed in 2.0.0
(suite/docs/CORE_BOUNDARY.md).
"""
import warnings

warnings.warn(
    "peclet.core.geom is peclet.geom since peclet 1.2.0 — use `from peclet import geom`. "
    "peclet.core is removed in peclet 2.0.0.", DeprecationWarning, stacklevel=2)

from peclet.geom import *  # noqa: F401,F403,E402
from peclet import geom as _canonical  # noqa: E402

__all__ = [n for n in dir(_canonical) if not n.startswith("_")]
