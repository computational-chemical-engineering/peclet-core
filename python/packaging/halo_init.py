"""peclet.halo — the distributed Lagrangian particle halo: ORB block decomposition, particle
migration, ghost exchange and weighted rebalancing over MPI.

Named for what it is, not for what it links against (suite/docs/CORE_BOUNDARY.md): `halo` is
already the thing's name on the C++ side — `peclet::halo`, `GridHalo`, `ParticleHalo`.

Needs an MPI toolchain to build and an MPI runtime to use, which is why it ships as an sdist behind
`pip install peclet[mpi]` while its former housemate `peclet.geom` ships wheels.

Until peclet 1.2.0 this was ``peclet.core.mpi`` in the ``peclet-core`` distribution. That spelling
still works and is the same object; it warns from 1.3.0 and is removed in 2.0.0.
"""
from importlib.metadata import PackageNotFoundError, version as _version

from ._halo import *  # noqa: F401,F403
from . import _halo as _ext

try:
    __version__ = _version("peclet-halo")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [n for n in dir(_ext) if not n.startswith("_")]
