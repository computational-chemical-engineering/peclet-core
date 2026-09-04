"""peclet.core — Python surface for the core shared infrastructure.

Submodules (compiled nanobind extensions, built by python/CMakeLists.txt):

* ``peclet.core.mpi`` — the Lagrangian particle halo: ORB block decomposition, particle
  migration/ghosts and ``rebalanceByParticleCount`` for an mpi4py driver. Always built.
* ``peclet.core.amr`` — the device (Kokkos) AMR octree flow. Built only when a Kokkos backend
  prefix and the ``peclet.morton`` headers are present; otherwise the import is absent.

``peclet`` itself is an implicit (PEP 420) namespace shared with the other ``peclet-*`` packages,
so it deliberately has no top-level ``__init__.py``.
"""

# The installed distribution's metadata (pyproject.toml) is the single source of truth for the version;
# a build-tree import (PYTHONPATH=<build>) has no metadata and reports "0+unknown".
try:
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("peclet-core")
except Exception:  # PackageNotFoundError (dev build), or a broken metadata install
    __version__ = "0+unknown"
__all__ = ["mpi", "amr"]  # noqa: F822 — compiled nanobind submodules, resolved lazily (amr optional)
