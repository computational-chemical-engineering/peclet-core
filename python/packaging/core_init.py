"""peclet.core — Python surface for the core shared infrastructure.

Submodules (compiled nanobind extensions, built by python/CMakeLists.txt):

* ``peclet.core.mpi`` — the Lagrangian particle halo: ORB block decomposition, particle
  migration/ghosts (``ParticleMigrator``, ``ParticleHalo``) and count-weighted rebalancing for an
  mpi4py driver. Always built.
* ``peclet.core.geom`` — analytic-SDF scene authoring (primitives, transforms, booleans) and
  rigid-body mass properties; host-only preprocessing tooling. Always built.

The AMR octree + flow (``Octree``, ``DistributedOctree``, ``Poisson``, ``Flow``) is the separate
``peclet-amr`` package, ``peclet.amr``, since 2026-09-10 — it was ``peclet.core.amr`` before.

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
__all__ = ["mpi", "geom"]  # noqa: F822 — compiled nanobind submodules, resolved lazily
