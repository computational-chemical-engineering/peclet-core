"""peclet.core — Python surface for the core shared infrastructure.

Submodules (compiled nanobind extensions, built by python/CMakeLists.txt):

* ``peclet.core.mpi`` — the Lagrangian particle halo: ORB block decomposition, particle
  migration/ghosts and ``rebalanceByParticleCount`` for an mpi4py driver. Always built.
* ``peclet.core.amr`` — the device (Kokkos) AMR octree flow. Built only when a Kokkos backend
  prefix and the ``peclet.morton`` headers are present; otherwise the import is absent.

``peclet`` itself is an implicit (PEP 420) namespace shared with the other ``peclet-*`` packages,
so it deliberately has no top-level ``__init__.py``.
"""

__version__ = "0.2.0"
__all__ = ["mpi", "amr"]  # noqa: F822 — compiled nanobind submodules, resolved lazily (amr optional)
