"""peclet.core — the pre-1.2 home of `peclet.geom` and `peclet.halo`.

This package is a thin compatibility shell. Its two submodules re-export, unchanged, the objects
that now live in the `peclet-geom` and `peclet-halo` distributions:

    peclet.core.geom  ->  peclet.geom     (package peclet-geom, wheels)
    peclet.core.mpi   ->  peclet.halo     (package peclet-halo, sdist, needs MPI)

They are the *same objects*, not wrappers: ``peclet.core.geom.SceneBuilder is
peclet.geom.SceneBuilder`` is True, so isinstance, pickling and docs all behave as before.

Why the split: `peclet.core.geom` is host-only SDF authoring with no MPI in it, but it shared a
distribution with the halo bindings whose build requires an MPI toolchain — so a pure-geometry API
could not be pip-installed without MPI. See suite/docs/CORE_BOUNDARY.md.

These spellings keep working unchanged in peclet 1.2.0, gain a DeprecationWarning in 1.3.0, and are
removed in 2.0.0. The canonical spellings are ``from peclet import geom`` and
``from peclet import halo``.
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("peclet-core")
except PackageNotFoundError:  # a source tree / build tree has no dist-info
    __version__ = "0+unknown"
