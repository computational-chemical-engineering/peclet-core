"""Contract: peclet.halo refuses a bad box, a non-finite position, a negative rcut or a wrong-shape array.

Validation lives in the binding (python/halo_bindings_checks.hpp, used from python/halo_bindings.cpp);
the ValueError names the call, the parameter, the admissible range with its dimension and the
offending value or shape. Each validated argument is covered twice: every bad value raises
ValueError naming the parameter, and a valid call still runs.

Registered under mpirun at np 1 and np 2 (ctest python_input_validation_np{1,2}). Every rank runs
the same cases in the same order with the same arguments, so the collective calls stay matched and
a refused call raises on every rank before any communication.

Run: PYTHONPATH=<python build tree> mpirun -np 2 python -m pytest -q python/test_input_validation.py
"""
import math
import re

import numpy as np
import pytest
from mpi4py import MPI  # initializes MPI first; peclet.halo never calls MPI_Init
from peclet import halo

NAN, INF = math.nan, math.inf
O, E, C, P = [0.0, 0.0, 0.0], [2.0, 1.0, 1.5], [8, 4, 6], [True, True, False]
CLASSES = [halo.ParticleMigrator, halo.ParticleHalo]


def _make(cls, **kw):
    args = {"origin": O, "extent": E, "cells": C, "periodic": P}
    args.update(kw)
    return cls(**args)


def _particles(n=10, k=2):
    rng = np.random.default_rng(0)
    return rng.random((n, 3)) * np.array(E), rng.random((n, k))


def _raises(call, text):
    with pytest.raises(ValueError, match=re.escape(text)):
        call()


def _built():
    h = _make(halo.ParticleHalo)
    pos, _ = _particles()
    h.build(pos, 0.3, True)
    return h, pos


# --- constructors (both classes take the same box) ------------------------------------------------

BAD_BOX = [
    ({"origin": [NAN, 0, 0]}, "origin[0] must be finite (length units)"),
    ({"origin": [0, -INF, 0]}, "origin[1] must be finite"),
    ({"extent": [0, 1, 1]}, "extent[0] must be finite and > 0 (length units)"),
    ({"extent": [2, -1, 1]}, "extent[1] must be finite and > 0"),
    ({"extent": [2, 1, NAN]}, "extent[2] must be finite and > 0"),
    ({"extent": [INF, 1, 1]}, "extent[0] must be finite and > 0"),
    ({"cells": [0, 4, 6]}, "cells[0] must be an integer >= 1 (decomposition cells)"),
    ({"cells": [8, -4, 6]}, "cells[1] must be an integer >= 1"),
    ({"cells": [2**22, 2**21, 2**21]}, "cells[2] must be such that the total cell count fits"),
]


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("kw, text", BAD_BOX)
def test_constructor_refuses(cls, kw, text):
    _raises(lambda: _make(cls, **kw), f"{cls.__name__}: {text}")


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("kw", [{}, {"origin": [-5.0, 1e3, -1e-3]}, {"cells": [1, 1, 1]},
                                {"extent": [1e-6, 1e6, 1.0]}])
def test_constructor_accepts(cls, kw):
    assert _make(cls, **kw).rank == MPI.COMM_WORLD.rank


# --- point queries ----------------------------------------------------------------------------------

QUERIES = [(halo.ParticleMigrator, "owner_of"), (halo.ParticleMigrator, "wrap_position"),
           (halo.ParticleMigrator, "cell_of"), (halo.ParticleHalo, "owner_of")]


@pytest.mark.parametrize("cls, meth", QUERIES, ids=lambda v: getattr(v, "__name__", v))
@pytest.mark.parametrize("x, param", [([NAN, 0.5, 0.5], "x[0]"), ([0.5, INF, 0.5], "x[1]"),
                                      ([0.5, 0.5, -INF], "x[2]")])
def test_query_refuses(cls, meth, x, param):
    obj = _make(cls)
    _raises(lambda: getattr(obj, meth)(x), f"{cls.__name__}.{meth}: {param} must be finite")


@pytest.mark.parametrize("cls, meth", QUERIES, ids=lambda v: getattr(v, "__name__", v))
def test_query_accepts(cls, meth):
    getattr(_make(cls), meth)([-3.25, 7.5, 0.75])  # outside the box: wrapped or clamped


# --- migrate / rebalance / gather_ghosts --------------------------------------------------------------

def _bad_particle_arrays():
    pos, pay = _particles()
    nanpos = pos.copy()
    nanpos[3, 1] = NAN
    infpos = pos.copy()
    infpos[0, 2] = INF
    return [
        ((pos[:, :2].copy(), pay), "positions must be a 2-D (N, 3) array, got shape (10, 2)"),
        ((pos.ravel().copy(), pay), "positions must be a 2-D (N, 3) array, got shape (30,)"),
        ((pos.reshape(10, 3, 1).copy(), pay), "positions must be a 2-D (N, 3) array"),
        ((pos, pay[:, 0].copy()), "payload must be a 2-D (N, K) array with N = 10"),
        ((pos, pay[:5].copy()), "payload must be a 2-D (N, K) array with N = 10"),
        ((pos, np.zeros((1000, 2))), "payload must be a 2-D (N, K) array with N = 10"),
        ((nanpos, pay), "positions[3, 1] must be finite (length units), got nan"),
        ((infpos, pay), "positions[0, 2] must be finite (length units), got inf"),
    ]


def _call(mig, meth, pos, pay):
    return getattr(mig, meth)(pos, pay, 0.3) if meth == "gather_ghosts" else getattr(mig, meth)(pos, pay)


@pytest.mark.parametrize("meth", ["migrate", "rebalance", "gather_ghosts"])
@pytest.mark.parametrize("case", range(len(_bad_particle_arrays())))
def test_particles_refused(meth, case):
    (pos, pay), text = _bad_particle_arrays()[case]
    mig = _make(halo.ParticleMigrator)
    _raises(lambda: _call(mig, meth, pos, pay), f"ParticleMigrator.{meth}: {text}")


@pytest.mark.parametrize("meth", ["migrate", "rebalance", "gather_ghosts"])
@pytest.mark.parametrize("n, k", [(10, 2), (0, 2), (10, 0)])
def test_particles_accepted(meth, n, k):
    pos, pay = _particles(n, k)
    out_pos, out_pay = _call(_make(halo.ParticleMigrator), meth, pos.astype(np.float32), pay)
    assert out_pos.shape[1] == 3 and out_pay.shape[1] == k


@pytest.mark.parametrize("rcut", [-0.3, NAN, INF])
def test_gather_ghosts_rcut_refused(rcut):
    pos, pay = _particles()
    mig = _make(halo.ParticleMigrator)
    _raises(lambda: mig.gather_ghosts(pos, pay, rcut),
            "ParticleMigrator.gather_ghosts: rcut must be finite and >= 0 (length units)")


def test_gather_ghosts_rcut_zero_accepted():
    pos, pay = _particles()
    gpos, _ = _make(halo.ParticleMigrator).gather_ghosts(pos, pay, 0.0)
    assert gpos.shape == (0, 3)


# --- ParticleHalo -----------------------------------------------------------------------------------

@pytest.mark.parametrize("rcut", [-0.3, NAN, INF])
def test_build_rcut_refused(rcut):
    pos, _ = _particles()
    h = _make(halo.ParticleHalo)
    _raises(lambda: h.build(pos, rcut, True),
            "ParticleHalo.build: rcut must be finite and >= 0 (length units)")


@pytest.mark.parametrize("bad, text", [
    (lambda p: p[:, :2].copy(), "positions must be a 2-D (N, 3) array, got shape (10, 2)"),
    (lambda p: p.ravel().copy(), "positions must be a 2-D (N, 3) array, got shape (30,)"),
    (lambda p: np.where(np.arange(30).reshape(10, 3) == 4, NAN, p), "positions[1, 1] must be finite"),
])
def test_build_positions_refused(bad, text):
    pos, _ = _particles()
    h = _make(halo.ParticleHalo)
    _raises(lambda: h.build(bad(pos), 0.3), f"ParticleHalo.build: {text}")


@pytest.mark.parametrize("n, rcut", [(10, 0.3), (10, 0.0), (0, 0.3)])
def test_build_accepted(n, rcut):
    pos, _ = _particles(n)
    h = _make(halo.ParticleHalo)
    assert h.build(pos, rcut, True) == h.num_ghost
    assert h.num_owned == n


@pytest.mark.parametrize("meth", ["forward", "forward_positions"])
@pytest.mark.parametrize("rows", [lambda p: p[:3].copy(), lambda p: np.zeros((1000, 3)),
                                  lambda p: p[:, :2].copy(), lambda p: p.ravel().copy()])
def test_forward_rows_refused(meth, rows):
    h, pos = _built()
    _raises(lambda: getattr(h, meth)(rows(pos)),
            f"ParticleHalo.{meth}: owned must be a 2-D (10, 3) array (N = num_owned")


@pytest.mark.parametrize("meth", ["forward", "forward_positions"])
def test_forward_before_build_refused(meth):
    pos, _ = _particles()
    h = _make(halo.ParticleHalo)
    _raises(lambda: getattr(h, meth)(pos), f"ParticleHalo.{meth}: owned must be a 2-D (0, 3) array")


@pytest.mark.parametrize("meth", ["forward", "forward_positions"])
def test_forward_accepted(meth):
    h, pos = _built()
    assert getattr(h, meth)(pos).shape == (h.num_ghost, 3)


def test_reverse_refused():
    h, pos = _built()
    g = h.forward(pos)
    _raises(lambda: h.reverse(np.zeros((g.shape[0] + 1, 3)), pos),
            "ParticleHalo.reverse: ghost must be a 2-D")
    _raises(lambda: h.reverse(g[:, :2].copy(), pos), "ParticleHalo.reverse: ghost must be a 2-D")
    _raises(lambda: h.reverse(g, pos[:2].copy()),
            "ParticleHalo.reverse: owned must be a 2-D (10, 3) array (N = num_owned")


def test_reverse_accepted():
    h, pos = _built()
    out = h.reverse(h.forward(pos), pos)
    assert out.shape == pos.shape
