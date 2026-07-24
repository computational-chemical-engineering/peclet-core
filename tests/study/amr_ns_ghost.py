"""Navier–Stokes with the ghost projection (ladder step 4) — device validation battery.

Z&H SC-sphere geometry at finite Re (advection ON, implicit-FOU + deferred SOU, advected by the
uf face field). Body force chosen for a target Reynolds number Re_D = rho*U*2R/mu ~ 10 using the
Stokes-K estimate U ~ f N^3/(6 pi mu R K).

  steady   — aperture vs ghost converged apparent K at N=32,64 (scheme A/B; no external truth at
             finite Re — the gate is closeness + health).
  unsteady — impulsive start at N=64: Umean(t) trajectory aperture vs ghost (max rel deviation
             after the initial transient) + residual-divergence traces (bounded).
  graded   — finest 64, band 5, ghost + cf-quadratic + advection: stable, converges near the
             uniform ghost value.

Run FOREGROUND on the CUDA build:
  python core/tests/study/amr_ns_ghost.py [steady|unsteady|graded ...]
"""
import math
import sys
import time

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
import numpy as np  # noqa: E402
from peclet.core import amr  # noqa: E402

PHI = 0.125
MU = 0.1
K_STOKES = 4.292


def setup(N, ghost, cf=0, band=None, dt=None, target_re=10.0):
    lmax = int(math.log2(N))
    t = amr.Octree([1, 1, 1], lmax, [0.0, 0.0, 0.0], 1.0)
    R = (PHI * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * N
    c = N / 2.0
    if band is None:
        t.refine_to_sdf(lambda x, y, z: 0.0, 0, 1e30, True)
    else:
        t.refine_to_sphere([c, c, c], R, 0, band, True)
    U = target_re * MU / (2.0 * R)  # rho = 1
    f = U * 6.0 * math.pi * MU * R * K_STOKES / N ** 3
    if dt is None:
        dt = 1.0 / U  # advective CFL ~ 1 on the finest cells
    fl = amr.Flow(t, 1.0, MU, dt)
    fl.set_body_force(f, 0.0, 0.0)
    fl.set_advection(True)
    if ghost:
        fl.set_ghost_projection(True, 1, 2)
    if cf:
        fl.set_cf_scheme(cf)
    fl.set_solid(lambda x, y, z:
                 math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) - R)
    w = np.asarray(t.sizes()) ** 3
    return t, fl, w, R, f


def umean(fl, w, N):
    return float((np.asarray(fl.velocity(0)) * w).sum()) / N ** 3


def run_steady(N, ghost, tol=1e-7, max_steps=8000):
    t, fl, w, R, f = setup(N, ghost)
    kprev, k, steps, confirm = None, None, 0, False
    t0 = time.time()
    while steps < max_steps:
        for _ in range(10):
            fl.step(100, 60)
        steps += 10
        um = umean(fl, w, N)
        k = f * N ** 3 / (6.0 * math.pi * MU * R * um)
        if kprev is not None and abs(k - kprev) < tol * abs(k):
            if confirm:
                break
            confirm = True
        else:
            confirm = False
        kprev = k
    re = umean(fl, w, N) * 2.0 * R / MU
    return k, re, steps, fl.last_pres_iters(), fl.divergence_norm(), time.time() - t0


def steady():
    print("== steady NS (apparent K = f N^3/(6 pi mu R U); Re ~10) ==", flush=True)
    print(f"{'N':>4} {'scheme':>9} {'K_app':>9} {'Re':>6} {'steps':>6} {'pres':>5} "
          f"{'div':>9} {'secs':>5}", flush=True)
    out = {}
    for N in (32, 64):
        for ghost in (False, True):
            k, re, s, pi, dv, el = run_steady(N, ghost)
            out[(N, ghost)] = k
            print(f"{N:>4} {'ghost' if ghost else 'aperture':>9} {k:>9.4f} {re:>6.2f} "
                  f"{s:>6} {pi:>5} {dv:>9.2e} {el:>5.0f}", flush=True)
        ga = 100 * (out[(N, True)] - out[(N, False)]) / out[(N, False)]
        print(f"     scheme gap at N={N}: {ga:+.3f}%", flush=True)


def unsteady(N=64, steps=600):
    print("== unsteady impulsive start: Umean(t) aperture vs ghost ==", flush=True)
    tr = {}
    for ghost in (False, True):
        t, fl, w, R, f = setup(N, ghost, dt=None)
        us, dvs = [], []
        for s in range(steps):
            fl.step(100, 60)
            us.append(umean(fl, w, N))
            if s % 20 == 19:
                dvs.append(fl.divergence_norm())
        tr[ghost] = (np.array(us), np.array(dvs))
        print(f"  {'ghost' if ghost else 'aperture'}: U_final={us[-1]:.6e} "
              f"div max/final={max(dvs):.2e}/{dvs[-1]:.2e}", flush=True)
    ua, ug = tr[False][0], tr[True][0]
    rel = np.abs(ug - ua) / np.maximum(np.abs(ua), 1e-30)
    i0 = 40  # skip the first few steps (tiny U, relative deviation meaningless)
    print(f"  trajectory max rel deviation (steps {i0}..{len(ua)}): {rel[i0:].max():.3e} "
          f"(final {rel[-1]:.3e})", flush=True)


def graded(N=64, band=5.0, tol=1e-7, max_steps=8000):
    print("== graded NS: ghost + cf-quadratic + advection (finest 64, band 5) ==", flush=True)
    for label, b, cf in (("uniform ghost", None, 0), ("graded ghost+cf", band, 1)):
        t, fl, w, R, f = setup(N, True, cf=cf, band=b)
        kprev, k, steps, confirm = None, None, 0, False
        while steps < max_steps:
            for _ in range(10):
                fl.step(100, 60)
            steps += 10
            um = umean(fl, w, N)
            k = f * N ** 3 / (6.0 * math.pi * MU * R * um)
            if kprev is not None and abs(k - kprev) < tol * abs(k):
                if confirm:
                    break
                confirm = True
            else:
                confirm = False
            kprev = k
        print(f"  {label}: K_app={k:.4f} leaves={t.num_leaves} steps={steps} "
              f"pres={fl.last_pres_iters()} div={fl.divergence_norm():.2e}", flush=True)


if __name__ == "__main__":
    which = [a for a in sys.argv[1:]] or ["steady", "unsteady", "graded"]
    if "steady" in which:
        steady()
    if "unsteady" in which:
        unsteady()
    if "graded" in which:
        graded()
