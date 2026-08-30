"""M2a — the cloud-economy study (docs/amr_march_perf_and_distributed_plan.md, M2 verdict).

M1 attributed the graded mesh's missing speed-up to the sampled overlay's least-squares CSR: it
GROWS as the mesh coarsens, and at depth 8 it is 4.3x the uniform band's while the mesh has 1.62x
fewer cells. The clouds are also ~10x oversampled — a degree-2 LS needs 12 points and the shipped
radius gathers 95-162 — so the obvious lever is to sample less. That changes the WEIGHTS, i.e.
march numerics, so it goes through a pre-registered a-priori ladder rather than a tuning loop.

Two knobs, read by `buildGhostOverlaySampled`, both inert at their defaults:
  PECLET_CORE_GPS_RHO   radius factor (default 2.2; rho = factor * max(h, H))
  PECLET_CORE_GPS_MAXN  keep the N nearest candidates (default 0 = no cap)

This script sweeps them on the RCP bed and reports, per variant: the overlay census (CSR size and
the LS2/LS1/degraded cascade — LS1 > 0 is the flag that matters, since M2 established degree-2 is
REQUIRED and degree-1 is an O(1) non-decaying perturbation), and the Darcy permeability against
the uniform-finest-band control on the same geometry. The mesh is built once and reused; only
set_solid and the march differ, so the comparison isolates the cloud.

  core/tests/study/amr_cloud_economy.py [--depth 7] [--tol 1e-7]
        [--variants 2.2:0,1.8:0,1.5:0,2.2:32,2.2:24,2.2:16] [--max-steps 6000]

Accuracy numbers here are contention-immune (they are converged solutions, not timings), so the
table is trustworthy on a shared box; the CSR column is exact. Wall-clock belongs to M0.
"""
import importlib.util
import math
import os
import sys
import time

ROOT = __file__.rsplit("/tests/", 1)[0]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def opt(args, nm, d):
    return args[args.index(nm) + 1] if nm in args else d


def march(bed, tree, C, R, N, tol, max_steps, np_):
    """Darcy k on an already-built mesh, marched to stationarity (the study's own criterion)."""
    amr = bed.amr
    fl = amr.Flow(tree, 1.0, bed.MU, 60.0)
    fl.set_body_force(bed.FX, 0.0, 0.0)
    fl.set_advection(False)
    fl.set_ghost_sampled(True)
    fl.set_cf_scheme(1)
    t0 = time.time()
    fl.set_solid_spheres(C, np_.array([R]), True)
    w = np_.asarray(tree.sizes()) ** 3
    kprev, k, steps, confirm = None, None, 0, False
    while steps < max_steps:
        for _ in range(10):
            fl.step(100, 60)
        steps += 10
        u = np_.asarray(fl.velocity(0))
        k = bed.MU * float((u * w).sum()) / N ** 3 / bed.FX
        if not math.isfinite(k) or abs(k) > 1e12:
            return dict(k=float("nan"), steps=steps, secs=time.time() - t0, diverged=True)
        if kprev is not None and abs(k - kprev) < tol * abs(k):
            if confirm:
                break
            confirm = True
        else:
            confirm = False
        kprev = k
    return dict(k=k, steps=steps, secs=time.time() - t0, diverged=False)


def main():
    args = sys.argv[1:]
    depth = int(opt(args, "--depth", 7))
    tol = float(opt(args, "--tol", 1e-7))
    max_steps = int(opt(args, "--max-steps", 6000))
    variants = opt(args, "--variants", "2.2:0,1.8:0,1.5:0,2.2:32,2.2:24,2.2:16").split(",")
    N = 2 ** depth

    bed = _load("bed", ROOT + "/tests/study/amr_bed_graded.py")
    import numpy as np

    gapfn, _ = bed.make_gap_lookup(N)
    C, R = bed.load_pack(N)
    print(f"# M2a cloud economy: RCP bed depth {depth} (N={N}), tol {tol:g}", flush=True)

    # The uniform-finest-band control: no sample slots at all, so it is variant-independent.
    os.environ.pop("PECLET_CORE_GPS_RHO", None)
    os.environ.pop("PECLET_CORE_GPS_MAXN", None)
    tu = bed.build("u", N, gapfn)
    ru = march(bed, tu, C, R, N, tol, max_steps, np)
    print(f"# uniform control: k = {ru['k']:.6e}  ({tu.num_leaves} leaves, {ru['steps']} steps)",
          flush=True)
    del tu

    tg = bed.build("g", N, gapfn)
    print(f"# graded mesh: {tg.num_leaves} leaves\n", flush=True)
    print(f"{'rho':>5} {'N':>4} {'k':>14} {'offset vs u':>12} {'steps':>6} {'secs':>7}",
          flush=True)
    for v in variants:
        rho, mx = v.split(":")
        os.environ["PECLET_CORE_GPS_RHO"] = rho
        os.environ["PECLET_CORE_GPS_MAXN"] = mx
        r = march(bed, tg, C, R, N, tol, max_steps, np)
        off = (r["k"] - ru["k"]) / ru["k"] * 100.0
        tag = "DIVERGED" if r["diverged"] else f"{off:+.3f}%"
        print(f"{rho:>5} {mx:>4} {r['k']:>14.6e} {tag:>12} {r['steps']:>6} {r['secs']:>7.0f}",
              flush=True)


if __name__ == "__main__":
    main()
