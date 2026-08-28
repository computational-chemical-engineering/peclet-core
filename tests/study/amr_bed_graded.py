"""Phase-3 sphere-bed run: the headline accuracy-matched cell-count reduction.

The measurement the mixed-level cut band exists to produce. A real RCP packing
(`tests/data/rcp_pack_seed3_unit.txt` — 180 monodisperse spheres, phi = 0.63, the M1 census
bed), Stokes flow along x, Darcy permeability and CELL COUNT for a gap-graded mesh against the
uniform-finest-band mesh at the same finest resolution.

Prior measurements this builds on:
  M1 (census, mesh only): map G costs 66.2% of the uniform band's cells at R/h0 = 24.1 and
    34.5% at R/h0 = 48.2 — the saving GROWS with resolution, so quote it at the highest
    resolution that fits.
  P3a (two-sphere): the gap floor's n = 4 is calibrated — 4-8 cells across the throat for
    0.1-1.3% policy error.
  P3b: cfDiv/cfGrad are gated to regular rows; without that gate throat-graded meshes diverge.

`--probe` builds the meshes only (CPU, no solver) and reports leaf counts, level histograms and
build times — run it FIRST at a new depth to size the solver run.

  core/tests/study/amr_bed_graded.py --probe [--depth 8] [--arms ug]
  core/tests/study/amr_bed_graded.py [--depth 8] [--arms ug] [--tol 1e-7]
"""
import math
import sys
import time

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
import numpy as np  # noqa: E402
from peclet.core import amr  # noqa: E402

PACK = __file__.rsplit("/tests/", 1)[0] + "/tests/data/rcp_pack_seed3_unit.txt"
LFAR = 3      # background level above finest
BAND = 4.0    # band margin in cells of the level being created
NGAP = 4.0    # the P3a-calibrated gap floor
MU, FX = 0.1, 1e-3


def load_pack(N):
    """Sphere centres/radius in CELL units (box [0,N)^3)."""
    c, r = [], None
    with open(PACK) as f:
        for line in f:
            if line.startswith("#"):
                continue
            v = [float(x) for x in line.split()]
            if len(v) == 4:
                c.append(v[:3])
                r = v[3]
    return np.asarray(c) * N, r * N


def make_sdf(N):
    """Union SDF, minimum-image periodic. Vectorised over the 180 spheres (one numpy pass)."""
    C, R = load_pack(N)

    def sdf(x, y, z):
        d = np.array([x, y, z]) - C            # (180, 3)
        d -= N * np.round(d / N)               # min-image
        return float(np.sqrt((d * d).sum(1)).min()) - R

    return sdf


def make_gap_lookup(N, gres=128, gapfile=None):
    """Gap proxy d1+d2 (plan §7 crit. 1) precomputed on a uniform gres^3 grid, nearest lookup.

    The gap field varies on the pore scale, so sampling it coarsely is ample for a refinement
    POLICY (it is not a discretization). Equal radii ⇒ the two closest surfaces are the two
    closest centres, so one vectorised distance pass per grid slab suffices.
    """
    # A precomputed UNIT-box table (gap_unit_<gres>.npy) is resolution-independent — gap scales
    # linearly with N — so one file serves every depth and the run needs no scipy. Used on
    # clusters where the venv is minimal, and it keeps the KD-tree build off billed GPU time.
    if gapfile:
        g = np.load(gapfile).astype(np.float32) * float(N)
        gres = g.shape[0]
        inv = gres / N

        def lookup_pre(x, y, z):
            return float(g[int(x * inv) % gres, int(y * inv) % gres, int(z * inv) % gres])

        return lookup_pre, g

    from scipy.spatial import cKDTree
    C, R = load_pack(N)
    ax = (np.arange(gres) + 0.5) * (N / gres)
    Cim = np.concatenate([C + np.array([i, j, k]) * N
                          for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], 0)
    tree = cKDTree(Cim)
    P = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    d, _ = tree.query(P, k=2, workers=-1)      # two nearest centres = two nearest surfaces
    gap = ((d[:, 0] - R) + (d[:, 1] - R)).astype(np.float32).reshape(gres, gres, gres)
    inv = gres / N

    def lookup(x, y, z):
        i = int(x * inv) % gres
        j = int(y * inv) % gres
        k = int(z * inv) % gres
        return float(gap[i, j, k])

    return lookup, gap


def build(arm, N, gapfn=None):
    lmax = int(math.log2(N))
    t = amr.Octree([1, 1, 1], lmax, [0.0, 0.0, 0.0], 1.0)
    t.refine_to_sdf(lambda x, y, z: 0.0, LFAR, 1e30, False)   # uniform background at LFAR
    sdf = make_sdf(N)
    if arm == "u":
        t.refine_to_sdf_graded(sdf, lambda x, y, z: 0, BAND, True)
    else:
        t.refine_to_gap_floor(sdf, gapfn, LFAR, NGAP, BAND, True)
    return t


def permeability(arm, N, gapfn, cf=1, tol=1e-7, max_steps=6000, dt=60.0):
    t = build(arm, N, gapfn)
    lev = np.asarray(t.levels())
    fl = amr.Flow(t, 1.0, MU, dt)
    fl.set_body_force(FX, 0.0, 0.0)
    fl.set_advection(False)
    fl.set_ghost_sampled(True)
    if cf:
        fl.set_cf_scheme(cf)
    C, R = load_pack(N)
    fl.set_solid_spheres(C, np.array([R]), True)   # native union-of-spheres SDF: set_solid with
    # a Python callback is sampled tens of times per leaf and dominates everything at bed scale
    # (measured 311s vs 33s at 262k leaves, bit-identical fields).
    w = np.asarray(t.sizes()) ** 3
    kprev, k, steps, confirm, diverged = None, None, 0, False, False
    t0 = time.time()
    while steps < max_steps:
        for _ in range(10):
            fl.step(100, 60)
        steps += 10
        u = np.asarray(fl.velocity(0))
        k = MU * float((u * w).sum()) / N ** 3 / FX
        if not math.isfinite(k) or abs(k) > 1e12:
            diverged = True
            print(f"    [{arm}] DIVERGED by step {steps}: k={k:.3e}", flush=True)
            break
        if steps % 500 == 0:
            print(f"    [{arm} N={N}] step {steps} k={k:.6e} "
                  f"dk={abs(k - kprev) / abs(k):.2e} pres={fl.last_pres_iters()} "
                  f"({(time.time() - t0) / steps * 1e3:.0f} ms/step)", flush=True)
        if kprev is not None and abs(k - kprev) < tol * abs(k):
            if confirm:
                break
            confirm = True
        else:
            confirm = False
        kprev = k
    return dict(k=k, leaves=t.num_leaves, steps=steps, secs=time.time() - t0,
                diverged=diverged, lev=np.bincount(lev, minlength=LFAR + 1).tolist())


if __name__ == "__main__":
    args = sys.argv[1:]

    def opt(nm, d):
        return args[args.index(nm) + 1] if nm in args else d

    depth = int(opt("--depth", "8"))
    arms = opt("--arms", "ug")
    tol = float(opt("--tol", "1e-7"))
    maxst = int(opt("--max-steps", "6000"))
    N = 1 << depth
    C, R = load_pack(N)
    phi = len(C) * (4.0 / 3.0) * math.pi * R ** 3 / N ** 3
    print(f"sphere bed: {len(C)} spheres, R/h0 = {R:.1f}, phi = {100 * phi:.1f}%, "
          f"N = {N} (depth {depth}), LFAR={LFAR} band={BAND} n_gap={NGAP} tol={tol:g}",
          flush=True)

    t0 = time.time()
    gapfn, gg = make_gap_lookup(N, gapfile=opt("--gap-file", None))
    print(f"gap field precomputed ({time.time() - t0:.0f}s): "
          f"min {gg.min():.1f} h0, median {np.median(gg):.1f} h0, max {gg.max():.1f} h0",
          flush=True)

    if "--probe" in args:
        for arm in arms:
            t0 = time.time()
            t = build(arm, N, gapfn)
            h = np.bincount(np.asarray(t.levels()), minlength=LFAR + 1).tolist()
            nm = "uniform finest band" if arm == "u" else f"gap-graded (n={NGAP:g})"
            print(f"  {nm:>24}: {t.num_leaves:>10} leaves ({100 * t.num_leaves / N ** 3:.1f}% "
                  f"of N^3)  levels {h}  built in {time.time() - t0:.0f}s", flush=True)
        sys.exit(0)

    res = {}
    for arm in arms:
        r = permeability(arm, N, gapfn, tol=tol, max_steps=maxst)
        res[arm] = r
        nm = "uniform" if arm == "u" else "graded"
        print(f"{nm:>8}: k={r['k']:.6e} leaves={r['leaves']} steps={r['steps']} "
              f"({r['secs']:.0f}s) levels {r['lev']}"
              + ("  *** DIVERGED ***" if r["diverged"] else ""), flush=True)
    if "u" in res and "g" in res:
        d = 100.0 * (res["g"]["k"] - res["u"]["k"]) / res["u"]["k"]
        print(f"\nHEADLINE: graded k differs by {d:+.3f}% at "
              f"{res['u']['leaves'] / res['g']['leaves']:.2f}x fewer cells "
              f"({res['g']['leaves']} vs {res['u']['leaves']})", flush=True)
