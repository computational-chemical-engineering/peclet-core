"""Phase-3 two-sphere gap unit case: calibrating the throat criterion n (plan §7 crit. 1, §Phase 3).

The gap floor says a cut cell at level L is permitted only where the local fluid gap satisfies
`gap >= n*h_L`. n has never been measured — the plan starts it at 4 on a guess, and the Phase-2
Z&H ladder could not calibrate it because a dilute sphere array has no throats (its gap contrast
is 1.44x, less than one level, so the floor is inert there). This is the smallest geometry where
n CAN be calibrated.

Geometry: a periodic chain of two equal spheres along z, sized so BOTH z-gaps are the same
throat width g (R = (N - 2g)/4), with the Stokes flow driven along x. The throat is then a
channel carrying flux, the open x/y directions give a gap ~N-2R away from it, so the gap field
has several levels of contrast and the floor genuinely grades.

Instrument: Darcy permeability k = mu*<u_x>/f (volume-weighted superficial mean) against the
UNIFORM FINEST BAND on the same mesh family and the same N. That difference isolates the
COARSENING-POLICY error — exactly the instrument Phase 2 validated — rather than the
discretization error the control also carries. Sweeping n from aggressive to conservative, the
calibration is the smallest n whose policy error is inside the ghost scheme's own ~0.2-0.3%
bias, and the cell count it buys.

Reported alongside, because the achieved resolution is the physically meaningful axis: the level
the floor actually selected in the throat and the resulting cells-across-gap (which the floor
constrains to [n, 2n) by construction), plus the overlay census — in particular whether a
genuinely throat-graded mesh generates the closed mixed faces that would make the unimplemented
sub-face closures blocking rather than merely untidy.

  core/tests/study/amr_two_sphere_gap.py [N] [--gaps 8,16] [--ns 1,2,3,4,6,8] [--cf 1]
"""
import math
import sys
import time

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
import numpy as np  # noqa: E402
from peclet.core import amr  # noqa: E402

LFAR = 3      # background level above finest (as in the Phase-2 ladder)
BAND = 4.0    # band margin in cells of the level being created
MU, FX = 0.1, 1e-3

IMG = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]


def geometry(N, g):
    """Two spheres on the z axis, both periodic z-gaps equal to g."""
    R = (N - 2.0 * g) / 4.0
    cz = [N / 4.0, 3.0 * N / 4.0]          # separation N/2 => gap = N/2 - 2R = g
    return R, [(N / 2.0, N / 2.0, z) for z in cz]


def make_sdf(N, g):
    R, cs = geometry(N, g)

    def sdf(x, y, z):
        m = 1e30
        for cx, cy, cz in cs:
            for i, j, k in IMG:
                d = math.sqrt((x - cx - i * N) ** 2 + (y - cy - j * N) ** 2
                              + (z - cz - k * N) ** 2) - R
                if d < m:
                    m = d
        return m

    return sdf


def make_gap(N, g):
    """Plan §7 criterion 1: two-closest-surfaces proxy d1 + d2, over both spheres' images."""
    R, cs = geometry(N, g)

    def gap(x, y, z):
        d0 = d1 = 1e30
        for cx, cy, cz in cs:
            for i, j, k in IMG:
                d = math.sqrt((x - cx - i * N) ** 2 + (y - cy - j * N) ** 2
                              + (z - cz - k * N) ** 2) - R
                if d < d0:
                    d1, d0 = d0, d
                elif d < d1:
                    d1 = d
        return d0 + d1

    return gap


def build(N, g, n):
    """n is None => the uniform finest band control; else the gap floor with that n."""
    lmax = int(math.log2(N))
    t = amr.Octree([1, 1, 1], lmax, [0.0, 0.0, 0.0], 1.0)
    t.refine_to_sdf(lambda x, y, z: 0.0, LFAR, 1e30, False)   # uniform background at LFAR
    sdf = make_sdf(N, g)
    if n is None:
        t.refine_to_sdf_graded(sdf, lambda x, y, z: 0, BAND, True)
    else:
        t.refine_to_gap_floor(sdf, make_gap(N, g), LFAR, float(n), BAND, True)
    return t


def throat_level(t, N):
    """Level of the leaf at the throat centre (z = N/2, midway between the two spheres)."""
    i = t.find([N / 2.0, N / 2.0, N / 2.0])
    return int(np.asarray(t.levels())[i]) if i >= 0 else -1


def permeability(N, g, n, cf, tol=1e-7, max_steps=20000, dt=60.0):
    t = build(N, g, n)
    lev = np.asarray(t.levels())
    fl = amr.Flow(t, 1.0, MU, dt)
    fl.set_body_force(FX, 0.0, 0.0)
    fl.set_advection(False)
    fl.set_ghost_sampled(True)
    if cf:
        fl.set_cf_scheme(cf)
    fl.set_solid(make_sdf(N, g))
    w = np.asarray(t.sizes()) ** 3
    kprev, k, steps, confirm = None, None, 0, False
    presMax = 0
    diverged = False
    t0 = time.time()
    while steps < max_steps:
        for _ in range(10):
            fl.step(100, 60)
        steps += 10
        presMax = max(presMax, fl.last_pres_iters())
        u = np.asarray(fl.velocity(0))
        umean = float((u * w).sum()) / N ** 3
        k = MU * umean / FX
        # A diverging march must bail immediately: NaN never satisfies the stationarity test, so
        # without this the run grinds out max_steps on garbage (measured: ~85 min a run).
        if not math.isfinite(k) or abs(k) > 1e12:
            diverged = True
            print(f"      [g={g} n={n}] DIVERGED by step {steps}: k={k:.3e}", flush=True)
            break
        if steps % 1000 == 0:
            print(f"      [g={g} n={n}] step {steps} k={k:.6e} "
                  f"dk={abs(k - kprev) / abs(k):.2e} pres={fl.last_pres_iters()} "
                  f"({(time.time() - t0) / steps * 1e3:.1f} ms/step)", flush=True)
        if kprev is not None and abs(k - kprev) < tol * abs(k):
            if confirm:
                break
            confirm = True
        else:
            confirm = False
        kprev = k
    L = throat_level(t, N)
    return dict(k=k, leaves=t.num_leaves, steps=steps, secs=time.time() - t0,
                throatLevel=L, gapCells=(g / (1 << L) if L >= 0 else float("nan")),
                pres=fl.last_pres_iters(), presMax=presMax, capped=(steps >= max_steps),
                diverged=diverged,
                lev=np.bincount(lev, minlength=LFAR + 1).tolist())


if __name__ == "__main__":
    args = sys.argv[1:]

    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default

    cf = int(opt("--cf", "1"))
    tol = float(opt("--tol", "1e-7"))
    gaps = [int(v) for v in opt("--gaps", "8,16").split(",")]
    ns = [float(v) for v in opt("--ns", "1,2,3,4,6,8").split(",")]
    skip = {"--cf", "--gaps", "--ns", "--tol"}
    vals, i = [], 0
    while i < len(args):
        if args[i] in skip:
            i += 2
            continue
        if not args[i].startswith("--"):
            vals.append(int(args[i]))
        i += 1
    N = vals[0] if vals else 128

    print(f"two-sphere gap unit case: N={N} cf={cf} gaps={gaps} n-sweep={ns} tol={tol:g} "
          f"LFAR={LFAR} band={BAND} sampled ghost (2,2)", flush=True)
    for g in gaps:
        R, cs = geometry(N, g)
        print(f"\n=== throat g = {g} h0  (R = {R:.1f} h0, two spheres on z, both gaps = g) ===",
              flush=True)
        ctl = permeability(N, g, None, cf, tol=tol)
        print(f"{'n':>5} {'k':>13} {'d(ctl)%':>9} {'Lthroat':>8} {'gap/h_L':>8} "
              f"{'leaves':>9} {'vs ctl':>7} {'steps':>7} {'pres':>5} {'s':>6}", flush=True)
        print(f"{'ctl':>5} {ctl['k']:>13.6e} {0.0:>9.4f} {ctl['throatLevel']:>8} "
              f"{ctl['gapCells']:>8.1f} {ctl['leaves']:>9} {1.0:>6.2f}x {ctl['steps']:>7} "
              f"{ctl['presMax']:>5} {ctl['secs']:>6.0f}"
              + ("  *** DIVERGED ***" if ctl['diverged']
                 else "  CAPPED" if ctl['capped'] else ""), flush=True)
        for n in ns:
            # The aggressive end can pinch the throat outright (that is the failure the floor
            # exists to prevent, so it is a RESULT, not a bug) — keep the sweep alive through it.
            try:
                r = permeability(N, g, n, cf, tol=tol)
            except Exception as e:  # noqa: BLE001
                print(f"{n:>5g} {'FAILED':>13}  {type(e).__name__}: {e}", flush=True)
                continue
            d = 100.0 * (r["k"] - ctl["k"]) / ctl["k"]
            print(f"{n:>5g} {r['k']:>13.6e} {d:>+9.4f} {r['throatLevel']:>8} "
                  f"{r['gapCells']:>8.1f} {r['leaves']:>9} "
                  f"{ctl['leaves'] / r['leaves']:>6.2f}x {r['steps']:>7} {r['presMax']:>5} "
                  f"{r['secs']:>6.0f}"
                  + ("  *** DIVERGED ***" if r['diverged']
                     else "  CAPPED (K NOT converged)" if r['capped'] else ""),
                  flush=True)
            print(f"      levels {r['lev']}", flush=True)
