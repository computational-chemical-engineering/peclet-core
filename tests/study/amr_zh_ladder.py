"""Phase-2 graded accuracy ladder for the mixed-level cut band (plan §8, Phase 2).

THE instrument that settles the seam accuracy question. Single-N K offsets against the
uniform control were measured to be weak (plan "Phase 1 progress"); convergence WITH
RESOLUTION is the honest one:

  Z&H simple-cubic sphere (phi = 0.125, K_ZH = 4.292), N = 64/128/256 finest, three arms
  that differ ONLY in the surface LEVEL MAP -- background, band margin and far field are
  identical by construction, so K(arm) - K(control) isolates the seam machinery:

    (a) uniform  -- finest band over the whole surface (today's policy; the control)
    (b) two-level -- latitude jump at z = c - R/2 (oblique to the wall: the M2 seam
                     geometry that actually produces sample rows), cap above at L1
    (c) gap-ordered three-level -- L0/L1/L2 ordered by the plan's §7 gap proxy
                     (two-closest-surfaces d1+d2 over the 27 nearest periodic images):
                     finest at the narrow pole directions, coarsest at the wide corner
                     directions. Thresholds split the geometry's OWN gap range in three,
                     because the absolute gap floor (n = 4 h_L) is INERT on a dilute SC
                     array -- measured by --gapfloor below: it coarsens the entire surface
                     to a fixed physical size, decoupled from N (a throat criterion needs
                     throats; that is Phase 3's two-sphere case). Arm (c)'s job here is to
                     stress MANY seams + a two-level jump, not to show the porous payoff.

Mesh family (self-similar in N, so every arm converges to the exact answer, not to a
graded plateau): background uniform at level LFAR above finest; surface refined by
`refine_to_sdf_graded` with the band margin measured in cells of the level being created.

Both C/F schemes are run: with the standard two-point flux the far-field C/F error
dominates the seam signal, so the quadratic arm (set_cf_scheme(1)) carries the verdict.

  core/tests/study/amr_zh_ladder.py [--cf 0|1|both] [--arms abc] [--tol 1e-8] N [N ...]
  core/tests/study/amr_zh_ladder.py --gapfloor 64 128 256    # policy census only, no solve
"""
import math
import sys
import time

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
import numpy as np  # noqa: E402
from peclet.core import amr  # noqa: E402

K_ZH = 4.292
PHI = 0.125
LFAR = 3      # background level above finest (physical background size = 8*h0 = N/8 cells)
BAND = 4.0    # band margin in cells of the level being created (>= 2 for the +-2 reach)

IMG = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]


def geometry(N):
    R = (PHI * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * N
    c = N / 2.0
    return R, c


def make_sdf(N):
    R, c = geometry(N)
    return lambda x, y, z: math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) - R


def make_gap(N):
    """Plan §7 criterion 1: two-closest-surfaces proxy d1 + d2 over the periodic images."""
    R, c = geometry(N)

    def gap(x, y, z):
        d0 = d1 = 1e30
        for i, j, k in IMG:
            d = math.sqrt((x - c - i * N) ** 2 + (y - c - j * N) ** 2 + (z - c - k * N) ** 2) - R
            if d < d0:
                d1, d0 = d0, d
            elif d < d1:
                d1 = d
        return d0 + d1

    return gap


def gap_range(N):
    """Analytic gap extremes on the sphere surface: pole (facing an image) vs corner direction."""
    R, _ = geometry(N)
    gmin = N - 2.0 * R                                       # +-x/y/z pole directions
    s = R / math.sqrt(3.0)
    gmax = math.sqrt((s - N) ** 2 + 2.0 * s * s) - R         # (1,1,1) corner direction
    return gmin, gmax


def target_fn(arm, N):
    R, c = geometry(N)
    if arm == "a":
        return lambda x, y, z: 0
    if arm == "b":
        zc = c - 0.5 * R
        return lambda x, y, z: 0 if z < zc else 1
    if arm == "c":
        gap = make_gap(N)
        gmin, gmax = gap_range(N)
        w = (gmax - gmin) / 3.0

        def tgt(x, y, z):
            b = int((gap(x, y, z) - gmin) / w)
            return 0 if b < 0 else (2 if b > 2 else b)

        return tgt
    raise ValueError(arm)


def build(arm, N):
    lmax = int(math.log2(N))
    t = amr.Octree([1, 1, 1], lmax, [0.0, 0.0, 0.0], 1.0)
    t.refine_to_sdf(lambda x, y, z: 0.0, LFAR, 1e30, False)   # uniform background at LFAR
    t.refine_to_sdf_graded(make_sdf(N), target_fn(arm, N), BAND, True)
    return t


def drag(arm, N, cf, sampled=True, tol=1e-8, max_steps=40000, dt=60.0, quiet=False):
    t = build(arm, N)
    R, c = geometry(N)
    mu, f = 0.1, 1e-3
    fl = amr.Flow(t, 1.0, mu, dt)
    fl.set_body_force(f, 0.0, 0.0)
    fl.set_advection(False)
    if sampled:
        fl.set_ghost_sampled(True)
    if cf:
        fl.set_cf_scheme(cf)
    fl.set_solid(make_sdf(N))
    w = np.asarray(t.sizes()) ** 3          # cell volumes in grid units
    lev = np.asarray(t.levels())
    kprev, k, steps, confirm = None, None, 0, False
    t0 = time.time()
    while steps < max_steps:
        for _ in range(10):
            fl.step(100, 60)
        steps += 10
        u = np.asarray(fl.velocity(0))
        umean = float((u * w).sum()) / N ** 3
        k = f * N ** 3 / (6.0 * math.pi * mu * R * umean)
        if not quiet and steps % 1000 == 0:
            print(f"    [{arm} N={N} cf={cf}] step {steps} K={k:.7f} "
                  f"({(time.time() - t0) / steps * 1e3:.1f} ms/step)", flush=True)
        if kprev is not None and abs(k - kprev) < tol * abs(k):
            if confirm:
                break
            confirm = True
        else:
            confirm = False
        kprev = k
    return dict(K=k, leaves=t.num_leaves, steps=steps, pres=fl.last_pres_iters(),
                secs=time.time() - t0, lev=np.bincount(lev, minlength=LFAR + 1).tolist())


def gapfloor_census(Ns):
    """What the ABSOLUTE gap floor (gap >= n*h_L, n = 4) actually does on a dilute SC array."""
    print("gap-floor policy census (plan §7 criterion 1, n=4, coarsest = lmax):", flush=True)
    print(f"{'N':>5} {'gap/h0 range':>16} {'leaves':>9} {'level histogram':>28}", flush=True)
    for N in Ns:
        lmax = int(math.log2(N))
        t = amr.Octree([1, 1, 1], lmax, [0.0, 0.0, 0.0], 1.0)
        t.refine_to_sdf(lambda x, y, z: 0.0, LFAR, 1e30, False)
        t.refine_to_gap_floor(make_sdf(N), make_gap(N), lmax, 4.0, BAND, True)
        gmin, gmax = gap_range(N)
        h = np.bincount(np.asarray(t.levels()), minlength=lmax + 1).tolist()
        print(f"{N:>5} {gmin:>7.1f}..{gmax:<8.1f} {t.num_leaves:>9} {str(h):>28}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--gapfloor" in args:
        Ns = [int(a) for a in args if not a.startswith("--")] or [64, 128, 256]
        gapfloor_census(Ns)
        sys.exit(0)
    cfarg = args[args.index("--cf") + 1] if "--cf" in args else "both"
    cfs = [0, 1] if cfarg == "both" else [int(cfarg)]
    arms = args[args.index("--arms") + 1] if "--arms" in args else "abc"
    tol = float(args[args.index("--tol") + 1]) if "--tol" in args else 1e-8
    skip = {"--cf", "--arms", "--tol"}
    vals, i = [], 0
    while i < len(args):
        if args[i] in skip:
            i += 2
            continue
        if not args[i].startswith("--"):
            vals.append(int(args[i]))
        i += 1
    Ns = vals or [64, 128]

    print(f"Z&H SC phi={PHI} ladder: N={Ns} arms={arms} cf={cfs} tol={tol:g} "
          f"LFAR={LFAR} band={BAND} sampled=True (2,2)", flush=True)
    res = {}
    for cf in cfs:
        print(f"\n=== cf_scheme = {cf} "
              f"({'standard two-point' if cf == 0 else 'Martin-Cartwright quadratic'}) ===",
              flush=True)
        print(f"{'arm':>4} {'N':>5} {'K':>12} {'err%':>9} {'d(a)%':>9} {'leaves':>9} "
              f"{'cells%':>7} {'steps':>7} {'pres':>5} {'s':>7}", flush=True)
        for N in Ns:
            for arm in arms:
                r = drag(arm, N, cf, tol=tol)
                res[(cf, N, arm)] = r
                base = res.get((cf, N, "a"))
                d = 100.0 * (r["K"] - base["K"]) / base["K"] if base else 0.0
                print(f"{arm:>4} {N:>5} {r['K']:>12.7f} "
                      f"{100 * (r['K'] - K_ZH) / K_ZH:>+9.4f} {d:>+9.4f} {r['leaves']:>9} "
                      f"{100 * r['leaves'] / N ** 3:>7.2f} {r['steps']:>7} {r['pres']:>5} "
                      f"{r['secs']:>7.0f}", flush=True)
            print(f"     levels " + "  ".join(
                f"{arm}:{res[(cf, N, arm)]['lev']}" for arm in arms), flush=True)

    # Convergence + seam-offset orders.
    print("\n--- orders (log2 ratio of successive N) ---", flush=True)
    for cf in cfs:
        for arm in arms:
            ks = [res[(cf, N, arm)]["K"] for N in Ns if (cf, N, arm) in res]
            e = [abs(k - K_ZH) for k in ks]
            o = [math.log2(e[i] / e[i + 1]) for i in range(len(e) - 1) if e[i + 1] > 0]
            print(f"cf={cf} arm {arm}: |K-K_ZH| = "
                  + " ".join(f"{v:.5f}" for v in e)
                  + ("   order " + " ".join(f"{v:.2f}" for v in o) if o else ""), flush=True)
        for arm in arms:
            if arm == "a":
                continue
            d = [abs(res[(cf, N, arm)]["K"] - res[(cf, N, "a")]["K"])
                 for N in Ns if (cf, N, arm) in res]
            o = [math.log2(d[i] / d[i + 1]) for i in range(len(d) - 1) if d[i + 1] > 0]
            print(f"cf={cf} SEAM OFFSET |K({arm}) - K(a)| = "
                  + " ".join(f"{v:.3e}" for v in d)
                  + ("   order " + " ".join(f"{v:.2f}" for v in o) if o else ""), flush=True)
