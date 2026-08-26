"""C2 acceptance battery ON A SEAMED MESH (mixed-level cut band, plan §3 / Phase 2).

amr_zh_c2.py's criteria, re-run on the ladder's arm (b) — the two-level latitude map, whose
jump plane is oblique to the wall and therefore actually produces LS2 sample rows — at N=128,
i.e. at scale rather than the M3 prototype's N=64. The campaign's instabilities are all of
lagged-stiff-term shape, and the mixed-level cut band adds two lagged terms (the momentum
xi-row seam correction and the wall-aware C/F tangential fallback), so this is the gate that
says the seam machinery has not re-opened the attractor family.

Criteria (all measured against the SAME protocol on the uniform-band control):
  1. dt-independence: relative K spread over dt = 60 / 600 / 1e20 below 1e-5 (amr_zh_c2's gate).
  2. stationarity: the per-step velocity residual max|u_{n+1} - u_n| / max|u| at the end of the
     march, at dt = 60 and dt = 1e20 (the M3 statRes probe).
  3. dt-cycling reversibility: 60 -> 1e20 -> 60 in fixed-length legs reproduces the pure-60
     field. This is the campaign's attractor-family DISCRIMINATOR: a family shows up as a
     protocol-dependent steady state, i.e. a non-zero cycling residual.

Trap honoured throughout: changing dt requires a re-run of set_solid (the momentum operator is
scaled 1/dt at build time), so every leg rebuilds the operator on the SAME Flow object.

  core/tests/study/amr_zh_c2_seam.py [N] [--arms ab] [--cf 0|1] [--legs 300]
"""
import math
import sys
import time

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
import numpy as np  # noqa: E402
from peclet.core import amr  # noqa: E402

from amr_zh_ladder import K_ZH, PHI, build, geometry, make_sdf  # noqa: E402

ARM_NAME = {"a": "uniform control", "b": "two-level latitude (seamed)"}


def make_flow(t, N, dt, cf, sampled=True):
    mu, f = 0.1, 1e-3
    fl = amr.Flow(t, 1.0, mu, dt)
    fl.set_body_force(f, 0.0, 0.0)
    fl.set_advection(False)
    if sampled:
        fl.set_ghost_sampled(True)
    if cf:
        fl.set_cf_scheme(cf)
    fl.set_solid(make_sdf(N))
    return fl


def kappa(fl, t, N):
    R, _ = geometry(N)
    w = np.asarray(t.sizes()) ** 3
    u = np.asarray(fl.velocity(0))
    umean = float((u * w).sum()) / N ** 3
    return 1e-3 * N ** 3 / (6.0 * math.pi * 0.1 * R * umean), u


def march(fl, t, N, steps=None, tol=1e-9, max_steps=30000):
    """March to K-stationarity (or a fixed number of steps); return (K, statRes, steps)."""
    kprev, confirm, done = None, False, 0
    uprev = None
    stat = float("nan")
    while done < (steps if steps else max_steps):
        for _ in range(10):
            fl.step(100, 60)
        done += 10
        k, u = kappa(fl, t, N)
        if uprev is not None:
            um = float(np.max(np.abs(u)))
            stat = float(np.max(np.abs(u - uprev))) / (um if um > 0 else 1.0)
        uprev = u
        if steps is None:
            if kprev is not None and abs(k - kprev) < tol * abs(k):
                if confirm:
                    break
                confirm = True
            else:
                confirm = False
            kprev = k
    return k, stat, done


def run_arm(arm, N, cf, legs):
    t = build(arm, N)
    lev = np.bincount(np.asarray(t.levels())).tolist()
    print(f"\n### arm {arm} ({ARM_NAME[arm]}) N={N} cf={cf}: {t.num_leaves} leaves, "
          f"levels {lev}", flush=True)
    out = {}

    # (1) + (2): dt-independence and stationarity.
    for dt in (60.0, 600.0, 1e20):
        t0 = time.time()
        fl = make_flow(t, N, dt, cf)
        k, stat, ns = march(fl, t, N)
        out[dt] = k
        print(f"  dt={dt:<8g} K={k:.9f} err={100 * (k - K_ZH) / K_ZH:+.4f}% "
              f"statRes={stat:.3e} steps={ns} ({time.time() - t0:.0f}s)", flush=True)
    ks = list(out.values())
    spread = (max(ks) - min(ks)) / abs(ks[0])
    print(f"  dt-spread {spread:.3e} rel  ({'PASS' if spread < 1e-5 else 'FAIL'} at 1e-5)",
          flush=True)

    # (3): dt-cycling reversibility (the attractor-family discriminator). dt is baked into the
    # momentum operator, and the device set_solid reallocates + zeroes the fields, so a dt
    # switch is read -> set_dt -> set_solid -> write back (amr_seam_march.cpp's protocol, whose
    # oracle setSolid keeps the fields).
    # Each leg marches to STATIONARITY (not a fixed length): a family shows up as a different
    # steady state after the excursion, and only converged legs can say that. `legs` is the
    # extra hold applied after each leg's stationarity test, so the excursion is not a no-op.
    t0 = time.time()
    fl = make_flow(t, N, 60.0, cf)
    march(fl, t, N)
    k1, u1 = kappa(fl, t, N)
    for dt in (1e20, 60.0):
        state = [np.array(fl.velocity(c)) for c in range(3)] + [np.array(fl.pressure())]
        fl.set_dt(dt)
        fl.set_solid(make_sdf(N))
        for c in range(3):
            fl.set_velocity(c, state[c])
        fl.set_pressure(state[3])
        march(fl, t, N)
        march(fl, t, N, steps=legs)
    k3, u3 = kappa(fl, t, N)
    rel = float(np.max(np.abs(u3 - u1))) / max(float(np.max(np.abs(u1))), 1e-300)
    print(f"  dt-cycling 60 -> 1e20 -> 60 (each leg to stationarity + {legs} hold): "
          f"K1={k1:.9f} K3={k3:.9f}  Krel={abs(k3 - k1) / abs(k1):.3e}  field rel {rel:.3e} "
          f"({time.time() - t0:.0f}s)", flush=True)
    out["spread"] = spread
    out["cycle"] = rel
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    arms = args[args.index("--arms") + 1] if "--arms" in args else "ba"
    cf = int(args[args.index("--cf") + 1]) if "--cf" in args else 1
    legs = int(args[args.index("--legs") + 1]) if "--legs" in args else 300
    skip = {"--arms", "--cf", "--legs"}
    vals, i = [], 0
    while i < len(args):
        if args[i] in skip:
            i += 2
            continue
        if not args[i].startswith("--"):
            vals.append(int(args[i]))
        i += 1
    N = vals[0] if vals else 128

    print(f"C2 battery on a SEAMED mesh: Z&H SC phi={PHI}, N={N}, cf={cf}, arms={arms}, "
          f"sampled ghost (2,2)", flush=True)
    res = {a: run_arm(a, N, cf, legs) for a in arms}
    print("\nC2 VERDICT (seamed vs control):", flush=True)
    for a in arms:
        print(f"  arm {a} ({ARM_NAME[a]}): dt-spread {res[a]['spread']:.3e} "
              f"({'PASS' if res[a]['spread'] < 1e-5 else 'FAIL'})", flush=True)
