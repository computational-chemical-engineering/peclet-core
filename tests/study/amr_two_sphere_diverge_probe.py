"""Attribution probe for the two-sphere gap divergence (amr_two_sphere_gap.py, g=8, n=3/4).

Those two graded meshes blow up (k ~ 1e12 by step 100-170) while their neighbours at 1, 2 and
8 cells across the throat are all stable. This bisects the blow-up against the pieces that
could carry it, WITHOUT changing any numerics — every knob here is an existing public option:

  cf   0 vs 1   : the wall-aware C/F tangential fallback is a LAGGED deferred-correction term
                  added in Phase-1 rung 2 and only active at cf=1. Stable at cf=0 => that term.
  dt   60 vs 1e20 : a lagged-stiff-term instability is dt-dependent; a structurally singular
                  operator is not.
  mom  MG on/off : separates the momentum solve from the pressure/constraint side.

Prints the step at which each configuration diverges (or that it survives the probe horizon).

  core/tests/study/amr_two_sphere_diverge_probe.py [--g 8] [--ns 3,4] [--steps 400]
"""
import math
import sys

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
from amr_two_sphere_gap import FX, MU, build, make_sdf  # noqa: E402
from peclet.core import amr  # noqa: E402


def probe(N, g, n, cf, dt, mom_mg, steps):
    t = build(N, g, n)
    fl = amr.Flow(t, 1.0, MU, dt)
    fl.set_body_force(FX, 0.0, 0.0)
    fl.set_advection(False)
    fl.set_ghost_sampled(True)
    if cf:
        fl.set_cf_scheme(cf)
    fl.set_momentum_mg(mom_mg)
    fl.set_solid(make_sdf(N, g))
    w = np.asarray(t.sizes()) ** 3
    done, kmax = 0, 0.0
    while done < steps:
        fl.step(100, 60)
        done += 1
        u = np.asarray(fl.velocity(0))
        k = MU * float((u * w).sum()) / N ** 3 / FX
        kmax = max(kmax, abs(k))
        if not math.isfinite(k) or abs(k) > 1e12:
            return done, k
    return None, k


if __name__ == "__main__":
    args = sys.argv[1:]

    def opt(name, d):
        return args[args.index(name) + 1] if name in args else d

    N = 128
    g = int(opt("--g", "8"))
    ns = [float(v) for v in opt("--ns", "3,4").split(",")]
    steps = int(opt("--steps", "400"))
    print(f"divergence attribution probe: N={N} g={g} ns={ns} horizon={steps} steps", flush=True)
    print(f"{'n':>4} {'cf':>3} {'dt':>7} {'momMG':>6} {'outcome':>26}", flush=True)
    for n in ns:
        for cf in (1, 0):
            for dt in (60.0, 1e20):
                for mom in (True, False):
                    try:
                        at, k = probe(N, g, n, cf, dt, mom, steps)
                    except Exception as e:  # noqa: BLE001
                        print(f"{n:>4g} {cf:>3} {dt:>7g} {str(mom):>6} "
                              f"{'THREW: ' + type(e).__name__:>26}", flush=True)
                        continue
                    out = (f"DIVERGED @ step {at}" if at is not None
                           else f"survived, k={k:.4e}")
                    print(f"{n:>4g} {cf:>3} {dt:>7g} {str(mom):>6} {out:>26}", flush=True)
