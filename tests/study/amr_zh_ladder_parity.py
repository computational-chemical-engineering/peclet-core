"""Parity of the ladder's control arm: sampled overlay vs classic overlay, at ladder scale.

Arm (a) — the uniform finest band — builds a sampled overlay whose slots are ALL IDENTITY
(no 2:1 boundary is ever crossed by a closure chain), so the sampled path must reproduce the
classic ghost projection BIT-FOR-BIT. `test_seam_sampled` pins this on a small mesh; this
re-checks it on the mesh the Phase-2 ladder actually reports, so the ladder's control arm is
provably the same number the classic path would have produced.

  core/tests/study/amr_zh_ladder_parity.py [N] [--cf 1]
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
sys.path.insert(0, __file__.rsplit("/", 1)[0])

from amr_zh_ladder import build, drag, geometry, make_sdf  # noqa: E402
from peclet.core import amr  # noqa: E402


def fields(N, cf, sampled, steps=200):
    t = build("a", N)
    fl = amr.Flow(t, 1.0, 0.1, 60.0)
    fl.set_body_force(1e-3, 0.0, 0.0)
    fl.set_advection(False)
    if sampled:
        fl.set_ghost_sampled(True)
    if cf:
        fl.set_cf_scheme(cf)
    fl.set_solid(make_sdf(N))
    for _ in range(steps // 10):
        fl.step(100, 60)
    return [np.array(fl.velocity(c)) for c in range(3)] + [np.array(fl.pressure())]


if __name__ == "__main__":
    args = sys.argv[1:]
    cf = int(args[args.index("--cf") + 1]) if "--cf" in args else 1
    N = next((int(a) for a in args if not a.startswith("--") and a.isdigit()), 128)
    print(f"ladder control-arm parity: N={N} cf={cf}, uniform finest band, 200 steps", flush=True)
    s = fields(N, cf, True)
    c = fields(N, cf, False)
    names = ["u", "v", "w", "p"]
    worst = 0.0
    for n, a, b in zip(names, s, c):
        d = float(np.max(np.abs(a - b)))
        worst = max(worst, d)
        print(f"  {n}: max|sampled - classic| = {d:.3e}  (max|classic| = "
              f"{float(np.max(np.abs(b))):.3e})", flush=True)
    print(f"VERDICT: {'BIT-IDENTICAL' if worst == 0.0 else f'DIFFERS ({worst:.3e})'}", flush=True)
