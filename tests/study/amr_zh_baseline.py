"""Uniform-grid Zick & Homsy sweep for the device AmrFlow (collocated AMR solver).

Baseline (docs/amr_collocated_projection.md, RTX 5080): phi=0.125, K_ZH=4.292,
N=32 +0.30%, N=64 +0.13%, N=128 +0.05% (order ~1.3).  --ghost enables the directional
ghost gradient (set_ghost_gradient — the AMR mode-9 hybrid) for the A/B.

Protocol: N^3 unrefined octree (bit-identical to the structured grid), SC sphere,
Stokes, incremental-rotational projection (dt-independent); iterate until |dK/K| < tol
over a 10-step stride, then confirm on one further stride (false-plateau guard).

Run FOREGROUND on the CUDA build:
  python core/tests/study/amr_zh_baseline.py [--ghost] N [N ...]
"""
import math
import sys
import time

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
from peclet.core import amr  # noqa: E402

K_ZH = 4.292
PHI = 0.125


def drag_k(N, ghost, ghostproj=False, tol=1e-7, max_steps=6000, mom_iters=100, pres_iters=60):
    lmax = int(math.log2(N))
    assert 2 ** lmax == N
    t0 = time.time()
    t = amr.Octree([1, 1, 1], lmax, [0.0, 0.0, 0.0], 1.0)
    t.refine_to_sdf(lambda x, y, z: 0.0, 0, 1e30, True)  # uniform N^3
    assert t.num_leaves == N ** 3, t.num_leaves
    R = (PHI * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * N
    mu, f, c = 0.1, 1e-3, N / 2.0
    fl = amr.Flow(t, 1.0, mu, 60.0)
    fl.set_body_force(f, 0.0, 0.0)
    fl.set_advection(False)
    if ghostproj:
        fl.set_ghost_projection(True, 2, 2)
    elif ghost:
        fl.set_ghost_gradient(True)
    fl.set_solid(lambda x, y, z:
                 math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) - R)
    print(f"  [N={N}] setup {time.time() - t0:.1f}s", flush=True)
    kprev, k = None, None
    n = t.num_leaves
    steps = 0
    confirm = False
    t0 = time.time()
    while steps < max_steps:
        for _ in range(10):
            fl.step(mom_iters, pres_iters)
        steps += 10
        u = fl.velocity(0)
        umean = float(u.sum()) / n
        k = f * N ** 3 / (6.0 * math.pi * mu * R * umean)
        if steps % 100 == 0:
            print(f"  [N={N}] step {steps}  K={k:.6f}  err={100*(k-K_ZH)/K_ZH:+.3f}%  "
                  f"({(time.time()-t0)/steps*1e3:.0f} ms/step)", flush=True)
        if kprev is not None and abs(k - kprev) < tol * abs(k):
            if confirm:
                break
            confirm = True  # hold for one more stride (false-plateau guard)
        else:
            confirm = False
        kprev = k
    return k, steps, fl.last_mom_iters(), fl.last_pres_iters()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    ghost = "--ghost" in args
    ghostproj = "--ghostproj" in args
    Ns = [int(a) for a in args if not a.startswith("--")] or [32, 64]
    print(f"ghost_gradient={ghost} ghost_projection={ghostproj}", flush=True)
    print(f"{'N':>5} {'K':>10} {'err%':>8} {'steps':>6} {'mom':>5} {'pres':>5}", flush=True)
    for N in Ns:
        k, s, mi, pi = drag_k(N, ghost, ghostproj)
        print(f"{N:>5} {k:>10.4f} {100 * (k - K_ZH) / K_ZH:>8.3f} {s:>6} {mi:>5} {pi:>5}",
              flush=True)
