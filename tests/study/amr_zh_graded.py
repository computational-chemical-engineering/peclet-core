"""Graded Zick & Homsy: finest band at the sphere + coarse far field (ladder step 3).

The ghost projection lives entirely in the finest band (buildGhostOverlay enforces the ±2
same-level reach); level boundaries carry only the smooth binary operator. Gate: the graded K
approaches the uniform ghost K at the same finest resolution as the band widens, at a fraction
of the cells. K uses the VOLUME-weighted mean (superficial velocity) on the graded mesh.

Run FOREGROUND on the CUDA build:
  python core/tests/study/amr_zh_graded.py [--base] N band [band ...]
"""
import math
import sys
import time

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
import numpy as np  # noqa: E402
from peclet.core import amr  # noqa: E402

K_ZH = 4.292
PHI = 0.125


def drag_graded(N, band, ghostproj, cf=0, tol=1e-7, max_steps=6000):
    lmax = int(math.log2(N))
    t = amr.Octree([1, 1, 1], lmax, [0.0, 0.0, 0.0], 1.0)
    R = (PHI * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * N
    c = N / 2.0
    if band >= 1e29:
        t.refine_to_sdf(lambda x, y, z: 0.0, 0, 1e30, True)  # uniform reference
    else:
        t.refine_to_sphere([c, c, c], R, 0, band, True)
    frac = t.num_leaves / N ** 3
    mu, f = 0.1, 1e-3
    fl = amr.Flow(t, 1.0, mu, 60.0)
    fl.set_body_force(f, 0.0, 0.0)
    fl.set_advection(False)
    if ghostproj:
        fl.set_ghost_projection(True, 1, 2)
    if cf:
        fl.set_cf_scheme(cf)
    fl.set_solid(lambda x, y, z:
                 math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) - R)
    w = np.asarray(t.sizes()) ** 3  # cell volumes (grid units)
    kprev, k, steps, confirm = None, None, 0, False
    t0 = time.time()
    while steps < max_steps:
        for _ in range(10):
            fl.step(100, 60)
        steps += 10
        u = np.asarray(fl.velocity(0))
        umean = float((u * w).sum()) / N ** 3  # superficial (volume-weighted)
        k = f * N ** 3 / (6.0 * math.pi * mu * R * umean)
        if steps % 200 == 0:
            print(f"  [N={N} band={band}] step {steps}  K={k:.6f} "
                  f"({(time.time()-t0)/steps*1e3:.0f} ms/step)", flush=True)
        if kprev is not None and abs(k - kprev) < tol * abs(k):
            if confirm:
                break
            confirm = True
        else:
            confirm = False
        kprev = k
    return k, frac, t.num_leaves, steps, fl.last_pres_iters()


if __name__ == "__main__":
    args = sys.argv[1:]
    ghostproj = "--base" not in args
    cf = 1 if "--cf" in args else 0
    vals = [float(a) for a in args if not a.startswith("--")]
    N = int(vals[0]) if vals else 64
    bands = vals[1:] or [3.0, 5.0, 8.0, 1e30]
    print(f"N={N} ghost_projection={ghostproj} cf={cf}  (band=1e30 -> uniform reference)", flush=True)
    print(f"{'band':>7} {'K':>10} {'err%':>8} {'cells%':>8} {'leaves':>9} {'steps':>6} "
          f"{'pres':>5}", flush=True)
    for b in bands:
        k, frac, nl, s, pi = drag_graded(N, b, ghostproj, cf)
        print(f"{b:>7.1f} {k:>10.4f} {100 * (k - K_ZH) / K_ZH:>8.3f} {100 * frac:>8.1f} "
              f"{nl:>9} {s:>6} {pi:>5}", flush=True)
