"""Dilute isolated-sphere demo (ladder step 3 payoff case): the same sphere in a 4x larger box
(phi = 0.125/64 ~ 0.002), where the far field is genuinely smooth Stokes flow — the setting the
finest-band + coarse-far-field trade is FOR (the periodic phi=0.125 SC array has shear
everywhere, so coarsening its far field costs real accuracy; docs/AMR.md already flags it as a
poor AMR showcase). Gate: graded K == uniform K at the same finest resolution, at a small cell
fraction. Self-consistency (no external reference needed at this phi).

Run FOREGROUND on the CUDA build:
  python core/tests/study/amr_zh_dilute.py [N] [band ...]
"""
import math
import sys
import time

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
import numpy as np  # noqa: E402
from peclet.core import amr  # noqa: E402

PHI_REF = 0.125  # the sphere of the N/4-cell reference case, in an N-cell box


def drag_dilute(N, band, ghostproj=True, cf=0, tol=1e-7, max_steps=8000):
    lmax = int(math.log2(N))
    t = amr.Octree([1, 1, 1], lmax, [0.0, 0.0, 0.0], 1.0)
    R = (PHI_REF * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * (N / 4.0)  # ~9.93 cells at N=128
    c = N / 2.0
    if band >= 1e29:
        t.refine_to_sdf(lambda x, y, z: 0.0, 0, 1e30, True)
    else:
        t.refine_to_sphere([c, c, c], R, 0, band, True)
    frac = t.num_leaves / N ** 3
    mu, f = 0.1, 1e-3
    fl = amr.Flow(t, 1.0, mu, 1e6)  # steady dt (rotational scheme dt-independent; the box is diffusion-limited at small dt)
    fl.set_body_force(f, 0.0, 0.0)
    fl.set_advection(False)
    if ghostproj:
        fl.set_ghost_projection(True, 1, 2)
    if cf:
        fl.set_cf_scheme(cf)
    fl.set_solid(lambda x, y, z:
                 math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) - R)
    w = np.asarray(t.sizes()) ** 3
    kprev, k, steps, confirm = None, None, 0, False
    t0 = time.time()
    while steps < max_steps:
        for _ in range(10):
            fl.step(100, 60)
        steps += 10
        u = np.asarray(fl.velocity(0))
        umean = float((u * w).sum()) / N ** 3
        k = f * N ** 3 / (6.0 * math.pi * mu * R * umean)
        if steps % 200 == 0:
            print(f"  [band={band}] step {steps}  K={k:.6f} "
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
    ghostproj = "--base" not in sys.argv
    cf = 1 if "--cf" in sys.argv else 0
    vals = [float(a) for a in sys.argv[1:] if not a.startswith("--")]
    N = int(vals[0]) if vals else 128
    bands = vals[1:] or [4.0, 1e30]
    print(f"dilute sphere: N={N}, R~{(PHI_REF*3/(4*math.pi))**(1/3)*(N/4):.2f} cells, "
          f"phi~{PHI_REF/64:.5f} (band=1e30 -> uniform)", flush=True)
    print(f"{'band':>7} {'K':>10} {'cells%':>8} {'leaves':>9} {'steps':>6} {'pres':>5}",
          flush=True)
    ks = {}
    for b in bands:
        k, frac, nl, s, pi = drag_dilute(N, b, ghostproj, cf)
        ks[b] = k
        print(f"{b:>7.1f} {k:>10.4f} {100 * frac:>8.2f} {nl:>9} {s:>6} {pi:>5}", flush=True)
    if 1e30 in ks:
        for b, k in ks.items():
            if b < 1e29:
                print(f"band {b}: K rel diff vs uniform = "
                      f"{100 * (k - ks[1e30]) / ks[1e30]:+.3f}%", flush=True)
