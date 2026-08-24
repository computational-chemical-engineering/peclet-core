"""C2 acceptance battery for the AMR ghost projection (the fluid-only scheme, (2,2)).

MEASURED 2026-08-24 (N=64, RTX 5080): ghost dt-spread 7.5e-9 rel over dt=60/600/1e20 (PASS;
K=-0.0566% == flow's uniform ghost to the digit) -- but the aperture arm ALSO reads 7.6e-9:
on the single smooth Z&H sphere the attractor family does not manifest (exactly as in flow,
where it needed the dense beds). This battery is therefore a NECESSARY-condition check +
the flow-equivalence anchor; the family CONTRAST rides on flow's bed evidence through the
unrefined-equivalence gate. Direct AMR bed demonstration = follow-up (needs a bed SDF path).
Bonus: ghost reaches K-stationarity in half the steps (5.7k vs 11.6k) and less wall time.

  python core/tests/study/amr_zh_c2.py [N]          # default 64, CUDA build
"""
import math
import sys
import time

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
from peclet.core import amr  # noqa: E402

K_ZH = 4.292
PHI = 0.125


def march(N, scheme, dt, tol=1e-9, max_steps=20000):
    lmax = int(math.log2(N))
    t = amr.Octree([1, 1, 1], lmax, [0.0, 0.0, 0.0], 1.0)
    t.refine_to_sdf(lambda x, y, z: 0.0, 0, 1e30, True)
    R = (PHI * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * N
    mu, f, c = 0.1, 1e-3, N / 2.0
    fl = amr.Flow(t, 1.0, mu, dt)
    fl.set_body_force(f, 0.0, 0.0)
    fl.set_advection(False)
    if scheme == "ghost":
        fl.set_ghost_projection(True)        # (2, 2) production default
    else:
        fl.set_ghost_projection(False)       # aperture (the current default)
    fl.set_solid(lambda x, y, z:
                 math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) - R)
    n = t.num_leaves
    kprev, k, steps, confirm = None, None, 0, False
    t0 = time.time()
    while steps < max_steps:
        for _ in range(10):
            fl.step(100, 60)
        steps += 10
        umean = float(fl.velocity(0).sum()) / n
        k = f * N ** 3 / (6.0 * math.pi * mu * R * umean)
        if kprev is not None and abs(k - kprev) < tol * abs(k):
            if confirm:
                break
            confirm = True
        else:
            confirm = False
        kprev = k
    print(f"  {scheme:>8} dt={dt:<8g} K={k:.8f} err={100*(k-K_ZH)/K_ZH:+.4f}% "
          f"steps={steps} ({time.time()-t0:.0f}s)", flush=True)
    return k


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    print(f"N={N} Z&H phi={PHI}: C2 (dt-independence) A/B", flush=True)
    out = {}
    for scheme in ("ghost", "aperture"):
        ks = [march(N, scheme, dt) for dt in (60.0, 600.0, 1e20)]
        spread = (max(ks) - min(ks)) / abs(ks[0])
        out[scheme] = spread
        print(f"{scheme}: dt-spread {spread:.2e} rel", flush=True)
    print(f"\nC2 VERDICT: ghost spread {out['ghost']:.2e} "
          f"vs aperture {out['aperture']:.2e} "
          f"({'PASS' if out['ghost'] < 1e-5 else 'FAIL'} at 1e-5)", flush=True)
