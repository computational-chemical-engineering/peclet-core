"""Solution-ADAPTIVE AMR on the two Zick & Homsy cases (the ladder's closing demo).

The packaged adapt policy (`adapt_cycle`): Löhner indicator on |u| drives refine/coarsen
(`Octree.adapt`), the GEOMETRY BAND is re-refined afterwards (`refine_to_sdf` — the cut band
must stay finest for the ghost overlay; also restores 2:1), and `Flow.finish_adapt`
conservatively remaps (u, p) + rebuilds the solver. Ghost projection + Martin–Cartwright C/F
scheme; Stokes (the Z&H ground truth is the Stokes drag).

  dense  — the phi=0.125 SC array at finest 64 (uniform ghost reference: K err -0.056%).
           Shear fills the inter-sphere channels, so the indicator should grow the mesh toward
           uniform — the honest outcome for this case.
  dilute — the same sphere in a 4x box (phi~0.002) at finest 128 (uniform ghost reference
           K=1.2785): the far field is smooth, so the indicator should hold a small mesh.

Run FOREGROUND on the CUDA build:
  python core/tests/study/amr_adaptive_zh.py [dense|dilute ...]
"""
import math
import sys
import time

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
import numpy as np  # noqa: E402
from peclet.core import amr  # noqa: E402

MU = 0.1
F = 1e-3
K_ZH = 4.292


def adapt_cycle(t, fl, sdf, band, refine_th=0.2, coarsen_th=0.05, reset_p=False):
    """One solution-adaptive step: Löhner on |u| + geometry-band protection + solver rebuild.

    reset_p: at STEADY-STATE dt (rho/dt -> 0) the accumulated rotational p is the load-bearing
    state, and its conservative remap under coarsening can leave a distorted field whose
    relaxation is SLOWER than re-accumulating from zero (measured: a mid-cycle collapse that
    took >> cold-start steps to recover). Zeroing p after the rebuild lets the rotational
    update re-accumulate it at the cold-start rate while u carries the flow state.
    """
    umag = np.linalg.norm(np.asarray(fl.velocities()), axis=1)
    fl.begin_adapt()
    t.adapt(umag, refine_th, coarsen_th, finest_level=0)
    t.refine_to_sdf(sdf, 0, band, True)  # keep the cut band finest (+ 2:1 balance)
    fl.finish_adapt(sdf)
    if reset_p:
        fl.set_pressure(np.zeros(fl.num_leaves))


def run_case(name, N, R, band, uniform_ref, ref_label, cycles=10, stride=100, tol=1e-7,
             max_tail=6000, dt=60.0, reset_p=False):
    lmax = int(math.log2(N))
    t = amr.Octree([1, 1, 1], lmax, [0.0, 0.0, 0.0], 1.0)
    c = N / 2.0
    sdf = lambda x, y, z: math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) - R
    t.refine_to_sphere([c, c, c], R, 0, band, True)
    fl = amr.Flow(t, 1.0, MU, dt)
    fl.set_body_force(F, 0.0, 0.0)
    fl.set_advection(False)
    fl.set_ghost_projection(True, 2, 2)  # ghost (2,2) — the production-candidate pair
    fl.set_cf_scheme(1)
    fl.set_solid(sdf)

    def kval():
        w = np.asarray(t.sizes()) ** 3
        um = float((np.asarray(fl.velocity(0)) * w).sum()) / N ** 3
        return F * N ** 3 / (6.0 * math.pi * MU * R * um)

    print(f"== {name}: finest {N}, sphere R={R:.2f}, start band {band} "
          f"({t.num_leaves} leaves, {100*t.num_leaves/N**3:.1f}%) ==", flush=True)
    t0 = time.time()
    for cyc in range(cycles):
        for _ in range(stride):
            fl.step(100, 60)
        k = kval()
        print(f"  cycle {cyc:2d}: K={k:.4f}  leaves={t.num_leaves:>8} "
              f"({100*t.num_leaves/N**3:5.1f}%)  pres={fl.last_pres_iters()}", flush=True)
        adapt_cycle(t, fl, sdf, band, reset_p=reset_p)
    # mesh frozen: converge to steadiness
    kprev, k, steps, confirm = None, None, 0, False
    while steps < max_tail:
        for _ in range(10):
            fl.step(100, 60)
        steps += 10
        k = kval()
        if kprev is not None and abs(k - kprev) < tol * abs(k):
            if confirm:
                break
            confirm = True
        else:
            confirm = False
        kprev = k
    frac = 100 * t.num_leaves / N ** 3
    print(f"  FINAL {name}: K={k:.4f}  vs {ref_label} {uniform_ref:.4f} "
          f"({100*(k-uniform_ref)/uniform_ref:+.3f}%)  cells={frac:.1f}%  "
          f"leaves={t.num_leaves}  tail_steps={steps}  {time.time()-t0:.0f}s", flush=True)
    return k, frac


if __name__ == "__main__":
    which = [a for a in sys.argv[1:]] or ["dense", "dilute"]
    if "dense" in which:
        N = 64
        R = (0.125 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * N
        # uniform ghost+cf reference at finest 64: K err -0.056% vs Z&H 4.292 -> K = 4.2896
        run_case("dense phi=0.125", N, R, band=3.0, uniform_ref=4.2896,
                 ref_label="uniform-ghost")
    if "dilute" in which:
        N = 128
        R = (0.125 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * (N / 4.0)
        # steady dt: the big dilute box is diffusion-limited at transient dt (the
        # rotational scheme is dt-independent at steady state)
        run_case("dilute phi~0.002", N, R, band=4.0, uniform_ref=1.2785,
                 ref_label="uniform-ghost", dt=1e6, stride=200, reset_p=True)
