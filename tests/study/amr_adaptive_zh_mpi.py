"""DISTRIBUTED solution-adaptive Z&H — the distributed-AmrFlow campaign's end-gate battery
(docs/amr_distributed_flow.md): the amr_adaptive_zh.py cases on AmrFlow::initMpi
(DistributedOctree + Flow(d, ...)), with mid-run distributed adapt (Löhner on |u| +
geometry-band protection, ownership-preserving) AND a weighted-ORB load rebalance every
cycle (state migrates with the leaves).

Differences vs the single-rank battery, both deliberate:
  * cf scheme 0 (the Martin-Cartwright quadratic C/F scheme is not distributed yet — the
    distributed setSolid throws on it), so the ABSOLUTE K carries the standard C/F scheme's
    graded offset; the GATE here is NP-PARITY: K(np) must match K(np=1). The np=1 path is
    bit-identical to the single-rank solver by construction (ctest-locked), so this is the
    distribution gate, with the cf accuracy layer still single-rank-validated.
  * The Löhner input is the SOLVED |u|, so meshes may differ near flag thresholds across
    rank counts (Krylov dot-order noise) — K is the mesh-robust comparable, as in the
    single-rank study.

Run FOREGROUND on the CUDA build:
  mpirun -np 1 python core/tests/study/amr_adaptive_zh_mpi.py [dense|dilute ...]
  mpirun -np 4 python core/tests/study/amr_adaptive_zh_mpi.py [dense|dilute ...]
"""
import math
import sys
import time

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0] + "/python/build_cuda2")
import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402
from peclet.core import amr  # noqa: E402

MU = 0.1
F = 1e-3
comm = MPI.COMM_WORLD


def run_case(name, N, R, band, cycles=10, stride=100, tol=1e-7, max_tail=6000, dt=60.0,
             reset_p=False):
    lmax = int(math.log2(N)) - 2  # 4^3 root cells so ORB has bricks to hand out
    d = amr.DistributedOctree([4, 4, 4], lmax, [0.0, 0.0, 0.0], 1.0, [True, True, True])
    c = N / 2.0
    sdf = lambda x, y, z: math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) - R  # noqa: E731
    d.refine_to_sphere([c, c, c], R, 0, band, True)
    fl = amr.Flow(d, 1.0, MU, dt)
    fl.set_body_force(F, 0.0, 0.0)
    fl.set_advection(False)
    fl.set_ghost_projection(True, 1, 2)
    fl.set_solid(sdf)

    def kval():
        w = np.asarray(d.sizes()) ** 3  # volume-weighted superficial velocity (graded mesh!)
        loc = float((np.asarray(fl.velocity(0)) * w).sum())
        um = comm.allreduce(loc, MPI.SUM) / N ** 3
        return F * N ** 3 / (6.0 * math.pi * MU * R * um)

    def gleaves():
        return comm.allreduce(int(fl.num_leaves), MPI.SUM)

    gl = gleaves()  # collective — NEVER call inside a rank-0-only block (deadlock)
    if comm.rank == 0:
        print(f"== {name}: finest {N}, R={R:.2f}, band {band}, np={comm.size} "
              f"({gl} global leaves) ==", flush=True)
    t0 = time.time()
    for cyc in range(cycles):
        for _ in range(stride):
            fl.step(100, 60)
        k = kval()
        gl = gleaves()
        if comm.rank == 0:
            print(f"  cycle {cyc:2d}: K={k:.4f}  leaves={gl:>8} "
                  f"({100*gl/N**3:5.1f}%)", flush=True)
        # Distributed adapt (ownership-preserving) + band protection + solver rebuild,
        # then a weighted-ORB load rebalance (state migrates with the leaves).
        umag = np.linalg.norm(np.asarray(fl.velocities()), axis=1)
        fl.begin_adapt()
        d.adapt(umag, 0.2, 0.05, finest_level=0)
        d.refine_to_sdf(sdf, 0, band, True)
        fl.finish_adapt(sdf)
        if reset_p:
            fl.set_pressure(np.zeros(fl.num_leaves))
        fl.rebalance_mpi(sdf)
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
    gl = gleaves()
    if comm.rank == 0:
        print(f"  FINAL {name} np={comm.size}: K={k:.4f}  cells={100*gl/N**3:.1f}%  "
              f"leaves={gl}  tail_steps={steps}  {time.time()-t0:.0f}s", flush=True)
    return k


if __name__ == "__main__":
    which = [a for a in sys.argv[1:]] or ["dense", "dilute"]
    if "dense" in which:
        N = 64
        R = (0.125 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * N
        run_case("dense phi=0.125", N, R, band=3.0)
    if "dilute" in which:
        N = 128
        R = (0.125 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) * (N / 4.0)
        run_case("dilute phi~0.002", N, R, band=4.0, dt=1e6, stride=200, reset_p=True)
