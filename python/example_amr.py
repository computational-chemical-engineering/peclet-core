"""Worked example of the transport-core AMR Python bindings (tpx_amr).

Run (after building the module — see python/CMakeLists.txt):
    PYTHONPATH=python/build python3 python/example_amr.py

Three short, self-contained workflows that together exercise the whole host AMR surface:

  1. Build an octree, refine it to a sphere, and write it (+ the refinement level) to a .vtu you can
     open in ParaView.
  2. Solve a Poisson problem on a graded octree with the multigrid V-cycle (manufactured RHS).
  3. Drive a planar Poiseuille flow between two immersed cut-cell walls and compare to the analytic
     parabola.

For the distributed (MPI) octree — DistributedOctree, rebalance, face_neighbor_gather, distributed
adapt — see python/test_tpx_amr.py, which drives it under mpi4py.
"""
import numpy as np

import tpx_amr


def example_refine_and_export() -> None:
    """Build -> refine to an SDF surface -> export a ParaView .vtu."""
    # 2x2x2 root cells, each refinable 4 levels deep, placed at the origin with finest cell h0=0.25.
    oct = tpx_amr.Octree(brick=[2, 2, 2], lmax=4, origin=[0.0, 0.0, 0.0], h0=0.25)
    center, radius = [4.0, 4.0, 4.0], 2.0
    nref = oct.refine_to_sphere(center=center, radius=radius, target_level=0, band=1.0)
    print(f"[1] refined a sphere: {nref} splits -> {oct.num_leaves} leaves, "
          f"balanced={oct.is_balanced()}")

    # Per-leaf field = refinement level; write it as an UnstructuredGrid.
    level = oct.levels().astype(np.float64)
    oct.write_vtu("amr_sphere.vtu", "level", level)
    print("    wrote amr_sphere.vtu (open in ParaView)")


def example_poisson() -> None:
    """Geometric-multigrid Poisson solve on a graded octree (manufactured RHS)."""
    oct = tpx_amr.Octree(brick=[2, 2, 2], lmax=3, origin=[0, 0, 0], h0=1.0)
    oct.refine_to_sphere(center=[8, 8, 8], radius=4.0, target_level=0, band=1.0)
    pois = tpx_amr.Poisson(oct, periodic=True)

    c = oct.centers()
    k = 2 * np.pi / 16.0
    u_exact = np.cos(k * c[:, 0]) + np.sin(2 * k * c[:, 1])
    u_exact -= u_exact.mean()
    rhs = pois.apply(u_exact)  # b = L u_exact  (exactly mean-zero)

    u, residual, cycles = pois.solve(rhs, cycles=25, tol=1e-12)
    err = u - u_exact
    err -= err.mean()
    print(f"[2] Poisson: {oct.num_leaves} leaves, {pois.num_levels} MG levels, "
          f"{cycles} V-cycles -> residual {residual:.2e}, "
          f"recovery error {np.linalg.norm(err)/np.linalg.norm(u_exact):.2e}")


def example_poiseuille() -> None:
    """Collocated Stokes flow between two immersed cut-cell walls vs the analytic parabola."""
    n, h0 = 16, 1.0 / 16
    a, b = 0.25, 0.75  # wall planes; fluid in the slab a < x < b
    oct = tpx_amr.Octree(brick=[n, n, n], lmax=0, origin=[0, 0, 0], h0=h0)
    flow = tpx_amr.Flow(oct, density=1.0, viscosity=1.0, dt=1e6)
    flow.set_solid(lambda x, y, z: min(x - a, b - x))  # >0 inside the channel
    flow.set_body_force(0.0, 1.0, 0.0)  # drive the flow along y
    for _ in range(5):
        flow.step(mom_sweeps=300, pres_iters=80)

    x = oct.centers()[:, 0]
    uy = flow.velocity(1)
    inside = flow.is_fluid() & (x > a) & (x < b)
    u_par = 0.5 * (x[inside] - a) * (b - x[inside])  # f/(2 mu) (x-a)(b-x), f=mu=1
    rel = np.abs(uy[inside] - u_par).max() / u_par.max()
    print(f"[3] Poiseuille: max u={uy[inside].max():.6f} (analytic {u_par.max():.6f}), "
          f"rel err {rel:.2e}, residual div {flow.divergence_norm():.2e}")


if __name__ == "__main__":
    example_refine_and_export()
    example_poisson()
    example_poiseuille()
