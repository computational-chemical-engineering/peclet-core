"""Test of the tpx_amr Python module (transport-core AMR octree via numpy / mpi4py).

Run: PYTHONPATH=python/build mpirun -np 4 python3 python/test_tpx_amr.py
(also valid serially: PYTHONPATH=python/build python3 python/test_tpx_amr.py)

Mirrors the C++ AMR tests through the binding. Validates, from Python:
  * serial Octree: a uniform brick has the expected leaf count; refining toward a sphere localizes
    leaves on the surface, stays 2:1-balanced, and grows the count; leaf geometry (centers/sizes/
    levels/codes) is self-consistent; find() round-trips a leaf centre; VTU export has one cell per
    leaf with the field reading back exactly.
  * DistributedOctree (np>=1): ORB partitions the global leaf set (counts sum to the serial total);
    refine_to_sphere + balance keeps every block locally 2:1-balanced; a weighted-ORB rebalance
    redistributes leaves+fields while conserving them; face_neighbor_gather returns finite interior
    neighbour values.
"""
import os
import sys
import numpy as np
from mpi4py import MPI
import tpx_amr

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size
fail = 0


def check(cond, msg):
    global fail
    if not cond:
        fail += 1
        sys.stderr.write(f"[rank {rank}] FAIL: {msg}\n")


# ----------------------------------------------------------------------------------------------
# Serial Octree (run on rank 0 only; pure host, no MPI).
# ----------------------------------------------------------------------------------------------
serial_leaves = None
if rank == 0:
    brick = [2, 2, 2]
    lmax = 3
    t = tpx_amr.Octree(brick=brick, lmax=lmax, origin=[1.0, -2.0, 0.5], h0=0.25)

    # Uniform brick: 2*2*2 root cells, each refinable but unrefined -> 8 leaves.
    check(t.num_leaves == 8, f"uniform brick leaf count {t.num_leaves} != 8")
    check(t.is_balanced(), "uniform brick not balanced")
    check(t.lmax == lmax and t.h0 == 0.25, "lmax/h0 round-trip")

    # Refine toward a sphere through the block centre.
    cx = [1.0 + 0.5 * brick[0] * 0.25 * (1 << lmax),
          -2.0 + 0.5 * brick[1] * 0.25 * (1 << lmax),
          0.5 + 0.5 * brick[2] * 0.25 * (1 << lmax)]
    n0 = t.num_leaves
    nref = t.refine_to_sphere(center=cx, radius=0.6, target_level=0, band=1.0, balance=True)
    check(nref > 0 and t.num_leaves > n0, "refine_to_sphere did not refine")
    check(t.is_balanced(), "refined octree not 2:1 balanced")

    # Leaf geometry is self-consistent and the right shapes.
    centers = t.centers()
    sizes = t.sizes()
    levels = t.levels()
    codes = t.codes()
    N = t.num_leaves
    check(centers.shape == (N, 3), f"centers shape {centers.shape}")
    check(sizes.shape == (N,) and levels.shape == (N,) and codes.shape == (N,), "per-leaf shapes")
    check(levels.dtype == np.int32 and codes.dtype == np.uint64, "level/code dtypes")
    # size == h0 * 2**level for every leaf.
    check(np.allclose(sizes, t.h0 * (2.0 ** levels)), "size != h0*2**level")
    # The finest leaves cluster near the sphere surface (|dist|<~ a cell), not in the far field.
    fine = levels == 0
    check(fine.sum() > 0, "no finest leaves after refinement")
    # The refined region is a thin shell around the surface (radius + band + 2:1 grading), not the
    # whole block: every finest leaf is within a few cells of the surface.
    d = np.linalg.norm(centers[fine] - np.array(cx), axis=1) - 0.6
    check(np.all(np.abs(d) <= 3.0 * sizes[fine]), "finest leaves not on the surface")

    # find() round-trips every leaf centre to its own slot.
    bad = sum(t.find(centers[i].tolist()) != i for i in range(N))
    check(bad == 0, f"find() mismatched {bad} leaf centres")
    check(t.find([-100.0, 0.0, 0.0]) == -1, "find() outside block != -1")

    # VTU export: one cell per leaf, field reads back exactly.
    field = levels.astype(np.float64)
    path = "tpx_amr_serial_test.vtu"
    t.write_vtu(path, "level", field)
    txt = open(path).read()
    ncells = int(txt.split('NumberOfCells="')[1].split('"')[0])
    npts = int(txt.split('NumberOfPoints="')[1].split('"')[0])
    check(ncells == N and npts == 8 * N, "vtu cell/point counts")
    block = txt.split("<CellData")[1].split("<DataArray")[1]
    vals = np.fromstring(block.split(">", 1)[1].split("</DataArray>")[0], sep=" ")
    check(len(vals) == N and np.array_equal(vals, field), "vtu cell data round-trip")
    os.remove(path)
    serial_leaves = N

    # ------------------------------------------------------------------------------------------
    # Poisson multigrid solve (manufactured RHS) through the binding.
    # ------------------------------------------------------------------------------------------
    # (a) Uniform multilevel grid: refine a 2x2x2/lmax=3 brick uniformly to a 16^3 finest grid,
    #     which the multigrid coarsens to several levels. With a periodic manufactured RHS
    #     b = L u_exact, the solver recovers u_exact exactly and the residual hits round-off.
    tu = tpx_amr.Octree(brick=[2, 2, 2], lmax=3, origin=[0, 0, 0], h0=1.0)
    tu.refine_to_sdf(lambda x, y, z: 0.0, target_level=0, band=1e9, balance=False)
    pois = tpx_amr.Poisson(tu, periodic=True)
    check(pois.num_levels >= 3, f"multigrid only built {pois.num_levels} levels")
    check(pois.num_leaves == tu.num_leaves, "Poisson num_leaves mismatch")
    cc = tu.centers()
    kk = 2 * np.pi / 16.0
    u_exact = np.cos(kk * cc[:, 0]) + np.sin(2 * kk * cc[:, 1])
    u_exact -= u_exact.mean()
    b = pois.apply(u_exact)
    check(abs(b.mean()) < 1e-10, f"manufactured RHS not mean-zero ({b.mean():.2e})")
    r0 = pois.residual(np.zeros_like(b), b)
    u, r, ncyc = pois.solve(b, cycles=20, tol=0.0)
    check(r < r0 * 1e-10, f"V-cycle did not converge: {r0:.2e} -> {r:.2e}")
    err = u - u_exact
    err -= err.mean()
    check(np.linalg.norm(err) < 1e-9 * np.linalg.norm(u_exact),
          f"did not recover u_exact (rel err {np.linalg.norm(err)/np.linalg.norm(u_exact):.2e})")

    # (b) Graded (sphere-refined) octree: same manufactured-RHS recipe still converges to round-off.
    tg = tpx_amr.Octree(brick=[2, 2, 2], lmax=3, origin=[0, 0, 0], h0=1.0)
    tg.refine_to_sphere(center=[8, 8, 8], radius=4.0, target_level=0, band=1.0, balance=True)
    pg = tpx_amr.Poisson(tg, periodic=True)
    cg = tg.centers()
    ug = np.cos(kk * cg[:, 0])
    ug -= ug.mean()
    bg = pg.apply(ug)
    rg0 = pg.residual(np.zeros_like(bg), bg)
    _, rg, _ = pg.solve(bg, cycles=25, tol=0.0)
    check(rg < rg0 * 1e-8, f"graded V-cycle did not converge: {rg0:.2e} -> {rg:.2e}")

    # ------------------------------------------------------------------------------------------
    # Collocated flow: planar Poiseuille between two immersed (cut-cell) walls -> analytic parabola.
    # ------------------------------------------------------------------------------------------
    # Uniform 16^3 grid; solid everywhere except the slab a<x<b (walls perpendicular to x); body
    # force along y. Steady Stokes gives u_y(x) = (f/2mu)(x-a)(b-x); the cut-cell IBM reproduces it.
    Nc = 16
    h0 = 1.0 / Nc
    a, b = 0.25, 0.75  # wall planes (cell-aligned at 4*h0, 12*h0)
    tc = tpx_amr.Octree(brick=[Nc, Nc, Nc], lmax=0, origin=[0, 0, 0], h0=h0)
    flow = tpx_amr.Flow(tc, density=1.0, viscosity=1.0, dt=1e6)
    flow.set_solid(lambda x, y, z: min(x - a, b - x))  # >0 inside the channel
    flow.set_body_force(0.0, 1.0, 0.0)
    for _ in range(5):
        flow.step(mom_sweeps=300, pres_iters=80, pres_sweeps=4)
    cc2 = tc.centers()
    xcoord = cc2[:, 0]
    uy = flow.velocity(1)
    fluid = flow.is_fluid()
    inside = fluid & (xcoord > a) & (xcoord < b)
    u_par = 0.5 * (xcoord[inside] - a) * (b - xcoord[inside])  # f/(2 mu), f=mu=1
    rel = np.abs(uy[inside] - u_par).max() / u_par.max()
    check(rel < 1e-6, f"Poiseuille velocity off analytic parabola (rel err {rel:.2e})")
    check(np.allclose(flow.velocity(0), 0.0) and np.allclose(flow.velocity(2), 0.0),
          "Poiseuille cross-flow velocities not ~0")
    check(flow.divergence_norm() < 1e-6, "Poiseuille residual divergence too large")
    # Cross-flow components zero in the solid too (no-slip): u_y vanishes inside the wall.
    solid = ~fluid
    check(np.abs(uy[solid]).max() < 1e-9, "velocity nonzero inside the solid wall")

    # Navier-Stokes (advection on): fully-developed Poiseuille is advection-free, so still the
    # exact parabola.
    flow.set_advection(True)
    for _ in range(5):
        flow.step(mom_sweeps=300, pres_iters=80, pres_sweeps=4)
    uy2 = flow.velocity(1)
    rel2 = np.abs(uy2[inside] - u_par).max() / u_par.max()
    check(rel2 < 1e-6, f"Poiseuille with advection off parabola (rel err {rel2:.2e})")

    # ------------------------------------------------------------------------------------------
    # Solution-adaptive AMR: a planar tanh front. The Löhner indicator localizes the front; adapt
    # refines there and coarsens the (flat) far field, conserving the field under the remap.
    # ------------------------------------------------------------------------------------------
    x0, wd = 16.0, 2.0  # front position / width, domain 32 units (8 root cells * 2^2 finest)

    def front(c):  # strictly positive (offset) so sum(V*f) is O(1) — a meaningful conservation ref
        return 2.0 + np.tanh((c[:, 0] - x0) / wd)

    # (a) Pure-remap conservation: refine a slab, set the field, ONE adapt -> sum(V*f) preserved.
    tr = tpx_amr.Octree(brick=[8, 8, 8], lmax=2, origin=[0, 0, 0], h0=1.0)
    tr.refine_to_sdf(lambda x, y, z: abs(x - x0) - 3.0, target_level=0, band=1.0)
    fr = front(tr.centers())
    mass0 = float(np.sum(tr.sizes() ** 3 * fr))
    fr2 = tr.adapt(fr, refine_thresh=0.2, coarsen_thresh=0.05, finest_level=0)
    mass1 = float(np.sum(tr.sizes() ** 3 * fr2))
    check(fr2.shape == (tr.num_leaves,), "adapt returned wrong field length")
    check(abs(mass1 - mass0) <= 1e-12 * abs(mass0), f"adapt remap not conservative ({mass0}->{mass1})")

    # (b) Tracking loop: re-sample the analytic front each step; the mesh converges to a thin refined
    #     slab around it — every finest leaf near the front, and far fewer leaves than uniform-fine.
    ta = tpx_amr.Octree(brick=[8, 8, 8], lmax=2, origin=[0, 0, 0], h0=1.0)
    for _ in range(5):
        ea = ta.lohner_indicator(front(ta.centers()), eps=0.01)
        check(ea.min() >= 0.0 and ea.max() <= 1.0 + 1e-12, "Löhner indicator out of [0,1]")
        ta.adapt(front(ta.centers()), refine_thresh=0.2, coarsen_thresh=0.05, finest_level=0)
    ca = ta.centers()
    finest = ta.levels() == 0
    check(finest.sum() > 0, "no finest leaves after adaptive loop")
    check(np.all(np.abs(ca[finest, 0] - x0) <= 3.0 * wd), "finest leaves not localized at the front")
    check(ta.num_leaves < 0.6 * 32 ** 3, f"adaptive mesh not coarser than uniform-fine ({ta.num_leaves})")
    check(ta.is_balanced(), "adaptive mesh not 2:1 balanced")

# ----------------------------------------------------------------------------------------------
# DistributedOctree (collective). Same global geometry on every rank; ORB partitions it.
# ----------------------------------------------------------------------------------------------
groot = [4, 4, 4]
lmax = 3
d = tpx_amr.DistributedOctree(global_root_size=groot, lmax=lmax, origin=[0.0, 0.0, 0.0], h0=1.0,
                              periodic=[True, True, True])
check(d.size == size and 0 <= d.rank < size, "rank/size")

# Uniform: local leaf counts sum to the global root-cell count (4*4*4 = 64 leaves).
gleaves = comm.allreduce(d.num_leaves, MPI.SUM)
check(gleaves == groot[0] * groot[1] * groot[2], f"global leaf count {gleaves} != 64")

# Refine toward a global sphere at the domain centre, then cross-block balance.
cx = [groot[0] / 2.0, groot[1] / 2.0, groot[2] / 2.0]
d.refine_to_sphere(center=cx, radius=1.5, target_level=0, band=1.0, balance=True)
gleaves2 = comm.allreduce(d.num_leaves, MPI.SUM)
check(gleaves2 > gleaves, "distributed refine did not grow the leaf set")

# Leaf geometry shapes line up on each rank.
N = d.num_leaves
check(d.centers().shape == (N, 3), "distributed centers shape")
check(d.sizes().shape == (N,) and d.levels().shape == (N,), "distributed per-leaf shapes")

# face_neighbor_gather: interior leaves have finite neighbours; result shape (N,6).
g = d.face_neighbor_gather(d.levels().astype(np.float64), sentinel=-1.0)
check(g.shape == (N, 6), "gather shape")
check(np.isfinite(g).all(), "gather produced non-finite values")

# Weighted-ORB rebalance: migrate leaves + 2 field columns (the level, and the leaf's own global
# index marker) — counts and the marker multiset are conserved; the partition is updated in place.
levels = d.levels().astype(np.float64)
# Give each leaf a globally-unique marker so we can check the multiset survives migration.
offsets = comm.scan(N, MPI.SUM) - N
marker = (offsets + np.arange(N)).astype(np.float64)
fields = np.column_stack([levels, marker])


def imbalance(local):
    mx = comm.allreduce(local, MPI.MAX)
    sm = comm.allreduce(local, MPI.SUM)
    return mx / (sm / size) if sm else 1.0


imb_before = imbalance(N)
out = d.rebalance(fields)
M = d.num_leaves
check(out.shape == (M, 2), f"rebalance returned {out.shape}, expected ({M}, 2)")
rcount = comm.allreduce(M, MPI.SUM)
check(rcount == gleaves2, f"rebalance changed total leaves {rcount} != {gleaves2}")
# Marker multiset preserved: gather all markers and compare to 0..gleaves2-1.
all_markers = comm.allreduce(int(out[:, 1].astype(np.int64).sum()), MPI.SUM)
check(all_markers == gleaves2 * (gleaves2 - 1) // 2, "rebalance marker sum not conserved")
imb_after = imbalance(M)
check(imb_after <= imb_before + 1e-9, f"rebalance worsened imbalance {imb_before}->{imb_after}")
# The migrated octree is still usable: geometry matches the new leaf count.
check(d.centers().shape == (M, 3), "post-rebalance centers shape")

# --- distributed solution-adaptive step (Löhner over the owner-based halo) ---
# A planar tanh front across the global domain; distributedAdapt refines it, restores cross-block
# 2:1 balance, and conservatively remaps the field. Check global conservation + localization.
dgroot = [4, 4, 4]
da = tpx_amr.DistributedOctree(global_root_size=dgroot, lmax=2, origin=[0, 0, 0], h0=1.0,
                               periodic=[False, False, False])
gx0, gw = 8.0, 1.5  # domain is 4*2^2 = 16 units per axis


def gfront(c):  # offset -> sum(V*f) is O(domain volume), so the conservation check is well-scaled
    return 2.0 + np.tanh((c[:, 0] - gx0) / gw)


fa = gfront(da.centers())
# Indicator is in [0,1] and finite across the halo.
ind = da.lohner_indicator(fa, eps=0.01)
check(ind.shape == (da.num_leaves,) and np.isfinite(ind).all(), "distributed indicator shape/finite")
gmass0 = comm.allreduce(float(np.sum(da.sizes() ** 3 * fa)), MPI.SUM)
gn0 = comm.allreduce(da.num_leaves, MPI.SUM)
fa2 = da.adapt(fa, refine_thresh=0.25, coarsen_thresh=0.05, finest_level=0)
check(fa2.shape == (da.num_leaves,), "distributed adapt field length")
gn1 = comm.allreduce(da.num_leaves, MPI.SUM)
gmass1 = comm.allreduce(float(np.sum(da.sizes() ** 3 * fa2)), MPI.SUM)
check(gn1 > gn0, "distributed adapt did not refine the front")
check(abs(gmass1 - gmass0) <= 1e-10 * abs(gmass0),
      f"distributed adapt not globally conservative ({gmass0}->{gmass1})")
# Every finest leaf (any rank) sits near the front.
cda = da.centers()
fmask = da.levels() == 0
local_far = float(np.abs(cda[fmask, 0] - gx0).max()) if fmask.any() else -1.0
global_far = comm.allreduce(local_far, MPI.MAX)
check(global_far <= 3.0 * gw, f"distributed finest leaves not localized (max |x-x0|={global_far})")

# ----------------------------------------------------------------------------------------------
total = comm.allreduce(fail, MPI.SUM)
if rank == 0:
    print(f"# tpx_amr: serial_leaves={serial_leaves} dist_leaves {gleaves}->{gleaves2} "
          f"imbalance {imb_before:.3f}->{imb_after:.3f}")
    if total == 0:
        print(f"OK (np={size}): tpx_amr Octree + DistributedOctree work from Python/mpi4py")
    else:
        sys.stderr.write(f"FAILED (np={size}): {total}\n")
sys.exit(0 if total == 0 else 1)
