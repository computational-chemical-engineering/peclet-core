"""Fixed-seed reference runs of every public entry path of the peclet.core Python modules, hashed.

The structural gate of suite/docs/QUALITY_PLAN.md §3.G: a refactor that moves code verbatim must
leave every final state BYTE-IDENTICAL. This script runs one deterministic scenario per public
entry path, hashes the final arrays (SHA-256 of the raw float64/int bytes) and prints them; with
``--save FILE`` it records them as JSON and with ``--check FILE`` it compares against a recording
and exits non-zero on any difference.

    PYTHONPATH=<python build tree> OMP_NUM_THREADS=1 python python/state_hash.py --save pre.json
    PYTHONPATH=<python build tree> OMP_NUM_THREADS=1 mpirun -np 2 python python/state_hash.py --check pre.json

Under ``mpirun -np N`` (N > 1) the distributed paths (ParticleMigrator / ParticleHalo across ranks,
DistributedOctree, the distributed Flow) run too; every per-rank array is gathered to rank 0 in
rank order before hashing, so the hash names carry the rank count (``.np2``). Run at
OMP_NUM_THREADS=1: the device reductions are order-dependent at more than one thread.

``--modules`` selects ``mpi``, ``geom`` and ``amr`` (default: all that import).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np


def sha(*arrays):
    h = hashlib.sha256()
    flat = []
    for a in arrays:  # a binding may return a tuple of arrays: hash each member in order
        flat += list(a) if isinstance(a, (tuple, list)) and not np.isscalar(a[0]) else [a]
    for a in flat:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def gather_rows(comm, a):
    """Concatenate a per-rank (n, k) array over ranks in rank order (rank 0 gets the result)."""
    if comm is None or comm.size == 1:
        return np.ascontiguousarray(a)
    parts = comm.gather(np.ascontiguousarray(a), root=0)
    return np.concatenate(parts, axis=0) if comm.rank == 0 else None


# ---------------------------------------------------------------------------------------------
# peclet.core.geom — a CSG scene evaluated on a grid, baked, and its mass properties.
# ---------------------------------------------------------------------------------------------
def run_geom(out, comm):
    from peclet.core import geom
    if comm is not None and comm.rank != 0:
        return
    s = geom.SceneBuilder()
    sph = s.add_leaf("sphere", [0.3])
    box = s.add_leaf("box", [0.25, 0.15, 0.2], translation=[0.2, 0.1, 0.0],
                     rotation=[0.0, 0.0, 0.3826834323650898, 0.9238795325112867])
    tor = s.add_leaf("torus", [0.35, 0.08], translation=[-0.1, 0.0, 0.2])
    u = s.add_union(sph, box)
    d = s.add_difference(u, tor)
    s.add_instance(d, translation=[0.5, 0.5, 0.5])
    s.add_instance(sph, translation=[0.15, 0.8, 0.3], scale=0.5)
    n = 24
    g = (np.arange(n) + 0.5) / n
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    out["geom.eval"] = sha(s.eval(pts))
    out["geom.eval_root"] = sha(s.eval_root(d, pts))
    out["geom.eval_root_grad"] = sha(s.eval_root_grad(d, pts))
    baked = s.bake(d, [-0.6, -0.6, -0.6], [0.05, 0.05, 0.05], [24, 24, 24])
    out["geom.bake"] = sha(np.asarray(baked))
    bp = s.body_properties(d, [-0.8, -0.8, -0.8], [0.8, 0.8, 0.8], n=24)
    out["geom.body_properties"] = sha(
        np.array([bp["volume"], bp["mass"]]), np.asarray(bp["com"]), np.asarray(bp["inertia_tensor"]),
        np.asarray(bp["principal"]), np.asarray(bp["rotation"]), np.asarray(bp["quat"]))


# ---------------------------------------------------------------------------------------------
# peclet.core.mpi — ParticleMigrator migrate / gather_ghosts / rebalance and ParticleHalo.
# ---------------------------------------------------------------------------------------------
def run_mpi(out, comm):
    from peclet.core import mpi as core_mpi
    size, rank = (comm.size, comm.rank) if comm is not None else (1, 0)
    tag = f".np{size}"
    origin, extent, cells = [0.0, 0.0, 0.0], [2.0, 1.0, 1.5], [8, 4, 6]
    mig = core_mpi.ParticleMigrator(origin=origin, extent=extent, cells=cells,
                                    periodic=[True, True, False])
    rng = np.random.default_rng(1234 + rank)
    n = 600
    # Deliberately spill outside the box (periodic wrap on x/y, clamp on z) and give every particle
    # a globally-unique id in payload column 0 so the gathered state can be put in canonical order.
    pos = rng.uniform([-0.3, -0.2, 0.0], [2.3, 1.2, 1.5], size=(n, 3))
    pay = np.column_stack([rank * n + np.arange(n, dtype=np.float64), rng.normal(size=n),
                           rng.normal(size=n)])

    def canon(p, q):
        p, q = gather_rows(comm, p), gather_rows(comm, q)
        if p is None:
            return None
        order = np.argsort(q[:, 0], kind="stable")
        return p[order], q[order]

    pos2, pay2 = mig.migrate(pos, pay)
    c = canon(pos2, pay2)
    if c is not None:
        out["mpi.migrate" + tag] = sha(*c)
    gpos, gpay = mig.gather_ghosts(pos2, pay2, 0.35)
    # Ghost sets are per rank by construction: hash the per-rank arrays in rank order (each
    # sorted by id, then by position for the periodic images of one particle).
    gp, gq = gather_rows(comm, gpos), gather_rows(comm, gpay)
    counts = comm.gather(gpos.shape[0], root=0) if comm is not None else [gpos.shape[0]]
    if gp is not None:
        pieces = []
        off = 0
        for cnt in counts:
            p, q = gp[off:off + cnt], gq[off:off + cnt]
            order = np.lexsort((p[:, 2], p[:, 1], p[:, 0], q[:, 0]))
            pieces += [p[order], q[order]]
            off += cnt
        out["mpi.gather_ghosts" + tag] = sha(*pieces)
    pos3, pay3 = mig.rebalance(pos2, pay2)
    c = canon(pos3, pay3)
    if c is not None:
        out["mpi.rebalance" + tag] = sha(*c)

    halo = core_mpi.ParticleHalo(origin=origin, extent=extent, cells=cells,
                                 periodic=[True, True, False])
    ng = halo.build(pos3, 0.35, include_periodic_self=(size == 1))
    fpos = halo.forward_positions(pos3)
    fval = halo.forward(pay3)
    ghost_field = np.ascontiguousarray(fpos * 0.5 + 1.0)
    acc = halo.reverse(ghost_field, np.zeros((pos3.shape[0], 3)))
    # Owned rows keep the canonical id order; ghost rows are sorted per rank (ids in column 0 of
    # the forwarded payload), in rank order.
    gp, gv = gather_rows(comm, fpos), gather_rows(comm, fval)
    counts = comm.gather(int(ng), root=0) if comm is not None else [int(ng)]
    ca = canon(acc, pay3)
    if gp is not None:
        pieces = []
        off = 0
        for cnt in counts:
            p, v = gp[off:off + cnt], gv[off:off + cnt]
            order = np.lexsort((p[:, 2], p[:, 1], p[:, 0], v[:, 0]))
            pieces += [p[order], v[order]]
            off += cnt
        out["mpi.halo_forward" + tag] = sha(*pieces)
        out["mpi.halo_reverse" + tag] = sha(*ca)


# ---------------------------------------------------------------------------------------------
# AMR — Octree refine/balance/adapt, Poisson, Flow (ghost projection; mixed-level sampled band),
# DistributedOctree + distributed Flow at np > 1.
# ---------------------------------------------------------------------------------------------
def import_amr():
    from peclet.core import amr
    return amr


def run_amr(out, comm):
    amr = import_amr()
    size, rank = (comm.size, comm.rank) if comm is not None else (1, 0)

    def sph(x, y, z, c=(0.5, 0.5, 0.5), r=0.22):
        return ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) ** 0.5 - r

    if rank == 0:
        # ---- Octree: refine to a sphere, balance, geometry, Löhner adapt ----
        t = amr.Octree(cells=[32, 32, 32], lmax=2, origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
        t.refine_to_sphere(center=[0.5, 0.5, 0.5], radius=0.22, target_level=0, band=1.0,
                           balance=False)
        t.balance()
        out["amr.octree"] = sha(t.centers(), t.sizes(), t.levels(), t.codes())
        f = 2.0 + np.tanh((t.centers()[:, 0] - 0.5) / 0.05)
        ind = t.lohner_indicator(f, eps=0.01)
        f2 = t.adapt(f, refine_thresh=0.2, coarsen_thresh=0.05, finest_level=0)
        out["amr.adapt"] = sha(ind, f2, t.centers(), t.levels())

        # ---- Poisson multigrid on a graded octree ----
        tg = amr.Octree(cells=[32, 32, 32], lmax=2, origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
        tg.refine_to_sphere(center=[0.5, 0.5, 0.5], radius=0.25, target_level=0, band=1.0)
        pg = amr.Poisson(tg, periodic=True)
        cc = tg.centers()
        k = 2 * np.pi
        ue = np.cos(k * cc[:, 0]) + np.sin(2 * k * cc[:, 1]) * np.cos(k * cc[:, 2])
        ue -= ue.mean()
        b = pg.apply(ue)
        u, r, ncyc = pg.solve(b, cycles=12, tol=0.0)
        out["amr.poisson"] = sha(b, u, np.array([r, float(ncyc)]))

        # ---- Flow, ghost projection (the default), uniform finest band around a sphere ----
        tf = amr.Octree(cells=[32, 32, 32], lmax=1, origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
        tf.refine_to_sdf(sph, target_level=0, band=3.0)
        fl = amr.Flow(tf, density=1.0, viscosity=0.05, dt=0.02)
        fl.set_advection(True)
        fl.set_solid(sph)
        fl.set_body_force(1.0, 0.0, 0.0)
        for _ in range(3):
            fl.step(mom_iters=60, pres_iters=40)
        out["amr.flow_ghost"] = sha(fl.velocities(), fl.pressure(), fl.face_field(),
                                    np.array([fl.divergence_norm()]))

        # ---- Flow, mixed-level sampled band: cut cells at TWO levels (graded refinement) ----
        ts = amr.Octree(cells=[32, 32, 32], lmax=1, origin=[0, 0, 0], extent=[1.0, 1.0, 1.0])
        ts.refine_to_sdf_graded(sph, lambda x, y, z: 0 if x < 0.5 else 1, band=2.0)
        fs = amr.Flow(ts, density=1.0, viscosity=0.05, dt=0.02)
        fs.set_ghost_sampled(True)
        fs.set_solid(sph)
        fs.set_body_force(1.0, 0.0, 0.0)
        for _ in range(3):
            fs.step(mom_iters=60, pres_iters=40)
        out["amr.flow_sampled"] = sha(fs.velocities(), fs.pressure(), fs.face_field(),
                                      np.array([fs.divergence_norm()]))

    # ---- DistributedOctree (collective; also meaningful at np=1) ----
    tag = f".np{size}"
    d = amr.DistributedOctree(cells=[32, 32, 32], lmax=2, origin=[0, 0, 0],
                              extent=[1.0, 1.0, 1.0], periodic=[True, True, True])
    d.refine_to_sphere(center=[0.5, 0.5, 0.5], radius=0.22, target_level=0, band=1.0,
                       balance=True)
    cen, lev = gather_rows(comm, d.centers()), gather_rows(comm, d.levels().reshape(-1, 1))
    if cen is not None:
        out["amr.distributed_octree" + tag] = sha(cen, lev)
    fields = np.column_stack([d.levels().astype(np.float64), d.centers()[:, 0]])
    moved = d.rebalance(fields)
    cen2, mv = gather_rows(comm, d.centers()), gather_rows(comm, moved)
    if cen2 is not None:
        out["amr.distributed_rebalance" + tag] = sha(cen2, mv)
    fa = 2.0 + np.tanh((d.centers()[:, 0] - 0.5) / 0.05)
    fa2 = d.adapt(fa, refine_thresh=0.2, coarsen_thresh=0.05, finest_level=0)
    cen3, f3 = gather_rows(comm, d.centers()), gather_rows(comm, fa2.reshape(-1, 1))
    if cen3 is not None:
        out["amr.distributed_adapt" + tag] = sha(cen3, f3)

    # ---- distributed Flow: the whole step multi-rank through the leaf halo ----
    df = amr.DistributedOctree(cells=[32, 32, 32], lmax=1, origin=[0, 0, 0],
                               extent=[1.0, 1.0, 1.0], periodic=[True, True, True])
    df.refine_to_sdf(sph, target_level=0, band=3.0, balance=True)
    fd = amr.Flow(df, density=1.0, viscosity=0.05, dt=0.02)
    fd.set_advection(True)
    fd.set_solid(sph)
    fd.set_body_force(1.0, 0.0, 0.0)
    for _ in range(3):
        fd.step(mom_iters=60, pres_iters=40)
    vc, vp = gather_rows(comm, df.centers()), gather_rows(comm, fd.velocities())
    pp = gather_rows(comm, fd.pressure().reshape(-1, 1))
    if vc is not None:
        # Slot order differs between partitions; sort by leaf centre so np=1 and np=2 hashes are
        # comparable across the same decomposition (the gate compares like with like anyway).
        order = np.lexsort((vc[:, 2], vc[:, 1], vc[:, 0]))
        out["amr.distributed_flow" + tag] = sha(vc[order], vp[order], pp[order])


RUNNERS = {"geom": run_geom, "mpi": run_mpi, "amr": run_amr}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modules", default="geom,mpi,amr", help="comma-separated subset of geom,mpi,amr")
    ap.add_argument("--save", metavar="FILE", help="write the hashes as JSON")
    ap.add_argument("--check", metavar="FILE", help="compare against a JSON recording")
    args = ap.parse_args()
    if os.environ.get("OMP_NUM_THREADS") != "1":
        sys.stderr.write("state_hash: run with OMP_NUM_THREADS=1 (device reductions are order-dependent)\n")
    comm = None
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        pass
    rank = comm.rank if comm is not None else 0
    out = {}
    for name in args.modules.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            RUNNERS[name](out, comm)
        except ImportError as e:
            if rank == 0:
                print(f"# {name}: not importable ({e}); skipped")
    if rank != 0:
        return 0
    for k in sorted(out):
        print(f"{k} {out[k]}")
    rc = 0
    if args.check:
        ref = json.load(open(args.check))
        for k in sorted(set(ref) | set(out)):
            if k not in out:
                print(f"MISSING {k}")
                rc = 1
            elif k not in ref:
                print(f"NEW {k}")
            elif ref[k] != out[k]:
                print(f"DIFFER {k}: {ref[k][:16]}... -> {out[k][:16]}...")
                rc = 1
        print("state_hash: " + ("IDENTICAL" if rc == 0 else "DIFFERENCES FOUND"))
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
