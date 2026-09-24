"""Fixed-seed reference runs of every public entry path of the peclet.halo module, hashed.

The structural gate of suite/docs/QUALITY_PLAN.md §3.G: a refactor that moves code verbatim must
leave every final state BYTE-IDENTICAL. This script runs one deterministic scenario per public
entry path, hashes the final arrays (SHA-256 of the raw float64/int bytes) and prints them; with
``--save FILE`` it records them as JSON and with ``--check FILE`` it compares against a recording
and exits non-zero on any difference.

    PYTHONPATH=<python build tree> OMP_NUM_THREADS=1 python python/state_hash.py --save pre.json
    PYTHONPATH=<python build tree> OMP_NUM_THREADS=1 mpirun -np 2 python python/state_hash.py --check pre.json

Under ``mpirun -np N`` (N > 1) the distributed paths (ParticleMigrator / ParticleHalo across ranks)
run too; every per-rank array is gathered to rank 0 in
rank order before hashing, so the hash names carry the rank count (``.np2``). Run at
OMP_NUM_THREADS=1: the device reductions are order-dependent at more than one thread.

Only ``mpi`` here: the geom entry paths moved to the peclet-geom package on 2026-09-21
(suite/docs/CORE_BOUNDARY.md) and carry the same script, with hashes byte-identical across the
move. The AMR ones are in peclet-amr, likewise since 2026-09-10.
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
# peclet.halo — ParticleMigrator migrate / gather_ghosts / rebalance and ParticleHalo.
# ---------------------------------------------------------------------------------------------
def run_mpi(out, comm):
    from peclet import halo as core_mpi
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


RUNNERS = {"mpi": run_mpi}


def toolchain():
    """The modules' compiler / version / build type (`peclet.halo.build_toolchain`). Hashes are
    comparable only between builds of one toolchain (FMA contraction, optimisation level), so a
    reference recorded elsewhere is SKIPPED, not failed."""
    from peclet import halo as core_mpi
    return getattr(core_mpi, "build_toolchain", "unknown")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modules", default="mpi", help="comma-separated subset of: mpi (geom moved to peclet-geom)")
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
        want = ref.pop("toolchain", None)
        have = toolchain()
        if want is not None and want != have:
            print(f"state_hash: reference recorded with toolchain '{want}', this build is '{have}' — "
                  "not comparable; SKIPPED (exit 77). Re-record with --save on this toolchain to gate it.")
            return 77
        # A recording may merge several rank counts; compare only the keys this run can produce
        # (no `.npN` suffix, or the suffix of the current communicator size).
        size = comm.size if comm is not None else 1
        ref = {k: v for k, v in ref.items() if ".np" not in k or k.endswith(f".np{size}")}
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
        out["toolchain"] = toolchain()
        with open(args.save, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
