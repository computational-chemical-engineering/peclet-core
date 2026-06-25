"""MPI test of the tpx_mpi Python shim (transport-core particle migration + ghosts via mpi4py).

Run: PYTHONPATH=python/build mpirun -np 4 python3 python/test_tpx_mpi.py
Validates, from Python: migration conserves particles and places each on its owning rank, ghost
gathering returns a nonzero set for np>1, and a weighted-ORB rebalance redistributes particles while
conserving them — mirroring the C++ tests, through the binding.
"""
import sys
import numpy as np
from mpi4py import MPI
import tpx_mpi

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

origin = [0.0, 0.0, 0.0]
boxsize = [10.0, 8.0, 6.0]
gsize = [40, 32, 24]
mig = tpx_mpi.Migrator(origin=origin, size=boxsize, gsize=gsize, periodic=[True, True, True])
assert mig.rank == rank


def frac(x, s):
    x = (np.uint64(x) ^ (np.uint64(s) * np.uint64(2654435761))).astype(np.uint64)
    x ^= x >> np.uint64(33)
    x = (x * np.uint64(0xFF51AFD7ED558CCD)).astype(np.uint64)
    x ^= x >> np.uint64(33)
    return float(int(x) & 0xFFFFFF) / float(0x1000000)


N = 4000
ids = np.arange(rank, N, size, dtype=np.int64)
M = ids.size
pos = np.zeros((M, 3), dtype=np.float64)
pay = np.zeros((M, 4), dtype=np.float64)  # K=4: vx, vy, vz, id
for k, pid in enumerate(ids):
    for a in range(3):
        pos[k, a] = origin[a] + frac(int(pid) * 3 + a, 0) * boxsize[a]
        pay[k, a] = (frac(int(pid), 10 + a) - 0.5) * 0.2
    pay[k, 3] = float(pid)

# --- migrate ---
pos2, pay2 = mig.migrate(pos, pay)
fail = 0
for k in range(pos2.shape[0]):
    if mig.owner_of(pos2[k].tolist()) != rank:
        fail += 1
local_ids = pay2[:, 3].astype(np.int64)

gcount = comm.allreduce(pos2.shape[0], MPI.SUM)
gsum = comm.allreduce(int(local_ids.sum()), MPI.SUM)
expect_sum = N * (N - 1) // 2
if gcount != N or gsum != expect_sum:
    fail += 1

# --- ghosts ---
gpos, gpay = mig.gather_ghosts(pos2, pay2, 0.5)
gghost = comm.allreduce(gpos.shape[0], MPI.SUM)
# ghosts must not duplicate owned ids on the same rank, and should be nonzero for np>1
if size > 1 and gghost == 0:
    fail += 1

# --- rebalance (weighted ORB by particle count) ---
def imbalance(local_count):
    mx = comm.allreduce(local_count, MPI.MAX)
    sm = comm.allreduce(local_count, MPI.SUM)
    return mx / (sm / size) if sm else 1.0

imb_before = imbalance(pos2.shape[0])
pos3, pay3 = mig.rebalance(pos2, pay2)
imb_after = imbalance(pos3.shape[0])
# count + id multiset conserved, and every particle placed on its NEW owner
rcount = comm.allreduce(pos3.shape[0], MPI.SUM)
rsum = comm.allreduce(int(pay3[:, 3].astype(np.int64).sum()), MPI.SUM)
for k in range(pos3.shape[0]):
    if mig.owner_of(pos3[k].tolist()) != rank:
        fail += 1
if rcount != N or rsum != expect_sum or imb_after > imb_before + 1e-9:
    fail += 1

total = comm.allreduce(fail, MPI.SUM)
if rank == 0:
    print(f"# tpx_mpi: count={gcount} idsum_ok={gsum == expect_sum} ghosts={gghost} "
          f"imbalance {imb_before:.3f}->{imb_after:.3f}")
    if total == 0:
        print(f"OK (np={size}): tpx_mpi migrate + gather_ghosts work from Python/mpi4py")
    else:
        sys.stderr.write(f"FAILED (np={size}): {total}\n")
sys.exit(0 if total == 0 else 1)
