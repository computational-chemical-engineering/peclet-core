// transport-core -- single-rank no-MPI stub. When the library is built without MPI (TPX_NO_MPI), the
// grid solver runs as ONE block: every ghost cell is a periodic self-copy onto the same block, so there
// are no remote neighbours. The only collectives ever EXECUTED at size 1 are the trivial ones --
// rank=0, size=1, Allreduce/Bcast are local identities. The point-to-point and neighbourhood-collective
// calls exist only so the single code path compiles and links; they are never reached at size 1.
//
// This keeps "one code": the same GridHaloTopology / GridHalo / NbxEngine run, with MPI replaced by
// these stubs. Build WITH MPI (the default) and this header is never included -- see tpx/common/mpi.hpp.
#pragma once

#include <cstdlib>
#include <cstring>

using MPI_Comm = int;
using MPI_Request = int;
using MPI_Datatype = int;  // value doubles as the element byte size (so Allreduce can memcpy)
using MPI_Op = int;
using MPI_Info = int;
struct MPI_Status {
  int MPI_SOURCE;
  int MPI_TAG;
};

#define MPI_COMM_WORLD 0
#define MPI_COMM_NULL (-1)
#define MPI_REQUEST_NULL 0
#define MPI_INFO_NULL 0
#define MPI_STATUS_IGNORE (static_cast<MPI_Status*>(nullptr))
#define MPI_STATUSES_IGNORE (static_cast<MPI_Status*>(nullptr))
#define MPI_ANY_SOURCE (-1)
#define MPI_UNDEFINED (-32766)
#define MPI_UNWEIGHTED (static_cast<int*>(nullptr))
#define MPI_SUCCESS 0
// datatypes carry their byte size (MPI_Allreduce copies count * datatype bytes)
#define MPI_BYTE 1
#define MPI_INT 4
#define MPI_LONG 8
#define MPI_DOUBLE 8
// reduction ops are ignored: at size 1 the local value IS the global value
#define MPI_SUM 0
#define MPI_MAX 1
#define MPI_MIN 2
#define MPI_LAND 3

inline int MPI_Init(int*, char***) { return MPI_SUCCESS; }
inline int MPI_Initialized(int* f) { *f = 1; return MPI_SUCCESS; }  // pretend initialised (no real MPI)
inline int MPI_Finalize() { return MPI_SUCCESS; }
inline int MPI_Finalized(int* f) { *f = 0; return MPI_SUCCESS; }
inline int MPI_Abort(MPI_Comm, int code) { std::abort(); return code; }
inline int MPI_Comm_rank(MPI_Comm, int* r) { *r = 0; return MPI_SUCCESS; }
inline int MPI_Comm_size(MPI_Comm, int* s) { *s = 1; return MPI_SUCCESS; }
inline int MPI_Comm_free(MPI_Comm*) { return MPI_SUCCESS; }
inline int MPI_Barrier(MPI_Comm) { return MPI_SUCCESS; }
// single rank: the local contribution is already the global result.
inline int MPI_Allreduce(const void* sbuf, void* rbuf, int count, MPI_Datatype dt, MPI_Op, MPI_Comm) {
  if (sbuf != rbuf) std::memcpy(rbuf, sbuf, static_cast<size_t>(count) * static_cast<size_t>(dt));
  return MPI_SUCCESS;
}
inline int MPI_Bcast(void*, int, MPI_Datatype, int, MPI_Comm) { return MPI_SUCCESS; }  // 1 rank: no-op

// --- never reached at size 1 (no remote neighbours); present only to compile/link the one code path ---
inline int MPI_Send(const void*, int, MPI_Datatype, int, int, MPI_Comm) { return MPI_SUCCESS; }
inline int MPI_Recv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status*) { return MPI_SUCCESS; }
inline int MPI_Isend(const void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request*) {
  return MPI_SUCCESS;
}
inline int MPI_Irecv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request*) { return MPI_SUCCESS; }
inline int MPI_Issend(const void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request*) {
  return MPI_SUCCESS;
}
inline int MPI_Iprobe(int, int, MPI_Comm, int* flag, MPI_Status*) { *flag = 0; return MPI_SUCCESS; }
inline int MPI_Get_count(const MPI_Status*, MPI_Datatype, int* c) { *c = 0; return MPI_SUCCESS; }
inline int MPI_Test(MPI_Request*, int* flag, MPI_Status*) { *flag = 1; return MPI_SUCCESS; }
inline int MPI_Testall(int, MPI_Request*, int* flag, MPI_Status*) { *flag = 1; return MPI_SUCCESS; }
inline int MPI_Wait(MPI_Request*, MPI_Status*) { return MPI_SUCCESS; }
inline int MPI_Waitall(int, MPI_Request*, MPI_Status*) { return MPI_SUCCESS; }
inline int MPI_Waitany(int, MPI_Request*, int* idx, MPI_Status*) {
  *idx = MPI_UNDEFINED;
  return MPI_SUCCESS;
}
inline int MPI_Ibarrier(MPI_Comm, MPI_Request*) { return MPI_SUCCESS; }
inline int MPI_Dist_graph_create_adjacent(MPI_Comm c, int, const int*, const int*, int, const int*,
                                          const int*, MPI_Info, int, MPI_Comm* out) {
  *out = c;
  return MPI_SUCCESS;
}
inline int MPI_Neighbor_alltoallv(const void*, const int*, const int*, MPI_Datatype, void*, const int*,
                                  const int*, MPI_Datatype, MPI_Comm) {
  return MPI_SUCCESS;
}
