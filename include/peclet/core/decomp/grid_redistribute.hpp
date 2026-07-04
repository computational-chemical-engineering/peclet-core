// core — redistribute structured grid fields between two ORB decompositions (load balancing).
//
// The Lagrangian half of dynamic load balancing already exists (weighted ORB + ParticleMigrator +
// rebalanceByParticleCount). This is the Eulerian counterpart: after a re-decomposition (e.g. a
// weighted ORB that moved the block boundaries), each rank's grid fields must move from the OLD
// block layout to the NEW one so a method code (flow) keeps a consistent partition shared with the
// particles. Pure data movement of the INNER cells — bit-exact by construction (no averaging); the
// caller refills ghosts with a halo exchange afterwards.
//
// Each field is a flat x-fastest buffer on the rank's LOCAL padded block: inner extent
// dec.block(rank).size, ghost width g, so inner cell (lx,ly,lz) sits at
// (lx+g) + (ly+g)*ex + (lz+g)*ex*ey with ex = size.x + 2g. `oldFields` live on the old block;
// `newFields` are caller-allocated on the NEW block's padded size. Redistributes N fields together
// (one message per rank pair carries every field's sub-box, laid out field-major). Host
// implementation (plain C++ + the NBX engine); a device-staged variant can wrap it later like
// GridHalo.
#ifndef PECLET_CORE_DECOMP_GRID_REDISTRIBUTE_HPP
#define PECLET_CORE_DECOMP_GRID_REDISTRIBUTE_HPP

#include <cstring>
#include <vector>

#include "peclet/core/common/mpi.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/nbx.hpp"

namespace peclet::core::decomp {

namespace detail {
// Intersection of two global cell boxes (origin,size); returns false if empty on any axis.
template <int Dim>
inline bool intersectBox(const Block<Dim>& a, const Block<Dim>& b, IVec<Dim>& origin,
                         IVec<Dim>& size) {
  for (int d = 0; d < Dim; ++d) {
    const Index lo = a.origin[d] > b.origin[d] ? a.origin[d] : b.origin[d];
    const Index ahi = a.origin[d] + a.size[d], bhi = b.origin[d] + b.size[d];
    const Index hi = ahi < bhi ? ahi : bhi;
    if (hi <= lo)
      return false;
    origin[d] = lo;
    size[d] = hi - lo;
  }
  return true;
}

// Flat x-fastest index of a global cell within a padded local block (ghost width g).
template <int Dim>
inline Index localFlat(const IVec<Dim>& gcell, const Block<Dim>& blk, int g) {
  Index stride = 1, idx = 0;
  for (int d = 0; d < Dim; ++d) {
    idx += (gcell[d] - blk.origin[d] + g) * stride;
    stride *= (blk.size[d] + 2 * g);
  }
  return idx;
}

inline Index boxCellCount(const IVec<3>& size) { return size[0] * size[1] * size[2]; }

// Visit every cell of a global box in x-fastest order: f(n, gcell) with n the box-linear index.
template <class F>
inline void forBox(const IVec<3>& origin, const IVec<3>& size, F&& f) {
  Index n = 0;
  for (Index z = 0; z < size[2]; ++z)
    for (Index y = 0; y < size[1]; ++y)
      for (Index x = 0; x < size[0]; ++x, ++n)
        f(n, IVec<3>{origin[0] + x, origin[1] + y, origin[2] + z});
}
}  // namespace detail

// Redistribute nFields grid fields from oldDec to newDec (this rank = `rank`). oldFields[f] is the
// old padded local buffer, newFields[f] the (caller-allocated) new padded local buffer. Dim==3.
template <class T>
void redistributeGridFields(const BlockDecomposer<3>& oldDec, const BlockDecomposer<3>& newDec,
                            int rank, int g, const std::vector<const T*>& oldFields,
                            const std::vector<T*>& newFields, MPI_Comm comm) {
  const int nF = static_cast<int>(oldFields.size());
  const Block<3> ob = oldDec.block(rank), nbSelf = newDec.block(rank);

  struct Task {
    int dst;
    IVec<3> origin, size;
  };
  std::vector<Task> sends;
  for (std::size_t d = 0; d < newDec.numBlocks(); ++d) {
    IVec<3> o, s;
    if (detail::intersectBox<3>(ob, newDec.block(d), o, s))
      sends.push_back({static_cast<int>(d), o, s});
  }

  // Self contribution: old local -> new local, directly.
  for (const auto& t : sends)
    if (t.dst == rank)
      detail::forBox(t.origin, t.size, [&](Index, const IVec<3>& c) {
        const Index si = detail::localFlat<3>(c, ob, g), di = detail::localFlat<3>(c, nbSelf, g);
        for (int f = 0; f < nF; ++f)
          newFields[f][di] = oldFields[f][si];
      });

  // Cross-rank: message = [origin(3), size(3)] then field-major box values (fp[f*cells + n]).
  halo::NbxEngine nbx(comm);
  std::size_t si = 0;
  nbx.exchange(
      [&](std::vector<char>& out) -> int {
        while (si < sends.size() && sends[si].dst == rank)
          ++si;
        if (si >= sends.size())
          return -1;
        const Task t = sends[si++];
        const Index cells = detail::boxCellCount(t.size);
        out.resize(sizeof(Index) * 6 + sizeof(T) * cells * nF);
        char* p = out.data();
        std::memcpy(p, t.origin.data(), sizeof(Index) * 3);
        std::memcpy(p + sizeof(Index) * 3, t.size.data(), sizeof(Index) * 3);
        T* fp = reinterpret_cast<T*>(p + sizeof(Index) * 6);
        detail::forBox(t.origin, t.size, [&](Index n, const IVec<3>& c) {
          const Index sidx = detail::localFlat<3>(c, ob, g);
          for (int f = 0; f < nF; ++f)
            fp[f * cells + n] = oldFields[f][sidx];
        });
        return t.dst;
      },
      [&](int, std::vector<char>& msg) {
        const char* p = msg.data();
        IVec<3> o, s;
        std::memcpy(o.data(), p, sizeof(Index) * 3);
        std::memcpy(s.data(), p + sizeof(Index) * 3, sizeof(Index) * 3);
        const Index cells = detail::boxCellCount(s);
        const T* fp = reinterpret_cast<const T*>(p + sizeof(Index) * 6);
        detail::forBox(o, s, [&](Index n, const IVec<3>& c) {
          const Index di = detail::localFlat<3>(c, nbSelf, g);
          for (int f = 0; f < nF; ++f)
            newFields[f][di] = fp[f * cells + n];
        });
      });
}

}  // namespace peclet::core::decomp

#endif  // PECLET_CORE_DECOMP_GRID_REDISTRIBUTE_HPP
