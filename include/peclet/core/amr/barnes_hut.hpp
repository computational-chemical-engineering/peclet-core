// core — Barnes–Hut tree over a BlockOctree (particle-tree mode).
//
// The same per-block octree that carries an AMR grid doubles as a general
// adaptive spatial tree. Here it is built top-down by particle insertion (refine
// any leaf holding more than one particle) over a single root cube (root brick
// 1x1x1), with per-node centre-of-mass / total-mass aggregates accumulated up the
// Morton hierarchy. A standard theta-criterion traversal then evaluates an
// approximate 1/r^2 (softened) interaction in O(N log N). This is the primitive a
// PBD / N-body consumer (e.g. dem) would use; theta = 0 recurses fully and
// reproduces the direct O(N^2) sum exactly.
//
// Header-only, guarded by PECLET_CORE_HAVE_MORTON. Serial/host (the device traversal is a
// follow-up; the aggregates + leaf arrays are already device-friendly).
#ifndef PECLET_CORE_AMR_BARNES_HUT_HPP
#define PECLET_CORE_AMR_BARNES_HUT_HPP

#ifdef PECLET_CORE_HAVE_MORTON

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <vector>

#include "peclet/core/amr/block_octree.hpp"
#include "peclet/core/amr/leaf_field.hpp"
#include "peclet/core/common/types.hpp"

namespace peclet::core::amr {

template <int Dim = 3, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class BarnesHut {
 public:
  using Octree = BlockOctree<Dim, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;
  using Coord = typename Octree::Coord;

  /// Build the tree from particle positions (+ masses) inside the world box
  /// described by `geo` (origin + h0; the box spans 2^lmax fine cells per axis).
  void build(const std::vector<Vec<Dim>>& pos, const std::vector<double>& mass,
             const AmrGeometry<Dim>& geo, unsigned lmax, double theta, double soft = 1e-3) {
    pos_ = pos;
    mass_ = mass;
    geo_ = geo;
    lmax_ = lmax;
    theta_ = theta;
    soft2_ = soft * soft;
    fineExt_ = Coord(Coord(1) << lmax_);
    octree_.init(IVec<Dim>{makeBrick()}, lmax_);

    // Top-down: refine any leaf holding more than one particle (until level 0).
    for (;;) {
      std::vector<int> cnt(static_cast<std::size_t>(octree_.numLeaves()), 0);
      for (std::size_t p = 0; p < pos_.size(); ++p)
        ++cnt[static_cast<std::size_t>(leafOf(p))];
      std::vector<Code> split;
      for (Index i = 0; i < octree_.numLeaves(); ++i)
        if (cnt[static_cast<std::size_t>(i)] > 1 && octree_.level(i) > 0)
          split.push_back(octree_.code(i));
      if (split.empty())
        break;
      std::sort(split.begin(), split.end());
      octree_.refineIf(
          [&](Code c, unsigned) { return std::binary_search(split.begin(), split.end(), c); });
    }

    // Leaf -> particle CSR.
    const Index nleaf = octree_.numLeaves();
    leafStart_.assign(static_cast<std::size_t>(nleaf) + 1, 0);
    std::vector<Index> ofLeaf(pos_.size());
    for (std::size_t p = 0; p < pos_.size(); ++p) {
      Index li = leafOf(p);
      ofLeaf[p] = li;
      ++leafStart_[static_cast<std::size_t>(li) + 1];
    }
    for (Index i = 0; i < nleaf; ++i)
      leafStart_[static_cast<std::size_t>(i) + 1] += leafStart_[static_cast<std::size_t>(i)];
    leafParts_.assign(pos_.size(), 0);
    std::vector<Index> cur(leafStart_.begin(), leafStart_.end() - 1);
    for (std::size_t p = 0; p < pos_.size(); ++p) {
      Index li = ofLeaf[p];
      leafParts_[static_cast<std::size_t>(cur[static_cast<std::size_t>(li)]++)] =
          static_cast<Index>(p);
    }

    // Node aggregates (mass + mass-weighted position) for every ancestor level.
    agg_.assign(lmax_ + 1, {});
    for (std::size_t p = 0; p < pos_.size(); ++p) {
      Index li = ofLeaf[p];
      Code lo = octree_.code(li);
      unsigned Ll = octree_.level(li);
      for (unsigned L = Ll; L <= lmax_; ++L) {
        Code anc = M::from_code(lo).ancestor(L).code();
        Agg& a = agg_[L][anc];
        a.m += mass_[p];
        for (int d = 0; d < Dim; ++d)
          a.com[d] += mass_[p] * pos_[p][d];
      }
    }
  }

  /// Approximate acceleration on particle `pi`.
  Vec<Dim> acceleration(Index pi) const {
    Vec<Dim> a{};
    walk(lmax_, Code(0), pi, a);
    return a;
  }

  std::vector<Vec<Dim>> accelerations() const {
    std::vector<Vec<Dim>> a(pos_.size());
    for (std::size_t p = 0; p < pos_.size(); ++p)
      a[p] = acceleration(static_cast<Index>(p));
    return a;
  }

  /// Direct O(N^2) reference acceleration on particle `pi` (for validation).
  Vec<Dim> accelerationDirect(Index pi) const {
    Vec<Dim> a{};
    for (std::size_t q = 0; q < pos_.size(); ++q)
      if (static_cast<Index>(q) != pi)
        addPair(pos_[static_cast<std::size_t>(pi)], pos_[q], mass_[q], a);
    return a;
  }

  const Octree& octree() const { return octree_; }

 private:
  struct Agg {
    double m = 0.0;
    std::array<double, Dim> com{};  // stores mass-weighted sum until normalized at use
  };

  std::array<Index, Dim> makeBrick() const {
    std::array<Index, Dim> b{};
    for (int d = 0; d < Dim; ++d)
      b[d] = 1;
    return b;
  }

  std::array<Coord, Dim> fineCoord(std::size_t p) const {
    std::array<Coord, Dim> c{};
    for (int d = 0; d < Dim; ++d) {
      long v = static_cast<long>(std::floor((pos_[p][d] - geo_.origin[d]) / geo_.h0));
      if (v < 0)
        v = 0;
      if (v >= static_cast<long>(fineExt_))
        v = static_cast<long>(fineExt_) - 1;
      c[d] = static_cast<Coord>(v);
    }
    return c;
  }
  Index leafOf(std::size_t p) const { return octree_.find(M::encode(fineCoord(p)).code()); }

  void addPair(const Vec<Dim>& x, const Vec<Dim>& y, double m, Vec<Dim>& a) const {
    double r2 = soft2_;
    Vec<Dim> dir{};
    for (int d = 0; d < Dim; ++d) {
      dir[d] = y[d] - x[d];
      r2 += dir[d] * dir[d];
    }
    double inv = 1.0 / (r2 * std::sqrt(r2));
    for (int d = 0; d < Dim; ++d)
      a[d] += m * dir[d] * inv;
  }

  bool isLeafNode(unsigned L, Code code, Index& li) const {
    li = octree_.find(code);
    return li >= 0 && octree_.code(li) == code && octree_.level(li) == L;
  }

  void walk(unsigned L, Code code, Index pi, Vec<Dim>& a) const {
    auto it = agg_[L].find(code);
    if (it == agg_[L].end())
      return;
    const Agg& g = it->second;

    Index li = -1;
    if (isLeafNode(L, code, li)) {
      for (Index k = leafStart_[static_cast<std::size_t>(li)];
           k < leafStart_[static_cast<std::size_t>(li) + 1]; ++k) {
        Index q = leafParts_[static_cast<std::size_t>(k)];
        if (q != pi)
          addPair(pos_[static_cast<std::size_t>(pi)], pos_[static_cast<std::size_t>(q)],
                  mass_[static_cast<std::size_t>(q)], a);
      }
      return;
    }

    // Internal node: try the multipole acceptance s/d < theta.
    const Vec<Dim>& x = pos_[static_cast<std::size_t>(pi)];
    Vec<Dim> com{};
    double r2 = soft2_;
    for (int d = 0; d < Dim; ++d) {
      com[d] = g.com[d] / g.m;
      double dd = com[d] - x[d];
      r2 += dd * dd;
    }
    const double width = geo_.h0 * static_cast<double>(Index(1) << L);
    if (width * width < theta_ * theta_ * r2) {
      addPair(x, com, g.m, a);
      return;
    }
    for (unsigned oct = 0; oct < Octree::octants; ++oct)
      walk(L - 1, M::from_code(code).child(L, oct).code(), pi, a);
  }

  Octree octree_;
  std::vector<Vec<Dim>> pos_;
  std::vector<double> mass_;
  AmrGeometry<Dim> geo_{};
  unsigned lmax_ = 0;
  Coord fineExt_ = 1;
  double theta_ = 0.5;
  double soft2_ = 1e-6;
  std::vector<Index> leafStart_;          // CSR offsets, length nleaf+1
  std::vector<Index> leafParts_;          // particle indices grouped by leaf
  std::vector<std::map<Code, Agg>> agg_;  // agg_[level][nodeOriginCode]
};

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_HAVE_MORTON
#endif  // PECLET_CORE_AMR_BARNES_HUT_HPP
