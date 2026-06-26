// transport-core — cell-centered finite-volume Poisson on a BlockOctree, with a
// geometric multigrid built from the octree's own levels.
//
// This is the "grid + multigrid" use of the AMR octree (serial, host; the device
// and distributed paths build on the same operator later). The operator is a
// conservative two-point finite-volume Laplacian: for a leaf i,
//
//     (L u)_i = (1/V_i) * sum_faces  A_f / d_f * (u_j - u_i),
//
// with V_i the cell volume, A_f the shared face area, d_f the centre-to-centre
// normal distance. At a 2:1 interface the coarse side sums the flux over all
// 2^(Dim-1) fine neighbours (each with the fine face area), so the discretisation
// is conservative (the global integral of L u is zero on a periodic domain). The
// two-point gradient ignores the tangential centre offset, so the interface flux
// is first-order there — a documented limitation; a tangential-gradient
// correction (full 2nd-order AMR flux) is a follow-up.
//
// Multigrid: the hierarchy is the octree coarsened uniformly one level at a time
// (coarsenIf over all sibling groups). Restriction averages children -> parent;
// prolongation is piecewise-constant (correction scheme). Smoother is
// lexicographic Gauss-Seidel over the Z-order leaf slots. Periodic BCs only (the
// natural first target); the singular null space is fixed by mean removal.
//
// Header-only, guarded by TPX_HAVE_MORTON.
#ifndef TPX_AMR_POISSON_HPP
#define TPX_AMR_POISSON_HPP

#ifdef TPX_HAVE_MORTON

#include <array>
#include <cmath>
#include <vector>

#include "tpx/amr/block_octree.hpp"
#include "tpx/amr/leaf_field.hpp"
#include "tpx/common/types.hpp"

namespace tpx::amr {

/// Cell-centered FV Poisson operator on one (periodic) block octree.
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class AmrPoisson {
 public:
  using Octree = BlockOctree<Dim, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;
  using Coord = typename Octree::Coord;

  AmrPoisson() = default;
  AmrPoisson(const Octree& t, Real h0) { init(t, h0); }

  void init(const Octree& t, Real h0) {
    t_ = &t;
    h0_ = h0;
    alpha_.clear();
    hasOpen_ = false;
    for (int d = 0; d < Dim; ++d) fineExt_[d] = static_cast<Coord>(t.brick()[d] * (Index(1) << t.lmax()));
  }

  void setOrigin(const Vec<Dim>& o) { origin_ = o; }

  // ---- cut-cell openness (per-leaf per-face fluid fraction in [0,1]) -------
  static constexpr int kFaces = 2 * Dim;
  static int faceIndex(int axis, int dir) { return 2 * axis + (dir > 0 ? 0 : 1); }

  /// Openness of leaf `i`'s face on (axis,dir); 1 if no openness has been set.
  double faceOpenness(Index i, int axis, int dir) const {
    if (!hasOpen_) return 1.0;
    return alpha_[static_cast<std::size_t>(i) * kFaces + faceIndex(axis, dir)];
  }

  bool hasOpenness() const { return hasOpen_; }
  const std::vector<double>& opennessRaw() const { return alpha_; }
  void setOpennessRaw(std::vector<double> a) {
    alpha_ = std::move(a);
    hasOpen_ = true;
  }

  /// Build face openness from a geometry callable openFn(faceCentreWorld, axis) ->
  /// [0,1] (1 = fully fluid, 0 = fully solid). Evaluated at each face centroid, so
  /// same-level neighbours see an identical value (consistent shared faces).
  template <class OpenFn>
  void buildOpenness(OpenFn&& openFn) {
    const Index n = numLeaves();
    alpha_.assign(static_cast<std::size_t>(n) * kFaces, 1.0);
    hasOpen_ = true;
    for (Index i = 0; i < n; ++i) {
      auto b = t_->bounds(i);
      const auto& lo = b[0];
      const Coord s = Coord(Coord(1) << t_->level(i));
      for (int axis = 0; axis < Dim; ++axis)
        for (int dir = -1; dir <= 1; dir += 2) {
          const long plane = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(s)
                                       : static_cast<long>(lo[axis]);
          Vec<Dim> fc{};
          for (int d = 0; d < Dim; ++d)
            fc[d] = (d == axis) ? origin_[d] + static_cast<Real>(plane) * h0_
                                : origin_[d] + (static_cast<Real>(lo[d]) + 0.5 * static_cast<Real>(s)) * h0_;
          double a = static_cast<double>(openFn(fc, axis));
          a = a < 0.0 ? 0.0 : (a > 1.0 ? 1.0 : a);
          alpha_[static_cast<std::size_t>(i) * kFaces + faceIndex(axis, dir)] = a;
        }
    }
  }

  Index numLeaves() const { return t_->numLeaves(); }

  /// Boundary condition: periodic (default) wraps every face; non-periodic treats a
  /// domain-boundary face as a homogeneous Dirichlet wall at half a cell (see
  /// `boundaryDiag`), which `forEachFaceNeighbor` then skips (no neighbour cell).
  void setPeriodic(bool p) { periodic_ = p; }
  bool periodic() const { return periodic_; }

  /// Immersed no-slip (Dirichlet) wall mode. OFF (default) ⇒ the Neumann/openness operator
  /// (the pressure Poisson): a solid-adjacent face just loses its flux (α<1). ON ⇒ the
  /// *velocity* operator: the solid fraction (1−α) of every interior face is a u=0 wall at
  /// half a cell, folded into the diagonal (`boundaryDiag`). This is the one difference
  /// between the pressure and velocity discretisations on the same openness geometry, and it
  /// makes the velocity operator strongly diagonally dominant (the wall pins u) — the basis
  /// for the velocity multigrid (DeviceMultigrid built with immersedWall + Helmholtz mass).
  void setImmersedWall(bool w) { immersedWall_ = w; }
  bool immersedWall() const { return immersedWall_; }

  /// Σ over leaf `i`'s Dirichlet-wall faces of the wall weight A_f/(½·cellWidth), folded
  /// into the operator diagonal so a wall cell sees a u=0 wall at half a cell (making the
  /// operator non-singular). Two contributions: domain-boundary faces weighted by α (only
  /// when non-periodic), and — when `immersedWall_` — the solid fraction (1−α) of every
  /// interior face (the immersed no-slip wall of the velocity operator).
  double boundaryDiag(Index i) const {
    if (periodic_ && !immersedWall_) return 0.0;
    auto b = t_->bounds(i);
    const auto& lo = b[0];
    const Coord si = Coord(Coord(1) << t_->level(i));
    const double wall = areaOf(si) / (0.5 * static_cast<Real>(si) * h0_);
    double s = 0.0;
    for (int axis = 0; axis < Dim; ++axis)
      for (int dir = -1; dir <= 1; dir += 2) {
        const long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si)
                                  : static_cast<long>(lo[axis]) - 1;
        const bool domainBoundary = !periodic_ && (pc < 0 || pc >= static_cast<long>(fineExt_[axis]));
        if (domainBoundary)
          s += faceOpenness(i, axis, dir) * wall;  // domain Dirichlet wall (open part)
        else if (immersedWall_)
          s += (1.0 - faceOpenness(i, axis, dir)) * wall;  // immersed no-slip wall (solid part)
      }
    return s;
  }

  Real cellWidth(Index i) const { return h0_ * static_cast<Real>(Index(1) << t_->level(i)); }
  Real cellVolume(Index i) const {
    Real w = cellWidth(i);
    Real v = 1;
    for (int d = 0; d < Dim; ++d) v *= w;
    return v;
  }

  /// Visit each face neighbour of leaf `i`: fn(neighbourSlot, coeff, axis, alpha)
  /// where coeff = A_f / d_f (physical) and alpha is the face openness (fluid
  /// fraction, from the finer side). Periodic wrap; 2:1 interfaces enumerated.
  template <class Fn>
  void forEachFaceNeighbor(Index i, Fn&& fn) const {
    auto b = t_->bounds(i);
    const auto& lo = b[0];
    const unsigned Li = t_->level(i);
    const Coord si = Coord(Coord(1) << Li);
    for (int axis = 0; axis < Dim; ++axis)
      for (int dir = -1; dir <= 1; dir += 2) {
        const long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si)
                                  : static_cast<long>(lo[axis]) - 1;
        // Non-periodic: a domain-boundary face has no neighbour cell (it is a Dirichlet
        // wall handled by boundaryDiag) — skip it.
        if (!periodic_ && (pc < 0 || pc >= static_cast<long>(fineExt_[axis]))) continue;
        std::array<Coord, Dim> p = lo;
        p[axis] = wrap(pc, axis);
        Index j = t_->find(M::encode(p).code());
        const unsigned Lj = t_->level(j);
        if (Lj >= Li) {
          // same level or coarser: one neighbour, shared face = this cell's face.
          // Openness lives on the finer side (here, this cell i).
          fn(j, coeff(si, Coord(Coord(1) << Lj)), axis, faceOpenness(i, axis, dir));
        } else {
          // finer neighbour: 2^(Dim-1) sub-faces, each the fine face area.
          const Coord sj = Coord(si >> 1);
          const int nsub = 1 << (Dim - 1);
          for (int k = 0; k < nsub; ++k) {
            std::array<Coord, Dim> q = lo;
            q[axis] = wrap(pc, axis);
            int bit = 0;
            for (int t = 0; t < Dim; ++t) {
              if (t == axis) continue;
              const Coord off = ((k >> bit) & 1) ? sj : Coord(0);
              q[t] = wrap(static_cast<long>(lo[t]) + static_cast<long>(off), t);
              ++bit;
            }
            Index jj = t_->find(M::encode(q).code());
            // Fine face area (min = sj) but the true centre-to-centre distance
            // (si+sj)/2 — same value the fine side computes, so the operator is
            // symmetric / conservative across the 2:1 interface. Openness lives on
            // the finer side (the neighbour jj), its face toward i is -dir.
            fn(jj, coeff(si, sj), axis, faceOpenness(jj, axis, -dir));
          }
        }
      }
  }

  /// Like forEachFaceNeighbor but exposes geometry for a consistent FV
  /// divergence/gradient: fn(neighbour, axis, dir, areaPhys, distPhys, alpha).
  /// Same face enumeration (2:1 sub-faces) and openness as the operator, so a
  /// collocated projection built on it stays consistent with the pressure Poisson.
  template <class Fn>
  void forEachFaceFull(Index i, Fn&& fn) const {
    auto b = t_->bounds(i);
    const auto& lo = b[0];
    const unsigned Li = t_->level(i);
    const Coord si = Coord(Coord(1) << Li);
    for (int axis = 0; axis < Dim; ++axis)
      for (int dir = -1; dir <= 1; dir += 2) {
        const long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si)
                                  : static_cast<long>(lo[axis]) - 1;
        std::array<Coord, Dim> p = lo;
        p[axis] = wrap(pc, axis);
        Index j = t_->find(M::encode(p).code());
        const unsigned Lj = t_->level(j);
        if (Lj >= Li) {
          const Coord sj = Coord(Coord(1) << Lj);
          fn(j, axis, dir, areaOf(si), 0.5 * (static_cast<Real>(si) + static_cast<Real>(sj)) * h0_,
             faceOpenness(i, axis, dir));
        } else {
          const Coord sj = Coord(si >> 1);
          const int nsub = 1 << (Dim - 1);
          for (int k = 0; k < nsub; ++k) {
            std::array<Coord, Dim> q = lo;
            q[axis] = wrap(pc, axis);
            int bit = 0;
            for (int t = 0; t < Dim; ++t) {
              if (t == axis) continue;
              const Coord off = ((k >> bit) & 1) ? sj : Coord(0);
              q[t] = wrap(static_cast<long>(lo[t]) + static_cast<long>(off), t);
              ++bit;
            }
            Index jj = t_->find(M::encode(q).code());
            fn(jj, axis, dir, areaOf(sj), 0.5 * (static_cast<Real>(si) + static_cast<Real>(sj)) * h0_,
               faceOpenness(jj, axis, -dir));
          }
        }
      }
  }

  /// Periodic face neighbour leaf (covering the cell just across the face).
  Index periodicNeighbor(Index i, int axis, int dir) const {
    auto b = t_->bounds(i);
    const auto& lo = b[0];
    const Coord si = Coord(Coord(1) << t_->level(i));
    const long pc = (dir > 0) ? static_cast<long>(lo[axis]) + static_cast<long>(si)
                              : static_cast<long>(lo[axis]) - 1;
    std::array<Coord, Dim> p = lo;
    p[axis] = wrap(pc, axis);
    return t_->find(M::encode(p).code());
  }

  /// Quadratic coarse-fine value: the coarse leaf `coarse`'s field, evaluated by
  /// tangential quadratic interpolation at the tangential position of fine leaf
  /// `fine` (Martin–Cartwright). Replacing the raw coarse value with this in the
  /// two-point flux makes the C/F flux 2nd-order; both sides of the face use the
  /// identical value, so the operator stays symmetric/conservative (refluxing is
  /// automatic). Falls back to the raw value on any tangential axis whose coarse
  /// neighbours aren't both same-level.
  double coarseStar(const std::vector<double>& u, Index coarse, Index fine, int axis) const {
    const double uc = u[static_cast<std::size_t>(coarse)];
    auto bc = t_->bounds(coarse);
    auto bf = t_->bounds(fine);
    const double H = cellWidth(coarse);
    const double sc = static_cast<double>(Index(1) << t_->level(coarse));
    const double sf = static_cast<double>(Index(1) << t_->level(fine));
    double val = uc;
    for (int t = 0; t < Dim; ++t) {
      if (t == axis) continue;
      const double dt = ((static_cast<double>(bf[0][t]) + 0.5 * sf) -
                         (static_cast<double>(bc[0][t]) + 0.5 * sc)) * h0_;
      Index cp = periodicNeighbor(coarse, t, +1);
      Index cm = periodicNeighbor(coarse, t, -1);
      if (cp < 0 || cm < 0) continue;
      if (t_->level(cp) != t_->level(coarse) || t_->level(cm) != t_->level(coarse)) continue;
      // Skip the correction near a solid: a nearly-closed tangential face means
      // the quadratic stencil would lean on a solid-side value. Drop to the raw
      // coarse value on this axis (locally lower order, but robust).
      if (faceOpenness(coarse, t, +1) < 0.5 || faceOpenness(coarse, t, -1) < 0.5) continue;
      const double up = u[static_cast<std::size_t>(cp)];
      const double um = u[static_cast<std::size_t>(cm)];
      const double Dt = (up - um) / (2.0 * H);
      const double Dtt = (up - 2.0 * uc + um) / (H * H);
      val += dt * Dt + 0.5 * dt * dt * Dtt;
    }
    return val;
  }

  /// out = L u with the quadratic coarse-fine flux (2nd-order at 2:1 interfaces).
  void applyLaplacianQuad(const std::vector<double>& u, std::vector<double>& out) const {
    const Index n = numLeaves();
    out.assign(static_cast<std::size_t>(n), 0.0);
    for (Index i = 0; i < n; ++i) {
      const double ui = u[static_cast<std::size_t>(i)];
      const unsigned Li = t_->level(i);
      double acc = 0.0;
      forEachFaceNeighbor(i, [&](Index j, Real c, int axis, double a) {
        const unsigned Lj = t_->level(j);
        double uj = u[static_cast<std::size_t>(j)];
        double uii = ui;
        if (Lj > Li)
          uj = coarseStar(u, j, i, axis);   // j coarser: correct its value
        else if (Lj < Li)
          uii = coarseStar(u, i, j, axis);  // i coarser: correct our value for this sub-face
        acc += a * c * (uj - uii);
      });
      out[static_cast<std::size_t>(i)] = acc / cellVolume(i);
    }
  }

  /// L2 norm of rhs - L_quad u.
  double residualQuad(const std::vector<double>& u, const std::vector<double>& rhs,
                      std::vector<double>& res) const {
    std::vector<double> lu;
    applyLaplacianQuad(u, lu);
    const Index n = numLeaves();
    res.assign(static_cast<std::size_t>(n), 0.0);
    double s = 0.0;
    for (Index i = 0; i < n; ++i) {
      double r = rhs[static_cast<std::size_t>(i)] - lu[static_cast<std::size_t>(i)];
      res[static_cast<std::size_t>(i)] = r;
      s += cellVolume(i) * r * r;
    }
    return std::sqrt(s);
  }

  /// out = L u (periodic FV Laplacian).
  void applyLaplacian(const std::vector<double>& u, std::vector<double>& out) const {
    const Index n = numLeaves();
    out.assign(static_cast<std::size_t>(n), 0.0);
    for (Index i = 0; i < n; ++i) {
      const double ui = u[static_cast<std::size_t>(i)];
      double acc = 0.0;
      forEachFaceNeighbor(i, [&](Index j, Real c, int, double a) {
        acc += a * c * (u[static_cast<std::size_t>(j)] - ui);
      });
      out[static_cast<std::size_t>(i)] = acc / cellVolume(i);
    }
  }

  /// res = rhs - L u, returns its L2 norm (sqrt(sum V_i res_i^2)).
  double residual(const std::vector<double>& u, const std::vector<double>& rhs,
                  std::vector<double>& res) const {
    const Index n = numLeaves();
    res.assign(static_cast<std::size_t>(n), 0.0);
    double s = 0.0;
    for (Index i = 0; i < n; ++i) {
      const double ui = u[static_cast<std::size_t>(i)];
      double acc = 0.0;
      forEachFaceNeighbor(i, [&](Index j, Real c, int, double a) {
        acc += a * c * (u[static_cast<std::size_t>(j)] - ui);
      });
      double r = rhs[static_cast<std::size_t>(i)] - acc / cellVolume(i);
      res[static_cast<std::size_t>(i)] = r;
      s += cellVolume(i) * r * r;
    }
    return std::sqrt(s);
  }

  /// `sweeps` lexicographic Gauss-Seidel relaxations of L u = rhs (in place).
  void gaussSeidel(std::vector<double>& u, const std::vector<double>& rhs, int sweeps) const {
    const Index n = numLeaves();
    for (int s = 0; s < sweeps; ++s)
      for (Index i = 0; i < n; ++i) {
        double sumOff = 0.0, diag = 0.0;
        forEachFaceNeighbor(i, [&](Index j, Real c, int, double a) {
          sumOff += a * c * u[static_cast<std::size_t>(j)];
          diag += a * c;
        });
        if (diag != 0.0)
          u[static_cast<std::size_t>(i)] = (sumOff - cellVolume(i) * rhs[static_cast<std::size_t>(i)]) / diag;
      }
  }

  /// Subtract the volume-weighted mean (fixes the periodic null space).
  void removeMean(std::vector<double>& u) const {
    const Index n = numLeaves();
    double sum = 0.0, vol = 0.0;
    for (Index i = 0; i < n; ++i) {
      sum += cellVolume(i) * u[static_cast<std::size_t>(i)];
      vol += cellVolume(i);
    }
    double m = sum / vol;
    for (Index i = 0; i < n; ++i) u[static_cast<std::size_t>(i)] -= m;
  }

  const Octree& octree() const { return *t_; }
  Real h0() const { return h0_; }

 private:
  Coord wrap(long c, int axis) const {
    long e = static_cast<long>(fineExt_[axis]);
    return static_cast<Coord>(((c % e) + e) % e);
  }
  // A_f / d_f (physical) for a cell of width-units `si` next to one of `sj`.
  Real coeff(Coord si, Coord sj) const {
    Real dist = 0.5 * (static_cast<Real>(si) + static_cast<Real>(sj)) * h0_;
    return areaOf(si < sj ? si : sj) / dist;
  }
  // Physical area of a face of a cell of width-units `s`: (s*h0)^(Dim-1).
  Real areaOf(Coord s) const {
    Real area = 1;
    for (int d = 0; d < Dim - 1; ++d) area *= static_cast<Real>(s) * h0_;
    return area;
  }

  const Octree* t_ = nullptr;
  Real h0_ = 1.0;
  std::array<Coord, Dim> fineExt_{};
  Vec<Dim> origin_{};
  std::vector<double> alpha_;  // per-leaf per-face openness (kFaces per leaf), or empty
  bool hasOpen_ = false;
  bool periodic_ = true;  // false ⇒ homogeneous Dirichlet domain walls (boundaryDiag)
  bool immersedWall_ = false;  // true ⇒ velocity operator: (1−α) interior faces are no-slip walls
};

/// Geometric multigrid for AmrPoisson over a uniformly-coarsened octree hierarchy.
template <int Dim, unsigned Bits = (Dim == 2 ? 32u : (Dim == 3 ? 21u : 16u))>
class AmrMultigrid {
 public:
  using Octree = BlockOctree<Dim, Bits>;
  using M = typename Octree::M;
  using Code = typename Octree::Code;

  /// Build the hierarchy from a finest octree by uniform coarsening until a single
  /// leaf remains (or no full sibling group can be merged).
  void build(const Octree& finest, Real h0) {
    levels_.clear();
    levels_.push_back(finest);
    for (;;) {
      Octree c = levels_.back();
      Index merged = c.coarsenIf([](Code, unsigned) { return true; });
      if (merged == 0 || c.numLeaves() == levels_.back().numLeaves()) break;
      levels_.push_back(c);
      if (c.numLeaves() == 1) break;
    }
    ops_.resize(levels_.size());
    // All levels share the finest h0: a coarse octree's leaves carry a higher
    // `level`, and cellWidth = h0 * 2^level already encodes the doubled width.
    for (std::size_t L = 0; L < levels_.size(); ++L) ops_[L].init(levels_[L], h0);
    // child(fine slot) -> parent(coarse slot) for each fine/coarse pair.
    c2p_.assign(levels_.size() ? levels_.size() - 1 : 0, {});
    for (std::size_t L = 0; L + 1 < levels_.size(); ++L) {
      const Octree& f = levels_[L];
      const Octree& c = levels_[L + 1];
      c2p_[L].resize(static_cast<std::size_t>(f.numLeaves()));
      for (Index i = 0; i < f.numLeaves(); ++i) {
        Code parent = M::from_code(f.code(i)).ancestor(f.level(i) + 1).code();
        c2p_[L][static_cast<std::size_t>(i)] = c.find(parent);
      }
    }
  }

  std::size_t numLevels() const { return levels_.size(); }
  const AmrPoisson<Dim, Bits>& op(std::size_t L = 0) const { return ops_[L]; }

  /// Apply a boundary condition to every level (periodic default, or non-periodic
  /// homogeneous Dirichlet). Call after build().
  void setPeriodic(bool p) {
    for (auto& o : ops_) o.setPeriodic(p);
  }

  /// Enable the immersed no-slip (Dirichlet) wall on every level — the velocity operator.
  /// Call after setOpenness() (the wall is derived per level from that level's coarsened
  /// aperture, so it is consistent down the hierarchy). Call after build().
  void setImmersedWall(bool w) {
    for (auto& o : ops_) o.setImmersedWall(w);
  }

  /// One V-cycle on level L solving L u = rhs (correction scheme).
  void vcycle(std::size_t L, std::vector<double>& u, const std::vector<double>& rhs, int pre = 2,
              int post = 2) {
    if (L + 1 == levels_.size()) {
      ops_[L].gaussSeidel(u, rhs, 40);  // coarsest: solve hard
      ops_[L].removeMean(u);
      return;
    }
    ops_[L].gaussSeidel(u, rhs, pre);
    std::vector<double> res;
    ops_[L].residual(u, rhs, res);

    // Restrict residual: volume-weighted average of children -> parent.
    const Octree& f = levels_[L];
    const Octree& c = levels_[L + 1];
    std::vector<double> crhs(static_cast<std::size_t>(c.numLeaves()), 0.0);
    std::vector<double> cvol(static_cast<std::size_t>(c.numLeaves()), 0.0);
    for (Index i = 0; i < f.numLeaves(); ++i) {
      Index p = c2p_[L][static_cast<std::size_t>(i)];
      if (p < 0) continue;
      Real v = ops_[L].cellVolume(i);
      crhs[static_cast<std::size_t>(p)] += v * res[static_cast<std::size_t>(i)];
      cvol[static_cast<std::size_t>(p)] += v;
    }
    for (Index p = 0; p < c.numLeaves(); ++p)
      if (cvol[static_cast<std::size_t>(p)] > 0) crhs[static_cast<std::size_t>(p)] /= cvol[static_cast<std::size_t>(p)];

    std::vector<double> ccorr(static_cast<std::size_t>(c.numLeaves()), 0.0);
    vcycle(L + 1, ccorr, crhs, pre, post);

    // Prolong correction (piecewise constant) and add.
    for (Index i = 0; i < f.numLeaves(); ++i) {
      Index p = c2p_[L][static_cast<std::size_t>(i)];
      if (p >= 0) u[static_cast<std::size_t>(i)] += ccorr[static_cast<std::size_t>(p)];
    }
    ops_[L].gaussSeidel(u, rhs, post);
    ops_[L].removeMean(u);
  }

  /// Solve the *quadratic* C/F operator L_quad u = rhs by deferred correction: the
  /// standard-operator V-cycle solves L_std u = rhs - (L_quad - L_std) u, with the
  /// cheap-to-evaluate quadratic correction lagged. Returns the final L_quad
  /// residual norm. `cyclesPerOuter` V-cycles per correction update.
  double solveQuad(std::vector<double>& u, const std::vector<double>& rhs, int outer = 30,
                   int cyclesPerOuter = 1) {
    AmrPoisson<Dim, Bits>& P = ops_[0];
    const std::size_t n = u.size();
    std::vector<double> lq, ls, rhsp(n), res;
    double r = 0.0;
    for (int o = 0; o < outer; ++o) {
      P.applyLaplacianQuad(u, lq);
      P.applyLaplacian(u, ls);
      for (std::size_t i = 0; i < n; ++i) rhsp[i] = rhs[i] - (lq[i] - ls[i]);
      for (int c = 0; c < cyclesPerOuter; ++c) vcycle(0, u, rhsp);
      r = P.residualQuad(u, rhs, res);
    }
    return r;
  }

  /// Set cut-cell face openness on the finest level from a geometry callable
  /// openFn(faceCentreWorld, axis) -> [0,1], then coarsen it to every coarser
  /// level by **area-averaging** the fine sub-faces (sdflow's coarsenOpenAvg). The
  /// per-level operators are thereby rediscretized with consistent openness, and
  /// the openness-weighted quadratic C/F flux applies on every level. Call after
  /// build().
  template <class OpenFn>
  void setOpenness(OpenFn&& openFn) {
    if (levels_.empty()) return;
    ops_[0].buildOpenness(openFn);
    for (std::size_t L = 0; L + 1 < levels_.size(); ++L) coarsenOpenness(L);
  }

 private:
  static int faceIdx(int axis, int dir) { return 2 * axis + (dir > 0 ? 0 : 1); }

  // Area-average the level-L face openness onto level L+1 (each coarse face is the
  // mean of the 2^(Dim-1) fine sub-faces covering it; equal areas => plain mean).
  void coarsenOpenness(std::size_t L) {
    const Octree& f = levels_[L];
    const Octree& c = levels_[L + 1];
    const int F = 2 * Dim;
    std::vector<double> ca(static_cast<std::size_t>(c.numLeaves()) * F, 0.0);
    std::vector<int> cnt(static_cast<std::size_t>(c.numLeaves()) * F, 0);
    for (Index i = 0; i < f.numLeaves(); ++i) {
      Index p = c2p_[L][static_cast<std::size_t>(i)];
      if (p < 0) continue;
      const std::size_t base = static_cast<std::size_t>(p) * F;
      if (c.level(p) == f.level(i)) {
        // identity (cell not coarsened this round): copy every face.
        for (int axis = 0; axis < Dim; ++axis)
          for (int dir = -1; dir <= 1; dir += 2) {
            int fi = faceIdx(axis, dir);
            ca[base + fi] += ops_[L].faceOpenness(i, axis, dir);
            cnt[base + fi] += 1;
          }
      } else {
        // merged child: each axis contributes its outward face to the parent face.
        unsigned oct = M::from_code(f.code(i)).child_index(f.level(i));
        for (int axis = 0; axis < Dim; ++axis) {
          int dir = ((oct >> axis) & 1) ? +1 : -1;
          int fi = faceIdx(axis, dir);
          ca[base + fi] += ops_[L].faceOpenness(i, axis, dir);
          cnt[base + fi] += 1;
        }
      }
    }
    for (std::size_t k = 0; k < ca.size(); ++k) ca[k] = cnt[k] ? ca[k] / cnt[k] : 1.0;
    ops_[L + 1].setOpennessRaw(std::move(ca));
  }

  std::vector<Octree> levels_;
  std::vector<AmrPoisson<Dim, Bits>> ops_;
  std::vector<std::vector<Index>> c2p_;
};

}  // namespace tpx::amr

#endif  // TPX_HAVE_MORTON
#endif  // TPX_AMR_POISSON_HPP
