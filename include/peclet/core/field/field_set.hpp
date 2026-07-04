// core — named registry of structured cell/face fields (the multiphysics field container).
//
// A method code (flow, and later coupling) accumulates an open-ended set of per-cell fields:
// velocity components, pressure, the SDF, transported scalars (temperature, concentration, phase
// fractions), and material-property fields (density, viscosity). Rather than hard-code each as a
// named member and hand-list them wherever the full set matters (halo exchange, load-balance
// redistribution), the FieldSet keeps them in one place keyed by name.
//
// It is a *directory*, not an owner of policy: it holds the device buffer, its ghost width and
// centering, and whether the FieldSet allocated it (add) or merely aliases a member the solver
// already owns (adopt). It carries NO halo objects — exchange stays per-call on the caller's
// topology (docs/INTERFACES.md: the Field concept is duck-typed on the exchanger side).
//
// Header-only, Kokkos-required (pulls in common/view.hpp). Payload type is fixed to double: Eulerian
// field state is double per docs/CONVENTIONS.md (§ precision policy).
#ifndef PECLET_CORE_FIELD_FIELD_SET_HPP
#define PECLET_CORE_FIELD_FIELD_SET_HPP

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "peclet/core/common/view.hpp"

namespace peclet::core {

/// Where a field's samples sit relative to a cell. Scalars/properties are cell-centred; the MAC
/// face-normal velocity components are face-centred. (The staggered solver stores its face fields as
/// cell-indexed arrays, so Centering is metadata for consumers — it does not change the buffer.)
enum class Centering { Cell, FaceX, FaceY, FaceZ };

/// One registered field: its flat x-fastest device buffer plus the metadata a consumer needs to
/// exchange or redistribute it. `ownStorage` distinguishes a FieldSet-allocated buffer (add) from an
/// aliased solver member (adopt) — redistribution reallocates the former and rebinds the latter.
struct FieldRec {
  View<double> data;
  int ghost = 0;
  Centering centering = Centering::Cell;
  bool ownStorage = false;
};

/// Name → FieldRec directory. Insertion is upsert (re-`add`/`adopt` of an existing name replaces the
/// record) so a solver can re-adopt its members after a redistribution reallocates them. `names()`
/// is sorted so every rank enumerates the set in the same order (required for collective
/// redistribution).
class FieldSet {
 public:
  /// Allocate a fresh zero-initialised device buffer of `n` elements and register it.
  FieldRec& add(const std::string& name, std::size_t n, int ghost,
                Centering c = Centering::Cell) {
    recs_[name] = FieldRec{View<double>(name, n), ghost, c, true};
    return recs_.at(name);
  }

  /// Register an existing buffer under `name` without taking ownership of its allocation.
  FieldRec& adopt(const std::string& name, View<double> v, int ghost,
                  Centering c = Centering::Cell) {
    recs_[name] = FieldRec{std::move(v), ghost, c, false};
    return recs_.at(name);
  }

  bool has(const std::string& name) const { return recs_.find(name) != recs_.end(); }

  FieldRec& at(const std::string& name) {
    auto it = recs_.find(name);
    if (it == recs_.end())
      throw std::out_of_range("FieldSet: no field named '" + name + "'");
    return it->second;
  }
  const FieldRec& at(const std::string& name) const {
    auto it = recs_.find(name);
    if (it == recs_.end())
      throw std::out_of_range("FieldSet: no field named '" + name + "'");
    return it->second;
  }

  std::size_t size() const { return recs_.size(); }

  /// Field names in a deterministic (sorted) order — the collective-safe enumeration.
  std::vector<std::string> names() const {
    std::vector<std::string> out;
    out.reserve(recs_.size());
    for (const auto& kv : recs_)
      out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
  }

 private:
  std::unordered_map<std::string, FieldRec> recs_;
};

}  // namespace peclet::core

#endif  // PECLET_CORE_FIELD_FIELD_SET_HPP
