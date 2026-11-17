/// @file
/// @brief Input validation of the `peclet.halo` Python binding (python/halo_bindings.cpp only).
///
/// The core halo headers trust their caller: a non-positive extent or cell count, a NaN position
/// or an array of the wrong shape reached them unchecked and read or wrote past an array, binned
/// a particle into an arbitrary cell, or (a negative rcut) acted as |rcut|. The binding refuses
/// these as ValueError (std::invalid_argument) before they reach the headers. A valid value takes
/// exactly the path it took before. The message names the call, the parameter, the admissible
/// range with its dimension (lengths are in the caller's consistent length units, those of
/// `origin` and `extent`) and the offending value or shape.
///
/// The checks are local to the rank: they add no communication. An array refused on one rank
/// raises there before the collective call starts, while the other ranks enter it; run an MPI
/// driver under `python -m mpi4py` so that an uncaught exception aborts the job instead of
/// leaving the other ranks waiting. Binding-only: no core header includes this file.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace halo_checks {

inline std::string num(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%g", v);
  return b;
}
[[noreturn]] inline void fail(const char* who, const std::string& name, const std::string& rule,
                              double v) {
  throw std::invalid_argument(std::string(who) + ": " + name + " must be " + rule + ", got " +
                              num(v));
}
inline void require(bool ok, const char* who, const std::string& name, const std::string& rule,
                    double v) {
  if (!ok) {
    fail(who, name, rule, v);
  }
}
inline std::string component(const char* name, std::size_t i) {
  return std::string(name) + "[" + std::to_string(i) + "]";
}

/// "(10, 2)": the shape of a nanobind ndarray, for a message.
template <class Array>
std::string shapeOf(const Array& a) {
  std::string s = "(";
  for (std::size_t d = 0; d < a.ndim(); ++d) {
    s += (d > 0 ? ", " : "") + std::to_string(a.shape(d));
  }
  return s + (a.ndim() == 1 ? ",)" : ")");
}
/// Refuses an array of the wrong shape. Called only on failure, so a valid call builds no string.
template <class Array>
[[noreturn]] void failShape(const char* who, const char* name, const std::string& rule,
                            const Array& a) {
  throw std::invalid_argument(std::string(who) + ": " + name + " must be " + rule + ", got shape " +
                              shapeOf(a));
}

/// Each component finite (a coordinate: any sign).
inline void finiteEach(const char* who, const char* name, const std::array<double, 3>& a) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i])) {
      fail(who, component(name, i), "finite (length units)", a[i]);
    }
  }
}

/// The box and decomposition grid of a ParticleMigrator / ParticleHalo: a finite origin, side
/// lengths finite and > 0, at least one cell per axis, and a total cell count that fits in 64
/// bits (rebalance allocates one weight per cell; a wrapped product would size it wrongly).
/// `Int` is the integer type nanobind hands the constructor.
template <class Int>
void box(const char* who, const std::array<double, 3>& origin, const std::array<double, 3>& extent,
         const std::array<Int, 3>& cells) {
  finiteEach(who, "origin", origin);
  for (std::size_t i = 0; i < extent.size(); ++i) {
    require(std::isfinite(extent[i]) && extent[i] > 0.0, who, component("extent", i),
            "finite and > 0 (length units)", extent[i]);
  }
  std::int64_t total = 1;
  for (std::size_t i = 0; i < cells.size(); ++i) {
    require(cells[i] >= 1, who, component("cells", i), "an integer >= 1 (decomposition cells)",
            static_cast<double>(cells[i]));
    require(!__builtin_mul_overflow(total, static_cast<std::int64_t>(cells[i]), &total), who,
            component("cells", i), "such that the total cell count fits in 64 bits",
            static_cast<double>(cells[i]));
  }
}

/// A query position x: three finite coordinates.
inline void point(const char* who, const std::array<double, 3>& x) {
  finiteEach(who, "x", x);
}

/// rcut: a distance, finite and >= 0 (0 gives no ghosts).
inline void rcut(const char* who, double r) {
  require(std::isfinite(r) && r >= 0.0, who, "rcut", "finite and >= 0 (length units)", r);
}

/// An (N, 3) array of positions with every coordinate finite (they are binned to cells). Returns N.
template <class Array>
std::size_t positions(const char* who, const char* name, const Array& a) {
  if (a.ndim() != 2 || a.shape(1) != 3) {
    failShape(who, name, "a 2-D (N, 3) array", a);
  }
  const double* p = a.data();
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(p[i])) {  // the name is built only on failure: no allocation per element
      fail(who,
           std::string(name) + "[" + std::to_string(i / 3) + ", " + std::to_string(i % 3) + "]",
           "finite (length units)", p[i]);
    }
  }
  return static_cast<std::size_t>(a.shape(0));
}

/// positions (N, 3) finite and payload a 2-D (N, K) array with the same N (any K >= 0).
template <class Array>
void particles(const char* who, const Array& pos, const Array& pay) {
  const std::size_t n = positions(who, "positions", pos);
  if (pay.ndim() != 2 || static_cast<std::size_t>(pay.shape(0)) != n) {
    failShape(who, "payload",
              "a 2-D (N, K) array with N = " + std::to_string(n) + " (the rows of positions)", pay);
  }
}

/// What the row count of a ParticleHalo field is, for a refusal message.
inline constexpr const char* kOwnedRows = "N = num_owned, the count passed to the last build()";
inline constexpr const char* kGhostRows = "G = num_ghost, the count the last build() returned";

/// An (n, 3) field over a fixed particle set: `count` names what n is.
template <class Array>
void rows(const char* who, const char* name, const Array& a, std::size_t n, const char* count) {
  if (a.ndim() != 2 || static_cast<std::size_t>(a.shape(0)) != n || a.shape(1) != 3) {
    failShape(who, name, "a 2-D (" + std::to_string(n) + ", 3) array (" + count + ")", a);
  }
}

}  // namespace halo_checks
