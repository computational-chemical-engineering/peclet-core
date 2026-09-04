// core — uniform Kokkos teardown for every peclet Python extension module (nanobind).
//
// THE PROBLEM. Kokkos::finalize() has to run from a Python `atexit` hook: the alternative — Kokkos's
// own static destructors, which run after the CUDA runtime has unloaded — aborts every CUDA process
// at exit with cudaErrorCudartUnloading. But Python runs atexit hooks BEFORE it tears down module
// globals, so any bound object (a Solver / Simulation / Tessellation at script or notebook scope)
// or any zero-copy array (a capsule holding a View) that is still referenced when the hook runs
// destroys its Kokkos View AFTER finalize, and Kokkos::Impl::SharedAllocationRecord::decrement calls
// Kokkos::abort — `Kokkos allocation "x" is being deallocated after Kokkos::finalize was called` +
// a backtrace, SIGABRT / exit 134 — on OpenMP exactly as on CUDA. That is the normal situation in a
// plain script, in `python -c`, and in every Jupyter / Quarto kernel; `del obj; gc.collect()` before
// exit was the only way out, and it does not exist for a notebook.
//
// THE PATTERN (the same in flow, dem, voro, pnm, coupling and core.amr):
//   * every binding-side owner of Kokkos Views derives from Releasable: its constructor registers
//     it in this module's live registry, its destructor unregisters it, release() drops its Views.
//     A wrapper that cannot drop its state piecemeal releases by destroying its own bound instance
//     (destruct_bound_instance): nanobind runs the C++ destructor now and marks the Python object
//     uninitialized, so a later method call raises a TypeError instead of touching freed memory,
//     and Python's final dealloc skips the destructor.
//   * every zero-copy capsule made by view_to_ndarray (ndarray_interop.hpp) is a Releasable too: it
//     keeps a reference to the exported View and release() drops it. The array's memory is gone
//     from then on — after shutdown it is only ever touched by Python's own dealloc, never read.
//   * a C++ class that keeps its own registry (dem's Simulation::releaseAll) plugs in via add_hook.
//   * install(m) initializes Kokkos, registers ONE atexit hook per module that runs release_all()
//     and THEN Kokkos::finalize(), exposes it as m.finalize() for deterministic teardown (idempotent;
//     after it, every bound object and zero-copy array of the module is dead), and publishes
//     m.execution_space.
// Each extension module statically links its own Kokkos (libkokkoscore.a), so the registry, the
// hook and the finalize are per module: modules tear down independently, in whatever order.
//
// Included by binding translation units only (they link nanobind + Kokkos); never by kernels.
#ifndef PECLET_CORE_PYTHON_KOKKOS_TEARDOWN_HPP
#define PECLET_CORE_PYTHON_KOKKOS_TEARDOWN_HPP

#include <nanobind/nanobind.h>

#include <functional>
#include <Kokkos_Core.hpp>
#include <set>
#include <vector>

namespace peclet::core::python {

namespace nb = nanobind;

/// Base of everything on the binding side that owns Kokkos Views on Python's behalf. Registration
/// is automatic (constructor/destructor); release() must leave the object safe to destroy after
/// Kokkos::finalize, i.e. holding no View that still has an allocation.
class Releasable {
 public:
  Releasable() noexcept { registry().insert(this); }
  Releasable(const Releasable&) noexcept { registry().insert(this); }
  Releasable(Releasable&&) noexcept { registry().insert(this); }
  Releasable& operator=(const Releasable&) noexcept { return *this; }
  Releasable& operator=(Releasable&&) noexcept { return *this; }
  virtual ~Releasable() { registry().erase(this); }
  virtual void release() noexcept = 0;

  /// This module's live registry. Heap-allocated and never destroyed, so the capsule deleters
  /// Python runs during its final module teardown can still unregister safely.
  static std::set<Releasable*>& registry() {
    static auto* s = new std::set<Releasable*>;
    return *s;
  }
};

/// Extra release callbacks (e.g. a C++ class's own `releaseAll`), run after the registry.
inline std::vector<std::function<void()>>& release_hooks() {
  static auto* v = new std::vector<std::function<void()>>;
  return *v;
}
inline void add_release_hook(std::function<void()> f) { release_hooks().push_back(std::move(f)); }

/// Drop every Kokkos View owned on Python's behalf in this module (registry + hooks). Idempotent.
/// Iterates a snapshot: a release() may destroy its object and thereby unregister it.
inline void release_all() noexcept {
  auto& reg = Releasable::registry();
  std::vector<Releasable*> snapshot(reg.begin(), reg.end());
  for (Releasable* r : snapshot)
    if (reg.count(r))
      r->release();
  for (auto& h : release_hooks())
    h();
}

/// release_all() and then Kokkos::finalize(). This is the module's atexit hook and its `finalize()`
/// function; safe to call more than once.
inline void shutdown() noexcept {
  release_all();
  if (Kokkos::is_initialized() && !Kokkos::is_finalized())
    Kokkos::finalize();
}

/// Release a bound wrapper by destroying its C++ instance in place (for types whose Kokkos state
/// cannot be dropped member by member). Runs the destructor now; nanobind marks the Python object
/// uninitialized (later calls raise TypeError, final dealloc skips the destructor). No-op if `self`
/// is not (or no longer) a live bound instance.
template <class T>
inline void destruct_bound_instance(T* self) noexcept {
  nb::handle h = nb::find(self);
  if (h.is_valid())
    nb::inst_destruct(h);
}

/// Module setup: initialize Kokkos (once per module — each module carries its own static Kokkos),
/// register the atexit shutdown, expose `finalize()` and `execution_space`.
inline void install(nb::module_& m) {
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
  m.def("finalize", &shutdown,
        "Release every live object and zero-copy array of this module, then Kokkos::finalize() "
        "(deterministic teardown; also run automatically at interpreter exit). Idempotent. After "
        "it, the module's solver objects and *_view arrays must not be used.");
  m.attr("execution_space") = nb::str(Kokkos::DefaultExecutionSpace::name());
  nb::module_::import_("atexit").attr("register")(nb::cpp_function(&shutdown));
}

}  // namespace peclet::core::python

#endif  // PECLET_CORE_PYTHON_KOKKOS_TEARDOWN_HPP
