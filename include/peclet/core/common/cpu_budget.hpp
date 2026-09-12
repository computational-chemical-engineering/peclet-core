// core — how many CPUs this process may actually USE, as opposed to how many it can SEE.
//
// THE PROBLEM. Kokkos sizes its host backend from the CPUs it can see. A container — Colab,
// Binder, Docker, a Slurm cgroup — normally shows the whole host while granting a fraction of it
// through a cgroup CPU *quota*, and a quota is invisible to OpenMP: omp_get_max_threads() returns
// the host count, the pool spin-waits at every barrier, and the run collapses. Measured on the
// 1.0.0 CPU wheel (suite docs/SCALING_ISSUES.md issue 7), a 42-step quick start that takes 25.7 s
// on two CPUs did NOT finish in fifteen minutes with 48 CPUs visible and two CPUs of quota — and
// took 26.0 s with the pool bounded. The trap is silent and it is the first thing a Colab user
// hits.
//
// An affinity MASK is not the same thing and is not the trap: OpenMP honours a mask and sizes
// itself correctly. What follows therefore reads the quota, and takes the affinity count only as
// the ceiling.
//
// WHERE THE QUOTA LIVES. Not, in general, at /sys/fs/cgroup/cpu.max — that is the cgroup ROOT, and
// on a systemd host it says "max" while the quota sits on the process's own scope further down.
// A container usually has its own cgroup namespace, so there the root IS the container's cgroup
// and the short path happens to work; outside one it does not. So: resolve the process's cgroup
// from /proc/self/cgroup and walk UP, taking the tightest quota on the chain, because a limit set
// on an ancestor binds just as hard as one set on the leaf.
#ifndef PECLET_CORE_COMMON_CPU_BUDGET_HPP
#define PECLET_CORE_COMMON_CPU_BUDGET_HPP

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#if defined(__linux__)
#include <sched.h>
#endif

namespace peclet::core {

namespace detail {

/// CPUs in this process's affinity mask, or 0 if that cannot be determined.
inline int affinityCpus() {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0)
    return CPU_COUNT(&set);
#endif
  return 0;
}

/// First whitespace-separated fields of a file, or an empty string if it cannot be read.
inline std::string readFirstLine(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  if (in && std::getline(in, line))
    return line;
  return {};
}

/// The cgroup path of this process for `controller` ("" = the v2 unified hierarchy), e.g.
/// "/user.slice/user-1000.slice/session-3.scope". Empty if absent.
inline std::string cgroupPath(const std::string& procSelfCgroup, const std::string& controller) {
  std::ifstream in(procSelfCgroup);
  std::string line;
  while (std::getline(in, line)) {
    // "hierarchy-ID:controller-list:path"
    const auto c1 = line.find(':');
    if (c1 == std::string::npos)
      continue;
    const auto c2 = line.find(':', c1 + 1);
    if (c2 == std::string::npos)
      continue;
    const std::string ctrl = line.substr(c1 + 1, c2 - c1 - 1);
    const std::string path = line.substr(c2 + 1);
    if (controller.empty()) {
      if (ctrl.empty())  // "0::/..." — the v2 unified hierarchy
        return path;
    } else if (ctrl.find(controller) != std::string::npos) {
      return path;  // "N:cpu,cpuacct:/..."
    }
  }
  return {};
}

/// CPUs allowed by one cgroup-v2 `cpu.max` file ("max 100000" or "200000 100000"), rounded UP so a
/// fractional budget still gets one thread. 0 = no limit here (or no such file).
inline int quotaFromCpuMax(const std::string& file) {
  std::istringstream in(readFirstLine(file));
  std::string quota;
  long period = 0;
  if (!(in >> quota >> period) || quota == "max" || period <= 0)
    return 0;
  const long q = std::strtol(quota.c_str(), nullptr, 10);
  return q > 0 ? static_cast<int>((q + period - 1) / period) : 0;
}

/// The same for cgroup v1's pair of files in `dir`. 0 = no limit here.
inline int quotaFromCfs(const std::string& dir) {
  const long q = std::strtol(readFirstLine(dir + "/cpu.cfs_quota_us").c_str(), nullptr, 10);
  const long p = std::strtol(readFirstLine(dir + "/cpu.cfs_period_us").c_str(), nullptr, 10);
  if (q <= 0 || p <= 0)
    return 0;  // -1 is "no quota"
  return static_cast<int>((q + p - 1) / p);
}

/// The tightest CPU quota binding this process, walking from its own cgroup up to the root. 0 = no
/// quota anywhere. Split out from usableCpus() so a test can point it at a fixture tree.
inline int cgroupQuota(const std::string& cgroupRoot, const std::string& procSelfCgroup) {
  int best = 0;
  const auto tighten = [&best](int n) {
    if (n > 0 && (best == 0 || n < best))
      best = n;
  };

  // v2: <root>/<path>/cpu.max, then every ancestor up to <root>/cpu.max.
  std::string path = cgroupPath(procSelfCgroup, "");
  for (;;) {
    tighten(quotaFromCpuMax(cgroupRoot + path + "/cpu.max"));
    if (path.empty())
      break;
    const auto slash = path.rfind('/');
    path = (slash == std::string::npos) ? std::string{} : path.substr(0, slash);
  }

  // v1: <root>/cpu/<path>/cpu.cfs_quota_us, likewise up the chain.
  path = cgroupPath(procSelfCgroup, "cpu");
  for (;;) {
    tighten(quotaFromCfs(cgroupRoot + "/cpu" + path));
    if (path.empty())
      break;
    const auto slash = path.rfind('/');
    path = (slash == std::string::npos) ? std::string{} : path.substr(0, slash);
  }
  return best;
}

}  // namespace detail

/// How many CPUs this process may actually use: the affinity count, lowered by any cgroup quota
/// binding it. At least 1. Falls back to the affinity count alone where no quota can be read.
inline int usableCpus() {
  const int affinity = detail::affinityCpus();
  const int quota = detail::cgroupQuota("/sys/fs/cgroup", "/proc/self/cgroup");
  int n = affinity > 0 ? affinity : quota;
  if (quota > 0 && (n == 0 || quota < n))
    n = quota;
  return std::max(1, n);
}

/// The host thread count to hand a backend that has not been told one, or 0 for "say nothing".
///
/// 0 whenever the user has expressed a preference (OMP_NUM_THREADS, KOKKOS_NUM_THREADS) — their
/// choice is never overridden — and 0 when the budget is the whole machine, so that on an ordinary
/// workstation this function changes NOTHING: same thread count, same schedule, same results. It
/// speaks up only in the case it exists for, a quota narrower than the visible machine.
inline int defaultHostThreads() {
  for (const char* var : {"OMP_NUM_THREADS", "KOKKOS_NUM_THREADS"})
    if (const char* v = std::getenv(var); v && *v)
      return 0;
  const int usable = usableCpus();
  const int affinity = detail::affinityCpus();
  return (affinity > 0 && usable < affinity) ? usable : 0;
}

}  // namespace peclet::core

#endif  // PECLET_CORE_COMMON_CPU_BUDGET_HPP
