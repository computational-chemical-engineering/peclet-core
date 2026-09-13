// peclet::core CPU budget (peclet/core/common/cpu_budget.hpp) — the cgroup quota a container
// imposes, which Kokkos would otherwise never see.
//
// Checks, against FIXTURE cgroup trees written into the test's own directory (the real
// /sys/fs/cgroup says whatever this machine says, and a test may not depend on that):
//  - cgroup v2 cpu.max: "max <period>" is no limit; "<quota> <period>" rounds UP so a fractional
//    budget still earns one thread;
//  - a quota on an ANCESTOR binds, and the TIGHTEST limit on the chain wins;
//  - cgroup v1's cfs pair, including -1 for "no quota";
//  - no cgroup files at all leaves the affinity count alone;
//  - usableCpus() on the real machine is >= 1 and never exceeds the affinity count;
//  - defaultHostThreads() says NOTHING (0) when OMP_NUM_THREADS is set AND the host backend's own
//    runtime will read it (OpenMP) — the user's choice stands; and passes it on when nothing else
//    will read it (the C++ threads / Serial backends a Windows or macOS wheel carries).
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "peclet/core/common/cpu_budget.hpp"
#include "test_util.hpp"

namespace fs = std::filesystem;
using namespace peclet::core;

static void write(const fs::path& p, const std::string& text) {
  fs::create_directories(p.parent_path());
  std::ofstream(p) << text << "\n";
}

int main() {
  const fs::path root = fs::temp_directory_path() / "peclet_cpu_budget_test";
  fs::remove_all(root);

  // --- 1. cgroup v2, quota on the leaf ------------------------------------------------------
  {
    const fs::path cg = root / "v2";
    write(cg / "proc_self_cgroup", "0::/user.slice/app.scope");
    write(cg / "sys" / "cpu.max", "max 100000");  // root: no limit
    write(cg / "sys/user.slice/cpu.max", "max 100000");
    write(cg / "sys/user.slice/app.scope/cpu.max", "200000 100000");  // 2 CPUs
    PECLET_CORE_CHECK_EQ(
        detail::cgroupQuota((cg / "sys").string(), (cg / "proc_self_cgroup").string()), 2);
  }

  // --- 2. the tightest limit on the chain wins, wherever it sits ----------------------------
  {
    const fs::path cg = root / "ancestor";
    write(cg / "proc_self_cgroup", "0::/slice/deep/app.scope");
    write(cg / "sys/cpu.max", "max 100000");
    write(cg / "sys/slice/cpu.max", "150000 100000");  // 1.5 -> 2 CPUs
    write(cg / "sys/slice/deep/cpu.max", "max 100000");
    write(cg / "sys/slice/deep/app.scope/cpu.max", "800000 100000");  // 8 CPUs, looser
    PECLET_CORE_CHECK_EQ(
        detail::cgroupQuota((cg / "sys").string(), (cg / "proc_self_cgroup").string()), 2);
  }

  // --- 3. a fractional budget still earns one thread, never zero ----------------------------
  {
    const fs::path cg = root / "fraction";
    write(cg / "proc_self_cgroup", "0::/app.scope");
    write(cg / "sys/app.scope/cpu.max", "50000 100000");  // half a CPU
    write(cg / "sys/cpu.max", "max 100000");
    PECLET_CORE_CHECK_EQ(
        detail::cgroupQuota((cg / "sys").string(), (cg / "proc_self_cgroup").string()), 1);
  }

  // --- 4. cgroup v1, and -1 meaning "no quota" ----------------------------------------------
  {
    const fs::path cg = root / "v1";
    write(cg / "proc_self_cgroup", "4:cpu,cpuacct:/docker/abc");
    write(cg / "sys/cpu/docker/abc/cpu.cfs_quota_us", "400000");
    write(cg / "sys/cpu/docker/abc/cpu.cfs_period_us", "100000");  // 4 CPUs
    write(cg / "sys/cpu/cpu.cfs_quota_us", "-1");
    write(cg / "sys/cpu/cpu.cfs_period_us", "100000");
    PECLET_CORE_CHECK_EQ(
        detail::cgroupQuota((cg / "sys").string(), (cg / "proc_self_cgroup").string()), 4);
  }

  // --- 5. nothing to read is not a limit ----------------------------------------------------
  {
    const fs::path cg = root / "empty";
    write(cg / "proc_self_cgroup", "0::/nowhere");
    PECLET_CORE_CHECK_EQ(
        detail::cgroupQuota((cg / "sys").string(), (cg / "proc_self_cgroup").string()), 0);
  }

  // --- 6. the real machine: sane, and never more than the affinity mask ----------------------
  {
    const int usable = usableCpus();
    const int affinity = detail::affinityCpus();
    PECLET_CORE_CHECK(usable >= 1);
    if (affinity > 0)
      PECLET_CORE_CHECK(usable <= affinity);
  }

  // --- 7. an explicit OMP_NUM_THREADS is never overridden, and never dropped either ------------
  //
  // On the OpenMP backend the runtime reads it, so we say nothing. On the C++ threads and Serial
  // backends -- what a Windows or macOS wheel carries -- NOTHING reads it (Kokkos itself reads only
  // KOKKOS_NUM_THREADS), so we have to pass it on or the user's setting vanishes silently.
  {
    setenv("OMP_NUM_THREADS", "3", 1);
    PECLET_CORE_CHECK_EQ(defaultHostThreads(true), 0);
    PECLET_CORE_CHECK_EQ(defaultHostThreads(false), 3);
    setenv("OMP_NUM_THREADS", "", 1);                  // empty: no preference expressed
    PECLET_CORE_CHECK_EQ(defaultHostThreads(false) >= 0, true);
    setenv("OMP_NUM_THREADS", "not-a-number", 1);      // garbage: fall through, do not pass 0 on
    PECLET_CORE_CHECK_EQ(defaultHostThreads(false), 0);
    unsetenv("OMP_NUM_THREADS");

    // KOKKOS_NUM_THREADS is Kokkos' own, on every backend: stay quiet for both.
    setenv("KOKKOS_NUM_THREADS", "5", 1);
    PECLET_CORE_CHECK_EQ(defaultHostThreads(true), 0);
    PECLET_CORE_CHECK_EQ(defaultHostThreads(false), 0);
    unsetenv("KOKKOS_NUM_THREADS");

    // Unset, on an unconstrained machine, it also stays quiet (usable == affinity).
    if (detail::cgroupQuota("/sys/fs/cgroup", "/proc/self/cgroup") == 0) {
      PECLET_CORE_CHECK_EQ(defaultHostThreads(true), 0);
      PECLET_CORE_CHECK_EQ(defaultHostThreads(false), 0);
    }
  }

  fs::remove_all(root);
  return ::peclet::core::test::g_failures == 0 ? 0 : 1;
}
