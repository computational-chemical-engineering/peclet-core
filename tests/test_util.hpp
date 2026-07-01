// Minimal dependency-free test helpers. A test binary returns non-zero on first failure so ctest
// (and `mpirun`) report it.
#ifndef PECLET_CORE_TEST_UTIL_HPP
#define PECLET_CORE_TEST_UTIL_HPP

#include <cstdio>
#include <cstdlib>

namespace peclet::core::test {
inline int g_failures = 0;
}

#define PECLET_CORE_CHECK(cond)                                                              \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++::peclet::core::test::g_failures;                                                    \
    }                                                                              \
  } while (0)

#define PECLET_CORE_CHECK_EQ(a, b)                                                          \
  do {                                                                              \
    auto _a = (a);                                                                  \
    auto _b = (b);                                                                  \
    if (!(_a == _b)) {                                                              \
      std::fprintf(stderr, "CHECK_EQ failed: %s == %s  (%lld vs %lld)\n  at %s:%d\n", \
                   #a, #b, (long long)_a, (long long)_b, __FILE__, __LINE__);       \
      ++::peclet::core::test::g_failures;                                                    \
    }                                                                              \
  } while (0)

#define PECLET_CORE_RETURN_TEST_RESULT()                                                    \
  do {                                                                              \
    if (::peclet::core::test::g_failures == 0) {                                            \
      std::printf("OK\n");                                                          \
      return 0;                                                                     \
    }                                                                              \
    std::fprintf(stderr, "%d failure(s)\n", ::peclet::core::test::g_failures);              \
    return 1;                                                                       \
  } while (0)

#endif  // PECLET_CORE_TEST_UTIL_HPP
