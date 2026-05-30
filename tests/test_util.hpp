// Minimal dependency-free test helpers. A test binary returns non-zero on first failure so ctest
// (and `mpirun`) report it.
#ifndef TPX_TEST_UTIL_HPP
#define TPX_TEST_UTIL_HPP

#include <cstdio>
#include <cstdlib>

namespace tpx::test {
inline int g_failures = 0;
}

#define TPX_CHECK(cond)                                                              \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++::tpx::test::g_failures;                                                    \
    }                                                                              \
  } while (0)

#define TPX_CHECK_EQ(a, b)                                                          \
  do {                                                                              \
    auto _a = (a);                                                                  \
    auto _b = (b);                                                                  \
    if (!(_a == _b)) {                                                              \
      std::fprintf(stderr, "CHECK_EQ failed: %s == %s  (%lld vs %lld)\n  at %s:%d\n", \
                   #a, #b, (long long)_a, (long long)_b, __FILE__, __LINE__);       \
      ++::tpx::test::g_failures;                                                    \
    }                                                                              \
  } while (0)

#define TPX_RETURN_TEST_RESULT()                                                    \
  do {                                                                              \
    if (::tpx::test::g_failures == 0) {                                            \
      std::printf("OK\n");                                                          \
      return 0;                                                                     \
    }                                                                              \
    std::fprintf(stderr, "%d failure(s)\n", ::tpx::test::g_failures);              \
    return 1;                                                                       \
  } while (0)

#endif  // TPX_TEST_UTIL_HPP
