// FieldSet — named registry of structured fields. Verifies allocation/zeroing (add), aliasing
// (adopt), lookup, upsert, deterministic sorted enumeration, and the ownStorage flag.
#include "test_util.hpp"

#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "peclet/core/field/field_set.hpp"

namespace {
using peclet::core::Centering;
using peclet::core::FieldSet;
using peclet::core::View;

int host0(const View<double>& v, std::size_t i) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  return static_cast<int>(h(i));
}

void run() {
  FieldSet fs;

  // add(): allocates a fresh buffer of the requested size, zero-initialised.
  auto& t = fs.add("temperature", 100, 2, Centering::Cell);
  PECLET_CORE_CHECK(fs.has("temperature"));
  PECLET_CORE_CHECK_EQ((long long)t.data.extent(0), 100);
  PECLET_CORE_CHECK_EQ(t.ghost, 2);
  PECLET_CORE_CHECK(t.ownStorage);
  PECLET_CORE_CHECK_EQ(host0(t.data, 0), 0);  // zero-initialised

  // adopt(): aliases an externally-owned buffer (no copy) — writes are visible through the record.
  View<double> ext("ext", 50);
  {
    auto e = ext;  // capture for the device lambda
    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<>(0, 50), KOKKOS_LAMBDA(int i) { e(i) = 7.0; });
    Kokkos::fence();
  }
  auto& a = fs.adopt("aliased", ext, 1, Centering::FaceX);
  PECLET_CORE_CHECK(!a.ownStorage);
  PECLET_CORE_CHECK_EQ(a.centering == Centering::FaceX, true);
  PECLET_CORE_CHECK_EQ(host0(a.data, 3), 7);  // sees ext's data (aliased, not copied)

  // at() lookup + missing-name throws.
  PECLET_CORE_CHECK_EQ((long long)fs.at("aliased").data.extent(0), 50);
  bool threw = false;
  try {
    fs.at("nope");
  } catch (const std::out_of_range&) {
    threw = true;
  }
  PECLET_CORE_CHECK(threw);

  // names(): sorted (collective-safe enumeration).
  auto names = fs.names();
  PECLET_CORE_CHECK_EQ((long long)names.size(), 2);
  PECLET_CORE_CHECK(names[0] == "aliased" && names[1] == "temperature");

  // upsert: re-adopt replaces the record (e.g. after a redistribution reallocates the buffer).
  View<double> ext2("ext2", 80);
  fs.adopt("aliased", ext2, 3, Centering::Cell);
  PECLET_CORE_CHECK_EQ((long long)fs.at("aliased").data.extent(0), 80);
  PECLET_CORE_CHECK_EQ((long long)fs.size(), 2);  // still two names, not three
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  PECLET_CORE_RETURN_TEST_RESULT();
}
