#!/usr/bin/env bash
# MPI-header manifest gate (suite docs/CORE_BOUNDARY.md §1.1, §2.1): a header is on the MPI side
# iff it includes peclet/core/common/mpi.hpp (or <mpi.h>). This script holds that manifest and
# fails if the tree drifts from it in either direction — a non-manifest header picking up the
# include, or a manifest header dropping it — so the boundary stays explicit and gated rather than
# merely documented. Run from the repo root: tools/check_mpi_manifest.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The manifest: every header (relative to include/peclet/core/) that is legitimately MPI-side.
# common/mpi.hpp itself is excluded by design -- it is the include being searched for, not a
# consumer of it.
MANIFEST=(
  # Coarse-level multigrid stages: host-staged, MPI only (core/CLAUDE.md).
  decomp/gather_by_global_id.hpp
  decomp/grid_redistribute.hpp
  decomp/redistribute_topology.hpp
  decomp/stage_comm.hpp

  # Grid/particle halo exchange: the async ghost-layer engines and their topology builders.
  halo/grid_halo.hpp
  halo/grid_halo_topology.hpp
  halo/nbx.hpp
  halo/particle_halo.hpp
  halo/particle_halo_topology.hpp
  halo/particle_migrator.hpp
  halo/particle_migrator_view.hpp
  halo/particle_rebalance.hpp

  # Anderson acceleration: the MPI side of the accelerator (core/CLAUDE.md).
  solver/anderson_mpi.hpp
)

INCLUDE_DIR="include/peclet/core"
fail=0

# (a) every manifest path must exist and must still include common/mpi.hpp or <mpi.h>.
for rel in "${MANIFEST[@]}"; do
  path="$INCLUDE_DIR/$rel"
  if [ ! -f "$path" ]; then
    echo "MPI manifest: listed header does not exist: $path" >&2
    fail=1
    continue
  fi
  if ! grep -qE '#include\s*[<"](peclet/core/common/mpi\.hpp|mpi\.h)[>"]' "$path"; then
    echo "MPI manifest: $path is on the manifest but no longer includes common/mpi.hpp or <mpi.h>" >&2
    fail=1
  fi
done

# (b) every header that includes common/mpi.hpp or <mpi.h> (other than common/mpi.hpp itself)
# must be on the manifest.
actual_file=$(mktemp)
manifest_file=$(mktemp)
trap 'rm -f "$actual_file" "$manifest_file"' EXIT

grep -rlE '#include\s*[<"](peclet/core/common/mpi\.hpp|mpi\.h)[>"]' "$INCLUDE_DIR" \
  | grep -v "^$INCLUDE_DIR/common/mpi\.hpp\$" \
  | sort >"$actual_file"

for rel in "${MANIFEST[@]}"; do
  echo "$INCLUDE_DIR/$rel"
done | sort >"$manifest_file"

extra=$(comm -23 "$actual_file" "$manifest_file")
if [ -n "$extra" ]; then
  while IFS= read -r f; do
    echo "MPI manifest: $f includes common/mpi.hpp or <mpi.h> but is not on the manifest" >&2
  done <<<"$extra"
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "MPI manifest check FAILED. Update tools/check_mpi_manifest.sh (and docs/CORE_BOUNDARY.md" >&2
  echo "in the umbrella) if this header is deliberately MPI-side; otherwise remove the include." >&2
  exit 1
fi

echo "MPI manifest check passed (${#MANIFEST[@]} headers)."
