// core — the assembled face-CSR row kernels, LIFTED to peclet/core/solver/face_csr.hpp
// (2026-09-10, suite/docs/QUALITY_PLAN.md G.2) so voro's mesh optimiser and the AMR package share
// one copy. This header keeps every AMR spelling resolving: the names below are the same entities.
#ifndef PECLET_CORE_AMR_FACE_CSR_HPP
#define PECLET_CORE_AMR_FACE_CSR_HPP

#include "peclet/core/solver/face_csr.hpp"

namespace peclet::core::amr {

using solver::faceCsrApplyRow;
using solver::faceCsrOffDiag;
using solver::FaceCsrOpT;
using solver::faceCsrPointUpdate;
using solver::fvApplyRow;
using solver::FvCsrOpT;
using solver::fvPointSolve;
using solver::HostArr;

}  // namespace peclet::core::amr

#endif  // PECLET_CORE_AMR_FACE_CSR_HPP
