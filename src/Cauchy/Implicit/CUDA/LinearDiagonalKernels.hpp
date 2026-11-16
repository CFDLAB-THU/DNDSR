#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Cauchy::CUDA
{
    using DiagonalRead = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using DiagonalWrite = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using DiagonalStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using DiagonalInfo = Runtime::ArrayView<const int, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using DiagonalModes = Runtime::ArrayView<int, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using DiagonalCount = Runtime::ArrayView<index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    /** @brief Experimental fixed-size LU and exceptional-row SVD inversion.
     * @details Same row normalization and reciprocal-condition guard as CPU.
     * All fallback decisions and arithmetic remain device-resident; the two
     * kernels are ordered on the supplied stream with no host observation.
     * fallbackMask is row scratch; fallbackCount is a reset, device-side count
     * of valid rows requiring SVD. Invalid input produces row status.
     * rowMajorBlocks describes native diagonal/inverse block order. Cross-block
     * Array layout is independent; no transpose or repacking buffer is created.
     */
    void InvertSmallBlocks(Runtime::ExecutionContext, DiagonalRead diagonal,
                           DiagonalWrite inverse, DiagonalStatus status,
                           DiagonalModes fallbackMask, DiagonalCount fallbackCount,
                           int components, int refinementSteps = 0, bool rowMajorBlocks = false);

    /// Prepare B=diag(1/D_ii)*D for batched SVD. Invalid rows emit status and
    /// feed identity to the provider, so invalid input never reaches its solver.
    /// Provider workspaces use column-major blocks regardless of native input
    /// order; preparation writes that normalization directly, without a mirror.
    void PrepareDiagonalSVD(Runtime::ExecutionContext, DiagonalRead diagonal,
                            DiagonalWrite normalized, DiagonalWrite scales, DiagonalStatus status, int components, bool rowMajorBlocks = false);
    /** @brief Recover D inverse from B=U*S*V^T using the native SVD filter.
     * @details Keep sigma_k > epsilon*sigma_max, invert retained modes and set
     * the others to zero, then right-multiply by diag(1/D_ii). This is the
     * existing HardEigen filter with svdTol=0, mode=0. All CUDA blocks use SVD
     * initially (including healthy blocks); CPU keeps its guarded LU/SVD path.
     * Each row reports numerical status and its number of truncated modes.
     * Optional refinement (0=off, up to 3 guarded corrections) preserves the
     * retained SVD subspace; it never restores discarded singular modes.
     */
    void CompleteDiagonalSVD(Runtime::ExecutionContext, DiagonalRead diagonal, DiagonalRead singularValues,
                             DiagonalRead left, DiagonalRead right, DiagonalRead scales,
                             DiagonalInfo info, DiagonalWrite inverse, DiagonalStatus status,
                             DiagonalModes truncated, int components, int refinementSteps = 0, bool rowMajorBlocks = false);
    /// Independent diagonal-entry inverse, with no provider/workspace needed.
    void InvertDiagonalEntries(Runtime::ExecutionContext, DiagonalRead diagonal,
                               DiagonalWrite inverse, DiagonalStatus status, int components);
}
