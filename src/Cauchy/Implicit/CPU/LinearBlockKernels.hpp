#pragma once

#include "DNDS/Core/RowRanges.hpp"

#include "Cauchy/Implicit/LinearPattern.hpp"
#include "Cauchy/Implicit/OffDiagonalView.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"

namespace DNDS::Cauchy::CPU
{
    using LinearReadRows = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using LinearWriteRows = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using LinearIndexRows = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using LinearDiagonalRows = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using LinearDofRead = Algebra::DofComponentView<const real>;
    using LinearDofWrite = Algebra::DofComponentView<real>;

    /** @brief Frozen local matrix operands for one arithmetic call.
     * @details diagonal supplies D for Apply and D inverse for Jacobi/SGS.
     * rows is the already selected logical scan region. Every payload and
     * evaluator remains retained by the caller; no access is acquired here.
     */
    struct LinearBlockInput
    {
        OffDiagonalView blocks;
        LinearPatternView pattern;
        LinearIndexRows rows;
        LinearDiagonalRows diagonal;
        index components;
        bool fullDiagonal;
        /// Immutable setup metadata; used only by CPU SGS. Each parallel block
        /// reads frozen values across its physical-row interval boundaries.
        SpanView<const WorkerRowRanges> sweepPartitions;
        bool rowMajorBlocks = false;
    };
    /// Apply A to the supplied rows; output is distinct from both inputs.
    void ApplyLinearRows(const LinearBlockInput &, LinearReadRows owned, LinearReadRows ghost, LinearWriteRows output);
    /// Zero-start skips every input/off-diagonal read at compile-time dispatch.
    void JacobiLinearRows(const LinearBlockInput &, LinearReadRows rhs, LinearReadRows owned, LinearReadRows ghost, LinearWriteRows output, bool zeroStart);
    /// Ordered lower scan; newly computed local predecessors read output.
    void LowerSGSRows(const LinearBlockInput &, LinearReadRows rhs, LinearReadRows old, LinearReadRows ghost, LinearWriteRows output, bool zeroStart);
    /// Ordered upper scan; forward is the completed lower result.
    void UpperSGSRows(const LinearBlockInput &, LinearReadRows rhs, LinearReadRows old, LinearReadRows forward, LinearReadRows ghost, LinearWriteRows output, bool zeroStart);
    /// Apply A to the supplied rows; output is distinct from both inputs.
    void ApplyLinearRows(const LinearBlockInput &, LinearDofRead owned, LinearDofRead ghost, LinearDofWrite output);
    /// Zero-start skips every input/off-diagonal read at compile-time dispatch.
    void JacobiLinearRows(const LinearBlockInput &, LinearDofRead rhs, LinearDofRead owned, LinearDofRead ghost, LinearDofWrite output, bool zeroStart);
    /// Ordered lower scan; newly computed local predecessors read output.
    void LowerSGSRows(const LinearBlockInput &, LinearDofRead rhs, LinearDofRead old, LinearDofRead ghost, LinearDofWrite output, bool zeroStart);
    /// Ordered upper scan; forward is the completed lower result.
    void UpperSGSRows(const LinearBlockInput &, LinearDofRead rhs, LinearDofRead old, LinearDofRead forward, LinearDofRead ghost, LinearDofWrite output, bool zeroStart);
    /// Fill the rank-local ILU template before the generic LU factorizer runs.
    void FillLinearFactor(LinearDiagonalRows diagonal, OffDiagonalView blocks, Algebra::BlockMatrixView<real> factor,
                          LinearColumnsView sourceBlocks,
                          index components, bool fullDiagonal, bool rowMajorBlocks = false);
}
