#pragma once

#include "DNDS/Algebra/BlockMatrixStorage.hpp"

namespace DNDS::Cauchy::CUDA
{
    using LinearReadRows = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using LinearWriteRows = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using VectorReadRows = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using LinearIndexRows = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    /** @brief Frozen assembled operator for a selected set of independent rows.
     * @details columns and blocks have identical sparse incidence ordering.
     * Diagonal rows contain D for matrix action and D inverse for Jacobi, with
     * column-major full blocks or one value per component. Selected rows are
     * unique. Core selection proves all columns owned; its ghost view may be
     * absent. No stream, lease, communicator or host evaluator is stored here.
     */
    struct LinearBlockInput
    {
        Algebra::BlockMatrixArrayView<const real> blocks;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::CSRRows> columns;
        LinearIndexRows rows;
        LinearReadRows diagonal;
        index ownedRows = 0;
        int components = 0;
        bool fullDiagonal = false;
    };
    /// Enqueue y = A*x on selected rows. Output must not alias any input.
    /// Combined AoS widths 5/6/7 only; no communication or host wait is hidden.
    void ApplyLinearRows(Runtime::ExecutionContext context, const LinearBlockInput &request,
                         VectorReadRows owned, VectorReadRows ghost, LinearWriteRows output);
    /// Enqueue x_new = D^-1*(b - A_off*x_old), with frozen remote inputs.
    /// Zero-start consumes only rhs, inverse and row selection: columns,
    /// blocks, owned and ghost may all be absent and are never dereferenced.
    void JacobiLinearRows(Runtime::ExecutionContext context, const LinearBlockInput &request,
                          VectorReadRows rhs, VectorReadRows owned, VectorReadRows ghost,
                          LinearWriteRows output, bool zeroStart);
}
