#pragma once

#include "DNDS/Core/NumericalStatus.hpp"
#include "DNDS/Runtime/Array/ArrayView.hpp"

namespace DNDS::Cauchy::CPU
{
    /// Frozen diagonal blocks or independent diagonal entries. The facade
    /// validates matrix generations and retains both access scopes.
    struct DiagonalInverseRequest
    {
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            diagonal;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            inverse;
        index components = 0;
        bool fullBlock = false;
        int refinementSteps = 0;
        bool rowMajor = false;
    };

    /// @return Number of rows that needed the existing SVD fallback.
    /// Errors propagate to the enclosing local/collective failure policy;
    /// no matrix generation is published by this arithmetic kernel.
    NumericalResult<index> InvertLinearDiagonal(const DiagonalInverseRequest &request) noexcept;
}
