#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include "DNDS/Runtime/Storage/Buffer.hpp"

namespace DNDS::Cauchy::CUDA
{
    /** @brief Enqueue the owned-field reconstruction update norm.
     * @details The result is [sum((next-current)^2), sum(next^2)], with a fixed
     * row traversal and tree reduction. Scratch has width two and
     * max(1,min(256,ceil(rows/128))) rows; result has one row. The enclosing
     * operation retains all operands through completion. This wrapper does no
     * host wait or MPI; the controller observes norms and decides stopping
     * only at its explicit convergence boundary.
     */
    void ReconstructionUpdateNorm(Runtime::ExecutionContext,
                                  Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> current,
                                  Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> next,
                                  Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> partial,
                                  Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> result);
}
