#pragma once

#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Implicit/LinearPattern.hpp"

namespace DNDS::Cauchy::CUDA
{
    /** @brief Build the local geometric aggregation graph for flow AmgX.
     * @details Weights share the supplied CSR slot order. Each face contributes
     * area divided by center-to-center distance in its common periodic image.
     * Physical boundaries and self-neighbors produce no edge. This completed
     * setup operation checks reciprocal strengths before returning the field.
     */
    Runtime::Array<real> BuildAMGGeometricWeights(Runtime::ExecutionContext context,
                                                  const FiniteVolumeStencilStorage::ReadAccess &stencil,
                                                  const LinearPatternStorage &pattern);
}
