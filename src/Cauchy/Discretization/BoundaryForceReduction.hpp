#pragma once
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Boundary/BoundaryInputs.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Cauchy
{
    /// Reusable rank-local reduction of the actual RHS boundary momentum flux.
    /// Owns scratch and the small selected-zone metadata; never recomputes flux
    /// or thermodynamics. Calls are controller-serialized, on the bound context.
    /// Returns after a scalar observation drain; domain MPI remains with caller.
    class BoundaryForceReduction
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        explicit BoundaryForceReduction(Runtime::ExecutionContext);
        Vector3 Evaluate(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                         const FiniteVolumeStencilStorage &, const Runtime::Array<BoundaryFaceRange> &,
                         const Runtime::Array<real> &momentum,
                         const std::vector<int> &zones);
    };
}
