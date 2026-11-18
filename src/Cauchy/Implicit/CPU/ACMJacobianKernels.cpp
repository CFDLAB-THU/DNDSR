#include "Cauchy/Implicit/CPU/ACMJacobianKernels.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy::CPU
{
    template <int C>
    static NumericalStatus Assemble(const ACMJacobianPreparation &request)
    {
        return OMP::ForRowsStatus(0, request.diagonal.RowCount(), [&](index cell)
                                  { return ACM::AssembleRow<C>(request, request.stencil, cell); });
    }
    NumericalStatus PrepareACMFluxJacobian(const ACMJacobianPreparation &request) noexcept
    {
        switch (request.input.ComponentNumber())
        {
        case 4:
            return Assemble<4>(request);
        case 5:
            return Assemble<5>(request);
        case 6:
            return Assemble<6>(request);
        default:
            return {NumericalError::InvalidInput};
        }
    }
}
