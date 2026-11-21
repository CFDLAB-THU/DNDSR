#pragma once

#include "Cauchy/Boundary/BoundaryInputs.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    /// Borrow the packet section indexed by this consumer's face stencil.
    /// A direct P1 receiver can also consume the center section of a modal donor packet.
    inline const Runtime::Array<real> &BoundaryFluxInput(const FiniteVolumeStencilStorage &stencil,
                                                         const BoundaryInputs &packet)
    {
        const auto &section = stencil.quadratureBoundaryValues ? packet.QuadratureValues() : packet.FaceCenterValues();
        return section.CoupledField().Owned();
    }
}
