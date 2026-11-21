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

    /// An auxiliary output must preserve both sections of the published packet,
    /// including the section reserved for another consumer of the same packet.
    inline void RequireDistinctBoundaryOutput(const Runtime::Array<real> &output, const BoundaryInputs &packet)
    {
        for (const auto *section : {&packet.FaceCenterValues(), &packet.QuadratureValues()})
            for (index field = 0; field < section->PhysicalFieldNumber(); ++field)
                DNDS_check_throw_info(!output.Storage().Overlaps(section->PhysicalField(field).Owned().Storage()),
                                      "auxiliary output aliases an interface packet");
    }
}
