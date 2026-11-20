#include "CFV/Detail/LimitedVRKernels.hpp"
#include "CFV/Math/LimitedVRStencil.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV::Detail
{
    void BuildLimitedVRSeeds(LimitedVRStencilView stencil, Runtime::RowSelection selection, LVRField scalars, LVRField gates,
                             LVRInput previous, LVROutput output, LimitedVRKernelParameters settings)
    {
        Runtime::ValidateViewDevices(Runtime::Device::Host(), stencil, selection, scalars, gates, previous, output);
        OMP::ForRows(0, selection.RowCount(), [&](index iWork) noexcept
                     { Math::BuildLimitedVRSeedCell(stencil, scalars, gates, previous, output, settings, selection.RowAt(iWork)); });
    }
    void LimitedVRMaximum(LimitedVRStencilView stencil, Runtime::RowSelection selection, LVRField input, LVROutput output,
                          std::array<bool, 3> active, bool excludeBoundary, bool saturatedOnly, std::array<real, 3> caps)
    {
        Runtime::ValidateViewDevices(Runtime::Device::Host(), stencil, selection, input, output);
        OMP::ForRows(0, selection.RowCount(), [&](index iWork) noexcept
                     { Math::LimitedVRMaximumCell(stencil, input, output, selection.RowAt(iWork), active, excludeBoundary, saturatedOnly, caps); });
    }
}
