#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/Discretization/LimitedVREvaluation.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "CFV/Plans/LimitedVRPlan.hpp"
namespace DNDS::Cauchy::CPU
{
    NumericalStatus LimitLVRTarget(FiniteVolumeStencilView stencil, RHSScalarView means, RHSOutputView coefficients,
                                   int dimension, const ThermalBounds &bounds, int betaPower, real betaCutoff) noexcept;
    void FillLVRScalars(RHSScalarView input, RHSOutputView output, const PerfectGas &gas);
    /// Accepted affine target evaluated on site at both parent images of the
    /// shared face center. Output has one [shock,contact,density support] row
    /// per active face; physical boundary rows are zero. No sampled field.
    NumericalStatus BuildLVRFaceSensors(FiniteVolumeStencilView stencil, RHSFieldView means,
                                        RHSFieldView targetCoefficients, int dimension,
                                        const PerfectGas &, const ThermalBounds &, const CFV::LimitedVRSettings &,
                                        RHSOutputView sensors) noexcept;
    /// Deterministic cell gather through canonical cell-face incidences.
    /// Repeated self-periodic incidences retain their semantic multiplicity.
    void GatherLVRBranches(FiniteVolumeStencilView stencil, RHSScalarView sensors, RHSOutputView branches,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base);
    void MergeLVRBase(Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base,
                      RHSOutputView seeds, int column);
    void PublishLVRPenalty(RHSScalarView branches,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> history,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> penalty);
}
