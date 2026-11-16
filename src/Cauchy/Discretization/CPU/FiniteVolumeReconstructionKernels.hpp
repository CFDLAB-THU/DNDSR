#pragma once
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Physics/PerfectGasCharacteristics.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "Geom/Access/MeshView.hpp"
namespace DNDS::Cauchy::CPU
{
    /// Flow-five characteristics only. Scalar extension components are not
    /// passed through identity matrices. Periodic equation frames are handled
    /// at the face, independently of polynomial modal transformations.
    /// Supply exactly one provider: inline perfect-gas parameters or frozen
    /// cell-mean p/a/H/beta views. The caller retains their leases and validates
    /// host residency before entry. No EOS call, inversion or throw in this leaf.
    NumericalResult<index> PrepareFaceCharacteristicMatrices(FiniteVolumeStencilView stencil, RHSFieldView u, index firstComponent,
                                                             Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> l,
                                                             Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> r,
                                                             std::optional<PerfectGas> gas,
                                                             std::optional<MeanThermodynamicView> thermodynamics = {}) noexcept;
    /// Opt-in diagnostic only; preserves the host finite/L*R identity check.
    /// The production producer and limiter do not need a numerical scan.
    NumericalStatus ValidateFaceCharacteristicMatrices(
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> forward,
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> inverse) noexcept;
}
