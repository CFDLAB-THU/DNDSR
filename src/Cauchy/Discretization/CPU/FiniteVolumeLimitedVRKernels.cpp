#include "Cauchy/Discretization/CPU/FiniteVolumeLimitedVRKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSampleKernels.hpp"
#include "DNDS/Core/OMP.hpp"
namespace DNDS::Cauchy::CPU
{
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;
    NumericalStatus LimitLVRTarget(FiniteVolumeStencilView stencil, RHSScalarView means, RHSOutputView coefficients,
                                   int dimension, const ThermalBounds &bounds, int betaPower, real betaCutoff) noexcept
    {
        const int modes = int(coefficients.Width() / 5);
        return OMP::ForRowsStatus(0, means.RowCount(), [&](index iCell) noexcept
                                  { return LimitLVRTargetCell(stencil, means, coefficients, iCell, dimension, modes, bounds, betaPower, betaCutoff); });
    }
    void FillLVRScalars(RHSScalarView u, RHSOutputView s, const PerfectGas &gas)
    {
        OMP::ForRows(0, s.RowCount(), [&](index iCell)
                     {
            const auto state = Load(u, iCell);
            s(iCell, 0) = gas.Pressure(state);
            s(iCell, 1) = state[0]; });
    }
    NumericalStatus BuildLVRFaceSensors(FiniteVolumeStencilView stencil, RHSFieldView means,
                                        RHSFieldView target, int dimension, const PerfectGas &gas,
                                        const ThermalBounds &bounds, const CFV::LimitedVRSettings &settings,
                                        RHSOutputView sensors) noexcept
    {
        DNDS_assert(sensors.RowCount() == stencil.stencilFaces.RowCount() && sensors.Width() == 3);
        const int modes = int(target.owned.Width() / 5);
        return OMP::ForRowsStatus(0, sensors.RowCount(), [&](index iAux) -> NumericalStatus
                                  {
            const index iFace = stencil.stencilFaces(iAux, 0);
            const auto parents = stencil.topology.FaceCells(iFace);
            if (parents[1] == UnInitIndex)
            {
                for (int j = 0; j < 3; ++j)
                    sensors(iAux, j) = 0;
                return {};
            }
            const auto center = stencil.metric.GetFaceCent(iFace);
            const auto left = EvaluateLVRLinearTarget(stencil, means, target, parents[0], center, dimension, modes);
            auto right = EvaluateLVRLinearTarget(stencil, means, target, parents[1],
                stencil.metric.periodic.FacePoint(iFace, 1, center), dimension, modes);
            if (stencil.metric.periodic.Rotates(iFace))
                right = RotateState(right, stencil.metric.periodic.OtherRotation(iFace, 0));
            if (!(ThermallyAdmissible(left, bounds) && ThermallyAdmissible(right, bounds)))
                return NumericalStatus{NumericalError::InvalidState, 1}.At(parents[0], iFace);
            const auto l = gas.Primitive(left), r = gas.Primitive(right);
            const auto normal = stencil.metric.GetFaceMeanNorm(iFace);
            real unLeft = 0, unRight = 0;
            for (int d = 0; d < 3; ++d)
            {
                unLeft += l.velocity[d] * normal[d];
                unRight += r.velocity[d] * normal[d];
            }
            const auto sensor = CFV::Math::EvaluateLimitedVRFaceSensor(l.density, r.density, l.pressure, r.pressure,
                unLeft, unRight, l.soundSpeed, r.soundSpeed, settings);
            const auto support = CFV::Math::EvaluateLimitedVRFaceSensor(means(parents[0], 0), means(parents[1], 0),
                1, 1, 0, 0, 1, 1, settings).contact;
            sensors(iAux, 0) = sensor.shock;
            sensors(iAux, 1) = sensor.contact;
            sensors(iAux, 2) = support;
            return {}; });
    }
    void GatherLVRBranches(FiniteVolumeStencilView stencil, RHSScalarView sensors, RHSOutputView branches,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base)
    {
        DNDS_assert(base.RowCount() == branches.RowCount());
        OMP::ForRows(0, branches.RowCount(), [&](index iCell)
                     {
            std::array<real, 3> values{};
            for (const index iFace : stencil.topology.CellFaces(iCell))
            {
                const index iAux = stencil.routing(iFace, 0).auxiliaryFace;
                for (int j = 0; j < 3; ++j)
                    values[j] = std::max(values[j], sensors(iAux, j));
            }
            for (int j = 0; j < 3; ++j)
                branches(iCell, j) = values[j];
            for (int j = 0; j < 2; ++j)
                base(iCell, j) = values[j]; });
    }
    void MergeLVRBase(Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> b,
                      RHSOutputView s, int column)
    {
        OMP::ForRows(0, s.RowCount(), [&](index iCell)
                     { s(iCell, column) = std::max(s(iCell, column), b(iCell, column)); });
    }
    void PublishLVRPenalty(RHSScalarView g,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> h,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> a)
    {
        OMP::ForRows(0, g.RowCount(), [&](index iCell)
                     {
            for (int j = 0; j < 2; ++j)
                h(iCell, j) = g(iCell, j);
            a(iCell, 0) = std::max(g(iCell, 0), g(iCell, 1)); });
    }
}
