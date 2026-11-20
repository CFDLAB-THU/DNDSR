#include "CFV/Plans/LimitedVRPlan.hpp"
#include "CFV/Detail/LimitedVRKernels.hpp"

namespace DNDS::CFV
{
    namespace R = Runtime;
    void LimitedVRPlan::ValidateRegion(ReconstructionRegion region) const
    {
        _metrics.Validate();
        _partition.Validate();
        DNDS_check_throw_info(region == ReconstructionRegion::All || region == ReconstructionRegion::Core || region == ReconstructionRegion::HaloDependent,
                              "invalid LVR work region");
    }
    void LimitedVRPlan::ValidateSeeds(const FVField &meanScalars, const FVField &baseGates, const R::Array<real> &previous,
                                      const FVField &output, ReconstructionRegion region) const
    {
        ValidateRegion(region);
        _metrics.ValidateField(meanScalars, 2);
        _metrics.ValidateField(baseGates, 3);
        _metrics.ValidateField(output, 3);
        DNDS_check_throw_info(previous.GetDevice() == Device() && !previous.Layout().IsVariable() && previous.RowCount() == _metrics.NumCell() && previous.Layout().Width() == 2,
                              "LVR previous-gate shape or placement differs");
        for (const auto *source : {&meanScalars.Owned(), &meanScalars.Ghost(), &baseGates.Owned(), &baseGates.Ghost(), &previous})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(source->Storage(), output.Owned().Storage()), "LVR seed output aliases input");
    }
    void LimitedVRPlan::BuildExtremumSeeds(const FVField &meanScalars, const FVField &baseGates, const R::Array<real> &previous,
                                           FVField &output, const LimitedVRSettings &settings, ReconstructionRegion region) const
    {
        ValidateSeeds(meanScalars, baseGates, previous, output, region);
        const auto parameters = LimitedVRKernelParameters::FromSettings(settings);
        DNDS_check_throw_info(Device() == R::Device::Host(), "synchronous LVR convenience API requires host placement; use its queued overload for CUDA");
        const auto geometry = _metrics.AcquireRead();
        auto scalarRead = meanScalars.Owned().Read(), gateRead = baseGates.Owned().Read(), history = previous.Read();
        std::optional<R::Array<real>::ReadLease> scalarGhost, gateGhost;
        if (region != ReconstructionRegion::Core)
            scalarGhost.emplace(meanScalars.Ghost().Read()), gateGhost.emplace(baseGates.Ghost().Read());
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> scalar{scalarRead.View<R::UniformTableRows>(), scalarGhost ? scalarGhost->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{}};
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> gate{gateRead.View<R::UniformTableRows>(), gateGhost ? gateGhost->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{}};
        const auto prior = history.View<R::UniformTableRows>();
        auto offsets = _offsets.Read(), neighbors = _neighbors.Read(), boundary = _boundary.Read();
        const auto o = offsets.View<R::FixedRows<1>>(), n = neighbors.View<R::FixedRows<1>>(), b = boundary.View<R::FixedRows<1>>();
        auto write = output.Owned().Write();
        const auto result = write.View<R::UniformTableRows>();
        const auto &selected = _regionRows.Get(region);
        const auto rows = selected.indices.Read();
        const Runtime::RowSelection selection{selected.shape, rows.View<Runtime::FixedRows<1>>()};
        Detail::BuildLimitedVRSeeds({o, n, b}, selection, scalar, gate, prior, result, parameters);
    }
    void LimitedVRPlan::ValidateMaximum(const FVField &input, const FVField &output, std::array<real, 3> caps, ReconstructionRegion region) const
    {
        ValidateRegion(region);
        _metrics.ValidateField(input, 3);
        _metrics.ValidateField(output, 3);
        for (auto cap : caps)
            DNDS_check_throw_info(std::isfinite(cap) && cap >= 0 && cap <= 1, "invalid LVR saturation cap");
        for (const auto *source : {&input.Owned(), &input.Ghost()})
            DNDS_check_throw_info(!Algebra::Detail::Overlaps(source->Storage(), output.Owned().Storage()), "LVR maximum output aliases input");
    }
    void LimitedVRPlan::NeighborMaximum(const FVField &input, FVField &output, std::array<bool, 3> active,
                                        bool excludeBoundary, bool saturatedOnly, std::array<real, 3> caps, ReconstructionRegion region) const
    {
        ValidateMaximum(input, output, caps, region);
        DNDS_check_throw_info(Device() == R::Device::Host(), "synchronous LVR convenience API requires host placement; use its queued overload for CUDA");
        const auto geometry = _metrics.AcquireRead();
        auto owned = input.Owned().Read();
        std::optional<R::Array<real>::ReadLease> ghost;
        if (region != ReconstructionRegion::Core)
            ghost.emplace(input.Ghost().Read());
        R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows> source{owned.View<R::UniformTableRows>(), ghost ? ghost->View<R::UniformTableRows>() : R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>{}};
        auto offsets = _offsets.Read(), neighbors = _neighbors.Read(), boundary = _boundary.Read();
        const auto o = offsets.View<R::FixedRows<1>>(), n = neighbors.View<R::FixedRows<1>>(), b = boundary.View<R::FixedRows<1>>();
        auto write = output.Owned().Write();
        const auto result = write.View<R::UniformTableRows>();
        const auto &selected = _regionRows.Get(region);
        const auto rows = selected.indices.Read();
        const Runtime::RowSelection selection{selected.shape, rows.View<Runtime::FixedRows<1>>()};
        Detail::LimitedVRMaximum({o, n, b}, selection, source, result, active, excludeBoundary, saturatedOnly, caps);
    }
}
