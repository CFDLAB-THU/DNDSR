#pragma once

#include "CFV/Detail/ReconstructionDeviceKernels.hpp"

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    /// Borrowed launch arguments; the operator retains all owners/reservations.
    struct GeneralSweepArguments
    {
        R::ExecutionContext context;
        ReconstructionDeviceView plan;
        DeviceWorkSelection selection;
        DevicePeriodicBlocks vectors;
        R::PairReadView<real> mean, input;
        R::ArrayView<const real> boundary;
        R::ArrayView<real> output;
        R::ArrayView<const int> status;
        R::ArrayView<const real> target, alpha;
    };

    struct StagedSweepArguments
    {
        R::ExecutionContext context;
        ReconstructionDeviceView plan;
        DeviceWorkSelection selection;
        DevicePeriodicBlocks vectors;
        R::PairReadView<real> mean, input;
        R::ArrayView<const real> boundary;
        R::ArrayView<real> output;
        R::ArrayView<const int> status;
        ReconstructionSweepLaunch options;
        bool execute;
        R::ArrayView<const real> target, alpha;
        bool limited = false;
    };
    /// Shared host-only structural checks for direct/borrowed sweep launchers.
    /// The operator's leases retain storage; these checks do not inspect CUDA
    /// payloads or turn local validation into a device synchronization point.
    template <class Arguments>
    void ValidateSweepArguments(const Arguments &a, bool limited)
    {
        const auto &p = a.plan;
        R::ValidateViewDevices(a.context.device, p, a.selection, a.vectors, a.mean, a.input, a.boundary,
                               a.output, a.status, a.target, a.alpha);
        const auto uniform = [](auto v)
        { return v.rows >= 0 && v.width >= 0 && !v.offsets && !v.sizes && (!v.rows || !v.width || v.data); };
        DNDS_check_throw_info(a.context.device.space == R::MemorySpace::CUDA && p.n > 0 && p.n <= 19 && p.owned >= 0 &&
                                  a.mean.owned.Width() > 0 && a.mean.owned.Width() <= INT_MAX / p.n &&
                                  a.output.Width() == a.mean.owned.Width() * p.n && a.input.owned.Width() == a.output.Width() &&
                                  a.mean.owned.RowCount() == p.owned && a.input.owned.RowCount() == p.owned && a.output.RowCount() == p.owned &&
                                  (!a.mean.ghost.RowCount() || a.mean.ghost.Width() == a.mean.owned.Width()) &&
                                  (!a.input.ghost.RowCount() || a.input.ghost.Width() == a.output.Width()) &&
                                  (!a.boundary.RowCount() || a.boundary.Width() == a.mean.owned.Width()) &&
                                  uniform(a.mean.owned) && uniform(a.mean.ghost) && uniform(a.input.owned) && uniform(a.input.ghost) &&
                                  uniform(a.boundary) && uniform(a.output) && a.status.RowCount() == 1 && a.status.Width() == 1 && a.status.data &&
                                  a.selection.count >= 0 && a.selection.count <= p.owned &&
                                  (!a.selection.indirect || (a.selection.rows.RowCount() >= a.selection.count && a.selection.rows.Width() == 1)) &&
                                  p.offsets.RowCount() == p.owned + 1 && p.offsets.Width() == 1 && p.neighbors.Width() == 1 &&
                                  p.boundaryOffsets.RowCount() == p.neighbors.RowCount() + 1,
                              "CUDA reconstruction field, selection or index shape differs");
        DNDS_check_throw_info(!a.output.RowCount() || (a.output.data != a.input.owned.data && a.output.data != a.input.ghost.data &&
                                                       a.output.data != a.mean.owned.data && a.output.data != a.mean.ghost.data &&
                                                       a.output.data != a.boundary.data && (!limited || a.output.data != a.target.data)),
                              "CUDA reconstruction output aliases a borrowed input");
        DNDS_check_throw_info(!limited || (a.target.RowCount() == p.owned && a.target.Width() == a.output.Width() &&
                                           a.alpha.RowCount() == p.owned && a.alpha.Width() == 1 && uniform(a.target) && uniform(a.alpha)),
                              "CUDA reconstruction target/penalty shape differs");
        const auto matrices = p.quantized ? p.matrixAInvB32.Width() : p.matrixAInvB.Width();
        const auto vectors = p.quantized ? p.vectorAInvB32.Width() : p.vectorAInvB.Width();
        DNDS_check_throw_info(matrices == p.n * p.n && vectors == p.n &&
                                  (!p.raw || (p.localInverse.RowCount() == p.owned && p.localInverse.Width() ==
                                                                                          (p.inverseTriangle ? p.n * (p.n + 1) / 2 : p.n * p.n))) &&
                                  (!p.faceCoupling || (p.faces.RowCount() == p.neighbors.RowCount() && p.faceSide.RowCount() == p.neighbors.RowCount())),
                              "CUDA reconstruction matrix payload shape differs");
    }
    ReconstructionSweepResources RunStagedFP64(const StagedSweepArguments &, bool tensor);
    ReconstructionSweepResources RunMatrixShared(const StagedSweepArguments &);
    ReconstructionSweepResources RunQuantizedComparison(const StagedSweepArguments &, int policy);
    ReconstructionSweepResources RunStagedFloatFloat(const StagedSweepArguments &);
    ReconstructionSweepResources RunStagedFP32(const StagedSweepArguments &, bool tensor);
    void RunGeneralFP64(const GeneralSweepArguments &);
    void RunGeneralLimited(const GeneralSweepArguments &);
}
