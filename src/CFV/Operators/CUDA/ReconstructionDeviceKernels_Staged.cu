#include "CFV/Operators/CUDA/ReconstructionDeviceKernels_Staged.cuh"

namespace DNDS::CFV::Detail
{
    static ReconstructionSweepResources DispatchStagedJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                                                             DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                                                             R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                                                             const ReconstructionSweepLaunch &options, ReconstructionSweepArithmetic arithmetic, bool execute)
    {
        options.Validate();
        DNDS_check_throw_info(arithmetic == ReconstructionSweepArithmetic::FP64 || arithmetic == ReconstructionSweepArithmetic::FloatFloat ||
                                  arithmetic == ReconstructionSweepArithmetic::FP64TensorCore || arithmetic == ReconstructionSweepArithmetic::FP32 || arithmetic == ReconstructionSweepArithmetic::TF32,
                              "unknown staged VR arithmetic");
        const bool quantized = arithmetic == ReconstructionSweepArithmetic::FP32 || arithmetic == ReconstructionSweepArithmetic::TF32;
        const bool tensor = arithmetic == ReconstructionSweepArithmetic::FP64TensorCore || arithmetic == ReconstructionSweepArithmetic::TF32;
#ifndef DNDS_USE_CCCL_COMPUTE
        DNDS_check_throw_info(arithmetic != ReconstructionSweepArithmetic::FloatFloat, "float-float VR requires DNDS_USE_CCCL_COMPUTE");
#endif
        // Native maps are contiguous tables, including padded row pitches.
        // Project their static access pattern once, rather than carrying CSR/
        // SoA/tile branches through every inner product. Solution fields stay
        // layout-generic; only their frozen row spans are staged.
        const auto table = [](auto view, index width)
        { return view.kind == R::RowLayoutKind::Table && view.width == width && view.stride >= width && !view.offsets && !view.sizes && (!view.rows || view.data); };
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && plan.quantized == quantized && !(quantized && plan.raw) &&
                                  plan.n > 0 && plan.n <= 19 && output.width % plan.n == 0 &&
                                  selection.count >= 0 && selection.count <= plan.owned &&
                                  (!selection.indirect || selection.rows.rows >= selection.count) &&
                                  output.width / plan.n >= 1 && output.width <= INT_MAX / 4 &&
                                  (quantized ? (table(plan.matrixAInvB32, plan.n * plan.n) && table(plan.vectorAInvB32, plan.n) && table(plan.boundaryInjection32, plan.n))
                                             : (table(plan.matrixAInvB, plan.n * plan.n) && table(plan.vectorAInvB, plan.n) && table(plan.boundaryInjection, plan.n))) &&
                                  table(plan.offsets, 1) && table(plan.neighbors, 1) && table(plan.boundaryOffsets, 1) &&
                                  (!plan.raw || table(plan.localInverse, plan.localInverse.width)) &&
                                  (!plan.faceCoupling || (table(plan.faces, 1) && table(plan.faceSide, 1))) &&
                                  (!plan.rotated || (table(plan.rotation, 9) && table(plan.rotationActive, 1))) &&
                                  plan.offsets.rows == plan.owned + 1 && plan.neighbors.width == 1 &&
                                  plan.boundaryOffsets.rows == plan.neighbors.rows + 1 &&
                                  (quantized ? plan.vectorAInvB32.rows : plan.vectorAInvB.rows) == plan.neighbors.rows &&
                                  table(status, 1) && status.rows == 1 &&
                                  (!plan.faceCoupling || (plan.faces.rows == plan.neighbors.rows && plan.faceSide.rows == plan.neighbors.rows)) &&
                                  (!plan.rotated || (plan.rotation.rows == plan.neighbors.rows && plan.rotationActive.rows == plan.neighbors.rows)) &&
                                  (!plan.raw || (plan.localInverse.rows == plan.owned &&
                                                 plan.localInverse.width == (plan.inverseTriangle ? plan.n * (plan.n + 1) / 2 : plan.n * plan.n))) &&
                                  input.owned.width == output.width && mean.owned.width == output.width / plan.n &&
                                  (!input.ghost.rows || input.ghost.width == output.width) &&
                                  (!mean.ghost.rows || mean.ghost.width == mean.owned.width) &&
                                  (!boundary.rows || boundary.width == mean.owned.width) &&
                                  mean.owned.rows == plan.owned && input.owned.rows == plan.owned && output.rows == plan.owned &&
                                  (!output.rows || (output.data != input.owned.data && output.data != input.ghost.data)) &&
                                  (!tensor || (quantized ? TF32TensorCoreAvailable(context.device) : FP64TensorCoreAvailable(context.device))),
                              "stage-once CUDA VR requires FP64 P1..P3, a valid component count and supported device and compute policy");
        R::ValidateViewDevices(context.device, plan, selection, vectors, mean, input, boundary, output, status);
        R::CheckKernelArgumentBudget<Staged::Sweep<real>, bool, decltype(selection), decltype(vectors), decltype(mean), decltype(input),
                                     decltype(boundary), decltype(output), decltype(status), int>();
        if (!selection.count)
            return {};
        const StagedSweepArguments args{context, plan, selection, vectors, mean, input, boundary, output, status, options, execute};
        ReconstructionSweepResources resource;
        if (quantized)
            resource = RunStagedFP32(args, tensor);
        else if (arithmetic == ReconstructionSweepArithmetic::FloatFloat)
            resource = RunStagedFloatFloat(args);
        else
            resource = RunStagedFP64(args, tensor);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "stage-once CUDA VR launch failed");
        return resource;
    }
    void LaunchStagedJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                            DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                            R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                            const ReconstructionSweepLaunch &options, ReconstructionSweepArithmetic arithmetic)
    {
        (void)DispatchStagedJacobi(context, plan, selection, vectors, mean, input, boundary, output, status, options, arithmetic, true);
    }
    ReconstructionSweepResources InspectStagedJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                                                     DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                                                     R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                                                     const ReconstructionSweepLaunch &options, ReconstructionSweepArithmetic arithmetic)
    {
        return DispatchStagedJacobi(context, plan, selection, vectors, mean, input, boundary, output, status, options, arithmetic, false);
    }
}
