#include "CFV/Detail/ReconstructionSweepArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace
    {
        ReconstructionSweepResources Dispatch(StagedSweepArguments args, ReconstructionSweepArithmetic arithmetic)
        {
            args.options.Validate();
            ValidateSweepArguments(args, args.limited);
            const auto &plan = args.plan;
            const auto table = [](auto v, index width)
            { return v.kind == R::RowLayoutKind::Table && v.width == width && v.stride >= width && !v.offsets && !v.sizes && (!v.rows || v.data); };
            const auto field = [&](auto v, index width)
            { return !v.rows || (table(v, width) && v.stride == width); };
            const bool quantized = arithmetic == ReconstructionSweepArithmetic::FP32 || arithmetic == ReconstructionSweepArithmetic::TF32 ||
                                   arithmetic == ReconstructionSweepArithmetic::FP32Reference || arithmetic == ReconstructionSweepArithmetic::FP32FullAccumulation ||
                                   arithmetic == ReconstructionSweepArithmetic::FP32FloatFloatAccumulation || arithmetic == ReconstructionSweepArithmetic::TF32Synchronous;
            const bool tensor = arithmetic == ReconstructionSweepArithmetic::FP64TensorCore || arithmetic == ReconstructionSweepArithmetic::TF32 ||
                                arithmetic == ReconstructionSweepArithmetic::TF32Synchronous;
            const bool comparison = arithmetic != ReconstructionSweepArithmetic::FP64;
            const auto matrixWidth = plan.n * plan.n;
            const bool matricesOK = plan.packedRawB ? table(plan.rawMatrixB32, matrixWidth)
                                    : quantized     ? table(plan.matrixAInvB32, matrixWidth)
                                                    : table(plan.matrixAInvB, matrixWidth);
            const auto inverseWidth = plan.inverseTriangle ? plan.n * (plan.n + 1) / 2 : plan.n * plan.n;
            const bool inverseOK = plan.packedInverse ? table(plan.localInverse32, inverseWidth)
                                                      : table(plan.localInverse, inverseWidth);
            DNDS_check_throw_info(arithmetic >= ReconstructionSweepArithmetic::FP64 && arithmetic <= ReconstructionSweepArithmetic::TF32Synchronous,
                                  "unknown VR arithmetic selector");
#ifndef DNDS_USE_CCCL_COMPUTE
            DNDS_check_throw_info(arithmetic != ReconstructionSweepArithmetic::FloatFloat && arithmetic != ReconstructionSweepArithmetic::FP32FloatFloatAccumulation,
                                  "float-float VR requires DNDS_USE_CCCL_COMPUTE");
#endif
            DNDS_check_throw_info(args.context.device.space == R::MemorySpace::CUDA && plan.quantized == quantized && !(quantized && plan.raw) &&
                                      plan.n > 0 && plan.n <= 19 && args.output.Width() % plan.n == 0 &&
                                      args.selection.RowCount() >= 0 && args.selection.RowCount() <= plan.owned &&
                                      args.mean.owned.Width() >= 1 && args.output.Width() <= INT_MAX / 4 &&
                                      matricesOK &&
                                      (quantized ? table(plan.vectorAInvB32, plan.n) && table(plan.boundaryInjection32, plan.n)
                                                 : table(plan.vectorAInvB, plan.n) && table(plan.boundaryInjection, plan.n)) &&
                                      table(plan.offsets, 1) && table(plan.neighbors, 1) && table(plan.boundaryOffsets, 1) &&
                                      (!plan.raw || inverseOK) &&
                                      (!plan.faceCoupling || (table(plan.faces, 1) && table(plan.faceSide, 1))) &&
                                      (!plan.rotated || (table(plan.rotation, 9) && table(plan.rotationActive, 1))) &&
                                      plan.offsets.RowCount() == plan.owned + 1 &&
                                      plan.boundaryOffsets.RowCount() == plan.neighbors.RowCount() + 1 &&
                                      (quantized ? plan.vectorAInvB32.RowCount() : plan.vectorAInvB.RowCount()) == plan.neighbors.RowCount() &&
                                      table(args.status, 1) && args.status.RowCount() == 1 &&
                                      (!plan.faceCoupling || (plan.faces.RowCount() == plan.neighbors.RowCount() && plan.faceSide.RowCount() == plan.neighbors.RowCount())) &&
                                      (!plan.rotated || (plan.rotation.RowCount() == plan.neighbors.RowCount() && plan.rotationActive.RowCount() == plan.neighbors.RowCount())) &&
                                      args.input.owned.Width() == args.output.Width() && args.mean.owned.Width() == args.output.Width() / plan.n &&
                                      field(args.mean.owned, args.mean.owned.Width()) && field(args.mean.ghost, args.mean.owned.Width()) &&
                                      field(args.input.owned, args.output.Width()) && field(args.input.ghost, args.output.Width()) &&
                                      field(args.boundary, args.mean.owned.Width()) && field(args.output, args.output.Width()) &&
                                      args.mean.owned.RowCount() == plan.owned && args.input.owned.RowCount() == plan.owned && args.output.RowCount() == plan.owned &&
                                      (!args.output.RowCount() || (args.output.data != args.input.owned.data && args.output.data != args.input.ghost.data)) &&
                                      (!tensor || (quantized ? TF32TensorCoreAvailable(args.context.device) : FP64TensorCoreAvailable(args.context.device))),
                                  "optimized VR needs compact resident AoS P1..P3 fields, compatible maps and supported device arithmetic; padded field rows are unsupported");
            DNDS_check_throw_info(!comparison || (plan.n == 19 && args.mean.owned.Width() == 5 && !plan.raw && !args.limited),
                                  "comparison VR selectors require C=5/M=19 precomputed non-LVR; use fp64Pipelined for other shapes");
            if (args.limited)
                DNDS_check_throw_info(field(args.target, args.output.Width()) && args.target.RowCount() == plan.owned &&
                                          field(args.alpha, 1) && args.alpha.RowCount() == plan.owned,
                                      "optimized LVR target/penalty shape differs");
            R::ValidateViewDevices(args.context.device, plan, args.selection, args.vectors, args.mean, args.input, args.boundary,
                                   args.output, args.status, args.target, args.alpha);
            if (!args.selection.RowCount())
                return {};
            ReconstructionSweepResources resource;
            switch (arithmetic)
            {
            case ReconstructionSweepArithmetic::FP64:
                resource = RunStagedFP64(args, false);
                break;
            case ReconstructionSweepArithmetic::FP64TensorCore:
                resource = RunStagedFP64(args, true);
                break;
            case ReconstructionSweepArithmetic::FP64MatrixShared:
                resource = RunMatrixShared(args);
                break;
            case ReconstructionSweepArithmetic::FloatFloat:
                resource = RunStagedFloatFloat(args);
                break;
            case ReconstructionSweepArithmetic::FP32:
                resource = RunStagedFP32(args, false);
                break;
            case ReconstructionSweepArithmetic::TF32:
                resource = RunStagedFP32(args, true);
                break;
            case ReconstructionSweepArithmetic::FP32Reference:
                resource = RunQuantizedComparison(args, 0);
                break;
            case ReconstructionSweepArithmetic::FP32FullAccumulation:
                resource = RunQuantizedComparison(args, 1);
                break;
            case ReconstructionSweepArithmetic::FP32FloatFloatAccumulation:
                resource = RunQuantizedComparison(args, 2);
                break;
            case ReconstructionSweepArithmetic::TF32Synchronous:
                resource = RunQuantizedComparison(args, 3);
                break;
            }
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "optimized VR launch failed");
            return resource;
        }
    }
    void LaunchStagedJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, Runtime::RowSelection selection,
                            DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                            R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                            const ReconstructionSweepLaunch &options, ReconstructionSweepArithmetic arithmetic)
    {
        (void)Dispatch({context, plan, selection, vectors, mean, input, boundary, output, status, options, true}, arithmetic);
    }
    void LaunchLimitedStagedJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, Runtime::RowSelection selection,
                                   DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                                   R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                                   R::ArrayView<const real> target, R::ArrayView<const real> alpha, const ReconstructionSweepLaunch &options)
    {
        (void)Dispatch({context, plan, selection, vectors, mean, input, boundary, output, status, options, true, target, alpha, true}, ReconstructionSweepArithmetic::FP64);
    }
    ReconstructionSweepResources InspectStagedJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, Runtime::RowSelection selection,
                                                     DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                                                     R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status,
                                                     const ReconstructionSweepLaunch &options, ReconstructionSweepArithmetic arithmetic)
    {
        return Dispatch({context, plan, selection, vectors, mean, input, boundary, output, status, options, false}, arithmetic);
    }
    void LaunchJacobiTF32(R::ExecutionContext context, ReconstructionDeviceView plan, Runtime::RowSelection selection,
                          DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                          R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status)
    {
        LaunchStagedJacobi(context, plan, selection, vectors, mean, input, boundary, output, status, {}, ReconstructionSweepArithmetic::TF32Synchronous);
    }
}
