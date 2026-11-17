#include "DNDS/Runtime/Execution/CUDAStream.hpp"
// Private CUDA implementation: included only by generated one-shape TUs.
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Shapes.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Staging.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Faces.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

namespace DNDS::Cauchy::CUDA::Detail
{
    using RHSGeometryView = Runtime::ArrayView<const RHSGeometry, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using RHSParametersView = Runtime::ArrayView<const RHSKernelParameters, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;

    /** @brief One CTA-packed face set with runtime launch width.
     * @details Exactly one kernel is instantiated for each C/M/D tuple. The
     * 512-thread bound is the maximum supported runtime width, not another
     * template axis. Shared staging/reduction preserves complete face order.
     */
    template <int Components, int Modes, int Dimension>
    __global__ __launch_bounds__(512) void PolynomialFaces(
        RHSGeometryView publication, RHSParametersView parameters, RHSKernelControls controls,
        RHSShapeFields<Components, Modes, Dimension> fields,
        RHSFaceOutputsT<Runtime::FixedRows<Components, Components>> output, RHSFacePackageView prepared)
    {
        extern __shared__ real shared[];
        const index stageWords = index(prepared.maximumFaces) * 2 * prepared.shape.width;
        SpanView<real, false> bank{shared, stageWords};
        real *high = shared + stageWords;
        real *low = high + Components * int(blockDim.x);
        unsigned char *tail = reinterpret_cast<unsigned char *>(low + (controls.produceLow ? Components * int(blockDim.x) : 0));
        int *codes = reinterpret_cast<int *>(tail);
        int *details = codes + int(blockDim.x);
        index *rows = reinterpret_cast<index *>(details + int(blockDim.x));
        index *points = rows + int(blockDim.x);
        index *fallback = points + int(blockDim.x);
        const auto &geometry = publication(0, 0);
        const auto &physics = parameters(0, 0);
        for (index iPackage = blockIdx.x; iPackage < prepared.packages.RowCount(); iPackage += gridDim.x)
        {
            const auto package = prepared.packages(iPackage, 0);
            const index words = index(package.faces) * 2 * prepared.shape.width;
            for (index iWord = threadIdx.x; iWord < words; iWord += int(blockDim.x))
            {
                const int slot = int(iWord / (2 * prepared.shape.width));
                const int side = int(iWord / prepared.shape.width) % 2;
                const int column = int(iWord % prepared.shape.width);
                const auto face = prepared.segments(package.firstSegment + slot, 0);
                const auto anchor = geometry.FaceAnchor(face.auxiliaryFace);
                const real *source = FaceStageSource(fields, controls, prepared.shape,
                                                     side ? anchor.other : anchor.cell, column);
                if (source)
                {
#if __CUDA_ARCH__ >= 800
                    __pipeline_memcpy_async(&bank[iWord], source, sizeof(real));
#else
                    bank[iWord] = *source;
#endif
                }
                else
                    bank[iWord] = 0;
            }
#if __CUDA_ARCH__ >= 800
            __pipeline_commit();
            __pipeline_wait_prior(0);
#endif
            __syncthreads();
            int ownSlot = -1;
            RHSFaceSegment segment{};
            for (int slot = 0; slot < package.faces; ++slot)
            {
                const auto candidate = prepared.segments(package.firstSegment + slot, 0);
                if (int(threadIdx.x) >= candidate.firstLane && int(threadIdx.x) < candidate.firstLane + candidate.lanes)
                {
                    ownSlot = slot;
                    segment = candidate;
                    break;
                }
            }
            const int localLane = ownSlot >= 0 ? int(threadIdx.x) - segment.firstLane : 0;
            StagedFaceFields<Components, Modes, Dimension> staged;
            index firstPoint = 0;
            if (ownSlot >= 0)
            {
                const auto anchor = geometry.FaceAnchor(segment.auxiliaryFace);
                staged = BindFaceStage(bank, prepared.shape, ownSlot, anchor, controls, fields);
                firstPoint = geometry.faceOffsets(anchor.face, 0);
            }
            std::array<real, Components> total{}, totalLow{};
            index totalFallback = 0;
            NumericalStatus faceFailure;
            for (int tile = 0; tile < package.tiles; ++tile)
            {
                std::array<real, Components> value{}, valueLow{};
                index localFallback = 0;
                NumericalStatus failure;
                const index q = index(tile) * segment.lanes + localLane;
                if (ownSlot >= 0 && q < segment.points)
                    failure = physics.physics.rans.Supported() && physics.physics.rans.ComponentNumber() + 5 == Components && physics.physics.dimension == Dimension &&
                                      (!controls.viscous || physics.physics.viscous)
                                  ? EvaluateFacePoint<Components>(geometry, physics, controls, staged, output,
                                                         segment.auxiliaryFace, firstPoint + q, value, valueLow, localFallback)
                                  : NumericalStatus{NumericalError::InvalidInput, 1};
                for (int j = 0; j < Components; ++j)
                {
                    high[Components * threadIdx.x + j] = value[j];
                    if (controls.produceLow)
                        low[Components * threadIdx.x + j] = valueLow[j];
                }
                codes[threadIdx.x] = int(failure.code);
                details[threadIdx.x] = failure.detail;
                rows[threadIdx.x] = failure.row;
                points[threadIdx.x] = failure.point;
                fallback[threadIdx.x] = localFallback;
                __syncthreads();
                if (ownSlot >= 0 && localLane == 0)
                    for (int lane = 0; lane < segment.lanes; ++lane)
                    {
                        const int row = segment.firstLane + lane;
                        const NumericalStatus pointFailure{NumericalError(codes[row]), details[row], rows[row], points[row]};
                        if (faceFailure && !pointFailure)
                            faceFailure = pointFailure;
                        totalFallback += fallback[row];
                    }
                if (controls.reduction == RHSFaceReduction::Serial)
                {
                    if (ownSlot >= 0 && localLane == 0)
                        for (int lane = 0; lane < segment.lanes; ++lane)
                            for (int j = 0; j < Components; ++j)
                            {
                                total[j] += high[Components * (segment.firstLane + lane) + j];
                                if (controls.produceLow)
                                    totalLow[j] += low[Components * (segment.firstLane + lane) + j];
                            }
                }
                else
                {
                    // CTA-packed faces may cross a warp. Each fixed tree
                    // level publishes shared partials before the next one;
                    // a warp-only segmented primitive cannot perform this
                    // cross-warp merge on its own.
                    for (int stride = 1; stride < int(blockDim.x); stride *= 2)
                    {
                        if (ownSlot >= 0 && localLane % (2 * stride) == 0 && localLane + stride < segment.lanes)
                            for (int j = 0; j < Components; ++j)
                            {
                                high[Components * threadIdx.x + j] += high[Components * (threadIdx.x + stride) + j];
                                if (controls.produceLow)
                                    low[Components * threadIdx.x + j] += low[Components * (threadIdx.x + stride) + j];
                            }
                        __syncthreads();
                    }
                    if (ownSlot >= 0 && localLane == 0)
                        for (int j = 0; j < Components; ++j)
                        {
                            total[j] += high[Components * threadIdx.x + j];
                            if (controls.produceLow)
                                totalLow[j] += low[Components * threadIdx.x + j];
                        }
                }
                __syncthreads();
            }
            if (ownSlot >= 0 && localLane == 0)
            {
                output.status(segment.auxiliaryFace, 0) = faceFailure;
                output.fallbackCounts(segment.auxiliaryFace, 0) = totalFallback;
                if (faceFailure)
                    for (int j = 0; j < Components; ++j)
                    {
                        output.high(segment.auxiliaryFace, j) = total[j];
                        if (controls.produceLow)
                            output.low(segment.auxiliaryFace, j) = totalLow[j];
                    }
            }
            __syncthreads(); // bank/scratch retirement before next package
        }
    }

    /// Resolve typed operand roles once; numerical work contains no host access,
    /// allocation, lease acquisition or synchronization.
    template <int Components, int Modes, int Dimension>
    void LaunchRHSShape(const RHSFaceLaunch &a)
    {
        static_assert(Components >= 5 && Components <= 7 && (Dimension == 2 || Dimension == 3));
        DNDS_check_throw_info(a.controls.components == Components && a.controls.dimension == Dimension &&
                                  ((Modes == 0 && (a.controls.direct || a.controls.firstOrder)) ||
                                   (Modes > 0 && !a.controls.direct && !a.controls.firstOrder && a.controls.modes == Modes)) &&
                                  a.controls.mapping == RHSFaceMapping::CTASegments && !a.controls.doubleBuffer,
                              "CUDA RHS static C/M/D launch does not match the validated profile");
        const auto fields = ProjectRHSShapeFields<Components, Modes, Dimension>(a.controls, a.fields);
        const auto output = ProjectRHSOutputs<Components>(a.output);
        Runtime::CheckKernelArgumentBudget<RHSGeometryView, RHSParametersView, RHSKernelControls,
                                           decltype(fields), decltype(output), RHSFacePackageView>();
        PolynomialFaces<Components, Modes, Dimension><<<a.segmentBlocks, a.controls.blockSize, a.packages.sharedBytes,
                                                         ::DNDS::Runtime::CUDAStream(a.context)>>>(
            a.geometry.View(), a.parameters, a.controls, fields, output, a.packages);
    }
}
