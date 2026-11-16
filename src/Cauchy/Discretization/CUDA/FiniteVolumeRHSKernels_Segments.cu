#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Segments.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Staging.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    namespace R = Runtime;
    namespace
    {
        // Same staged view contract, distinct local arithmetic instantiation.
        // The capped segment callee must not be weak-linked to the uncapped
        // buffered-face instantiation from another translation unit.
        struct SegmentFields : StagedFaceFields
        {
        };

        template <int Threads, bool WarpContained>
        // The 512-thread candidate needs a compiler-visible block bound: an
        // unconstrained register allocation can exceed the block register file
        // even though its explicitly budgeted shared staging fits.
        __global__ __launch_bounds__(Threads) void SegmentFaces(RHSGeometryView publication, RHSParametersView parameters,
                                                                RHSKernelControls controls, RHSFields fields, RHSFaceOutputs output, RHSFacePackageView prepared)
        {
            extern __shared__ real shared[];
            const index stageWords = index(prepared.maximumFaces) * 2 * prepared.shape.width;
            SpanView<real, false> bank{shared, stageWords};
            real *high = shared + stageWords;
            real *low = high + 5 * Threads;
            unsigned char *tail = reinterpret_cast<unsigned char *>(low + (controls.produceLow ? 5 * Threads : 0));
            int *codes = reinterpret_cast<int *>(tail);
            int *details = codes + Threads;
            index *rows = reinterpret_cast<index *>(details + Threads);
            index *points = rows + Threads;
            index *fallback = points + Threads;
            const auto &geometry = publication(0, 0);
            const auto &physics = parameters(0, 0);
            for (index iPackage = blockIdx.x; iPackage < prepared.packages.RowCount(); iPackage += gridDim.x)
            {
                const auto package = prepared.packages(iPackage, 0);
                const index words = index(package.faces) * 2 * prepared.shape.width;
                for (index iWord = threadIdx.x; iWord < words; iWord += Threads)
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
                SegmentFields staged;
                index firstPoint = 0;
                if (ownSlot >= 0)
                {
                    const auto anchor = geometry.FaceAnchor(segment.auxiliaryFace);
                    static_cast<StagedFaceFields &>(staged) = BindFaceStage(bank, prepared.shape, ownSlot, anchor, controls, fields);
                    firstPoint = geometry.faceOffsets(anchor.face, 0);
                }
                std::array<real, 5> total{}, totalLow{};
                index totalFallback = 0;
                NumericalStatus faceFailure;
                for (int tile = 0; tile < package.tiles; ++tile)
                {
                    std::array<real, 5> value{}, valueLow{};
                    index localFallback = 0;
                    NumericalStatus failure;
                    const index q = index(tile) * segment.lanes + localLane;
                    if (ownSlot >= 0 && q < segment.points)
                        failure = physics.physics.rans.ComponentNumber() == 0 && physics.physics.dimension == controls.dimension &&
                                          (!controls.viscous || physics.physics.viscous)
                                      ? EvaluateFacePoint<5>(geometry, physics, controls, staged, output,
                                                             segment.auxiliaryFace, firstPoint + q, value, valueLow, localFallback)
                                      : NumericalStatus{NumericalError::InvalidInput, 1};
                    for (int j = 0; j < 5; ++j)
                    {
                        high[5 * threadIdx.x + j] = value[j];
                        if (controls.produceLow)
                            low[5 * threadIdx.x + j] = valueLow[j];
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
                                for (int j = 0; j < 5; ++j)
                                {
                                    total[j] += high[5 * (segment.firstLane + lane) + j];
                                    if (controls.produceLow)
                                        totalLow[j] += low[5 * (segment.firstLane + lane) + j];
                                }
                    }
                    else if constexpr (WarpContained)
                    {
                        // Every warp lane, including idle/tail lanes, executes
                        // each shuffle. Segment-local bounds prevent crossing
                        // adjacent faces even for non-power-of-two widths.
                        for (int stride = 1; stride < 32; stride *= 2)
                            for (int j = 0; j < 5; ++j)
                            {
                                const real other = __shfl_down_sync(0xffffffffu, value[j], stride);
                                const real otherLow = __shfl_down_sync(0xffffffffu, valueLow[j], stride);
                                if (ownSlot >= 0 && localLane % (2 * stride) == 0 && localLane + stride < segment.lanes)
                                {
                                    value[j] += other;
                                    valueLow[j] += otherLow;
                                }
                            }
                        if (ownSlot >= 0 && localLane == 0)
                            for (int j = 0; j < 5; ++j)
                            {
                                total[j] += value[j];
                                if (controls.produceLow)
                                    totalLow[j] += valueLow[j];
                            }
                    }
                    else
                    {
                        // CTA-packed faces may cross a warp. Each fixed tree
                        // level publishes shared partials before the next one;
                        // a warp-only segmented primitive cannot perform this
                        // cross-warp merge on its own.
                        for (int stride = 1; stride < Threads; stride *= 2)
                        {
                            if (ownSlot >= 0 && localLane % (2 * stride) == 0 && localLane + stride < segment.lanes)
                                for (int j = 0; j < 5; ++j)
                                {
                                    high[5 * threadIdx.x + j] += high[5 * (threadIdx.x + stride) + j];
                                    if (controls.produceLow)
                                        low[5 * threadIdx.x + j] += low[5 * (threadIdx.x + stride) + j];
                                }
                            __syncthreads();
                        }
                        if (ownSlot >= 0 && localLane == 0)
                            for (int j = 0; j < 5; ++j)
                            {
                                total[j] += high[5 * threadIdx.x + j];
                                if (controls.produceLow)
                                    totalLow[j] += low[5 * threadIdx.x + j];
                            }
                    }
                    __syncthreads();
                }
                if (ownSlot >= 0 && localLane == 0)
                {
                    output.status(segment.auxiliaryFace, 0) = faceFailure;
                    output.fallbackCounts(segment.auxiliaryFace, 0) = totalFallback;
                    if (faceFailure)
                        for (int j = 0; j < 5; ++j)
                        {
                            output.high(segment.auxiliaryFace, j) = total[j];
                            if (controls.produceLow)
                                output.low(segment.auxiliaryFace, j) = totalLow[j];
                        }
                }
                __syncthreads(); // bank/scratch retirement before next package
            }
        }
        template <int Threads>
        void Launch(R::ExecutionContext context, const RHSGeometryPublication &geometry, RHSParametersView parameters,
                    RHSKernelControls controls, RHSFields fields, RHSFaceOutputs output, RHSFacePackageView packages, unsigned blocks)
        {
            const auto stream = static_cast<cudaStream_t>(context.stream);
            if (controls.mapping == RHSFaceMapping::WarpSegments)
                SegmentFaces<Threads, true><<<blocks, Threads, packages.sharedBytes, stream>>>(geometry.View(), parameters, controls, fields, output, packages);
            else
                SegmentFaces<Threads, false><<<blocks, Threads, packages.sharedBytes, stream>>>(geometry.View(), parameters, controls, fields, output, packages);
        }
    }
    void LaunchSegmentedFluxFaces(R::ExecutionContext context, const RHSGeometryPublication &geometry,
                                  RHSParametersView parameters, RHSKernelControls controls, RHSFields fields, RHSFaceOutputs output,
                                  RHSIndices selected, RHSFacePackageView packages)
    {
        packages.ValidateDevice(context.device);
        const auto shape = RHSFacePackages::StageShape(controls);
        const size_t sharedBytes = size_t(controls.blockSize) * (5 * sizeof(real) * (controls.produceLow ? 2 : 1) + 2 * sizeof(int) + 3 * sizeof(index)) +
                                   size_t(packages.maximumFaces) * 2 * shape.width * sizeof(real);
        DNDS_check_throw_info(controls.components == 5 && packages.mapping == controls.mapping && packages.blockSize == controls.blockSize &&
                                  packages.produceLow == controls.produceLow && packages.selection.data == selected.data &&
                                  packages.selection.RowCount() == selected.RowCount() && packages.selection.stride == selected.stride &&
                                  packages.shape.reconstruction == shape.reconstruction && packages.shape.viscous == shape.viscous && packages.shape.width == shape.width &&
                                  packages.sharedBytes == sharedBytes && packages.sharedBytes <= 48 * 1024 && packages.maximumFaces > 0 &&
                                  packages.packages.RowCount() > 0 && packages.segments.RowCount() == selected.RowCount(),
                              "RHS segments were not prepared for this exact selection/profile");
        R::CheckKernelArgumentBudget<RHSGeometryView, RHSParametersView, RHSKernelControls, RHSFields, RHSFaceOutputs, RHSFacePackageView>();
        const unsigned blocks = unsigned(std::min<index>(packages.packages.RowCount(), controls.gridBlocks ? controls.gridBlocks : geometry.MultiprocessorCount() * 4));
        switch (controls.blockSize)
        {
        case 128:
            Launch<128>(context, geometry, parameters, controls, fields, output, packages, blocks);
            break;
        case 256:
            Launch<256>(context, geometry, parameters, controls, fields, output, packages, blocks);
            break;
        case 512:
            Launch<512>(context, geometry, parameters, controls, fields, output, packages, blocks);
            break;
        default:
            DNDS_check_throw_info(false, "unsupported RHS segment block size");
        }
    }
}
