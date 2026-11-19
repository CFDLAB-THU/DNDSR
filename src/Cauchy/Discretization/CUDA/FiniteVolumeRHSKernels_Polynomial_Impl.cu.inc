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
    __global__ __launch_bounds__(512) void PerfectGasFluxCTASegments(
        RHSGeometryView publication, RHSParametersView parameters, RHSKernelControls controls,
        RHSShapeFields<Components, Modes, Dimension> fields,
        RHSFaceOutputsT<Runtime::FixedRows<Components, Components>> output, RHSFacePackageView prepared)
    {
        extern __shared__ real shared[];
        const index stageWords = index(prepared.maximumFaces) * 2 * prepared.shape.width;
        real *high = shared + stageWords * (controls.doubleBuffer ? 2 : 1);
        real *low = high + Components * int(blockDim.x);
        unsigned char *tail = reinterpret_cast<unsigned char *>(low + (controls.produceLow ? Components * int(blockDim.x) : 0));
        int *codes = reinterpret_cast<int *>(tail);
        int *details = codes + int(blockDim.x);
        index *rows = reinterpret_cast<index *>(details + int(blockDim.x));
        index *points = rows + int(blockDim.x);
        index *fallback = points + int(blockDim.x);
        const auto &geometry = publication(0, 0);
        const auto &physics = parameters(0, 0);
        const auto stage = [&](RHSFacePackage package, SpanView<real, false> bank)
        {
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
#endif
        };
        int activeBank = 0;
        if (controls.doubleBuffer && blockIdx.x < prepared.packages.RowCount())
            stage(prepared.packages(blockIdx.x, 0), {shared, stageWords});
        for (index iPackage = blockIdx.x; iPackage < prepared.packages.RowCount(); iPackage += gridDim.x)
        {
            const auto package = prepared.packages(iPackage, 0);
            const SpanView<real, false> bank{shared + activeBank * stageWords, stageWords};
            if (!controls.doubleBuffer)
                stage(package, bank);
#if __CUDA_ARCH__ >= 800
            __pipeline_wait_prior(0);
#endif
            __syncthreads();
            const index next = iPackage + gridDim.x;
            if (controls.doubleBuffer && next < prepared.packages.RowCount())
                stage(prepared.packages(next, 0), {shared + (1 - activeBank) * stageWords, stageWords});
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
            const bool warpTree = controls.reduction == RHSFaceReduction::WarpTree && package.warpContained;
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
                if (!warpTree)
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
                else if (warpTree)
                {
                    // All lanes participate. The setup proves that every
                    // active addition stays inside its face segment and warp.
                    for (int stride = 1; stride < package.maximumLanes; stride *= 2)
                    {
                        const bool add = ownSlot >= 0 && localLane % (2 * stride) == 0 &&
                                         localLane + stride < segment.lanes;
                        for (int j = 0; j < Components; ++j)
                        {
                            const real peer = __shfl_down_sync(0xffffffffu, value[j], stride);
                            if (add)
                                value[j] += peer;
                            if (controls.produceLow)
                            {
                                const real peerLow = __shfl_down_sync(0xffffffffu, valueLow[j], stride);
                                if (add)
                                    valueLow[j] += peerLow;
                            }
                        }
                    }
                    if (ownSlot >= 0 && localLane == 0)
                        for (int j = 0; j < Components; ++j)
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
                    const int treeWidth = controls.reduction == RHSFaceReduction::Segmented ? int(blockDim.x) : package.maximumLanes;
                    for (int stride = 1; stride < treeWidth; stride *= 2)
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
            if (controls.doubleBuffer)
                activeBank = 1 - activeBank;
        }
    }

    template <int Width>
    using NativeThreadRows = Runtime::PairReadView<real, Runtime::AccessBoundsChecks,
                                                   Runtime::FixedRows<(Width ? Width : 1), (Width ? Width : 1)>>;

    template <int Width>
    struct BoundFaceRows
    {
        using Row = typename NativeThreadRows<Width>::RowType;
        index own = UnInitIndex;
        Row left, right;
        __device__ Row operator[](index cell) const { return cell == own ? left : right; }
        __device__ real operator()(index cell, index column) const { return (*this)[cell][column]; }
    };

    template <int C, int M, int D, bool Bind>
    __device__ auto ThreadOperands(const RHSShapeFields<C, M, D> &input, const RHSKernelControls &controls,
                                   const FiniteVolumeDetail::SampleRecord &anchor)
    {
        using Native = StaticFaceOperands<C, M, D, NativeThreadRows>;
        using Bound = StaticFaceOperands<C, M, D, BoundFaceRows>;
        std::conditional_t<Bind, Bound, Native> fields{};
        const auto source = [&](auto pair)
        {
            if constexpr (Bind)
            {
                constexpr int width = int(decltype(pair)::row_width);
                using Row = typename decltype(pair)::RowType;
                return BoundFaceRows<width>{
                    anchor.cell, pair.RowCount() ? pair[anchor.cell] : Row{},
                    pair.RowCount() && anchor.other != UnInitIndex ? pair[anchor.other] : Row{}};
            }
            else
                return pair;
        };
        fields.mean = source(input.mean);
        if constexpr (M == 0)
        {
            fields.gradients = source(input.reconstruction);
            if (controls.distinctViscous && !controls.primitiveGradient)
                fields.viscousGradients = source(input.viscous.gradients);
        }
        else
        {
            fields.coefficients = source(input.reconstruction);
            if (controls.distinctViscous && !controls.primitiveGradient)
                fields.viscousCoefficients = source(input.viscous.coefficients);
        }
        if (controls.primitiveGradient)
            fields.primitiveGradients = source(input.viscous.primitive);
        fields.waveJump = source(input.waveJump);
        fields.pressureMinimum = source(input.pressureMinimum);
        fields.ransP1Gradients = input.ransP1Gradients;
        fields.incoming = input.incoming;
        fields.hasIncoming = input.hasIncoming;
        fields.hasWave = input.hasWave;
        fields.hasPressure = input.hasPressure;
        return fields;
    }

    /// The prepared means are a whole-array local-only prefix whose parent
    /// pitch may exceed its width. Preserve that borrowed view in both modes.
    template <class Base, class Prepared>
    struct ThreadPointOperands : Base
    {
        Prepared preparedMean;
    };

    template <bool Bind, int C, int M, int D, class Output>
    __device__ NumericalStatus EvaluateThreadFace(const RHSGeometry &geometry, const RHSKernelParameters &physics,
                                                   RHSKernelControls controls, const RHSShapeFields<C, M, D> &input,
                                                   const Output &output, index face, index &fallbacks)
    {
        const auto anchor = geometry.FaceAnchor(face);
        auto fields = ThreadOperands<C, M, D, Bind>(input, controls, anchor);
        fields.hasPreparedMean = input.hasPreparedMean;
        const ThreadPointOperands<decltype(fields), decltype(input.preparedMean)> point{fields, input.preparedMean};
        return EvaluateFace<C>(geometry, physics, controls, point, output, face, fallbacks);
    }

    template <int Components, int Modes, int Dimension, ReconstructionForm Form>
    __global__ void PerfectGasFluxFaceThreads(
        RHSGeometryView publication, RHSParametersView parameters, RHSKernelControls controls,
        RHSShapeFields<Components, Modes, Dimension> fields,
        RHSFaceOutputsT<Runtime::FixedRows<Components, Components>> output, RHSIndices selected)
    {
        controls.direct = Form != ReconstructionForm::Modal;
        controls.linear = Form == ReconstructionForm::DirectGradient;
        controls.firstOrder = Form == ReconstructionForm::MeanOnly;
        const auto &geometry = publication(0, 0);
        const auto &physics = parameters(0, 0);
        const index thread = index(blockIdx.x) * blockDim.x + threadIdx.x;
        for (index row = thread; row < selected.RowCount(); row += index(gridDim.x) * blockDim.x)
        {
            const index face = selected(row, 0);
            index fallbacks = 0;
            NumericalStatus status{NumericalError::InvalidInput, 1};
            if (physics.physics.rans.Supported() && physics.physics.rans.ComponentNumber() + 5 == Components &&
                physics.physics.dimension == Dimension && (!controls.viscous || physics.physics.viscous))
            {
                status = controls.hoistFaceRows
                             ? EvaluateThreadFace<true>(geometry, physics, controls, fields, output, face, fallbacks)
                             : EvaluateThreadFace<false>(geometry, physics, controls, fields, output, face, fallbacks);
            }
            output.status(face, 0) = status;
            output.fallbackCounts(face, 0) = fallbacks;
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
                                  (a.controls.mapping == RHSFaceMapping::CTASegments || a.controls.mapping == RHSFaceMapping::Thread),
                              "CUDA RHS static C/M/D launch does not match the validated profile");
        const auto fields = ProjectRHSShapeFields<Components, Modes, Dimension>(a.controls, a.fields);
        const auto output = ProjectRHSOutputs<Components>(a.output);
        if (a.controls.mapping == RHSFaceMapping::Thread)
        {
            Runtime::CheckKernelArgumentBudget<RHSGeometryView, RHSParametersView, RHSKernelControls,
                                               decltype(fields), decltype(output), RHSIndices>();
            const index blocks = std::min<index>((a.selected.RowCount() + a.controls.blockSize - 1) / a.controls.blockSize,
                                                 a.controls.gridBlocks ? a.controls.gridBlocks : 1024);
            const auto run = [&](auto form)
            {
                PerfectGasFluxFaceThreads<Components, Modes, Dimension, decltype(form)::value>
                    <<<unsigned(blocks), a.controls.blockSize, 0, Runtime::CUDAStream(a.context)>>>(
                        a.geometry.View(), a.parameters, a.controls, fields, output, a.selected);
            };
            if constexpr (Modes > 0)
                run(std::integral_constant<ReconstructionForm, ReconstructionForm::Modal>{});
            else if (!a.controls.firstOrder && a.controls.linear)
                run(std::integral_constant<ReconstructionForm, ReconstructionForm::DirectGradient>{});
            else
                run(std::integral_constant<ReconstructionForm, ReconstructionForm::MeanOnly>{});
            return;
        }
        Runtime::CheckKernelArgumentBudget<RHSGeometryView, RHSParametersView, RHSKernelControls,
                                           decltype(fields), decltype(output), RHSFacePackageView>();
        PerfectGasFluxCTASegments<Components, Modes, Dimension><<<a.segmentBlocks, a.controls.blockSize, a.packages.sharedBytes,
                                                         ::DNDS::Runtime::CUDAStream(a.context)>>>(
            a.geometry.View(), a.parameters, a.controls, fields, output, a.packages);
    }
}
