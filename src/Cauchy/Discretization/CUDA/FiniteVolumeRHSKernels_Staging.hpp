#pragma once

#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Launch.hpp"
#include "DNDS/Core/Span.hpp"
#include <cuda_pipeline.h>

namespace DNDS::Cauchy::CUDA::Detail
{
    /** @brief One face's two parent rows in a shared-memory staging bank.
     * @details This is a borrowed numerical adapter, not a Runtime allocation.
     * The launch owns both banks until every consuming lane finishes. Cell IDs
     * are retained, including periodic self-neighbors; identical IDs denote
     * the same native state before the point arithmetic applies its image map.
     */
    struct StagedFaceField
    {
        SpanView<const real, false> values;
        index own = UnInitIndex, other = UnInitIndex, width = 0, pitch = 0, first = 0;
        DNDS_DEVICE_CALLABLE real operator()(index iCell, index j) const
        {
            return values[(iCell == own ? 0 : 1) * pitch + first + j];
        }
    };
    struct StagedFaceFields
    {
        StagedFaceField mean, reconstruction, viscousData, waveJump, pressureMinimum, preparedMean;
        RHSInput incoming;
        bool hasPreparedMean = false, hasIncoming = false, hasWave = false, hasPressure = false;
    };
    using FaceStageShape = RHSFaceStageShape;

    template <class Fields>
    __device__ const real *FaceStageSource(const Fields &fields, const RHSKernelControls &controls,
                                           FaceStageShape shape, index cell, int column)
    {
        if (cell == UnInitIndex)
            return nullptr;
        if (column < 5)
            return &fields.mean(cell, column);
        column -= 5;
        if (column < shape.reconstruction)
            return &fields.reconstruction(cell, column);
        column -= shape.reconstruction;
        if (column < shape.viscous)
            return &fields.viscousData(cell, column);
        column -= shape.viscous;
        if (column < 2)
            return fields.hasPreparedMean ? &fields.preparedMean(cell, column) : nullptr;
        if (column == 2)
            return controls.useStencil && controls.fluxPolicy.transverseSignal ? &fields.waveJump(cell, 0) : nullptr;
        return controls.useStencil && controls.fluxPolicy.pressureStencil ? &fields.pressureMinimum(cell, 0) : nullptr;
    }
    template <class Fields>
    __device__ void StageFaceBatch(SpanView<real, false> bank, FaceStageShape shape, index begin, int batchFaces,
                                   const RHSGeometry &geometry, const RHSKernelControls &controls,
                                   const Fields &fields, RHSIndices selected)
    {
        const index words = index(batchFaces) * 2 * shape.width;
        for (index iWord = threadIdx.x; iWord < words; iWord += blockDim.x)
        {
            const index iRow = begin + iWord / (2 * shape.width);
            const int side = (iWord / shape.width) % 2, column = iWord % shape.width;
            const real *source = nullptr;
            if (iRow < selected.RowCount())
            {
                const auto &sample = geometry.FaceAnchor(selected(iRow, 0));
                source = FaceStageSource(fields, controls, shape, side ? sample.other : sample.cell, column);
            }
            if (source)
            {
#if __CUDA_ARCH__ >= 800
                // Shared destination and FP64 source are naturally 8-byte
                // aligned, including tiled input layouts. No host staging.
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
    }
    __device__ inline StagedFaceFields BindFaceStage(SpanView<const real, false> bank, FaceStageShape shape,
                                                     int slot, const FiniteVolumeDetail::SampleRecord &sample,
                                                     const RHSKernelControls &controls, const RHSFields &fields)
    {
        const auto values = bank.Subspan(index(slot) * 2 * shape.width, 2 * shape.width);
        const auto field = [&](int first, int width)
        { return StagedFaceField{values, sample.cell, sample.other, width, shape.width, first}; };
        StagedFaceFields result;
        result.mean = field(0, 5);
        result.reconstruction = field(5, shape.reconstruction);
        result.viscousData = controls.distinctViscous || controls.primitiveGradient
                                 ? field(shape.ViscousFirst(), shape.viscous)
                                 : result.reconstruction;
        result.preparedMean = field(shape.AuxFirst(), 2);
        result.waveJump = field(shape.AuxFirst() + 2, 1);
        result.pressureMinimum = field(shape.AuxFirst() + 3, 1);
        result.incoming = fields.incoming;
        result.hasPreparedMean = fields.hasPreparedMean;
        result.hasIncoming = fields.hasIncoming;
        result.hasWave = fields.hasWave;
        result.hasPressure = fields.hasPressure;
        return result;
    }
    /** Two shared banks and explicit block-uniform retirement.
     * Prefetch the next batch with cp.async before consuming the current one.
     * A bank is reused only after all consumers and all producers have joined.
     * Immutable geometry/quad/basis remains read-only in the common plan; each
     * slot stages changing field inputs, not another quadrature/basis cache.
     */
    template <bool WarpQuadrature>
    __global__ void FluxFacesBuffered(RHSGeometryView published, RHSParametersView parameters,
                                      RHSKernelControls controls, RHSFields fields,
                                      RHSFaceOutputs output, RHSIndices selected, FaceStageShape shape, int batchFaces)
    {
        extern __shared__ real storage[];
        const index bankWords = index(batchFaces) * 2 * shape.width;
        SpanView<real, false> banks[2]{{storage, bankWords}, {storage + bankWords, bankWords}};
        const auto &geometry = published(0, 0);
        const index stride = index(gridDim.x) * batchFaces;
        index begin = index(blockIdx.x) * batchFaces;
        if (begin >= selected.RowCount())
            return;
        int active = 0;
        StageFaceBatch(banks[active], shape, begin, batchFaces, geometry, controls, fields, selected);
#if __CUDA_ARCH__ >= 800
        __pipeline_wait_prior(0);
#endif
        __syncthreads();
        for (; begin < selected.RowCount(); begin += stride, active = 1 - active)
        {
            const bool next = begin + stride < selected.RowCount();
            if (next)
                StageFaceBatch(banks[1 - active], shape, begin + stride, batchFaces, geometry, controls, fields, selected);
            const int slot = WarpQuadrature ? int(threadIdx.x / 32) : int(threadIdx.x);
            if (slot < batchFaces && begin + slot < selected.RowCount())
            {
                const auto face = selected(begin + slot, 0);
                const auto &sample = geometry.FaceAnchor(face);
                const auto staged = BindFaceStage(banks[active], shape, slot, sample, controls, fields);
                index fallbacks = 0;
                const auto &physics = parameters(0, 0).physics;
                const auto status = physics.rans.Supported() && physics.rans.ComponentNumber() == 0 &&
                                            physics.dimension == controls.dimension && (!controls.viscous || physics.viscous)
                                        ? EvaluateFace<5, real, real, real, WarpQuadrature>(
                                              geometry, parameters(0, 0), controls, staged, output, face, fallbacks)
                                        : NumericalStatus{NumericalError::InvalidInput, 1};
                if (!WarpQuadrature || threadIdx.x % 32 == 0)
                {
                    output.status(face, 0) = status;
                    output.fallbackCounts(face, 0) = fallbacks;
                }
            }
#if __CUDA_ARCH__ >= 800
            __pipeline_wait_prior(0);
#endif
            __syncthreads();
        }
    }
    inline void LaunchBufferedFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                                        RHSParametersView parameters, RHSKernelControls controls, RHSFields fields,
                                        RHSFaceOutputs output, RHSIndices selected)
    {
        const auto shape = RHSFacePackages::StageShape(controls);
        // Bound the two banks to 32 KiB without requiring opt-in shared memory.
        // Thread/face batches are small at P3; report this occupancy tradeoff.
        const int capacity = (32 * 1024) / (int(sizeof(real)) * 4 * shape.width);
        const int batchFaces = std::min((controls.mapping == RHSFaceMapping::WarpQuadrature) ? controls.blockSize / 32 : controls.blockSize, capacity);
        DNDS_check_throw_info(batchFaces > 0, "buffered face batch does not fit shared memory");
        const unsigned blocks = unsigned(std::min<index>((selected.RowCount() + batchFaces - 1) / batchFaces,
                                                         controls.gridBlocks ? index(controls.gridBlocks) : index(geometry.MultiprocessorCount() * 4)));
        const auto bytes = index(batchFaces) * 4 * shape.width * sizeof(real);
        Runtime::CheckKernelArgumentBudget<RHSGeometryView, RHSParametersView, RHSKernelControls, RHSFields, RHSFaceOutputs, RHSIndices, FaceStageShape, int>();
        const auto stream = static_cast<cudaStream_t>(context.stream);
        if ((controls.mapping == RHSFaceMapping::WarpQuadrature))
            FluxFacesBuffered<true><<<blocks, controls.blockSize, bytes, stream>>>(geometry.View(), parameters, controls, fields, output, selected, shape, batchFaces);
        else
            FluxFacesBuffered<false><<<blocks, controls.blockSize, bytes, stream>>>(geometry.View(), parameters, controls, fields, output, selected, shape, batchFaces);
    }
}
