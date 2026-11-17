#pragma once

#include "CFV/Detail/ReconstructionSweepArguments.hpp"
#include "CFV/Math/ReconstructionMatrixAccess.hpp"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;

    /** @brief The single runtime-shaped FP64 reference sweep.
     * @details One CTA owns a cell. Matrix and field physical layouts remain
     * runtime-described; scratch scales with actual C*M, never a seven-value
     * ceiling. Raw B is accumulated before the filtered inverse, while
     * precomputed maps already contain that inverse. LVR's target bypasses
     * the inverse. At alpha=1 no neighbor, BC or inverse value may be read.
     * Index/placement checks belong to the wrapper; no host resource is used.
     */
    __global__ void GeneralJacobi(RawReconstructionSweepView plan, DeviceWorkSelection selection,
                                  DevicePeriodicBlocks vectors, R::PairReadView<real> mean,
                                  R::PairReadView<real> input, R::ArrayView<const real> boundary,
                                  R::ArrayView<real> output, R::ArrayView<const int> status,
                                  R::ArrayView<const real> target, R::ArrayView<const real> alpha,
                                  bool raw, bool limited)
    {
        if (status(0, 0))
            return;
        extern __shared__ real right[];
        const index n = plan.n, width = output.Width();
        for (index iWork = blockIdx.x; iWork < selection.count; iWork += gridDim.x)
        {
            const index iCell = selection.Cell(iWork);
            const real penalty = limited ? alpha(iCell, 0) : 0;
            const real scale = plan.relax * (1 - penalty);
            for (index i = threadIdx.x; i < width; i += blockDim.x)
                right[i] = 0;
            __syncthreads();
            if (scale != 0)
                for (index i = threadIdx.x; i < width; i += blockDim.x)
                {
                    const index mode = i % n, component = i / n;
                    real value = 0;
                    for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
                    {
                        const index other = plan.neighbors(entry, 0);
                        if (other == UnInitIndex)
                        {
                            for (index q = plan.boundaryOffsets(entry, 0); q < plan.boundaryOffsets(entry + 1, 0); ++q)
                                value += plan.injection(q, mode) * (boundary(q, component) - mean(iCell, component));
                            continue;
                        }
                        const bool rotates = plan.rotated && plan.rotationActive(entry, 0);
                        const index first = rotates ? vectors.First(component) : -1;
                        if (!raw || other != iCell || rotates)
                        {
                            real product = 0;
                            for (index j = 0; j < n; ++j)
                                product += Math::ReconstructionCouplingEntry(plan.matrices, plan.faces, plan.faceSide,
                                                                             entry, mode, j, n, plan.faceCoupling) *
                                           input(other, component * n + j);
                            value += product;
                            if (first >= 0)
                            {
                                real correction = 0;
                                for (index j = 0; j < n; ++j)
                                {
                                    real rotated = 0;
                                    for (index d = 0; d < 3; ++d)
                                        rotated += input(other, (first + d) * n + j) *
                                                   (plan.rotation(entry, d * 3 + component - first) - (component - first == d ? 1 : 0));
                                    correction += Math::ReconstructionCouplingEntry(plan.matrices, plan.faces, plan.faceSide,
                                                                                    entry, mode, j, n, plan.faceCoupling) *
                                                  rotated;
                                }
                                value += correction;
                            }
                        }
                        real average = mean(other, component);
                        if (first >= 0)
                        {
                            average = 0;
                            for (index d = 0; d < 3; ++d)
                                average += plan.rotation(entry, d * 3 + component - first) * mean(other, first + d);
                        }
                        value += plan.vectors(entry, mode) * (average - mean(iCell, component));
                    }
                    right[i] = value;
                }
            __syncthreads();
            for (index i = threadIdx.x; i < width; i += blockDim.x)
            {
                real value = (1 - plan.relax) * input(iCell, i);
                if (limited && penalty > 0)
                    value += plan.relax * penalty * target(iCell, i);
                if (scale != 0)
                {
                    real update = right[i];
                    if (raw)
                    {
                        update = 0;
                        for (index j = 0; j < n; ++j)
                            update += Math::ReconstructionInverseEntry(plan.inverse, iCell, i % n, j, n, plan.triangle) * right[(i / n) * n + j];
                    }
                    value += scale * update;
                }
                output(iCell, i) = value;
            }
            __syncthreads();
        }
    }

    void RunGeneralFP64(const GeneralSweepArguments &);
    void RunGeneralLimited(const GeneralSweepArguments &);
}
