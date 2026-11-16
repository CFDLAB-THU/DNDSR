#pragma once

#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "DNDS/Math/ComputeScalar.hpp"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    /// Sweep-only argument projection. Limiter tables and work lists do not
    /// belong in the kernel argument block (especially with LVR operands).
    template <class Scalar, class Layout = R::RowLayout>
    struct JacobiView
    {
        R::ArrayView<const index> offsets, neighbors, boundaryOffsets;
        R::ArrayView<const uint8_t> rotationActive;
        R::ArrayView<const Scalar, R::AccessBoundsChecks, Layout> matrixAInvB, vectorAInvB, boundaryInjection;
        R::ArrayView<const real> rotation;
        index n;
        real relax;
        bool rotated;
        explicit JacobiView(const ReconstructionDeviceView &p)
            : offsets(p.offsets), neighbors(p.neighbors), boundaryOffsets(p.boundaryOffsets),
              rotationActive(p.rotationActive), rotation(p.rotation), n(p.n), relax(p.relax), rotated(p.rotated)
        {
            const auto project = [](R::ArrayView<const Scalar> v) -> R::ArrayView<const Scalar, R::AccessBoundsChecks, Layout>
            { return {v.data, v.offsets, v.rows, v.width, v.stride, v.sizes, v.kind, v.device}; };
            if constexpr (std::is_same_v<Scalar, float>)
                matrixAInvB = project(p.matrixAInvB32), vectorAInvB = project(p.vectorAInvB32), boundaryInjection = project(p.boundaryInjection32);
            else
                matrixAInvB = project(p.matrixAInvB), vectorAInvB = project(p.vectorAInvB), boundaryInjection = project(p.boundaryInjection);
        }
    };
    // Matrix storage, dense-product arithmetic and whole-update arithmetic are
    // independent. The defaults preserve the original FP32-product/FP64-update
    // policy, including the FP64 periodic rotation before its FP32 product.
    template <bool limited, class Scalar, class AccumScalar = real, class ProductScalar = Scalar, int FixedModes = 0, int Components = 0, class Layout = R::RowLayout, class Status>
    __global__ void Jacobi(JacobiView<Scalar, Layout> plan, DeviceWorkSelection selection, DevicePeriodicBlocks blocks,
                           R::PairReadView<real> mean, R::PairReadView<real> input,
                           R::ArrayView<const real> boundary, R::ArrayView<real> output, Status status,
                           R::ArrayView<const real> target, R::ArrayView<const real> alpha)
    {
        if (status(0, 0))
            return;
        const index n = FixedModes ? FixedModes : plan.n;
        for (index iWork = blockIdx.x; iWork < selection.count; iWork += gridDim.x)
        {
            const auto iCell = selection.Cell(iWork);
            AccumScalar penalty(0);
            if constexpr (limited)
                penalty = AccumScalar(alpha(iCell, 0));
            const AccumScalar relax(plan.relax);
            const AccumScalar scale = relax * (AccumScalar(1) - penalty);
            for (index i = threadIdx.x; i < (Components ? Components * n : output.width); i += blockDim.x)
            {
                const index mode = i % n, component = i / n;
                AccumScalar value = AccumScalar(input(iCell, i)) * (AccumScalar(1) - relax);
                if constexpr (limited)
                    if (penalty > 0)
                        value += (relax * penalty) * AccumScalar(target(iCell, i));
                // A fully constrained cell must not read a neighbor or BC.
                // Post-blending an unlimited sweep would propagate NaNs from
                // values which the CPU LVR operator deliberately never reads.
                if (scale == AccumScalar(0))
                {
                    output(iCell, i) = Math::ScalarCast<real>(value);
                    continue;
                }
                for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
                {
                    const auto other = plan.neighbors(entry, 0);
                    if (other != UnInitIndex)
                    {
                        // The original quantized policy uses an FP32 product
                        // and FP64 outer sum. The full policies instead use the
                        // chosen scalar throughout, casting FP64 inputs directly
                        // and retaining the already quantized FP32 maps.
                        ProductScalar product(0);
                        for (index j = 0; j < n; ++j)
                            product += ProductScalar(plan.matrixAInvB(entry, j * n + mode)) * ProductScalar(input(other, component * n + j));
                        value += scale * AccumScalar(product);
                        const auto first = plan.rotated && plan.rotationActive(entry, 0)
                                               ? blocks.First(component)
                                               : -1;
                        if (first >= 0)
                        {
                            ProductScalar correction(0);
                            for (index j = 0; j < n; ++j)
                            {
                                AccumScalar rotated(0);
                                for (index d = 0; d < 3; ++d)
                                    rotated += AccumScalar(input(other, (first + d) * n + j)) *
                                               (AccumScalar(plan.rotation(entry, d * 3 + component - first)) - AccumScalar(component - first == d ? 1 : 0));
                                correction += ProductScalar(plan.matrixAInvB(entry, j * n + mode)) * ProductScalar(rotated);
                            }
                            value += scale * AccumScalar(correction);
                        }
                        AccumScalar otherMean(mean(other, component));
                        if (first >= 0)
                        {
                            otherMean = AccumScalar(0);
                            for (index d = 0; d < 3; ++d)
                                otherMean += AccumScalar(plan.rotation(entry, d * 3 + component - first)) * AccumScalar(mean(other, first + d));
                        }
                        value += AccumScalar(plan.vectorAInvB(entry, mode)) * (scale * (otherMean - AccumScalar(mean(iCell, component))));
                    }
                    else
                        for (index q = plan.boundaryOffsets(entry, 0); q < plan.boundaryOffsets(entry + 1, 0); ++q)
                            value += AccumScalar(plan.boundaryInjection(q, mode)) * (scale * (AccumScalar(boundary(q, component)) - AccumScalar(mean(iCell, component))));
                }
                output(iCell, i) = Math::ScalarCast<real>(value);
            }
        }
    }
    /// Borrowed host-only launch descriptor. Never retained past submission;
    /// the public operator owns scheduled operands and completion lifetimes.
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
    template <class Storage, class Accum, class Product, bool Limited = false>
    void RunGeneralShapes(const GeneralSweepArguments &a)
    {
        const auto modes = [&](auto n)
        {
            const auto components = [&](auto c)
            {
                Jacobi<Limited, Storage, Accum, Product, decltype(n)::value, decltype(c)::value><<<
                    unsigned(std::min<index>(a.selection.count, 256)), 128, 0, static_cast<cudaStream_t>(a.context.stream)>>>(
                    JacobiView<Storage>(a.plan), a.selection, a.vectors, a.mean, a.input, a.boundary, a.output, a.status, a.target, a.alpha);
            };
            switch (a.mean.owned.width)
            {
            case 5:
                components(std::integral_constant<int, 5>{});
                break;
            case 6:
                components(std::integral_constant<int, 6>{});
                break;
            case 7:
                components(std::integral_constant<int, 7>{});
                break;
            default:
                components(std::integral_constant<int, 0>{});
                break;
            }
        };
        switch (a.plan.n)
        {
        case 2:
            modes(std::integral_constant<int, 2>{});
            break;
        case 3:
            modes(std::integral_constant<int, 3>{});
            break;
        case 5:
            modes(std::integral_constant<int, 5>{});
            break;
        case 9:
            modes(std::integral_constant<int, 9>{});
            break;
        case 19:
            modes(std::integral_constant<int, 19>{});
            break;
        default:
            modes(std::integral_constant<int, 0>{});
            break;
        }
    }
    void RunGeneralFP64(const GeneralSweepArguments &);
    void RunGeneralFP32(const GeneralSweepArguments &, bool full);
    void RunGeneralFloatFloat(const GeneralSweepArguments &);
    void RunGeneralLimited(const GeneralSweepArguments &);
}
