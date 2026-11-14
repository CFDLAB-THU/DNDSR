#pragma once

#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    /// Host launch projection after the public wrapper's shape/residency check.
    /// Retains exact parent pitch/width; does not acquire access or retain data.
    /// Empty optional/ghost views carry no elements and need no matching layout.
    template <class Policy>
    Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Policy> ProjectRHSField(const RHSField &field)
    {
        static_assert(Policy::layout_kind == Runtime::RowLayoutKind::Table ||
                      Policy::layout_kind == Runtime::RowLayoutKind::TiledColMajorTable);
        const auto project = [](auto view)
        {
            DNDS_check_throw_info(!view.RowCount() ||
                                      (view.kind == Policy::layout_kind &&
                                       (Policy::layout_kind != Runtime::RowLayoutKind::TiledColMajorTable ||
                                        view.stride == index(Policy::component_stride))),
                                  "RHS static input projection differs from physical layout");
            return Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Policy>{
                view.data, view.offsets, view.rows, view.width, view.stride,
                view.sizes, Policy::layout_kind, view.device};
        };
        return {project(field.owned), project(field.ghost)};
    }

    /// Select static mean/coefficient indexing once per launch. Reconstruction
    /// and distinct viscous coefficient inputs share the chosen coefficient
    /// policy. Scalar stencil signals keep their uniform-row views unchanged.
    template <class MeanPolicy, class CoefficientPolicy>
    RHSFieldsT<MeanPolicy, CoefficientPolicy> ProjectRHSFields(const RHSFields &fields)
    {
        RHSFieldsT<MeanPolicy, CoefficientPolicy> result;
        result.mean = ProjectRHSField<MeanPolicy>(fields.mean);
        result.reconstruction = ProjectRHSField<CoefficientPolicy>(fields.reconstruction);
        result.viscousData = ProjectRHSField<CoefficientPolicy>(fields.viscousData);
        result.waveJump = fields.waveJump;
        result.pressureMinimum = fields.pressureMinimum;
        result.preparedMean = fields.preparedMean;
        result.incoming = fields.incoming;
        result.hasPreparedMean = fields.hasPreparedMean;
        result.hasIncoming = fields.hasIncoming;
        result.hasWave = fields.hasWave;
        result.hasPressure = fields.hasPressure;
        return result;
    }
}
