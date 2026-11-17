#pragma once

#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels.hpp"
#include "DNDS/Runtime/Execution/KernelViews.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    /// Host launch projection after the public wrapper's shape/residency check.
    /// Retains exact parent pitch/width; does not acquire access or retain data.
    /// Empty optional/ghost views carry no elements and need no matching layout.
    template <class Policy>
    Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Policy> ProjectRHSField(const RHSField &field)
    {
        return Runtime::ProjectKernelPair<Policy>(field.DynamicView());
    }

    /** @brief Compact physical operands for one C/M/D specialization.
     * @details The public descriptor uses the same allocation slot for modal
     * coefficients or direct gradients. Resolve that role once on the host.
     * The optional viscous union retains exactly one active typed whole-array
     * view: C*M coefficients, C*D conservative gradients, or 5*D primitive
     * gradients. Carrying all three would exceed the launch argument budget.
     * Shared-memory point operands expose the roles separately, without new
     * allocations, uploads, template axes or per-element validation.
     */
    template <int Components, int Modes, int Dimension>
    struct RHSShapeFields
    {
        static constexpr int static_modes = Modes, static_dimension = Dimension;
        template <int Width>
        using Pair = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<Width, Width>>;
        static constexpr int reconstruction_width = Components * (Modes ? Modes : Dimension);
        using Reconstruction = Pair<reconstruction_width>;
        struct ConservativeGradientTag
        {
        };
        struct PrimitiveGradientTag
        {
        };
        union ViscousInput
        {
            Reconstruction coefficients;
            Pair<Components * Dimension> gradients;
            Pair<5 * Dimension> primitive;
            ViscousInput() : coefficients{} {}
            explicit ViscousInput(Reconstruction value) : coefficients(value) {}
            ViscousInput(ConservativeGradientTag, Pair<Components * Dimension> value) : gradients(value) {}
            ViscousInput(PrimitiveGradientTag, Pair<5 * Dimension> value) : primitive(value) {}
        };
        static_assert(std::is_trivially_copyable_v<ViscousInput>);
        Pair<Components> mean;
        Reconstruction reconstruction;
        ViscousInput viscous;
        Pair<1> waveJump, pressureMinimum;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> preparedMean;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<Components, Components>> incoming;
        bool hasPreparedMean = false, hasIncoming = false, hasWave = false, hasPressure = false;
    };

    template <int Components, int Modes, int Dimension>
    RHSShapeFields<Components, Modes, Dimension> ProjectRHSShapeFields(const RHSKernelControls &controls, const RHSFields &fields)
    {
        using Fields = RHSShapeFields<Components, Modes, Dimension>;
        using R = Runtime::FixedRows<Components, Components>;
        using Reconstruction = Runtime::FixedRows<Fields::reconstruction_width, Fields::reconstruction_width>;
        using G = Runtime::FixedRows<Components * Dimension, Components * Dimension>;
        Fields result;
        result.mean = ProjectRHSField<R>(fields.mean);
        const bool reconstruct = !controls.firstOrder && (!controls.direct || controls.linear);
        if (reconstruct)
        {
            result.reconstruction = ProjectRHSField<Reconstruction>(fields.reconstruction);
            if (controls.viscous && controls.distinctViscous && !controls.primitiveGradient)
            {
                if constexpr (Modes == 0)
                    result.viscous = typename Fields::ViscousInput(typename Fields::ConservativeGradientTag{}, ProjectRHSField<G>(fields.viscousData));
                else
                    result.viscous = typename Fields::ViscousInput(ProjectRHSField<Reconstruction>(fields.viscousData));
            }
        }
        if (controls.primitiveGradient)
            result.viscous = typename Fields::ViscousInput(typename Fields::PrimitiveGradientTag{},
                                                           ProjectRHSField<Runtime::FixedRows<5 * Dimension, 5 * Dimension>>(fields.viscousData));
        if (fields.hasPreparedMean)
        {
            // Auxiliary storage may also contain diffusion/enthalpy columns.
            // RHS stages only [pressure, soundSpeed]; narrow that prefix while
            // retaining its real parent pitch and device descriptor.
            auto prefix = fields.preparedMean.DynamicView();
            prefix.width = 2;
            result.preparedMean = Runtime::ProjectKernelView<Runtime::FixedRows<2>>(prefix);
        }
        if (fields.hasIncoming)
            result.incoming = Runtime::ProjectKernelView<R>(fields.incoming.DynamicView());
        if (fields.hasWave)
            result.waveJump = ProjectRHSField<Runtime::FixedRows<1, 1>>(fields.waveJump);
        if (fields.hasPressure)
            result.pressureMinimum = ProjectRHSField<Runtime::FixedRows<1, 1>>(fields.pressureMinimum);
        result.hasPreparedMean = fields.hasPreparedMean;
        result.hasIncoming = fields.hasIncoming;
        result.hasWave = fields.hasWave;
        result.hasPressure = fields.hasPressure;
        return result;
    }

    /// Output projection changes only typed metadata. Row pitch is retained.
    template <int Components>
    RHSFaceOutputsT<Runtime::FixedRows<Components, Components>> ProjectRHSOutputs(const RHSFaceOutputs &output)
    {
        return {Runtime::ProjectKernelView<Runtime::FixedRows<Components, Components>>(output.high.DynamicView()),
                Runtime::ProjectKernelView<Runtime::FixedRows<Components, Components>>(output.low.DynamicView()),
                output.wallForce, output.status, output.fallbackCounts};
    }
}
