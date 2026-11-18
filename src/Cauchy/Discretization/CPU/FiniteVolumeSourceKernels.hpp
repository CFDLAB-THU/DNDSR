#pragma once
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Motion/RigidMotion.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::Cauchy::CPU
{
    struct SourceGradientView
    {
        RHSScalarView values;
        int componentStride = 1, direction = 0;
        real operator()(index iPoint, index component) const
        {
            return values(iPoint, component * componentStride + direction);
        }
    };
    struct SourceKernelRequest
    {
        CFV::FiniteVolumeHostView metric;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> offsets;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> weights, lengths;
        SpanView<const EquationBlockReadView> blocks;
        SpanView<const EquationBlockLocation> locations;
        SpanView<const index> logicalFirst;
        SpanView<const RHSScalarView> samples;
        SpanView<const std::array<SourceGradientView, 3>> gradients;
        WallDistanceView walls;
        EquationBlockReadView ransMean;
        EquationBlockArrayView<real> speciesOutput, ransOutput;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> matrix;
        std::optional<EquationFVDetail::MeanLinearizationOutput> linearization;
        index cells = 0, components = 0, flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        int dimension = 0, independentSpecies = 0;
        bool chemical = false, fullBlock = false;
        bool cellInputs = false; ///< Direct FV takes its mean and physical gradient from the parent cell.
        real scale = 1;
        bool rowMajorBlocks = false;
    };
    /// Exclusive physics worker and point scratch retained by the operator.
    /// Construct before submission with the same property request as the serial
    /// source path; no chemistry state is shared between worker slots.
    struct SourceWorkerScratch
    {
        PointEvaluator evaluator;
        EquationFVDetail::PointScratch point;
        NumericalStatus status;
        SourceWorkerScratch(PointEvaluator value, index components)
            : evaluator(std::move(value)), point(components) {}
    };
    NumericalStatus AssembleCenterSourceJacobian(const SourceKernelRequest &, const EquationSystem &, SpanView<SourceWorkerScratch>) noexcept;
    NumericalStatus AddQuadratureSourceJacobian(const SourceKernelRequest &, const EquationSystem &, SpanView<SourceWorkerScratch>) noexcept;
    NumericalStatus EvaluateVolumeSource(const SourceKernelRequest &, const EquationSystem &, SpanView<SourceWorkerScratch>) noexcept;
    NumericalStatus AssembleCenterSourceJacobian(const SourceKernelRequest &, const EquationSystem &, PointEvaluator &) noexcept;
    NumericalStatus AddQuadratureSourceJacobian(const SourceKernelRequest &, const EquationSystem &, PointEvaluator &) noexcept;
    NumericalStatus EvaluateVolumeSource(const SourceKernelRequest &, const EquationSystem &, PointEvaluator &) noexcept;
    void AddEquationRotationSource(CFV::FiniteVolumeHostView metric, const RigidFrame &frame,
                                   EquationBlockReadView flow, EquationBlockArrayView<real> output,
                                   std::optional<EquationBlockArrayView<real>> lowOutput);
}
