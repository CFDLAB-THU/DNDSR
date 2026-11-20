#pragma once

#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"

namespace DNDS::Cauchy::CPU
{
    /// Explicit numeric operands for one auxiliary phase. All field views are
    /// pinned by AuxiliaryPreparation, which also owns the worker lease.
    struct AuxiliaryKernelRequest
    {
        FiniteVolumeStencilView stencil;
        CFV::FiniteVolumeHostView metric;
        SpanView<const EquationBlockReadView> blocks;
        SpanView<const RHSFieldView> gradients;
        SpanView<const EquationBlockLocation> locations;
        SpanView<const index> vectorStarts;
        SpanView<const BoundaryCondition> conditions;
        WallDistanceView wall;
        bool hasWall = false;
        std::optional<MeanThermodynamicView> derived;
        std::array<bool, 8> enabled{};
        std::array<RHSOutputView, 8> values;
        RHSOutputView scratch;
        bool useScratch = false;
        RHSScalarView incoming;
        RigidFrame frame;
        index components = 0, flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        int dimension = 0, independentSpecies = 0;
        real time = 0;
    };
    /// Operator-allocated point scratch. Each worker has an exclusive evaluator;
    /// no chemistry state or dynamic point storage is shared across OMP rows.
    struct AuxiliaryWorkerScratch
    {
        PointEvaluator evaluator;
        EquationFVDetail::PointScratch left, right, face;
        NumericalStatus status;
        AuxiliaryWorkerScratch(PointEvaluator value, index components)
            : evaluator(std::move(value)), left(components), right(components), face(components) {}
    };
    NumericalStatus PrepareAuxiliaryMeans(const AuxiliaryKernelRequest &request,
                                          const EquationSystem &equation, SpanView<AuxiliaryWorkerScratch> workers,
                                          index first, index end) noexcept;
    NumericalStatus PrepareAuxiliaryFaces(const AuxiliaryKernelRequest &request,
                                          const EquationSystem &equation, const BoundaryPointOperator &boundary,
                                          SpanView<AuxiliaryWorkerScratch> workers,
                                          FiniteVolumeStencilView::Indices faces) noexcept;
    void ReduceAuxiliaryCells(const AuxiliaryKernelRequest &request,
                              FiniteVolumeStencilView::Indices cells);
}
