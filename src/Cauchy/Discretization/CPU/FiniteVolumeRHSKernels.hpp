#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Discretization/FiniteVolumeKernelPolicy.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "Cauchy/State/EquationViews.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Physics/ViscousFlux.hpp"

namespace DNDS::Cauchy::CPU
{
    /// Geometry, flags and explicit operands of the fixed-size flow5 leaf.
    /// No FiniteVolume/Workspace owner or access acquisition crosses this boundary.
    struct FlowRHSRequest
    {
        FiniteVolumeStencilView stencil;
        CFV::FiniteVolumeHostView metric;
        std::optional<CFV::ReconstructionBasisEvaluationView> basis;
        SpanView<const BoundaryCondition> conditions;
        FiniteVolumeDetail::FluxKernelPolicy fluxPolicy;
        ThermalBounds thermalBounds;
        RigidFrame frame;
        index ownedCells = 0, allCells = 0, forceSlots = 0;
        int dimension = 0, modes = 0;
        bool direct = false, linear = false, firstOrder = false;
        bool polynomialViscosity = false, hasLocalInterfaceRows = false, useStencil = false;
        real time = 0;
    };
    /// Small host worker scratch, allocated/reset by the C++ operator wrapper.
    /// One disjoint slot per worker, merged after this kernel returns. Field
    /// outputs remain explicit Array views; numerical loops acquire no leases.
    struct FlowRHSWorkerBuffers
    {
        SpanView<RHSKernelDiagnostics> diagnostics;
        SpanView<NumericalStatus> status;
    };
    NumericalStatus EvaluateFlowFlux(const FlowRHSRequest &, const EquationSystem &,
                                     const Riemann::RiemannFlux &, const RHSReadViews &,
                                     const RHSWriteViews &, CFV::ReconstructionRegion,
                                     FlowRHSWorkerBuffers) noexcept;
    NumericalStatus AddFlowRotationSource(const FlowRHSRequest &, const RHSReadViews &,
                                          const RHSWriteViews &, CFV::ReconstructionRegion) noexcept;

    /// Layout-independent equation leaf input. Per-equation descriptor tables
    /// are small immutable metadata; mesh-sized fields are explicit Array views.
    struct EquationRHSRequest
    {
        FiniteVolumeStencilView stencil;
        CFV::FiniteVolumeHostView metric;
        std::optional<CFV::ReconstructionBasisEvaluationView> basis;
        SpanView<const BoundaryCondition> conditions;
        SpanView<const EquationBlockReadView> meanBlocks, recBlocks, viscousRecBlocks;
        SpanView<const EquationBlockLocation> directLocations;
        SpanView<const RHSFieldView> directViews, directViscousViews;
        SpanView<const index> logicalFirst, vectorStarts;
        EquationBlockReadView ransMean;
        std::optional<RHSScalarView> preparedMean;
        RHSFieldView waveJump, pressureMinimum, primitiveGradientValues;
        RHSScalarView incoming;
        RHSOutputView faceValues, lowFaceValues;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> wallFlux;
        bool captureBoundaryFlux = false, genericFluxOracle = false;
        WallDistanceView wall;
        ThermalView thermal;
        RigidFrame frame;
        FiniteVolumeDetail::FluxKernelPolicy fluxPolicy;
        index ownedCells = 0, width = 0, flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        int dimension = 0, modes = 0, species = 0;
        bool direct = false, linear = false, firstOrder = false, viscous = false;
        bool primitiveGradient = false, hasWall = false, positivity = false, faceFallback = false;
        bool useStencil = false, preparedWaveJump = false, preparedPressureMinimum = false, produceLow = false;
        real time = 0;
    };
    /// Point-local dynamic storage allocated once by the operator. Face output
    /// and cell output fields remain explicit views in EquationRHSRequest.
    struct EquationRHSPointScratch
    {
        EquationFVDetail::PointScratch left, right, face, cellMean;
        std::vector<real> integrated, flux, speciesFlux, integratedLow, lowFlux;
        ViscousFluxScratch transport;
        CFV::Math::SmallMatrix basis;
        EquationRHSPointScratch(index width, int species, bool low)
            : left(width), right(width), face(width), cellMean(width), integrated(size_t(width)),
              flux(size_t(width)), speciesFlux(size_t(std::max(0, species - 1))),
              integratedLow(low ? size_t(width) : 0), lowFlux(low ? size_t(width) : 0), transport(species) {}
    };
    struct EquationRHSWorkerScratch
    {
        PointEvaluator evaluator;
        EquationRHSPointScratch scratch;
        NumericalResult<index> result{index(0)};
        EquationRHSWorkerScratch(PointEvaluator value, index width, int species, bool low)
            : evaluator(std::move(value)), scratch(width, species, low) {}
    };
    /// Two independent passes separated by the OMP join: face evaluation first,
    /// then cell gather. Mutable physics and scratch are exclusive per slot.
    NumericalResult<index> EvaluateEquationFlux(const EquationRHSRequest &, const EquationSystem &,
                                                const BoundaryPointOperator &, const Riemann::RiemannFlux &,
                                                SpanView<EquationRHSWorkerScratch>, EquationLayoutView<const real>,
                                                EquationLayoutView<const real>, EquationLayoutView<real>,
                                                EquationLayoutView<real>, CFV::ReconstructionRegion) noexcept;
    /// Resolve the compact generated catalog or the layout-generic CPU body at entry.
    /// Returns local fallback count; the facade publishes diagnostics afterward.
    NumericalResult<index> EvaluateEquationFlux(const EquationRHSRequest &, const EquationSystem &,
                                                const BoundaryPointOperator &, const Riemann::RiemannFlux &,
                                                PointEvaluator &, EquationLayoutView<const real>,
                                                EquationLayoutView<const real>, EquationLayoutView<real>,
                                                EquationLayoutView<real>, CFV::ReconstructionRegion) noexcept;
}
