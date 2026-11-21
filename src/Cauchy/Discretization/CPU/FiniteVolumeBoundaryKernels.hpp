#pragma once
#include "DNDS/Core/NumericalStatus.hpp"

#include "Cauchy/Discretization/BoundaryTraceGeometry.hpp"
#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Core/Span.hpp"
#include "Cauchy/Boundary/BoundaryInputs.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Physics/PointProperties.hpp"

namespace DNDS::Cauchy::CPU
{
    struct BoundaryTraceRequest
    {
        FiniteVolumeStencilView stencil;
        Runtime::ArrayView<const BoundaryFaceRange, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faces;
        SpanView<const EquationBlockReadView> means, coefficients;
        SpanView<const EquationBlockArrayView<real>> output;
        SpanView<const index> logicalFirst;
        CFV::ReconstructionBasisEvaluationView basis;
        index componentNumber = 0;
        bool polynomial = false, centers = false, repairRANS = true;
        /// Point-local repair of an unlimited VR iterate before physical BC
        /// evaluation. Never modifies means or reconstruction coefficients.
        bool allowPointFallback = false;
        ThermalView thermal{};
        /// Resolved by the operator. Mirroring is a VR-only choice, not a
        /// property of polynomial sampling or of the downstream CFV kernel.
        BoundaryPointStateKind stateKind = BoundaryPointStateKind::FaceTrace;
        RANSStateView rans;
        EquationBlockReadView ransMeans;
        WallDistanceView wallDistance;
        RigidFrame frame;
        InterfaceBoundaryRows incoming;
        index ransFirst = -1;
        int modes = 0;
        real time = 0;
    };

    struct BoundaryTraceScratch
    {
        CFV::Math::SmallMatrix basis;
        std::vector<real> point, mean;
        explicit BoundaryTraceScratch(index components)
            : point(size_t(components)), mean(size_t(components)) {}
    };
    /// Each slot owns its chemistry lease and all mutable point scratch.
    /// Operator setup resolves the evaluator property request before queuing.
    struct BoundaryTraceWorkerScratch
    {
        PointEvaluator evaluator;
        BoundaryTraceScratch scratch;
        NumericalStatus status;
        BoundaryTraceWorkerScratch(PointEvaluator value, index components)
            : evaluator(std::move(value)), scratch(components) {}
    };
    NumericalStatus FillBoundaryTraces(const BoundaryTraceRequest &, SpanView<const BoundaryCondition>,
                                       const BoundaryPointOperator &, SpanView<BoundaryTraceWorkerScratch>) noexcept;
    /// CPU physical producer context is explicit; it contains no array access
    /// or scheduling. All sampled field operands are frozen in request.
    NumericalStatus FillBoundaryTraces(const BoundaryTraceRequest &request,
                                       SpanView<const BoundaryCondition> conditions,
                                       const BoundaryPointOperator &boundary,
                                       PointEvaluator &evaluator) noexcept;

    void CopyBoundaryRows(
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            source,
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            destination,
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks,
                           Runtime::FixedRows<1>>
            sourceRows);

    struct PrimitiveBoundaryRequest
    {
        FiniteVolumeStencilView stencil;
        Runtime::ArrayView<const BoundaryFaceRange, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faces;
        SpanView<const EquationBlockReadView> means;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            output;
        PerfectGas gas;
        RigidFrame frame;
        InterfaceBoundaryRows incoming;
        index componentNumber = 0;
        index flowFirst = -1;
        real time = 0;
    };

    NumericalStatus FillPrimitiveBoundary(const PrimitiveBoundaryRequest &request,
                                          SpanView<const BoundaryCondition> conditions) noexcept;

    NumericalStatus ValidateInterfacePacketRanges(Algebra::DofComponentView<const real> rows,
                                                  Runtime::ArrayView<const BoundaryFaceRange, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> ranges,
                                                  bool centers, AdmissibilityView admissibility) noexcept;
    /// Optional caller-owned worker slots enable independent contiguous scans
    /// and an ordered FP sum. Empty slots retain the serial reference path.
    Vector3 AccumulateBoundaryForce(FiniteVolumeStencilView stencil,
                                    Runtime::ArrayView<const BoundaryFaceRange, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> ranges,
                                    Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> values,
                                    SpanView<const int> zones, SpanView<Vector3> workerSums = {});
}
