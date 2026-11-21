#include "Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "CFV/CPU/ReconstructionBasisKernels.hpp"
#include "Physics/EvaluationStatus.hpp"
#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy::CPU
{
    namespace
    {
        void Gather(SpanView<const EquationBlockReadView> blocks, index iCell,
                    std::vector<real> &point)
        {
            index first = 0;
            for (const auto &block : blocks)
                for (const auto value : block[iCell])
                    point[size_t(first++)] = value;
            DNDS_assert(first == index(point.size()));
        }
    }

    static NumericalStatus FillBoundaryTraceRows(const BoundaryTraceRequest &request,
                                                 SpanView<const BoundaryCondition> conditions,
                                                 const BoundaryPointOperator &boundary,
                                                 PointEvaluator &evaluator, BoundaryTraceScratch &worker,
                                                 index first, index last) noexcept
    {
        DNDS_assert((request.polynomial ? request.means.size() == request.coefficients.size()
                                        : request.coefficients.empty()) &&
                    request.means.size() == request.output.size() &&
                    request.means.size() == request.logicalFirst.size());
        auto &scratch = worker.basis;
        auto &point = worker.point;
        auto &cellMean = worker.mean;
        DNDS_assert(index(point.size()) == request.componentNumber && index(cellMean.size()) == request.componentNumber);
        for (index iBoundaryFace = first; iBoundaryFace < last; ++iBoundaryFace)
        {
            const auto range = request.faces(iBoundaryFace, 0);
            const index count = request.centers ? 1 : range.quadratureCount;
            for (index iLocalPoint = 0; iLocalPoint < count; ++iLocalPoint)
            {
                const index iPoint = request.centers ? iBoundaryFace : range.quadratureBegin + iLocalPoint;
                const auto sample = BoundaryTracePoint(request.stencil, range, iBoundaryFace, iLocalPoint, request.centers);
                const auto &condition = conditions[sample.condition];
                Gather(request.means, sample.cell, cellMean);
                point = cellMean;
                if (request.polynomial)
                {
                    const auto basis = CFV::CPU::EvaluateFaceBasis(
                        request.basis, sample.face, sample.side, sample.point,
                        0, scratch);
                    for (index iBlock = 0; iBlock < request.means.size(); ++iBlock)
                    {
                        const auto row = request.coefficients[iBlock][sample.cell];
                        for (index j = 0;
                             j < request.means[iBlock].ComponentNumber(); ++j)
                            for (int mode = 0; mode < request.modes; ++mode)
                                point[size_t(request.logicalFirst[iBlock] + j)] +=
                                    row[j * request.modes + mode] * basis(0, mode);
                    }
                }
                // PP POLICY -- VR's warm iterate is deliberately unlimited. Like
                // Euler's FBoundary -> CompressRecPart, repair only the temporary
                // boundary input before evaluating a physical state. The flux
                // polynomial's beta limiter runs separately, after reconstruction.
                // Species bounds are not a polynomial condition: the thermal test
                // uses temporary clipped composition, as in pointwise RHS physics.
                if (!boundary.IsACM() && request.polynomial && request.allowPointFallback &&
                    !request.thermal.ClippedReferenceAdmissible(point))
                    point = cellMean;
                if (request.rans.components && request.repairRANS)
                    RANSReconstructionView{request.rans.kind, request.rans.components}.RepairPoint(
                        request.ransMeans[sample.cell],
                        point.data() + request.ransFirst);
                {
                    const auto evaluate = [&]
                    { return boundary.TryEvaluate(
                          point, condition, sample.interfaceRow, sample.normal,
                          sample.location, evaluator, request.frame, request.time,
                          request.stateKind,
                          OptionalRef<const std::vector<real>>(cellMean),
                          request.rans.components == 2
                              ? request.wallDistance.cellMean(sample.cell, 0)
                              : 0,
                          request.incoming); };
                    auto status = evaluate();
                    if (!status && request.polynomial && request.allowPointFallback &&
                        Physics::AllowsMeanFallback(status))
                    {
                        // A reported EOS/T failure may retry from the admissible
                        // mean; malformed BC requests/backend failures must escape.
                        point = cellMean;
                        status = evaluate();
                    }
                    if (!status)
                        return status.At(sample.cell, iPoint);
                }
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                // Preserve Euler's fixUL=true reconstruction target. RHS exterior
                // reflection remains a separate boundary-producer operation.
                if (!boundary.IsACM() && request.rans.components &&
                    request.rans.kind == Physics::RANS::ModelKind::SA &&
                    condition.kind == BoundaryKind::NoSlipWall)
                    point[size_t(request.ransFirst)] = 0;
#endif
                for (index iBlock = 0; iBlock < request.output.size(); ++iBlock)
                {
                    const auto target = request.output[iBlock];
                    for (index j = 0; j < target.ComponentNumber(); ++j)
                        target(iPoint, j) =
                            point[size_t(request.logicalFirst[iBlock] + j)];
                }
            }
        }
        return {};
    }

    NumericalStatus FillBoundaryTraces(const BoundaryTraceRequest &request, SpanView<const BoundaryCondition> conditions,
                                       const BoundaryPointOperator &boundary, PointEvaluator &evaluator) noexcept
    {
        BoundaryTraceScratch scratch(request.componentNumber);
        return FillBoundaryTraceRows(request, conditions, boundary, evaluator, scratch, 0,
                                     request.faces.RowCount());
    }
    NumericalStatus FillBoundaryTraces(const BoundaryTraceRequest &request, SpanView<const BoundaryCondition> conditions,
                                       const BoundaryPointOperator &boundary, SpanView<BoundaryTraceWorkerScratch> workers) noexcept
    {
        if (workers.empty())
            return NumericalStatus{NumericalError::InvalidInput};
        const index count = request.faces.RowCount();
        OMP::ForRows(0, workers.size(), [&](index iWorker)
                     {
            const index quotient = count / workers.size(), remainder = count % workers.size();
            const index first = iWorker * quotient + std::min(iWorker, remainder);
            auto &worker = workers[iWorker];
            worker.status = FillBoundaryTraceRows(request, conditions, boundary, worker.evaluator, worker.scratch,
                                                 first, first + quotient + (iWorker < remainder)); });
        for (const auto &worker : workers)
            if (!worker.status)
                return worker.status;
        return {};
    }

    void CopyBoundaryRows(
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            source,
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            destination,
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks,
                           Runtime::FixedRows<1>>
            sourceRows)
    {
        DNDS_assert(sourceRows.RowCount() == destination.RowCount());
        OMP::ForRows(0, destination.RowCount(), [&](index iPoint)
                     {
            for (index j = 0; j < destination.Width(); ++j)
                destination(iPoint, j) = source(sourceRows(iPoint, 0), j); });
    }

    NumericalStatus FillPrimitiveBoundary(const PrimitiveBoundaryRequest &request,
                                          SpanView<const BoundaryCondition> conditions) noexcept
    {
        return OMP::ForRowsStatus(0, request.output.RowCount(), [&](index iPoint) -> NumericalStatus
                                  {
            const auto sample = BoundaryTracePoint(request.stencil, request.faces(iPoint, 0), iPoint, 0, true);
            const auto &condition = conditions[sample.condition];
            FlowState inside{};
            index first = 0;
            for (const auto &block : request.means)
            {
                for (index j = std::max<index>(0, request.flowFirst - first);
                     j < std::min<index>(block.ComponentNumber(), request.flowFirst + 5 - first); ++j)
                    inside[first + j - request.flowFirst] = block(sample.cell, j);
                first += block.ComponentNumber();
            }
            FlowState state;
            if (condition.kind == BoundaryKind::Interface)
            {
                for (int j = 0; j < 5; ++j)
                    state[j] = request.incoming(sample.interfaceRow, j);
            }
            else
            {
                const auto exterior = condition.TryFaceValue(
                    request.gas, inside, sample.normal, sample.location,
                    request.time, OptionalRef<const RigidFrame>(request.frame));
                if (!exterior)
                    return exterior.status.At(sample.cell, iPoint);
                state = exterior.value;
            }
            const auto checked = TryPerfectGasPoint(request.gas, state, false);
            if (!checked)
                return checked.status.At(sample.cell, iPoint);
            const auto value = request.gas.Primitive(state);
            const real converted[5]{value.density, value.velocity[0],
                                    value.velocity[1], value.velocity[2],
                                    value.temperature};
            for (int j = 0; j < 5; ++j)
                request.output(iPoint, j) = converted[j];
            return {}; });
    }

    NumericalStatus ValidateInterfacePacketRanges(Algebra::DofComponentView<const real> rows,
                                                  Runtime::ArrayView<const BoundaryFaceRange, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> ranges,
                                                  bool centers, AdmissibilityView admissibility) noexcept
    {
        return OMP::ForRowsStatus(0, ranges.RowCount(), [&](index iFace) -> NumericalStatus
                                  {
            const auto range = ranges(iFace, 0);
            if (!range.interface)
                return {};
            const index begin = centers ? iFace : range.quadratureBegin;
            const index end = begin + (centers ? 1 : range.quadratureCount);
            for (index iPoint = begin; iPoint < end; ++iPoint)
            {
                if (!admissibility.Admissible(ThermalAdmissibilityDetail::PointRow<decltype(rows)>{rows, iPoint}))
                    return NumericalStatus{NumericalError::InvalidState, 1, iPoint};
            }
            return {}; });
    }
    Vector3 AccumulateBoundaryForce(FiniteVolumeStencilView stencil,
                                    Runtime::ArrayView<const BoundaryFaceRange, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> ranges,
                                    Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> values,
                                    SpanView<const int> zones, SpanView<Vector3> workerSums)
    {
        const auto accumulate = [&](index first, index end)
        {
            Vector3 force{};
            for (index iBoundary = first; iBoundary < end; ++iBoundary)
            {
                const auto range = ranges(iBoundary, 0);
                const int zone = stencil.topology.GetFaceZone(range.face);
                if (std::find(zones.begin(), zones.end(), zone) == zones.end())
                    continue;
                const index begin = stencil.quadratureBoundaryValues ? range.quadratureBegin : iBoundary;
                const index last = begin + (stencil.quadratureBoundaryValues ? range.quadratureCount : 1);
                for (index iPoint = begin; iPoint < last; ++iPoint)
                    for (int d = 0; d < 3; ++d)
                        force[d] += values(iPoint, d);
            }
            return force;
        };
        const index count = ranges.RowCount();
        if (workerSums.empty())
            return accumulate(0, count);
        const index quotient = count / workerSums.size(), remainder = count % workerSums.size();
        OMP::ForRows(0, workerSums.size(), [&](index iWorker)
                     {
            const index first = iWorker * quotient + std::min(iWorker, remainder);
            workerSums[iWorker] = accumulate(first, first + quotient + (iWorker < remainder)); });
        // Fixed slot order, no floating-point atomic or runtime tree choice.
        auto force = workerSums[0];
        for (index iWorker = 1; iWorker < workerSums.size(); ++iWorker)
            for (int d = 0; d < 3; ++d)
                force[d] += workerSums[iWorker][d];
        return force;
    }
}
