#include "Cauchy/Discretization/CPU/FiniteVolumeSourceKernels.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "DNDS/Core/OMP.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "Physics/Incompressible/RANSPoint.hpp"

namespace DNDS::Cauchy::CPU
{
    using namespace EquationFVDetail;
    static NumericalStatus AddMeanSourceJacobianRows(const SourceKernelRequest &request, const EquationSystem &equation, PointEvaluator &evaluator,
                                                     PointScratch &point, index first, index last) noexcept
    {
        // Direction descriptors have exactly three entries. Validate once,
        // before both the point scratch fill and any output mutation.
        const int dimension = request.dimension;
        if (dimension < 1 || dimension > 3)
            return NumericalStatus{NumericalError::InvalidInput};
        const index nv = request.components;
        const int independent = request.independentSpecies;
        for (index iCell = first; iCell < last; ++iCell)
        {
            Gather(request.blocks, iCell, point, false);
            if (request.ransFirst >= 0)
                for (index iBlock = 0; iBlock < request.locations.size(); ++iBlock)
                {
                    const auto location = request.locations[iBlock];
                    for (index j = 0; j < location.componentNumber; ++j)
                        for (int d = 0; d < dimension; ++d)
                            point.gradient(request.logicalFirst[iBlock] + j, d) = request.gradients[location.physicalField][d](iCell, location.firstComponent + j);
                }
            PointProperties properties;
            const auto status = evaluator.TryEvaluate(point.Flow(request.flowFirst), point.Species(request.speciesFirst, independent), 0,
                                                      {request.ransFirst >= 0, false, bool(request.linearization), request.chemical, request.chemical}, properties);
            if (!status)
                return status.At(iCell);
            if (request.linearization)
            {
                const auto stored = request.linearization->FillRow(iCell, point, properties);
                if (!stored)
                    return stored;
            }
            const real volume = request.metric.GetCellVol(iCell);
            if (request.ransFirst >= 0)
            {
                const auto &model = equation.RANS();
                const auto input = RANSPoint(point, request.flowFirst, request.ransFirst, model, properties, request.walls.cellMean(iCell, 0), request.lengths(iCell, 0));
                const auto destruction = model.TrySource(input, 1);
                if (!destruction)
                    return destruction.status.At(iCell);
                for (int j = 0; j < model.ComponentNumber(); ++j)
                {
                    const index row = request.ransFirst + j;
                    request.matrix(iCell, request.fullBlock ? row + nv * row : row) += request.scale * volume * destruction.value[j];
                }
            }
            if (request.chemical)
                for (int j = 0; j < 5 + independent; ++j)
                {
                    const index column = j < 5 ? request.flowFirst + j : request.speciesFirst + j - 5;
                    for (int k = 0; k < independent; ++k)
                    {
                        const index row = request.speciesFirst + k;
                        const real derivative = properties.sourceJacobian(k, j);
                        if (!std::isfinite(derivative))
                            return NumericalStatus{NumericalError::InvalidState}.At(iCell);
                        if (request.fullBlock || row == column)
                            request.matrix(iCell, request.fullBlock ? (request.rowMajorBlocks ? row * nv + column : row + nv * column) : row) -= request.scale * volume * derivative;
                    }
                }
        }
        return {};
    }
    static NumericalStatus AddQuadratureSourceJacobianRows(const SourceKernelRequest &request, const EquationSystem &equation, PointEvaluator &evaluator,
                                                           PointScratch &point, index first, index last) noexcept
    {
        const int dimension = request.dimension;
        if (dimension < 1 || dimension > 3)
            return NumericalStatus{NumericalError::InvalidInput};
        const index nv = request.components;
        const int independent = request.independentSpecies;
        if (request.linearization)
        {
            for (index iCell = first; iCell < last; ++iCell)
            {
                Gather(request.blocks, iCell, point, false);
                PointProperties properties;
                const auto status = evaluator.TryEvaluate(
                    point.Flow(request.flowFirst), point.Species(request.speciesFirst, independent), 0,
                    {request.ransFirst >= 0, false, true, false, false}, properties);
                if (!status)
                    return status.At(iCell);
                const auto stored = request.linearization->FillRow(iCell, point, properties);
                if (!stored)
                    return stored;
            }
        }
        const auto addTerms = [&](index iCell, const PointScratch &at,
                                  const PointProperties &properties, real weight,
                                  real wallDistance) -> NumericalStatus
        {
            if (request.ransFirst >= 0)
            {
                const auto &model = equation.RANS();
                const auto input = RANSPoint(at, request.flowFirst, request.ransFirst, model, properties, wallDistance, request.lengths(iCell, 0));
                const auto destruction = model.TrySource(input, 1);
                if (!destruction)
                    return destruction.status.At(iCell);
                for (int j = 0; j < model.ComponentNumber(); ++j)
                {
                    const index row = request.ransFirst + j;
                    request.matrix(iCell, request.fullBlock ? row + nv * row : row) +=
                        request.scale * weight * destruction.value[j];
                }
            }
            if (request.chemical)
                for (int j = 0; j < 5 + independent; ++j)
                {
                    const index column = j < 5 ? request.flowFirst + j : request.speciesFirst + j - 5;
                    for (int k = 0; k < independent; ++k)
                    {
                        const index row = request.speciesFirst + k;
                        const real derivative = properties.sourceJacobian(k, j);
                        if (!std::isfinite(derivative))
                            return NumericalStatus{NumericalError::InvalidState}.At(iCell);
                        if (request.fullBlock || row == column)
                            request.matrix(iCell, request.fullBlock ? (request.rowMajorBlocks ? row * nv + column : row + nv * column) : row) -=
                                request.scale * weight * derivative;
                    }
                }
            return {};
        };
        for (index iCell = first; iCell < last; ++iCell)
        {
            for (index iPoint = request.offsets(iCell, 0); iPoint < request.offsets(iCell + 1, 0); ++iPoint)
            {
                for (index iBlock = 0; iBlock < request.locations.size(); ++iBlock)
                {
                    const auto location = request.locations[iBlock];
                    const auto values = request.samples[location.physicalField][iPoint];
                    for (index j = 0; j < location.componentNumber; ++j)
                        point.values[request.logicalFirst[iBlock] + j] = values[location.firstComponent + j];
                    if (request.ransFirst >= 0)
                        for (int d = 0; d < dimension; ++d)
                        {
                            const auto values = request.gradients[location.physicalField][d].values[iPoint];
                            for (index j = 0; j < location.componentNumber; ++j)
                                point.gradient(request.logicalFirst[iBlock] + j, d) =
                                    values[location.firstComponent + j];
                        }
                }
                if (request.ransFirst >= 0)
                {
                    const auto &model = equation.RANS();
                    RANSReconstructionView{model.Kind(), model.ComponentNumber()}.RepairPoint(
                        request.ransMean[iCell], point.values.data() + request.ransFirst);
                }
                PointProperties properties;
                const auto status = evaluator.TryEvaluate(
                    point.Flow(request.flowFirst), point.Species(request.speciesFirst, independent), 0,
                    {request.ransFirst >= 0, false, false, request.chemical, request.chemical}, properties);
                if (!status)
                    return status.At(iCell, iPoint);
                const real wallDistance = request.ransFirst >= 0
                                              ? request.walls.cells(iCell, iPoint - request.offsets(iCell, 0))
                                              : real(1);
                const auto added = addTerms(iCell, point, properties, request.weights(iPoint, 0), wallDistance);
                if (!added)
                    return added.At(iCell, iPoint);
            }
        }
        return {};
    }
    static NumericalStatus AddVolumeSourcesRows(const SourceKernelRequest &request, const EquationSystem &equation, PointEvaluator &evaluator,
                                                PointScratch &point, index first, index last) noexcept
    {
        const int dimension = request.dimension;
        if (dimension < 1 || dimension > 3)
            return NumericalStatus{NumericalError::InvalidInput};
        const index nv = request.components;
        const int independent = request.independentSpecies;
        for (index iCell = first; iCell < last; ++iCell)
            for (index iPoint = request.offsets(iCell, 0); iPoint < request.offsets(iCell + 1, 0); ++iPoint)
            {
                for (index iBlock = 0; iBlock < request.locations.size(); ++iBlock)
                {
                    const auto location = request.locations[iBlock];
                    // Resolve each borrowed row before traversing its
                    // components. The span retains physical element stride;
                    // no packing or AoS storage assumption is introduced.
                    const auto values = request.samples[location.physicalField][iPoint];
                    for (index j = 0; j < location.componentNumber; ++j)
                        point.values[request.logicalFirst[iBlock] + j] = values[location.firstComponent + j];
                    if (request.ransFirst >= 0)
                        for (int d = 0; d < dimension; ++d)
                        {
                            const auto &gradient = request.gradients[location.physicalField][d];
                            const auto values = gradient.values[iPoint];
                            for (index j = 0; j < location.componentNumber; ++j)
                                point.gradient(request.logicalFirst[iBlock] + j, d) = values[(location.firstComponent + j) * gradient.componentStride + gradient.direction];
                        }
                }
                if (request.ransFirst >= 0)
                {
                    const auto &model = equation.RANS();
                    RANSReconstructionView{model.Kind(), model.ComponentNumber()}.RepairPoint(request.ransMean[iCell], point.values.data() + request.ransFirst);
                }
                PointProperties properties;
                if (!equation.IsACM())
                {
                    const auto status = evaluator.TryEvaluate(point.Flow(request.flowFirst), point.Species(request.speciesFirst, independent), 0,
                                                              {request.ransFirst >= 0, false, false, request.chemical, false}, properties);
                    if (!status)
                        return status.At(iCell, iPoint);
                }
                if (request.chemical)
                    for (int k = 0; k < independent; ++k)
                        request.speciesOutput(iCell, k) += request.weights(iPoint, 0) * properties.chemicalSource[k];
                if (request.ransFirst >= 0)
                {
                    const auto &model = equation.RANS();
                    const real distance = request.walls.cells(iCell, iPoint - request.offsets(iCell, 0));
                    const auto input = equation.IsACM()
                                           ? Physics::Incompressible::RANSPoint(point.values, point.gradient, model.ComponentNumber(), equation.ACMParameters().kinematicViscosity, distance, request.lengths(iCell, 0))
                                           : RANSPoint(point, request.flowFirst, request.ransFirst, model, properties, distance, request.lengths(iCell, 0));
                    const auto source = model.TrySource(input);
                    if (!source)
                        return source.status.At(iCell, iPoint);
                    for (int j = 0; j < model.ComponentNumber(); ++j)
                    {
                        request.ransOutput(iCell, j) += request.weights(iPoint, 0) * source.value[j];
                    }
                }
            }
        return {};
    }
    namespace
    {
        template <class Operation>
        NumericalStatus SourceWorkers(const SourceKernelRequest &request, const EquationSystem &equation,
                                      SpanView<SourceWorkerScratch> workers, Operation operation) noexcept
        {
            if (workers.empty())
                return NumericalStatus{NumericalError::InvalidInput};
            OMP::ForRows(0, workers.size(), [&](index iWorker)
                         {
                const index quotient = request.cells / workers.size(), remainder = request.cells % workers.size();
                const index first = iWorker * quotient + std::min(iWorker, remainder);
                auto &worker = workers[iWorker];
                DNDS_assert(index(worker.point.values.size()) == request.components);
                worker.status = operation(request, equation, worker.evaluator, worker.point,
                                          first, first + quotient + (iWorker < remainder)); });
            // Contiguous ranges make the first failing slot the earliest cell.
            for (const auto &worker : workers)
                if (!worker.status)
                    return worker.status;
            return {};
        }
    }
    NumericalStatus AddMeanSourceJacobian(const SourceKernelRequest &request, const EquationSystem &equation, PointEvaluator &evaluator) noexcept
    {
        PointScratch point(request.components);
        return AddMeanSourceJacobianRows(request, equation, evaluator, point, 0, request.cells);
    }
    NumericalStatus AddMeanSourceJacobian(const SourceKernelRequest &request, const EquationSystem &equation, SpanView<SourceWorkerScratch> workers) noexcept
    {
        return SourceWorkers(request, equation, workers, AddMeanSourceJacobianRows);
    }
    NumericalStatus AddQuadratureSourceJacobian(const SourceKernelRequest &request, const EquationSystem &equation, PointEvaluator &evaluator) noexcept
    {
        PointScratch point(request.components);
        return AddQuadratureSourceJacobianRows(request, equation, evaluator, point, 0, request.cells);
    }
    NumericalStatus AddQuadratureSourceJacobian(const SourceKernelRequest &request, const EquationSystem &equation, SpanView<SourceWorkerScratch> workers) noexcept
    {
        return SourceWorkers(request, equation, workers, AddQuadratureSourceJacobianRows);
    }
    NumericalStatus AddVolumeSources(const SourceKernelRequest &request, const EquationSystem &equation, PointEvaluator &evaluator) noexcept
    {
        PointScratch point(request.components);
        return AddVolumeSourcesRows(request, equation, evaluator, point, 0, request.cells);
    }
    NumericalStatus AddVolumeSources(const SourceKernelRequest &request, const EquationSystem &equation, SpanView<SourceWorkerScratch> workers) noexcept
    {
        return SourceWorkers(request, equation, workers, AddVolumeSourcesRows);
    }
    void ReplicateSourceValues(Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> offsets,
                               RHSScalarView input, RHSOutputView output, int stride, int direction)
    {
        OMP::ForRows(0, offsets.RowCount() - 1, [&](index iCell)
                     {
            for (index iPoint = offsets(iCell, 0); iPoint < offsets(iCell + 1, 0); ++iPoint)
                for (index j = 0; j < output.Width(); ++j)
                    output(iPoint, j) = input(iCell, j * stride + direction); });
    }
    void AddEquationRotationSource(CFV::FiniteVolumeHostView metric, const RigidFrame &frame,
                                   EquationBlockReadView flow, EquationBlockArrayView<real> output,
                                   std::optional<EquationBlockArrayView<real>> lowOutput)
    {
        OMP::ForRows(0, output.RowCount(), [&](index iCell)
                     {
            FlowState mean{};
            for (int j = 0; j < 5; ++j)
                mean[j] = flow(iCell, j);
            const auto source = frame.RotationSource(mean);
            for (int j = 0; j < 5; ++j)
            {
                output(iCell, j) += metric.GetCellVol(iCell) * source[j];
                if (lowOutput)
                    (*lowOutput)(iCell, j) += metric.GetCellVol(iCell) * source[j];
            } });
    }
}
