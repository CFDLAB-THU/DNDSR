#include "Cauchy/Discretization/CPU/FiniteVolumeSourceKernels.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "DNDS/Algebra/BlockIndex.hpp"
#include "DNDS/Core/OMP.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "Physics/Incompressible/RANSPoint.hpp"

namespace DNDS::Cauchy::CPU
{
    using namespace EquationFVDetail;
    static NumericalStatus AssembleCenterSourceJacobianRows(const SourceKernelRequest &request, const EquationSystem &equation, PointEvaluator &evaluator,
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
            if (!equation.IsACM())
            {
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
            }
            const real volume = request.metric.GetCellVol(iCell);
            if (request.ransFirst >= 0)
            {
                const auto &model = equation.RANS();
                const auto distance = request.walls.cellMean(iCell, 0);
                const auto length = request.lengths(iCell, 0);
                const auto input = equation.IsACM()
                                       ? Physics::Incompressible::RANSPoint(point.values, point.gradient, model.ComponentNumber(), equation.ACMParameters().kinematicViscosity, distance, length)
                                       : RANSPoint(point, request.flowFirst, request.ransFirst, model, properties, distance, length);
                const auto destruction = model.TrySource(input, 1);
                if (!destruction)
                    return destruction.status.At(iCell);
                for (int j = 0; j < model.ComponentNumber(); ++j)
                {
                    const index row = request.ransFirst + j;
                    auto &entry = request.matrix(iCell, request.fullBlock ? row + nv * row : row);
                    const real diagonal = request.scale * volume * destruction.value[j];
                    if (equation.IsACM())
                        entry = diagonal; // ACM assembles its complete flux/temporal row next.
                    else
                        entry += diagonal;
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
    template <bool ProduceResidual, bool ProduceDerivative>
    static NumericalStatus EvaluateVolumeSourceRows(const SourceKernelRequest &request, const EquationSystem &equation, PointEvaluator &evaluator,
                                                    SourceRowScratch &worker, index first, index last) noexcept
    {
        auto &point = worker.point;
        auto &rowRHS = worker.rowRHS;
        auto &rowDerivative = worker.rowDerivative;
        const int dimension = request.dimension;
        if (dimension < 1 || dimension > 3)
            return NumericalStatus{NumericalError::InvalidInput};
        const index nv = request.components;
        const int independent = request.independentSpecies;
        if constexpr (ProduceDerivative)
            rowDerivative.resize(request.fullBlock ? nv * nv : nv);
        for (index iCell = first; iCell < last; ++iCell)
        {
            if constexpr (ProduceResidual)
            {
                if (request.chemical)
                    for (int k = 0; k < independent; ++k)
                        rowRHS[request.speciesFirst + k] = request.speciesOutput(iCell, k);
                if (request.ransFirst >= 0)
                    for (int k = 0; k < equation.RANS().ComponentNumber(); ++k)
                        rowRHS[request.ransFirst + k] = request.ransOutput(iCell, k);
            }
            if constexpr (ProduceDerivative)
                std::fill(rowDerivative.begin(), rowDerivative.end(), real(0));
            for (index iPoint = request.offsets(iCell, 0); iPoint < request.offsets(iCell + 1, 0); ++iPoint)
            {
                const int modes = request.basis.polynomial.size - 1;
                real basis[4][19];
                if (modes > 0)
                    request.basis.Fill(iCell, iPoint, request.ransFirst >= 0 ? dimension + 1 : 1, basis);
                for (index iBlock = 0; iBlock < request.locations.size(); ++iBlock)
                {
                    const auto location = request.locations[iBlock];
                    const index inputRow = request.cellInputs || modes > 0 ? iCell : iPoint;
                    const auto values = request.samples[location.physicalField][inputRow];
                    for (index j = 0; j < location.componentNumber; ++j)
                    {
                        const index logical = request.logicalFirst[iBlock] + j, physical = location.firstComponent + j;
                        point.values[logical] = values[physical];
                        if (modes > 0)
                        {
                            const auto coefficients = request.coefficients[location.physicalField][iCell];
                            for (int m = 0; m < modes; ++m)
                                point.values[logical] += basis[0][m] * coefficients[physical * modes + m];
                            if (request.ransFirst >= 0)
                                for (int d = 0; d < dimension; ++d)
                                {
                                    real derivative = 0;
                                    for (int m = 0; m < modes; ++m)
                                        derivative += basis[1 + d][m] * coefficients[physical * modes + m];
                                    point.gradient(logical, d) = derivative;
                                }
                        }
                        else if (request.ransFirst >= 0)
                            for (int d = 0; d < dimension; ++d)
                                point.gradient(logical, d) = request.gradients[location.physicalField][d](inputRow, physical);
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
                                                              {request.ransFirst >= 0, false, false, request.chemical, ProduceDerivative && request.chemical}, properties);
                    if (!status)
                        return status.At(iCell, iPoint);
                }
                const real weight = request.weights(iPoint, 0);
                if (request.chemical)
                {
                    if constexpr (ProduceResidual)
                        for (int k = 0; k < independent; ++k)
                            rowRHS[request.speciesFirst + k] += weight * properties.chemicalSource[k];
                    if constexpr (ProduceDerivative)
                        for (int k = 0; k < independent; ++k)
                            for (int j = 0; j < 5 + independent; ++j)
                            {
                                const index column = j < 5 ? request.flowFirst + j : request.speciesFirst + j - 5;
                                const index row = request.speciesFirst + k;
                                const real derivative = properties.sourceJacobian(k, j);
                                if (!std::isfinite(derivative))
                                    return NumericalStatus{NumericalError::InvalidState}.At(iCell, iPoint);
                                if (request.fullBlock || row == column)
                                    rowDerivative[request.fullBlock ? row * nv + column : row] -= weight * derivative;
                            }
                }
                if (request.ransFirst >= 0)
                {
                    const auto &model = equation.RANS();
                    const real distance = request.walls.cells(iCell, iPoint - request.offsets(iCell, 0));
                    const auto input = equation.IsACM()
                                           ? Physics::Incompressible::RANSPoint(point.values, point.gradient, model.ComponentNumber(), equation.ACMParameters().kinematicViscosity, distance, request.lengths(iCell, 0))
                                           : RANSPoint(point, request.flowFirst, request.ransFirst, model, properties, distance, request.lengths(iCell, 0));
                    if constexpr (ProduceResidual)
                    {
                        const auto source = model.TrySource(input);
                        if (!source)
                            return source.status.At(iCell, iPoint);
                        for (int j = 0; j < model.ComponentNumber(); ++j)
                            rowRHS[request.ransFirst + j] += weight * source.value[j];
                    }
                    if constexpr (ProduceDerivative)
                    {
                        const auto destruction = model.TrySource(input, 1);
                        if (!destruction)
                            return destruction.status.At(iCell, iPoint);
                        for (int j = 0; j < model.ComponentNumber(); ++j)
                        {
                            const index row = request.ransFirst + j;
                            rowDerivative[request.fullBlock ? row * nv + row : row] += weight * destruction.value[j];
                        }
                    }
                }
            }
            if constexpr (ProduceResidual)
            {
                if (request.chemical)
                    for (int k = 0; k < independent; ++k)
                        request.speciesOutput(iCell, k) = rowRHS[request.speciesFirst + k];
                if (request.ransFirst >= 0)
                    for (int k = 0; k < equation.RANS().ComponentNumber(); ++k)
                        request.ransOutput(iCell, k) = rowRHS[request.ransFirst + k];
            }
            if constexpr (ProduceDerivative)
            {
                for (index k = 0; k < index(rowDerivative.size()); ++k)
                    if constexpr (ProduceResidual)
                        request.preparedDerivative(iCell, k) = rowDerivative[k];
                    else
                    {
                        const index entry = request.fullBlock && !request.rowMajorBlocks ? (k % nv) * nv + k / nv : k;
                        request.matrix(iCell, entry) += request.scale * rowDerivative[k];
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
                worker.status = operation(request, equation, worker.evaluator, worker,
                                          first, first + quotient + (iWorker < remainder)); });
            // Contiguous ranges make the first failing slot the earliest cell.
            for (const auto &worker : workers)
                if (!worker.status)
                    return worker.status;
            return {};
        }
    }
    NumericalStatus AssembleCenterSourceJacobian(const SourceKernelRequest &request, const EquationSystem &equation, PointEvaluator &evaluator) noexcept
    {
        PointScratch point(request.components);
        return AssembleCenterSourceJacobianRows(request, equation, evaluator, point, 0, request.cells);
    }
    NumericalStatus AssembleCenterSourceJacobian(const SourceKernelRequest &request, const EquationSystem &equation, SpanView<SourceWorkerScratch> workers) noexcept
    {
        return SourceWorkers(request, equation, workers, [](const auto &r, const auto &eq, auto &e, auto &w, index first, index last)
                             { return AssembleCenterSourceJacobianRows(r, eq, e, w.point, first, last); });
    }
    NumericalStatus AddQuadratureSourceJacobian(const SourceKernelRequest &request, const EquationSystem &equation, PointEvaluator &evaluator) noexcept
    {
        SourceRowScratch row(request.components);
        return EvaluateVolumeSourceRows<false, true>(request, equation, evaluator, row, 0, request.cells);
    }
    NumericalStatus AddQuadratureSourceJacobian(const SourceKernelRequest &request, const EquationSystem &equation, SpanView<SourceWorkerScratch> workers) noexcept
    {
        return SourceWorkers(request, equation, workers, EvaluateVolumeSourceRows<false, true>);
    }
    NumericalStatus EvaluateVolumeSource(const SourceKernelRequest &request, const EquationSystem &equation, PointEvaluator &evaluator) noexcept
    {
        SourceRowScratch row(request.components);
        return request.preparedDerivative.data
                   ? EvaluateVolumeSourceRows<true, true>(request, equation, evaluator, row, 0, request.cells)
                   : EvaluateVolumeSourceRows<true, false>(request, equation, evaluator, row, 0, request.cells);
    }
    NumericalStatus EvaluateVolumeSource(const SourceKernelRequest &request, const EquationSystem &equation, SpanView<SourceWorkerScratch> workers) noexcept
    {
        return request.preparedDerivative.data
                   ? SourceWorkers(request, equation, workers, EvaluateVolumeSourceRows<true, true>)
                   : SourceWorkers(request, equation, workers, EvaluateVolumeSourceRows<true, false>);
    }
    void ApplyVolumeSourceDerivative(Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> source, RHSOutputView diagonal,
                                     index n, bool full, bool rowMajor, real scale)
    {
        OMP::ForRows(0, source.RowCount(), [&](index cell)
                     {
            for (index row=0;row<n;++row)
                if (full)
                    for(index col=0;col<n;++col)
                        diagonal(cell, Algebra::DenseBlockEntry(row,col,n,rowMajor)) += scale*source(cell,row*n+col);
                else
                    diagonal(cell,row) += scale*source(cell,row*n+row); });
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
