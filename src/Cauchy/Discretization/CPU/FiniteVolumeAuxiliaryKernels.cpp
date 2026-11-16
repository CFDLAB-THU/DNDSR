#include "Cauchy/Discretization/CPU/FiniteVolumeAuxiliaryKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "DNDS/Core/OMP.hpp"
#include <limits>

namespace DNDS::Cauchy::CPU
{
    void LoadGradients(const AuxiliaryKernelRequest &request, index iCell, EquationFVDetail::PointScratch &point)
    {
        if (request.gradients.empty())
            return;
        index first = 0;
        const int dim = request.dimension;
        for (const auto &location : request.locations)
            for (index j = 0; j < location.componentNumber; ++j, ++first)
                for (int d = 0; d < dim; ++d)
                    point.gradient(first, d) = request.gradients[location.physicalField](iCell, (location.firstComponent + j) * dim + d);
    }
    NumericalResult<std::array<real, 5>> Properties(const AuxiliaryKernelRequest &request, const EquationSystem &equation, EquationFVDetail::PointScratch &point, PointEvaluator &evaluator,
                                                    real wallDistance = 1, bool turbulent = true) noexcept
    {
        const auto flow = point.Flow(request.flowFirst);
        PointProperties result;
        const auto status = evaluator.TryEvaluate(flow, point.Species(request.speciesFirst, request.independentSpecies), 0, result);
        if (!status)
            return status;
        real diffusion = result.diffusionBound;
        if (request.ransFirst >= 0 && turbulent)
        {
            const auto &model = equation.RANS();
            const auto transport = model.TryTransport(EquationFVDetail::RANSPoint(point, request.flowFirst, request.ransFirst, model, result, wallDistance, 0));
            if (!transport)
                return transport.status;
            diffusion = std::max({diffusion, real(4. / 3.) * (result.transport.viscosity + transport.value.viscosity) / flow[0],
                                  (result.transport.conductivity + transport.value.conductivity) / (flow[0] * result.thermo.cv)});
        }
        // Characteristic quantities reuse this cell's completed EOS solve.
        // H preserves the conservative energy origin, including formation
        // energy; beta is a thermodynamic derivative, not gamma-equivalent.
        return std::array<real, 5>{result.thermo.pressure, result.thermo.soundSpeed, diffusion,
                                   (flow[4] + result.thermo.pressure) / flow[0], result.thermo.gasConstant / result.thermo.cv};
    }
    NumericalStatus PrepareAuxiliaryMeans(const AuxiliaryKernelRequest &request, const EquationSystem &equation, SpanView<AuxiliaryWorkerScratch> workers, index first, index last) noexcept
    {
        if (!request.enabled[0])
            return {};
        if (workers.empty() || (request.values[0].Width() != 3 && request.values[0].Width() != 5))
            return NumericalStatus{NumericalError::InvalidInput};
        OMP::ForRows(0, workers.size(), [&](index iWorker)
                     {
          auto &worker = workers[iWorker];
          worker.status = {};
          auto &point = worker.left;
          const index count = last - first, quotient = count / workers.size(), remainder = count % workers.size();
          const index begin = first + iWorker * quotient + std::min(iWorker, remainder);
          const index end = begin + quotient + (iWorker < remainder);
          for (index iCell = begin; iCell < end; ++iCell)
          {
            EquationFVDetail::Gather(request.blocks, iCell, point);
            LoadGradients(request, iCell, point);
            const auto result = Properties(request, equation, point, worker.evaluator, request.hasWall ? request.wall.cellMean(iCell, 0) : 1);
            if (!result)
            {
                worker.status = result.status.At(iCell);
                break;
            }
            for (int j = 0; j < request.values[0].Width(); ++j)
                request.values[0](iCell, j) = result.value[j];
          } });
        for (const auto &worker : workers)
            if (!worker.status)
                return worker.status;
        return {};
    }
    NumericalStatus PrepareAuxiliaryFaces(const AuxiliaryKernelRequest &request, const EquationSystem &equation, const BoundaryPointOperator &boundary, SpanView<AuxiliaryWorkerScratch> workers, FiniteVolumeStencilView::Indices faces) noexcept
    {
        const bool spectral = request.enabled[1] || request.enabled[4], wave = request.enabled[2] || request.enabled[5], pressure = request.enabled[3] || request.enabled[6];
        if (!spectral && !wave && !pressure)
            return {};
        const auto metric = request.metric;
        const auto &derived = request.derived;
        if (workers.empty())
            return NumericalStatus{NumericalError::InvalidInput};
        const auto compute = [&](AuxiliaryWorkerScratch &worker, index begin, index end) -> NumericalStatus
        {
            auto &evaluator = worker.evaluator;
            auto &left = worker.left;
            auto &right = worker.right;
            auto &face = worker.face;
            const auto prepared = [&](index iCell, auto &point) -> NumericalResult<std::array<real, 5>>
            {
                if (derived && request.ransFirst < 0)
                    return std::array<real, 5>{(*derived)(iCell, 0), (*derived)(iCell, 1), (*derived)(iCell, 2)};
                if (request.enabled[0])
                    return std::array<real, 5>{request.values[0](iCell, 0), request.values[0](iCell, 1), request.values[0](iCell, 2)};
                return Properties(request, equation, point, evaluator, request.hasWall ? request.wall.cellMean(iCell, 0) : 1);
            };
            for (index iRow = begin; iRow < end; ++iRow)
            {
                const index iAux = faces(iRow, 0);
                const auto iFace = request.stencil.stencilFaces(iAux, 0), first = metric.FaceQuadratureBegin(iFace);
                const auto &anchor = FiniteVolumeDetail::ExpandSample(request.stencil.RHSFaceSample(iFace, 0, first), request.conditions);
                EquationFVDetail::Gather(request.blocks, anchor.cell, left);
                LoadGradients(request, anchor.cell, left);
                const auto leftResult = prepared(anchor.cell, left);
                if (!leftResult)
                    return leftResult.status.At(anchor.cell);
                const auto &leftProperties = leftResult.value;
                const Riemann::AcousticState leftData{{left.Flow(request.flowFirst), leftProperties[0]}, leftProperties[1]};
                real convection = 0, diffusion = 0, jump = 0, minimum = leftProperties[0];
                const auto count = metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
                for (index q = 0; q < count; ++q)
                {
                    const auto &sample = FiniteVolumeDetail::ExpandSample(request.stencil.RHSFaceSample(iFace, 0, first + q), request.conditions);
                    if (sample.other == UnInitIndex)
                    {
                        right.values = left.values;
                        right.gradient = left.gradient;
                        // Euler's EvaluateDt uses the owned mean on both sides
                        // of an ordinary physical boundary. A coupling
                        // interface supplies its explicitly published remote
                        // point state.
                        if (sample.boundary.kind == BoundaryKind::Interface)
                        {
                            if (const auto status = boundary.TryEvaluate(
                                    right.values, sample.boundary,
                                    sample.boundaryPoint,
                                    {sample.normal[0], sample.normal[1],
                                     sample.normal[2]},
                                    {sample.ownPoint[0], sample.ownPoint[1],
                                     sample.ownPoint[2]},
                                    evaluator, request.frame, request.time,
                                    BoundaryPointStateKind::FaceTrace, {}, 0,
                                    request.incoming);
                                !status)
                                return status;
                        }
                    }
                    else
                    {
                        EquationFVDetail::Gather(request.blocks, sample.other, right);
                        LoadGradients(request, sample.other, right);
                        // Setup proves exact identity, including translational
                        // periodic images. Keep the full transform for every
                        // genuinely rotated component request.frame.
                        if (sample.changesFrame)
                            EquationFVDetail::RotatePoint(right, sample.rotation,
                                                          request.vectorStarts);
                    }
                    const auto rightResult = sample.other == UnInitIndex
                                                 ? (sample.boundary.kind == BoundaryKind::Interface
                                                        ? Properties(request, equation, right, evaluator)
                                                        : leftResult)
                                                 : prepared(sample.other, right);
                    if (!rightResult)
                        return rightResult.status.At(sample.other == UnInitIndex ? sample.cell : sample.other, sample.point);
                    const auto &rightProperties = rightResult.value;
                    const Riemann::AcousticState rightData{{right.Flow(request.flowFirst), rightProperties[0]}, rightProperties[1]};
                    const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                    const auto grid = request.frame.GridVelocity({sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]});
                    if (spectral)
                    {
                        convection = std::max({convection, Riemann::SignalSpeed(leftData, normal, grid), Riemann::SignalSpeed(rightData, normal, grid)});
                        real bound = std::max(leftProperties[2], rightProperties[2]);
                        if (request.hasWall)
                        {
                            for (index j = 0; j < request.components; ++j)
                                face.values[j] = .5 * (left.values[j] + right.values[j]);
                            face.gradient = .5 * (left.gradient + right.gradient);
                            if (sample.other == UnInitIndex)
                            {
                                face.values = left.values;
                            }
                            const auto faceResult = Properties(request, equation, face, evaluator, request.wall.faces(sample.face, q));
                            if (!faceResult)
                                return faceResult.status.At(sample.cell, sample.point);
                            bound = std::max(bound, faceResult.value[2]);
                        }
                        diffusion = std::max(diffusion, bound * metric.GetFaceArea(request.stencil.stencilFaces(iAux, 0)) *
                                                            (1 / metric.GetCellVol(sample.cell) + 1 / metric.GetCellVol(sample.other == UnInitIndex ? sample.cell : sample.other)));
                    }
                    if (wave)
                        jump = std::max(jump, Riemann::TransverseSignalJump(leftData, rightData, normal));
                    if (pressure)
                        minimum = std::min(minimum, rightProperties[0]);
                }
                if (request.enabled[1])
                {
                    request.values[1](iAux, 0) = convection;
                    request.values[1](iAux, 1) = diffusion;
                }
                if (request.enabled[2])
                    request.values[2](iAux, 0) = jump;
                if (request.enabled[3])
                    request.values[3](iAux, 0) = minimum;
                if (request.useScratch)
                {
                    request.scratch(iAux, 0) = convection + diffusion;
                    request.scratch(iAux, 1) = jump;
                    request.scratch(iAux, 2) = minimum;
                }
            }
            return {};
        };
        OMP::ForRows(0, workers.size(), [&](index iWorker)
                     {
            const index count = faces.RowCount(), quotient = count / workers.size(), remainder = count % workers.size();
            const index begin = iWorker * quotient + std::min(iWorker, remainder);
            workers[iWorker].status = compute(workers[iWorker], begin, begin + quotient + (iWorker < remainder)); });
        for (const auto &worker : workers)
            if (!worker.status)
                return worker.status;
        return {};
    }
    void ReduceAuxiliaryCells(const AuxiliaryKernelRequest &request, FiniteVolumeStencilView::Indices cells)
    {
        if (!request.enabled[4] && !request.enabled[5] && !request.enabled[6])
            return;
        const auto metric = request.metric;
        const real rotationBound = Geom::tPoint{request.frame.angularVelocity[0], request.frame.angularVelocity[1], request.frame.angularVelocity[2]}.norm();
        OMP::ForRows(0, cells.RowCount(), [&](index iRow)
                     {
            const index iCell = cells(iRow, 0);
            real sum = 0, wave = 0, pressure = std::numeric_limits<real>::max();
            for (index entry = request.stencil.auxiliaryCellOffsets(iCell, 0);
                 entry < request.stencil.auxiliaryCellOffsets(iCell + 1, 0); ++entry)
            {
                const index iAux = request.stencil.auxiliaryCellFaces(entry, 0);
                if (request.enabled[4])
                    sum += metric.GetFaceArea(request.stencil.stencilFaces(iAux, 0)) *
                           (request.enabled[1] ? request.values[1](iAux, 0) + request.values[1](iAux, 1) : request.scratch(iAux, 0));
                if (request.enabled[5])
                    wave = std::max(wave, request.enabled[2] ? request.values[2](iAux, 0) : request.scratch(iAux, 1));
                if (request.enabled[6])
                    pressure = std::min(pressure, request.enabled[3] ? request.values[3](iAux, 0) : request.scratch(iAux, 2));
            }
            if (request.enabled[4])
                request.values[4](iCell, 0) = sum + metric.GetCellVol(iCell) * rotationBound;
            if (request.enabled[5])
                request.values[5](iCell, 0) = wave;
            if (request.enabled[6])
                request.values[6](iCell, 0) = pressure; });
    }

}
