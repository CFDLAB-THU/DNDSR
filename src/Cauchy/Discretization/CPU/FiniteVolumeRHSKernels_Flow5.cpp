#include "Cauchy/Discretization/CPU/FiniteVolumeRHSKernels.hpp"
#include "DNDS/Core/OMP.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSampleKernels.hpp"
#include "CFV/CPU/ReconstructionBasisKernels.hpp"
#include "Cauchy/Discretization/ViscousFlux.hpp"
#include "Cauchy/Riemann/RiemannFluxMath.hpp"
#include <cstdio>
#include <cstdlib>

namespace DNDS::Cauchy::CPU
{
    namespace
    {
        using FiniteVolumeDetail::Load;
        using FiniteVolumeDetail::RotateState;
        FlowGradient RotateGradient(const FlowGradient &gradient, const Geom::tGPoint &rotation)
        {
            FlowGradient result{};
            for (int component = 0; component < 5; ++component)
                for (int d = 0; d < 3; ++d)
                    for (int k = 0; k < 3; ++k)
                        if (component == 0 || component == 4)
                            result[component][d] += rotation(d, k) * gradient[component][k];
                        else
                            for (int v = 0; v < 3; ++v)
                                result[component][d] += rotation(component - 1, v) * gradient[1 + v][k] * rotation(d, k);
            return result;
        }

    }

    NumericalStatus EvaluateFlowFlux(const FlowRHSRequest &request, const EquationSystem &equation,
                                     const Riemann::RiemannFlux &fluxOperator, const RHSReadViews &input,
                                     const RHSWriteViews &output, CFV::ReconstructionRegion region,
                                     FlowRHSWorkerBuffers workers) noexcept
    {
        if (!(std::isfinite(request.time)))
            return NumericalStatus{NumericalError::InvalidState, 1, -1};
        const auto metric = request.metric;
        const int modes = request.modes;
        const int dim = request.dimension;
        const auto &gas = equation.Gas();
        const auto &thermalBounds = request.thermalBounds;
        const auto frame = request.frame;
        if (!(!request.hasLocalInterfaceRows || input.boundary))
            return NumericalStatus{NumericalError::InvalidState, 2, -1};
        const auto incoming = input.boundary.value_or(RHSScalarView{});
        if (input.boundary)
            if (!(incoming.Width() == 5))
                return NumericalStatus{NumericalError::InvalidState, 3, -1};
        const bool viscous = equation.Viscous() && input.includeViscosity;
        const bool direct = request.direct;
        const auto physicalGradient = input.conservativeGradient.value_or(RHSFieldView{});
        const auto physicalViscousGradient = input.viscousConservativeGradient.value_or(physicalGradient);
        // P0 always uses the conservative normal jump, without reconstructed
        // face derivatives; higher orders retain their selected formulation.
        const bool polynomialViscosity = viscous && request.polynomialViscosity;
        const auto viscousCoefficient = input.viscousCoefficients.value_or(input.coefficients);
        const auto &mean = input.mean;
        const auto &coefficient = input.coefficients;
        const auto &gradientInput = input.primitiveGradient;
        const auto result = output.residual;
        const bool produceLow = bool(output.lowOrder), hasThermo = bool(input.auxiliary.meanThermodynamics);
        const bool useWave = bool(input.auxiliary.cellWaveJump), usePressure = bool(input.auxiliary.cellPressureMinimum);
        const auto lowOutput = output.lowOrder.value_or(RHSFlowOutputView{});
        const auto thermo = input.auxiliary.meanThermodynamics.value_or(RHSScalarView{});
        const auto waveInput = input.auxiliary.cellWaveJump.value_or(RHSFieldView{});
        const auto pressureInput = input.auxiliary.cellPressureMinimum.value_or(RHSFieldView{});
        const auto &fluxPolicy = request.fluxPolicy;
        const auto baseFluxSettings = fluxPolicy.settings;
        const bool writeFace = bool(output.faceFlux);
        const bool noRsOnWall = fluxPolicy.noRsOnWall;
        const bool noGRPOnWall = fluxPolicy.noGRPOnWall;
        const bool distinctViscousState = direct ? bool(input.viscousConservativeGradient) : bool(input.viscousCoefficients);
        if (!(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
              region == CFV::ReconstructionRegion::HaloDependent))
            return NumericalStatus{NumericalError::InvalidState, 4, -1};
        const bool needGhost = region != CFV::ReconstructionRegion::Core && request.stencil.dependentCells.RowCount() != 0;
        const auto checkField = [&](const auto &field, index width) -> NumericalStatus
        {
            if (!(field.owned.RowCount() == request.ownedCells && field.owned.Width() == width))
                return NumericalStatus{NumericalError::InvalidState, 5, -1};
            if (needGhost)
                if (!(field.ghost.RowCount() == request.allCells - request.ownedCells && field.ghost.Width() == width))
                    return NumericalStatus{NumericalError::InvalidState, 6, -1};
            return {};
        };
        {
            if (const auto status = checkField(mean, 5); !status)
                return status;
        }
        if (!direct)
        {
            if (const auto status = checkField(coefficient, 5 * modes); !status)
                return status;
        }
        else if (!input.meanOnly && !request.firstOrder && request.linear)
        {
            if (!(bool(input.conservativeGradient)))
                return NumericalStatus{NumericalError::InvalidState, 7, -1};
            {
                if (const auto status = checkField(physicalGradient, 5 * dim); !status)
                    return status;
            }
            if (polynomialViscosity)
            {
                if (const auto status = checkField(physicalViscousGradient, 5 * dim); !status)
                    return status;
            }
        }
        if (viscous && !polynomialViscosity)
        {
            if (const auto status = checkField(gradientInput, 5 * dim); !status)
                return status;
        }
        if (polynomialViscosity && !direct)
        {
            if (const auto status = checkField(viscousCoefficient, 5 * modes); !status)
                return status;
        }
        if (useWave)
        {
            if (const auto status = checkField(waveInput, 1); !status)
                return status;
        }
        if (usePressure)
        {
            if (const auto status = checkField(pressureInput, 1); !status)
                return status;
        }
        if (!(result.RowCount() == request.ownedCells && result.Width() == 5))
            return NumericalStatus{NumericalError::InvalidState, 8, -1};
        if (produceLow)
            if (!(lowOutput.RowCount() == result.RowCount() && lowOutput.Width() == 5))
                return NumericalStatus{NumericalError::InvalidState, 9, -1};
        if (output.faceFlux)
            if (!(output.faceFlux->RowCount() == request.stencil.stencilFaces.RowCount() && output.faceFlux->Width() == 5))
                return NumericalStatus{NumericalError::InvalidState, 10, -1};
        if (hasThermo)
            if (!(thermo.Width() == 3 && thermo.RowCount() >= (needGhost ? request.allCells : request.ownedCells)))
                return NumericalStatus{NumericalError::InvalidState, 11, -1};
        if (!(!request.useStencil ||
              ((!fluxPolicy.transverseSignal || useWave) && (!fluxPolicy.pressureStencil || usePressure))))
            return NumericalStatus{NumericalError::InvalidState, 12, -1};
        auto compute = [&](index iCell, RHSKernelDiagnostics &diagnostics) -> NumericalStatus
        {
            FlowState integrated{};
            FlowState integratedLow{};
            const auto faces = request.stencil.topology.CellFaces(iCell);
            for (rowsize slot = 0; slot < faces.size(); ++slot)
            {
                const index iFace = faces[slot];
                const int side = request.stencil.topology.CellIsFaceBack(iCell, iFace, slot) ? 0 : 1;
                for (index iPoint = metric.FaceQuadratureBegin(iFace); iPoint < metric.FaceQuadratureEnd(iFace); ++iPoint)
                {
                    const auto sample = FiniteVolumeDetail::ExpandSample(request.stencil.RHSFaceSample(iFace, side, iPoint), request.conditions);
                    const bool internal = sample.other != UnInitIndex;
                    const bool recordFace = writeFace && (!output.boundaryFacesOnly || !internal);
                    const auto boundaryKind = sample.boundary.kind;
                    const real weight = metric.GetFaceWeight(sample.point);
                    if (recordFace && sample.point == metric.FaceQuadratureBegin(sample.face))
                        for (int j = 0; j < 5; ++j)
                            (*output.faceFlux)(sample.auxiliaryFace, j) = 0;
                    auto reconstruct = [&](index cell, int parent)
                    {
                        auto state = Load(mean, cell);
                        auto viscousState = state;
                        FlowGradient gradient{};
                        if (direct && !input.meanOnly && request.linear && !request.firstOrder)
                        {
                            const auto displacement = FiniteVolumeDetail::DirectFVDisplacement(metric, sample, parent != 0);
                            const auto row = physicalGradient[cell];
                            const auto viscousRow = physicalViscousGradient[cell];
                            for (int component = 0; component < 5; ++component)
                                for (int d = 0; d < dim; ++d)
                                {
                                    state[component] += row[component * dim + d] * displacement[d];
                                    if (polynomialViscosity)
                                    {
                                        viscousState[component] += viscousRow[component * dim + d] * displacement[d];
                                        gradient[component][d] = viscousRow[component * dim + d];
                                    }
                                }
                        }
                        else if (!direct && !input.meanOnly && !request.firstOrder)
                        {
                            CFV::Math::SmallMatrix basisScratch;
                            const auto basis = CFV::CPU::EvaluateFaceBasis(*request.basis, sample.face, parent == 0 ? sample.side : 1 - sample.side,
                                                                           sample.point, polynomialViscosity ? 1 : 0, basisScratch);
                            const auto row = coefficient[cell];
                            const auto viscousRow = polynomialViscosity ? viscousCoefficient[cell] : decltype(viscousCoefficient[cell]){};
                            for (int component = 0; component < 5; ++component)
                                for (int mode = 0; mode < modes; ++mode)
                                {
                                    const real value = row[component * modes + mode];
                                    state[component] += value * basis(0, mode);
                                    if (polynomialViscosity)
                                    {
                                        const real viscousValue = input.viscousCoefficients ? viscousRow[component * modes + mode] : value;
                                        if (input.viscousCoefficients)
                                            viscousState[component] += viscousValue * basis(0, mode);
                                        for (int d = 0; d < dim; ++d)
                                            gradient[component][d] += viscousValue * basis(1 + d, mode);
                                    }
                                }
                        }
                        if (!distinctViscousState)
                            viscousState = state;
                        // Explicit local first-order fallback keeps the mean
                        // unchanged when a high-order face value is inadmissible.
                        if (!ThermallyAdmissible(state, thermalBounds))
                        {
                            diagnostics.AddFaceFallback();
                            state = Load(mean, cell);
                        }
                        if (viscous && !polynomialViscosity)
                            for (int component = 0; component < 5; ++component)
                                for (int d = 0; d < dim; ++d)
                                    gradient[component][d] = gradientInput(cell, component * dim + d);
                        struct PointState
                        {
                            FlowState state, viscousState;
                            FlowGradient gradient;
                        };
                        return PointState{state, viscousState, gradient};
                    };
                    const auto left = reconstruct(iCell, 0);
                    FlowState right{}, viscousRight{};
                    FlowGradient rightGradient{};
                    if (internal)
                    {
                        auto other = reconstruct(sample.other, 1);
                        right = sample.changesFrame ? RotateState(other.state, sample.rotation) : other.state;
                        if (viscous)
                        {
                            viscousRight = sample.changesFrame ? RotateState(other.viscousState, sample.rotation) : other.viscousState;
                            rightGradient = sample.changesFrame ? RotateGradient(other.gradient, sample.rotation) : other.gradient;
                        }
                    }
                    else
                    {
                        const auto exterior = FiniteVolumeDetail::BoundaryExterior(sample, gas, left.state, request.time, frame, incoming);
                        if (!exterior)
                            return exterior.status.At(iCell, sample.point);
                        right = exterior.value;
                        viscousRight = right;
                        rightGradient = left.gradient;
                    }
                    const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                    const Vector3 point{sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]};
                    const bool wallFluxState = noRsOnWall && !internal && boundaryKind == BoundaryKind::NoSlipWall;
                    const auto convectiveLeftResult = wallFluxState ? FiniteVolumeDetail::BoundaryFaceValue(sample, gas, left.state, request.time, frame, incoming)
                                                                    : NumericalResult<FlowState>{left.state};
                    if (!convectiveLeftResult)
                        return convectiveLeftResult.status.At(iCell, sample.point);
                    const auto &convectiveLeft = convectiveLeftResult.value;
                    const auto convectiveRight = wallFluxState ? convectiveLeft : right;
                    const auto leftFlux = equation.PrepareFluxState(convectiveLeft), rightFlux = equation.PrepareFluxState(convectiveRight);
                    const auto gridVelocity = frame.GridVelocity(point);
                    const auto leftMean = Load(mean, iCell);
                    const auto otherMean = internal ? Load(mean, sample.other) : FlowState{};
                    const auto rightMeanResult = !internal ? FiniteVolumeDetail::BoundaryExterior(sample, gas, leftMean, request.time, frame, incoming)
                                                           : NumericalResult<FlowState>{sample.changesFrame ? RotateState(otherMean, sample.rotation) : otherMean};
                    if (!rightMeanResult)
                        return rightMeanResult.status.At(iCell, sample.point);
                    const auto &rightMean = rightMeanResult.value;
                    const auto leftMeanData = hasThermo ? Riemann::AcousticState{{leftMean, thermo(iCell, 0)}, thermo(iCell, 1)} : equation.PrepareFluxState(leftMean);
                    const auto rightMeanData = !internal || !hasThermo ? equation.PrepareFluxState(rightMean) : Riemann::AcousticState{{rightMean, thermo(sample.other, 0)}, thermo(sample.other, 1)};
                    auto fluxSettings = baseFluxSettings;
                    if (useWave)
                    {
                        real jump = waveInput(iCell, 0);
                        if (internal)
                            jump = std::max(jump, waveInput(sample.other, 0));
                        fluxSettings.dLambda = std::max(fluxSettings.dLambda, jump);
                    }
                    if (usePressure)
                    {
                        real minimum = pressureInput(iCell, 0);
                        if (internal)
                            minimum = std::min(minimum, pressureInput(sample.other, 0));
                        fluxSettings.pressureStencilMin = std::min(fluxSettings.pressureStencilMin, minimum);
                    }
                    const auto preparedFlux = PreparePerfectGasRiemann(gas, fluxOperator.Kind(), leftFlux, rightFlux, leftMeanData, rightMeanData, normal, gridVelocity);
                    auto flux = Riemann::Math::Evaluate(fluxOperator.Kind(), preparedFlux.Input(leftFlux, rightFlux, normal, gridVelocity), fluxSettings).flux;
                    FlowState lowFlux{};
                    if (produceLow)
                        lowFlux = wallFluxState ? Riemann::Rusanov(leftFlux, rightFlux, normal, gridVelocity) : Riemann::Rusanov(leftMeanData, rightMeanData, normal, gridVelocity);
                    if (polynomialViscosity && (internal || boundaryKind == BoundaryKind::NoSlipWall || boundaryKind == BoundaryKind::Interface))
                    {
                        FlowState rightLinearization = convectiveRight;
                        FlowState faceMean{};
                        for (int j = 0; j < 5; ++j)
                            faceMean[j] = .5 * (convectiveLeft[j] + rightLinearization[j]);
                        if (!(ThermallyAdmissible(faceMean, thermalBounds)))
                            return NumericalStatus{NumericalError::InvalidState, 13, iCell, sample.point};
                        const bool wall = !internal && boundaryKind == BoundaryKind::NoSlipWall;
                        const real inverseDistance = wall && noGRPOnWall ? 0 : 1 / sample.viscousDistance;
                        const auto diffusion = PolynomialViscousFlux(equation, convectiveLeft, rightLinearization, left.viscousState, viscousRight,
                                                                     left.gradient, rightGradient, normal, inverseDistance, wall && !sample.boundary.isothermal);
                        for (int component = 0; component < 5; ++component)
                        {
                            flux[component] -= diffusion[component];
                            lowFlux[component] -= diffusion[component];
                        }
                    }
                    else if (viscous && !polynomialViscosity &&
                             (internal || boundaryKind == BoundaryKind::NoSlipWall || boundaryKind == BoundaryKind::Interface))
                    {
                        auto primitive = gas.Primitive(left.state);
                        auto gradient = left.gradient;
                        const auto ownPrimitive = gas.Primitive(Load(mean, iCell));
                        const auto neighborResult = !internal ? FiniteVolumeDetail::BoundaryGradientValue(sample, gas, Load(mean, iCell), request.time, frame, incoming)
                                                              : NumericalResult<FlowState>{sample.changesFrame ? RotateState(Load(mean, sample.other), sample.rotation) : Load(mean, sample.other)};
                        if (!neighborResult)
                            return neighborResult.status.At(iCell, sample.point);
                        const auto rightPrimitive = gas.Primitive(neighborResult.value);
                        // Evaluate into an owning vector: an Eigen expression
                        // would retain a dangling temporary cell barycenter.
                        const Geom::tPoint distance = !internal ? Geom::tPoint(sample.ownPoint - metric.GetCellBary(iCell)) : Geom::tPoint(sample.otherPoint - metric.GetCellBary(iCell));
                        const real lengthSquared = distance.squaredNorm();
                        if (!(lengthSquared > 0))
                            return NumericalStatus{NumericalError::InvalidState, 14, iCell, sample.point};
                        const std::array<real, 5> own{ownPrimitive.density, ownPrimitive.velocity[0], ownPrimitive.velocity[1], ownPrimitive.velocity[2], ownPrimitive.temperature};
                        const std::array<real, 5> neighbor{rightPrimitive.density, rightPrimitive.velocity[0], rightPrimitive.velocity[1], rightPrimitive.velocity[2], rightPrimitive.temperature};
                        for (int component = 0; component < 5; ++component)
                        {
                            real projected = 0;
                            for (int d = 0; d < 3; ++d)
                            {
                                if (internal)
                                    gradient[component][d] = .5 * (gradient[component][d] + rightGradient[component][d]);
                                projected += gradient[component][d] * distance[d];
                            }
                            for (int d = 0; d < 3; ++d)
                                gradient[component][d] += (neighbor[component] - own[component] - projected) * distance[d] / lengthSquared;
                        }
                        if (internal)
                        {
                            const auto p = gas.Primitive(right);
                            for (int d = 0; d < 3; ++d)
                                primitive.velocity[d] = .5 * (primitive.velocity[d] + p.velocity[d]);
                        }
                        else if (boundaryKind == BoundaryKind::SlipWall || boundaryKind == BoundaryKind::NoSlipWall)
                        {
                            FlowState faceMean{};
                            for (int j = 0; j < 5; ++j)
                                faceMean[j] = .5 * (convectiveLeft[j] + convectiveRight[j]);
                            primitive = gas.Primitive(faceMean);
                            if (!sample.boundary.isothermal)
                            {
                                real heat = 0;
                                for (int d = 0; d < 3; ++d)
                                    heat += gradient[4][d] * normal[d];
                                for (int d = 0; d < 3; ++d)
                                    gradient[4][d] -= heat * normal[d];
                            }
                        }
                        const auto viscous = gas.ViscousFlux(primitive, gradient, normal);
                        for (int component = 0; component < 5; ++component)
                        {
                            flux[component] -= viscous[component];
                            lowFlux[component] -= viscous[component];
                        }
                    }
                    if (sample.forceSlot != UnInitIndex)
                        for (int d = 0; d < 3; ++d)
                            diagnostics.AddBoundaryForce(sample.forceSlot, d, weight * flux[1 + d]);
                    if (output.boundaryMomentum.RowCount() && sample.boundaryPoint != UnInitIndex)
                        for (int d = 0; d < 3; ++d)
                            output.boundaryMomentum(sample.boundaryPoint, d) = weight * flux[1 + d];
                    FlowState canonical{};
                    if (recordFace)
                    {
                        canonical = sample.side == 0 || !sample.changesFrame ? flux : RotateState(flux, sample.rotation.transpose());
                        if (sample.side != 0)
                            for (auto &value : canonical)
                                value = -value;
                    }
                    for (int component = 0; component < 5; ++component)
                    {
                        integrated[component] -= weight * flux[component];
                        if (recordFace)
                            (*output.faceFlux)(sample.auxiliaryFace, component) -= weight * canonical[component];
                        if (produceLow)
                            integratedLow[component] -= weight * lowFlux[component];
                    }
                }
            }
            for (int component = 0; component < 5; ++component)
            {
                result(iCell, component) = integrated[component];
                if (produceLow)
                    lowOutput(iCell, component) = integratedLow[component];
            }
            return {};
        };
        const auto rows = request.stencil.RHSRows(region, false, request.ownedCells);
        const index count = rows.RowCount();
        // Optional face diagnostics currently revisit a shared face from both
        // parents. Keep their established serial publication until the common
        // face-owned flux/gather path replaces it; never race those writes.
        const index threads = writeFace && !output.boundaryFacesOnly ? 1 : std::min<index>(OMP::DistributedThreads(), std::max<index>(1, count));
        if (workers.diagnostics.size() < threads || workers.status.size() < threads)
            return NumericalStatus{NumericalError::InvalidState, 40, -1};
        return rows.Visit([&](auto cellAt) -> NumericalStatus
                          {
        if (threads == 1)
        {
            for (index iRow = 0; iRow < count; ++iRow)
                if (const auto status = compute(cellAt(iRow), workers.diagnostics[0]); !status)
                    return status;
        }
        else
        {
            // Small per-worker diagnostics; no field-sized packing or counter
            // atomics. All point/basis/flux scratch is local to compute's row.
            OMP::ForRows(0, threads, [&](index worker)
                         {
                const index quotient = count / threads, remainder = count % threads;
                const index begin = worker * quotient + std::min(worker, remainder);
                const index end = begin + quotient + (worker < remainder);
                for (index iRow = begin; iRow < end; ++iRow)
                    if (const auto status = compute(cellAt(iRow), workers.diagnostics[worker]); !status)
                    {
                        workers.status[worker] = status;
                        break;
                    } });
            for (index worker = 0; worker < threads; ++worker)
            {
                if (!workers.status[worker])
                    return workers.status[worker];
            }
        }
            return {}; });
    }

    NumericalStatus AddFlowRotationSource(const FlowRHSRequest &request, const RHSReadViews &input,
                                          const RHSWriteViews &output, CFV::ReconstructionRegion region) noexcept
    {
        if (!(std::isfinite(request.time)))
            return NumericalStatus{NumericalError::InvalidState, 1, -1};
        if (!(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
              region == CFV::ReconstructionRegion::HaloDependent))
            return NumericalStatus{NumericalError::InvalidState, 2, -1};
        if (!(input.mean.owned.RowCount() == request.ownedCells && input.mean.owned.Width() == 5 &&
              output.residual.RowCount() == request.ownedCells && output.residual.Width() == 5))
            return NumericalStatus{NumericalError::InvalidState, 3, -1};
        if (output.lowOrder)
            if (!(output.lowOrder->RowCount() == output.residual.RowCount() && output.lowOrder->Width() == 5))
                return NumericalStatus{NumericalError::InvalidState, 4, -1};
        const auto frame = request.frame;
        if (!frame.moving)
            return {};
        const auto metric = request.metric;
        const auto compute = [&](index iCell)
        {
            const auto source = frame.RotationSource(Load(input.mean, iCell));
            for (int j = 0; j < 5; ++j)
            {
                const real value = metric.GetCellVol(iCell) * source[j];
                output.residual(iCell, j) += value;
                if (output.lowOrder)
                    (*output.lowOrder)(iCell, j) += value;
            }
        };
        const auto rows = request.stencil.RHSRows(region, false, request.ownedCells);
        rows.Visit([&](auto cellAt)
                   { OMP::ForRows(0, rows.RowCount(), [&](index iRow)
                                  { compute(cellAt(iRow)); }); });
        return {};
    }
}
