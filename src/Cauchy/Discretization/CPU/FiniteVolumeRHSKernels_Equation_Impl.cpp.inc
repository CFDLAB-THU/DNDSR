#include "Cauchy/Discretization/CPU/FiniteVolumeFluxShapes.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeSampleKernels.hpp"
#include "CFV/CPU/ReconstructionBasisKernels.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "Physics/Incompressible/RANSPoint.hpp"
#include "Cauchy/Physics/ACMFlux.hpp"
#include "Cauchy/Physics/ViscousFlux.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "Cauchy/Riemann/RiemannFluxMath.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy::CPU
{
    using namespace EquationFVDetail;
    template <EquationPhysicsProfile Model, ReconstructionForm Form, int C, int D, int M, class Input, class Output>
    NumericalResult<index> EvaluateEquationFluxRows(const EquationRHSRequest &request, const EquationSystem &equation,
                                                     const BoundaryPointOperator &boundary, const Riemann::RiemannFlux &fluxOperator,
                                                     PointEvaluator &evaluator, Input owned, Input remote,
                                                     Output destination, Output lowDestination, CFV::ReconstructionRegion region,
                                                     EquationRHSPointScratch &scratch, index firstRow, index lastRow, bool facePhase) noexcept
    {
        index localFallbacks = 0;
        const auto metric = request.metric;
        const bool acm = Model == EquationPhysicsProfile::Generic ? equation.IsACM() : IsACMProfile(Model);
        const auto gatherMean = [&](index iCell, PointScratch &point)
        {
            for (index j = 0; j < owned.ComponentNumber(); ++j)
                point.values[j] = iCell < owned.RowCount() ? owned(iCell, j) : remote(iCell - owned.RowCount(), j);
            point.gradient.setZero();
        };
        const int modes = M < 0 ? request.modes : M;
        const int dim = D < 0 ? request.dimension : D;
        const bool viscous = request.viscous;
        const auto &fluxPolicy = request.fluxPolicy;
        const bool noRsOnWall = fluxPolicy.noRsOnWall;
        const bool noGRPOnWall = fluxPolicy.noGRPOnWall;
        const bool rusanovFlux = fluxPolicy.rusanov;
        const auto &fluxRequirements = fluxPolicy.required;
        const auto &baseFluxSettings = fluxPolicy.settings;
        const bool useWaveJump = request.useStencil && fluxPolicy.transverseSignal;
        const bool usePressureMinimum = request.useStencil && fluxPolicy.pressureStencil;
        const bool preparedWaveJump = request.preparedWaveJump;
        const bool preparedPressureMinimum = request.preparedPressureMinimum;
        if ((useWaveJump && !preparedWaveJump) || (usePressureMinimum && !preparedPressureMinimum))
            return NumericalStatus{NumericalError::InvalidInput};
        const int species = request.species;
        const int independent = std::max(0, species - 1);
        const index width = C < 0 ? request.width : C;
        DNDS_assert(index(scratch.integrated.size()) == width && index(scratch.flux.size()) == width &&
                    index(scratch.left.values.size()) == width && index(scratch.right.values.size()) == width &&
                    index(scratch.face.values.size()) == width && index(scratch.cellMean.values.size()) == width &&
                    index(scratch.speciesFlux.size()) == independent &&
                    (!request.produceLow || (index(scratch.integratedLow.size()) == width && index(scratch.lowFlux.size()) == width)));
        const auto frame = request.frame;
        auto &left = scratch.left, &right = scratch.right, &face = scratch.face, &cellMean = scratch.cellMean;
        const auto *ransModel = equation.HasRANS() ? &equation.RANS() : nullptr;
        const auto thermalBounds = request.thermal;
        const auto ransMean = request.ransMean;
        auto &integrated = scratch.integrated, &flux = scratch.flux, &speciesFlux = scratch.speciesFlux;
        auto &integratedLow = scratch.integratedLow, &lowFlux = scratch.lowFlux;
        auto &transport = scratch.transport;
        const auto wallValues = request.wall;
        auto &basisScratch = scratch.basis;
        const auto faceValues = FluxTable<C>(request.faceValues);
        const auto lowFaces = FluxTable<C>(request.lowFaceValues);
        // Bind the two complete, compact coefficient parents once per worker.
        // The dynamic leg below preserves block-specific offsets and layouts.
        constexpr int fieldWidth = Form == ReconstructionForm::Modal ? C * M : C * D;
        const auto bindCoefficients = [&](bool useViscous)
        {
            RHSFieldView field{};
            if constexpr (Form == ReconstructionForm::Modal)
            {
                const auto blocks = useViscous && !request.viscousRecBlocks.empty() ? request.viscousRecBlocks : request.recBlocks;
                field = {blocks[0].owned.parent, blocks[0].ghost.parent};
            }
            if constexpr (Form == ReconstructionForm::DirectGradient)
                field = (useViscous && !request.directViscousViews.empty() ? request.directViscousViews : request.directViews)[0];
            return Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<fieldWidth, fieldWidth>>{
                FluxTable<fieldWidth>(field.owned), FluxTable<fieldWidth>(field.ghost)};
        };
        const auto coefficients = bindCoefficients(false), viscousCoefficients = bindCoefficients(true);
        const auto primitiveGradients = [&]
        {
            if constexpr (C > 0)
                return Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<5 * D, 5 * D>>{
                    FluxTable<5 * D>(request.primitiveGradientValues.owned), FluxTable<5 * D>(request.primitiveGradientValues.ghost)};
            else
                return request.primitiveGradientValues;
        }();
        const auto reconstruct = [&](index iCell, const FiniteVolumeDetail::SampleValue &sample, int side, PointScratch &point)
        {
            gatherMean(iCell, point);
            // P0 uses mean states and only the normal face jump for
            // viscosity. This does not suppress volume-source gradients:
            // RANS/chemistry sampling retains the configured derivatives.
            if constexpr (Form == ReconstructionForm::DirectGradient)
            {
                const auto dx = FiniteVolumeDetail::DirectFVDisplacement(metric, sample, iCell != sample.cell || side != sample.side);
                const auto gradient = coefficients[iCell], viscousGradient = viscousCoefficients[iCell];
                for (int j = 0; j < C; ++j)
                    for (int d = 0; d < D; ++d)
                    {
                        point.values[j] += gradient[j * D + d] * dx[d];
                        if (viscous)
                            point.gradient(j, d) = viscousGradient[j * D + d];
                    }
            }
            else if constexpr (Form == ReconstructionForm::Modal)
            {
                const auto basis = CFV::CPU::EvaluateFaceBasis(*request.basis, sample.face, side, sample.point, viscous ? 1 : 0, basisScratch);
                const auto modes = coefficients[iCell], viscousModes = viscousCoefficients[iCell];
                for (int j = 0; j < C; ++j)
                    for (int mode = 0; mode < M; ++mode)
                    {
                        point.values[j] += modes[j * M + mode] * basis(0, mode);
                        if (viscous)
                            for (int d = 0; d < D; ++d)
                                point.gradient(j, d) += viscousModes[j * M + mode] * basis(1 + d, mode);
                    }
            }
            else if constexpr (Form == ReconstructionForm::Generic)
            if (!request.firstOrder && (!request.direct || request.linear))
            {
                if (request.direct)
                {
                    const auto displacement = FiniteVolumeDetail::DirectFVDisplacement(metric, sample, iCell != sample.cell || side != sample.side);
                    for (index iBlock = 0; iBlock < request.directLocations.size(); ++iBlock)
                    {
                        const auto location = request.directLocations[iBlock];
                        const auto row = request.directViews[location.physicalField][iCell];
                        const auto viscousRow = request.directViscousViews.empty() ? row : request.directViscousViews[location.physicalField][iCell];
                        for (index j = 0; j < location.componentNumber; ++j)
                            for (int d = 0; d < dim; ++d)
                            {
                                const real value = row[(location.firstComponent + j) * dim + d];
                                point.values[request.logicalFirst[iBlock] + j] += value * displacement[d];
                                if (viscous)
                                    point.gradient(request.logicalFirst[iBlock] + j, d) = viscousRow[(location.firstComponent + j) * dim + d];
                            }
                    }
                }
                else
                {
                    const auto basis = CFV::CPU::EvaluateFaceBasis(*request.basis, sample.face, side, sample.point, viscous ? 1 : 0, basisScratch);
                    for (index iBlock = 0; iBlock < index(request.meanBlocks.size()); ++iBlock)
                    {
                        const auto row = request.recBlocks[iBlock][iCell];
                        const auto viscousRow = request.viscousRecBlocks.empty() ? row : request.viscousRecBlocks[iBlock][iCell];
                        for (index j = 0; j < request.meanBlocks[iBlock].ComponentNumber(); ++j)
                            for (int mode = 0; mode < modes; ++mode)
                            {
                                const real value = row[j * modes + mode];
                                const index logical = request.logicalFirst[iBlock] + j;
                                point.values[logical] += value * basis(0, mode);
                                if (viscous)
                                    for (int d = 0; d < dim; ++d)
                                        point.gradient(logical, d) += viscousRow[j * modes + mode] * basis(1 + d, mode);
                            }
                    }
                }
            }
            if (ransModel)
                RANSReconstructionView{ransModel->Kind(), ransModel->ComponentNumber()}.RepairPoint(ransMean[iCell], point.values.data() + request.ransFirst);
            if (request.primitiveGradient)
            {
                const auto row = primitiveGradients[iCell];
                for (int j = 0; j < 5; ++j)
                    for (int d = 0; d < dim; ++d)
                        point.gradient(request.flowFirst + j, d) = row[j * dim + d];
            }
            // PP POLICY -- point-local physical fallback. Reconstruction
            // deliberately retains raw rhoY for conservative advection and
            // diffusion; its final limiter check only evaluates a temporary
            // clipped composition matching EOS/T. No simplex condition is
            // imposed on the polynomial. If the actual point preparation
            // still fails, Cauchy may replace this temporary point by the
            // valid mean without mutating coefficients. Euler's separate
            // point policy is unchanged. See CauchyPositivity.md.
            if (acm || (!request.faceFallback && !request.positivity))
                return;
            const bool pointAdmissible = request.positivity
                                             ? thermalBounds.ClippedReferenceAdmissible(point.values)
                                             : thermalBounds.Admissible(point.values);
            if ((request.faceFallback || request.positivity) && !pointAdmissible)
            {
                ++localFallbacks;
                gatherMean(iCell, point);
            }
        };
        const bool allowPointFallback = request.faceFallback ||
                                        request.positivity;
        const auto evaluateReconstructedPoint = [&](index iCell, PointScratch &point, FlowState &flow,
                                                    const FiniteVolumeDetail::SampleValue &sample, bool neighborFrame,
                                                    Riemann::AcousticState &output) -> NumericalStatus
        {
            PointProperties properties;
            const auto evaluate = [&]
            {
                return evaluator.TryEvaluate(flow, point.Species(request.speciesFirst, independent), 0, {}, properties);
            };
            auto status = evaluate();
            if (!status && allowPointFallback && Physics::AllowsMeanFallback(status))
            {
                // PP POLICY -- the limiter has already validated the same
                // clipped-reference input. A remaining EOS/T inversion
                // failure is point-local numerical failure: retry from the
                // admissible mean without changing raw polynomial storage.
                // Only an explicit thermodynamic status enables this fallback;
                // invalid requests/resource/backend failures are propagated.
                ++localFallbacks;
                gatherMean(iCell, point);
                // The mean is stored in its cell frame, whereas the flux
                // consumes both states in the anchor face frame. Restore the
                // same vector-component transform used by reconstruction.
                // Use the explicit side flag: periodic self-neighbors can
                // have the same local cell index on both sides.
                if (neighborFrame && sample.changesFrame)
                    RotatePoint(point, sample.rotation, request.vectorStarts);
                flow = point.Flow(request.flowFirst);
                status = evaluate();
            }
            if (!status)
                return status.At(iCell, sample.point);
            output = PointEvaluator::FluxState(flow, properties);
            return {};
        };
        const auto computeFace = [&](index iAux) -> NumericalStatus
        {
            const index iFace = request.stencil.stencilFaces(iAux, 0);
            const index first = metric.FaceQuadratureBegin(iFace), end = metric.FaceQuadratureEnd(iFace);
            const auto anchor = FiniteVolumeDetail::ExpandSample(request.stencil.RHSFaceSample(iFace, 0, first), request.conditions);
            const index iCell = anchor.cell;
            std::fill(integrated.begin(), integrated.end(), 0);
            if (request.produceLow)
                std::fill(integratedLow.begin(), integratedLow.end(), 0);
            for (index entry = first; entry < end; ++entry)
            {
                const auto &sample = FiniteVolumeDetail::ExpandSample(request.stencil.RHSFaceSample(iFace, 0, entry), request.conditions);
                const bool internal = sample.other != UnInitIndex;
                const auto boundaryKind = sample.boundary.kind;
                const real weight = metric.GetFaceWeight(sample.point);
                reconstruct(iCell, sample, sample.side, left);
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                real wallSAViscousInterior = 0;
#endif
                if (internal)
                {
                    reconstruct(sample.other, sample, 1 - sample.side, right);
                    if (sample.changesFrame)
                        RotatePoint(right, sample.rotation, request.vectorStarts);
                }
                else
                {
                    right.values = left.values;
                    right.gradient = left.gradient;
                    OptionalRef<const std::vector<real>> boundaryMean;
                    if (ransModel && ransModel->ComponentNumber() == 2)
                    {
                        gatherMean(sample.cell, cellMean);
                        boundaryMean.Emplace(cellMean.values);
                    }
                    {
                        if (const auto status = boundary.TryEvaluate(
                                right.values, sample.boundary, sample.boundaryPoint,
                                {sample.normal[0], sample.normal[1], sample.normal[2]},
                                {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]},
                                evaluator, frame, request.time,
                                BoundaryPointStateKind::ExteriorState, boundaryMean,
                                request.hasWall ? wallValues.cellMean(sample.cell, 0) : 0,
                                request.incoming);
                            !status)
                            return status;
                    }
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                    if (!acm && ransModel && ransModel->Kind() == Physics::RANS::ModelKind::SA && sample.boundary.kind == BoundaryKind::NoSlipWall)
                    {
                        // ACM keeps both scalar traces for the diffusive wall jump.
                        // Compressible Euler retains ULxyUnlim before its BC zeros the
                        // convective SA pair. Preserve that interior value
                        // for the viscous jump without mutating cell means.
                        wallSAViscousInterior = left.values[request.ransFirst];
                        left.values[request.ransFirst] = right.values[request.ransFirst] = 0;
                    }
#endif
                }
                if (acm)
                {
                    auto params = equation.ACMParameters();
                    params.roe = !rusanovFlux;
                    ACM::NumericalFlux(params, left.values, right.values, sample.normal, flux, int(width));
                    // An ACM slip wall has inviscid pressure flux only. The copied
                    // interior gradient would otherwise leak shear and RANS diffusion.
                    if (viscous && boundaryKind != BoundaryKind::SlipWall)
                    {
                        for (int j = 0; j < width; ++j)
                        {
                            face.values[j] = .5 * (left.values[j] + right.values[j]);
                            for (int d = 0; d < 3; ++d)
                                face.gradient(j, d) = .5 * (left.gradient(j, d) + right.gradient(j, d)) +
                                                      sample.normal[d] * (right.values[j] - left.values[j]) / sample.viscousDistance;
                        }
                        real mu = params.kinematicViscosity;
                        if (ransModel)
                        {
                            const auto at = Physics::Incompressible::RANSPoint(face.values, face.gradient, ransModel->ComponentNumber(),
                                                                               mu, wallValues.faces(sample.face, sample.point - first), 0);
                            const auto transport = ransModel->TryTransport(at);
                            if (!transport)
                                return NumericalStatus{transport.status.code, 6201, iCell, sample.point};
                            const auto diffusion = ransModel->TryDiffusionFlux(at, sample.normal, transport.value.viscosity);
                            if (!diffusion)
                                return NumericalStatus{diffusion.status.code, 6202, iCell, sample.point};
                            for (int j = 0; j < ransModel->ComponentNumber(); ++j)
                                flux[4 + j] -= diffusion.value[j];
                            mu += transport.value.viscosity;
                        }
                        const auto traction = Physics::Transport::NewtonianTraction(face.gradient, sample.normal, mu);
                        // Cauchy omits the former isotropic 2*rho*k/3 normal stress.
                        for (int d = 0; d < 3; ++d)
                            flux[d] -= traction[d];
                    }
                    if (sample.boundaryPoint != UnInitIndex)
                        for (int d = 0; d < 3; ++d)
                            request.wallFlux(sample.boundaryPoint, d) = request.captureBoundaryFlux || boundaryKind == BoundaryKind::SlipWall || boundaryKind == BoundaryKind::NoSlipWall ? weight * flux[d] : 0;
                    for (int j = 0; j < width; ++j)
                    {
                        if (!std::isfinite(flux[j]))
                            return NumericalStatus{NumericalError::InvalidState, 6210 + j, iCell, sample.point};
                        integrated[j] -= weight * flux[j];
                    }
                    continue;
                }
                auto leftFlow = left.Flow(request.flowFirst), rightFlow = right.Flow(request.flowFirst);
                const bool wallFluxState = noRsOnWall && !internal && boundaryKind == BoundaryKind::NoSlipWall;
                const auto prepareWallState = [&]
                {
                    face.values = left.values;
                    const auto original = left.Flow(request.flowFirst);
                    const Physics::Chemistry::SpeciesBufferView wallSpecies{independent ? face.values.data() + request.speciesFirst : nullptr, independent};
                    const auto wall = sample.boundary.TryFaceValue(evaluator, original, wallSpecies,
                                                                   {sample.normal[0], sample.normal[1], sample.normal[2]},
                                                                   {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, request.time,
                                                                   OptionalRef<const RigidFrame>(frame));
                    if (!wall)
                        return wall;
                    for (int j = 0; j < 5; ++j)
                        face.values[request.flowFirst + j] = wall.value[j];
                    if (ransModel)
                        for (int j = 0; j < ransModel->ComponentNumber(); ++j)
                            face.values[request.ransFirst + j] *= wall.value[0] / original[0];
                    return wall;
                };
                if (wallFluxState)
                {
                    const auto wall = prepareWallState();
                    if (!wall)
                        return wall.status.At(iCell, sample.point);
                    leftFlow = rightFlow = wall.value;
                }
                Riemann::AcousticState leftData{}, rightData{};
                const auto evaluateDirect = [&](const FlowState &flow, const PointScratch &point,
                                                Riemann::AcousticState &output) -> NumericalStatus
                {
                    PointProperties properties;
                    const auto status = evaluator.TryEvaluate(flow, point.Species(request.speciesFirst, independent), 0, {}, properties);
                    if (!status)
                        return status.At(iCell, sample.point);
                    output = PointEvaluator::FluxState(flow, properties);
                    return {};
                };
                auto status = wallFluxState ? evaluateDirect(leftFlow, face, leftData)
                                            : evaluateReconstructedPoint(iCell, left, leftFlow, sample, false, leftData);
                if (!status)
                    return status;
                status = wallFluxState ? evaluateDirect(rightFlow, face, rightData)
                         : internal    ? evaluateReconstructedPoint(sample.other, right, rightFlow, sample, true, rightData)
                                       : evaluateDirect(rightFlow, right, rightData);
                if (!status)
                    return status;
                const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                const auto grid = frame.GridVelocity({sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]});
                auto meanLeftData = leftData, meanRightData = rightData;
                if (fluxRequirements.meanStates || request.produceLow)
                {
                    // Mean eigenvalue preparation follows the same path as
                    // flow5. Storage/block traversal stays in this adapter.
                    Gather(request.meanBlocks, iCell, cellMean);
                    const auto meanLeft = cellMean.Flow(request.flowFirst);
                    auto meanRight = meanLeft;
                    if (internal)
                    {
                        Gather(request.meanBlocks, sample.other, cellMean);
                        meanRight = sample.changesFrame ? FiniteVolumeDetail::RotateState(cellMean.Flow(request.flowFirst), sample.rotation) : cellMean.Flow(request.flowFirst);
                    }
                    else
                    {
                        // The shared boundary dispatch also handles interface
                        // traces. Pass its real input instead of duplicating
                        // the kind test around an empty interface view.
                        const auto exterior = FiniteVolumeDetail::BoundaryExterior(
                            sample, equation.Gas(), meanLeft, request.time, frame, request.incoming);
                        if (!exterior)
                            return exterior.status.At(iCell, sample.point);
                        meanRight = exterior.value;
                    }
                    meanLeftData = request.preparedMean
                                       ? Riemann::AcousticState{{meanLeft, (*request.preparedMean)(iCell, 0)}, (*request.preparedMean)(iCell, 1)}
                                       : equation.PrepareFluxState(meanLeft);
                    meanRightData = request.preparedMean && internal
                                        ? Riemann::AcousticState{{meanRight, (*request.preparedMean)(sample.other, 0)}, (*request.preparedMean)(sample.other, 1)}
                                        : equation.PrepareFluxState(meanRight);
                }
                auto fluxSettings = baseFluxSettings;
                if (useWaveJump)
                {
                    real jump = request.waveJump(iCell, 0);
                    if (internal)
                        jump = std::max(jump, request.waveJump(sample.other, 0));
                    fluxSettings.dLambda = std::max(fluxSettings.dLambda, jump);
                }
                if (usePressureMinimum)
                {
                    real minimum = request.pressureMinimum(iCell, 0);
                    if (internal)
                        minimum = std::min(minimum, request.pressureMinimum(sample.other, 0));
                    fluxSettings.pressureStencilMin = std::min(fluxSettings.pressureStencilMin, minimum);
                }
                const auto preparedFlux = equation.PrepareRiemann(fluxOperator, leftData, rightData, meanLeftData, meanRightData, normal, grid);
                const auto convection = Riemann::Math::Evaluate(fluxOperator.Kind(), preparedFlux.Input(leftData, rightData, normal, grid), fluxSettings);
                const auto &flowFlux = convection.flux;
                FlowState lowFlowFlux{};
                if (request.produceLow)
                    lowFlowFlux = Riemann::Rusanov(wallFluxState ? leftData : meanLeftData,
                                                   wallFluxState ? rightData : meanRightData, normal, grid);
                real speedLeft = 0, speedRight = 0;
                for (int d = 0; d < 3; ++d)
                {
                    speedLeft += (leftFlow[1 + d] / leftFlow[0] - grid[d]) * normal[d];
                    speedRight += (rightFlow[1 + d] / rightFlow[0] - grid[d]) * normal[d];
                }
                // The scalar kernel receives a fixed speed, not the flow5
                // acoustic flux. Roe's prepared contact magnitude is the
                // Euler velocity/entropy-fix policy. Rusanov's diagnostic
                // contact is acoustic, so use an explicit velocity speed.
                const real alpha = rusanovFlux
                                       ? .5 * (std::abs(speedLeft) + std::abs(speedRight))
                                       : convection.contact;
                for (index j = 0; j < width; ++j)
                    if (j < request.flowFirst || j >= request.flowFirst + 5)
                        flux[j] = wallFluxState ? face.values[j] * speedLeft : AdvectedScalarFlux(left.values[j], right.values[j], speedLeft, speedRight, alpha);
                for (int j = 0; j < 5; ++j)
                {
                    flux[request.flowFirst + j] = flowFlux[j];
                    if (request.produceLow)
                        lowFlux[request.flowFirst + j] = lowFlowFlux[j];
                }
                // Cauchy omits the former isotropic 2*rho*k/3 momentum flux and energy work.
                const bool wall = !internal && boundaryKind == BoundaryKind::NoSlipWall;
                if (viscous && (internal || wall))
                {
                    if (request.primitiveGradient)
                    {
                        DNDS_assert_info(!ransModel && independent == 0 && request.flowFirst == 0 && width == 5,
                                         "primitive-gradient leaf requires pure flow5");
                        const auto &gas = equation.Gas();
                        auto primitive = gas.Primitive(leftFlow);
                        FlowGradient gradient{};
                        for (int j = 0; j < 5; ++j)
                            for (int d = 0; d < 3; ++d)
                                gradient[j][d] = left.gradient(request.flowFirst + j, d);
                        Gather(request.meanBlocks, iCell, cellMean);
                        const auto ownMean = cellMean.Flow(request.flowFirst);
                        const auto ownPrimitive = gas.Primitive(ownMean);
                        FlowState rightMean;
                        if (internal)
                        {
                            Gather(request.meanBlocks, sample.other, cellMean);
                            rightMean = sample.changesFrame
                                            ? FiniteVolumeDetail::RotateState(cellMean.Flow(request.flowFirst), sample.rotation)
                                            : cellMean.Flow(request.flowFirst);
                        }
                        else
                        {
                            const auto trace = FiniteVolumeDetail::BoundaryGradientValue(sample, gas, ownMean, request.time, frame, request.incoming);
                            if (!trace)
                                return trace.status.At(iCell, sample.point);
                            rightMean = trace.value;
                        }
                        const auto rightPrimitive = gas.Primitive(rightMean);
                        const Geom::tPoint distance = !internal
                                                          ? Geom::tPoint(sample.ownPoint - metric.GetCellBary(iCell))
                                                          : Geom::tPoint(sample.otherPoint - metric.GetCellBary(iCell));
                        const real lengthSquared = distance.squaredNorm();
                        DNDS_assert_info(lengthSquared > 0, "zero primitive viscous stencil displacement");
                        const std::array<real, 5> own{ownPrimitive.density, ownPrimitive.velocity[0], ownPrimitive.velocity[1],
                                                      ownPrimitive.velocity[2], ownPrimitive.temperature};
                        const std::array<real, 5> neighbor{rightPrimitive.density, rightPrimitive.velocity[0], rightPrimitive.velocity[1],
                                                           rightPrimitive.velocity[2], rightPrimitive.temperature};
                        for (int j = 0; j < 5; ++j)
                        {
                            real projected = 0;
                            for (int d = 0; d < 3; ++d)
                            {
                                if (internal)
                                    gradient[j][d] = .5 * (gradient[j][d] + right.gradient(request.flowFirst + j, d));
                                projected += gradient[j][d] * distance[d];
                            }
                            for (int d = 0; d < 3; ++d)
                                gradient[j][d] += (neighbor[j] - own[j] - projected) * distance[d] / lengthSquared;
                        }
                        if (internal)
                        {
                            const auto other = gas.Primitive(rightFlow);
                            for (int d = 0; d < 3; ++d)
                                primitive.velocity[d] = .5 * (primitive.velocity[d] + other.velocity[d]);
                        }
                        else
                        {
                            FlowState faceMean{};
                            for (int j = 0; j < 5; ++j)
                                faceMean[j] = .5 * (leftFlow[j] + rightFlow[j]);
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
                        const auto diffusion = gas.ViscousFlux(primitive, gradient, normal);
                        for (int j = 0; j < 5; ++j)
                        {
                            flux[j] -= diffusion[j];
                            if (request.produceLow)
                                lowFlux[j] -= diffusion[j];
                        }
                    }
                    else
                    {
                        const real inverseDistance = wall && noGRPOnWall ? 0 : 1 / sample.viscousDistance;
                        for (index j = 0; j < width; ++j)
                        {
                            face.values[j] = .5 * (left.values[j] + right.values[j]);
                            for (int d = 0; d < 3; ++d)
                                face.gradient(j, d) = .5 * (left.gradient(j, d) + right.gradient(j, d)) +
                                                      normal[d] * (right.values[j] - left.values[j]) * inverseDistance;
                        }
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                        if (wallSAViscousInterior != 0)
                            for (int d = 0; d < 3; ++d)
                                face.gradient(request.ransFirst, d) -= normal[d] * wallSAViscousInterior * inverseDistance;
#endif
                        if (wallFluxState)
                        {
                            const auto wallState = prepareWallState();
                            if (!wallState)
                                return wallState.status.At(iCell, sample.point);
                        }
                        const auto faceFlow = face.Flow(request.flowFirst);
                        PointProperties point;
                        const auto propertyStatus = evaluator.TryEvaluate(faceFlow, face.Species(request.speciesFirst, independent),
                                                                          0, {true, true, true, false, false}, point);
                        if (!propertyStatus)
                            return propertyStatus.At(iCell, sample.point);
                        struct SpeciesGradient
                        {
                            const PointScratch &point;
                            index first;
                            real operator()(int k, int d) const { return point.gradient(first + k, d); }
                        };
                        ViscousFluxOptions transportOptions{wall && !sample.boundary.isothermal, wall};
                        if (request.ransFirst >= 0)
                        {
                            const auto &model = equation.RANS();
                            const index iQ = sample.point - metric.FaceQuadratureBegin(sample.face);
                            // Transport does not consume the DES source length.
                            const auto ransPoint = RANSPoint(face, request.flowFirst, request.ransFirst, model, point, wallValues.faces(sample.face, iQ), 0);
                            const auto turbulent = model.TryTransport(ransPoint);
                            if (!turbulent)
                                return turbulent.status.At(iCell, sample.point);
                            transportOptions.turbulentViscosity = turbulent.value.viscosity;
                            transportOptions.turbulentConductivity = turbulent.value.conductivity;
                            transportOptions.turbulentSpeciesDiffusivity = turbulent.value.speciesDiffusivity;
                            transportOptions.useQCR = model.Settings().useQCR;
                            const auto diffusionRANS = model.TryDiffusionFlux(ransPoint, Eigen::Vector3d{normal[0], normal[1], normal[2]}, turbulent.value.viscosity);
                            if (!diffusionRANS)
                                return diffusionRANS.status.At(iCell, sample.point);
                            for (int j = 0; j < model.ComponentNumber(); ++j)
                                flux[request.ransFirst + j] -= diffusionRANS.value[j];
                        }
                        const auto diffusion = PrepareViscousFlux(faceFlow, face.FlowDerivative(request.flowFirst), point, face.Species(request.speciesFirst, independent),
                                                                  SpeciesGradient{face, request.speciesFirst}, normal, speciesFlux, transport, transportOptions);
                        for (int j = 0; j < 5; ++j)
                        {
                            flux[request.flowFirst + j] -= diffusion[j];
                            if (request.produceLow)
                                lowFlux[request.flowFirst + j] -= diffusion[j];
                        }
                        for (int k = 0; k < independent; ++k)
                            flux[request.speciesFirst + k] -= speciesFlux[k];
                    }
                }
                if (sample.boundaryPoint != UnInitIndex)
                    for (int d = 0; d < 3; ++d)
                        request.wallFlux(sample.boundaryPoint, d) = request.captureBoundaryFlux || wall || boundaryKind == BoundaryKind::SlipWall
                                                                        ? weight * flux[request.flowFirst + 1 + d]
                                                                        : 0;
                for (index j = 0; j < width; ++j)
                {
                    integrated[j] -= weight * flux[j];
                    if (request.produceLow)
                        integratedLow[j] -= weight * lowFlux[j];
                }
            }
            for (index j = 0; j < width; ++j)
            {
                faceValues(iAux, j) = integrated[j];
                if (request.produceLow)
                    lowFaces(iAux, j) = integratedLow[j];
            }
            return {};
        };
        if (facePhase)
        {
            const auto rows = region == CFV::ReconstructionRegion::Core ? request.stencil.rhsCoreFaces : request.stencil.rhsDependentFaces;
            for (index iRow = firstRow; iRow < lastRow; ++iRow)
            {
                const auto status = computeFace(region == CFV::ReconstructionRegion::All ? iRow : rows(iRow, 0));
                if (!status)
                    return status;
            }
            if (!allowPointFallback && localFallbacks)
                return NumericalStatus{NumericalError::InvalidState};
            return localFallbacks;
        }
        const auto compute = [&](index iCell)
        {
            std::fill(integrated.begin(), integrated.end(), 0);
            if (request.produceLow)
                std::fill(integratedLow.begin(), integratedLow.end(), 0);
            const auto faces = request.stencil.topology.CellFaces(iCell);
            for (rowsize slot = 0; slot < faces.size(); ++slot)
            {
                const auto sample = request.stencil.GradientSample(iCell, slot);
                const index iAux = sample.auxiliaryFace;
                const bool opposite = sample.side != 0;
                const Eigen::Map<const Geom::tGPoint> rotation(sample.rotation.data());
                for (index j = 0; j < width; ++j)
                {
                    flux[j] = (opposite ? -1 : 1) * faceValues(iAux, j);
                    if (request.produceLow)
                        lowFlux[j] = (opposite ? -1 : 1) * lowFaces(iAux, j);
                }
                // Scalar fluxes change sign; vector fluxes additionally
                // rotate from the other parent's periodic component frame.
                // Side, not cell identity, distinguishes self-periodic faces.
                if (opposite && sample.changesFrame)
                    for (const index first : request.vectorStarts)
                    {
                        const Geom::tPoint value{flux[first], flux[first + 1], flux[first + 2]};
                        const Geom::tPoint rotated = rotation * value;
                        for (int d = 0; d < 3; ++d)
                            flux[first + d] = rotated[d];
                        if (request.produceLow)
                        {
                            const Geom::tPoint lowValue{lowFlux[first], lowFlux[first + 1], lowFlux[first + 2]};
                            const Geom::tPoint lowRotated = rotation * lowValue;
                            for (int d = 0; d < 3; ++d)
                                lowFlux[first + d] = lowRotated[d];
                        }
                    }
                for (index j = 0; j < width; ++j)
                {
                    integrated[j] += flux[j];
                    if (request.produceLow)
                        integratedLow[j] += lowFlux[j];
                }
            }
            for (index j = 0; j < destination.ComponentNumber(); ++j)
            {
                destination(iCell, j) = integrated[j];
                if (request.produceLow)
                    lowDestination(iCell, j) = integratedLow[j];
            }
        };
        const auto rows = region == CFV::ReconstructionRegion::Core ? request.stencil.coreCells : request.stencil.dependentCells;
        for (index iRow = firstRow; iRow < lastRow; ++iRow)
            compute(region == CFV::ReconstructionRegion::All ? iRow : rows(iRow, 0));
        return index(0);
    }
}
