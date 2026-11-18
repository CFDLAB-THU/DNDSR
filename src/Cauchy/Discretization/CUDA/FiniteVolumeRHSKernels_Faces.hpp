#pragma once

#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Reconstruction.hpp"
#include "Cauchy/Physics/RiemannPreparation.hpp"
#include "Cauchy/Physics/PhysicalFlux.hpp"
#include "Cauchy/Riemann/RiemannFluxMath.hpp"
#include "Cauchy/Riemann/RiemannCompute.hpp"
#include "Cauchy/Physics/PerfectGasCompute.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    template <int Components, class Fields>
    DNDS_DEVICE_CALLABLE NumericalStatus WallPoint(const RHSGeometry &geometry, const RHSKernelParameters &parameters,
                                                   const RHSKernelControls &controls,
                                                   const Fields &fields, const FiniteVolumeDetail::SampleRecord &sample,
                                                   const PerfectGasKernelPoint<Components> &left, PerfectGasKernelPoint<Components> &face) noexcept
    {
        const auto wall = BoundaryFlow(geometry, parameters, controls, fields, sample, left.Flow(), false);
        if (!wall)
            return wall.status;
        face.values = left.values;
        for (int j = 0; j < 5; ++j)
            face.values[j] = wall.value[j];
        for (int j = 5; j < Components; ++j)
            face.values[j] *= wall.value[0] / left.values[0];
        return {};
    }
    template <int Components, class Fields>
    DNDS_DEVICE_CALLABLE NumericalResult<FlowState> PrimitiveDiffusion(const RHSGeometry &geometry, const RHSKernelParameters &parameters,
                                                                       const RHSKernelControls &controls,
                                                                       const Fields &fields, const FiniteVolumeDetail::SampleRecord &sample,
                                                                       const PerfectGasKernelPoint<Components> &left,
                                                                       const PerfectGasKernelPoint<Components> &right) noexcept
    {
        const bool internal = sample.other != UnInitIndex;
        const auto &gas = parameters.physics.gas;
        auto primitive = gas.Primitive(left.Flow());
        PerfectGasKernelPoint<Components> mean;
        MeanPoint(fields, sample.cell, mean);
        const auto own = gas.Primitive(mean.Flow());
        FlowState otherFlow;
        if (internal)
        {
            MeanPoint(fields, sample.other, mean);
            if (sample.changesFrame)
                mean.Rotate(sample.rotation);
            otherFlow = mean.Flow();
        }
        else
        {
            const auto boundary = BoundaryFlow(geometry, parameters, controls, fields, sample, mean.Flow(), true);
            if (!boundary)
                return boundary.status;
            otherFlow = boundary.value;
            if (geometry.conditions(sample.condition, 0).kind != BoundaryKind::Interface)
                for (int j = 0; j < 5; ++j)
                    otherFlow[j] = .5 * (otherFlow[j] + mean.values[j]);
        }
        const auto other = gas.Primitive(otherFlow);
        const FlowState a{own.density, own.velocity[0], own.velocity[1], own.velocity[2], own.temperature},
            b{other.density, other.velocity[0], other.velocity[1], other.velocity[2], other.temperature};
        Vector3 distance{};
        real lengthSquared = 0;
        for (int d = 0; d < 3; ++d)
        {
            distance[d] = (internal ? sample.otherPoint[d] : sample.ownPoint[d]) - geometry.cells(sample.cell, 4 + d);
            lengthSquared += sqr(distance[d]);
        }
        if (!(lengthSquared > 0))
            return NumericalStatus{NumericalError::InvalidState, 14};
        FlowGradient gradient;
        for (int j = 0; j < 5; ++j)
        {
            real projected = 0;
            for (int d = 0; d < 3; ++d)
            {
                gradient[j][d] = internal ? .5 * (left.gradients[j][d] + right.gradients[j][d]) : left.gradients[j][d];
                projected += gradient[j][d] * distance[d];
            }
            for (int d = 0; d < 3; ++d)
                gradient[j][d] += (b[j] - a[j] - projected) * distance[d] / lengthSquared;
        }
        if (internal)
        {
            const auto otherPrimitive = gas.Primitive(right.Flow());
            for (int d = 0; d < 3; ++d)
                primitive.velocity[d] = .5 * (primitive.velocity[d] + otherPrimitive.velocity[d]);
        }
        else
        {
            FlowState average;
            for (int j = 0; j < 5; ++j)
                average[j] = .5 * (left.values[j] + right.values[j]);
            primitive = gas.Primitive(average);
            if (!geometry.conditions(sample.condition, 0).isothermal)
            {
                real heat = 0;
                for (int d = 0; d < 3; ++d)
                    heat += gradient[4][d] * sample.normal[d];
                for (int d = 0; d < 3; ++d)
                    gradient[4][d] -= heat * sample.normal[d];
            }
        }
        return gas.ViscousFlux(primitive, gradient, sample.normal);
    }

    /** @brief One weighted quadrature contribution shared by every mapping.
     * @details The caller supplies local accumulators. No global point-flux
     * intermediate or ownership action is hidden here. Keeping this numerical
     * leaf common prevents mapping experiments from changing flux/PP/BC policy.
     */
    template <int Components, class PolynomialScalar = real, class FluxScalar = real,
              class DissipationScalar = real, class Fields, class Outputs>
    DNDS_DEVICE_CALLABLE NumericalStatus EvaluateFacePoint(
        const RHSGeometry &geometry, const RHSKernelParameters &parameters,
        const RHSKernelControls &controls, const Fields &fields,
        const Outputs &output, index iFace, index iPoint,
        std::array<real, Components> &integrated, std::array<real, Components> &integratedLow,
        index &fallbacks) noexcept
    {
        const auto &gas = parameters.physics.gas;
        const auto &sample = geometry.FaceSample(iFace, iPoint);
        const auto &condition = geometry.conditions(sample.condition, 0);
        const bool internal = sample.other != UnInitIndex;
        const bool wall = !internal && condition.kind == BoundaryKind::NoSlipWall;
        const bool wallFluxState = wall && controls.fluxPolicy.noRsOnWall;
        PerfectGasKernelPoint<Components> left, right, face;
        ReconstructPoint<Components, PolynomialScalar>(geometry, parameters, controls, fields, sample, false, left, fallbacks);
        if (internal)
            ReconstructPoint<Components, PolynomialScalar>(geometry, parameters, controls, fields, sample, true, right, fallbacks);
        else
        {
            right = left;
            const auto status = ExteriorPoint(geometry, parameters, controls, fields, sample, right);
            if (!status)
                return status.At(sample.cell, sample.point);
        }
        real wallSAViscousInterior = 0;
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
        if constexpr (Components == 6)
            if (wall)
            {
                wallSAViscousInterior = left.values[5];
                left.values[5] = right.values[5] = 0;
            }
#endif
        if (wallFluxState)
        {
            const auto status = WallPoint(geometry, parameters, controls, fields, sample, left, face);
            if (!status)
                return status.At(sample.cell, sample.point);
        }
        auto leftData = wallFluxState ? Acoustic(gas, face.Flow()) : PointAcoustic(parameters, controls, fields, sample, false, left, fallbacks);
        if (!leftData)
            return leftData.status.At(sample.cell, sample.point);
        auto rightData = wallFluxState ? Acoustic(gas, face.Flow()) : internal ? PointAcoustic(parameters, controls, fields, sample, true, right, fallbacks)
                                                                               : Acoustic(gas, right.Flow());
        if (!rightData)
            return rightData.status.At(sample.cell, sample.point);
        const auto &leftFlow = leftData.value.flow.conservative, &rightFlow = rightData.value.flow.conservative;
        auto meanLeftData = leftData.value, meanRightData = rightData.value;
        if (Riemann::RequiredInputs(controls.solver).meanStates || controls.produceLow)
        {
            PerfectGasKernelPoint<Components> mean;
            MeanPoint(fields, sample.cell, mean);
            const auto leftMean = mean.Flow();
            FlowState rightMean;
            if (internal)
            {
                MeanPoint(fields, sample.other, mean);
                if (sample.changesFrame)
                    mean.Rotate(sample.rotation);
                rightMean = mean.Flow();
            }
            else
            {
                const auto result = BoundaryFlow(geometry, parameters, controls, fields, sample, leftMean, true);
                if (!result)
                    return result.status.At(sample.cell, sample.point);
                rightMean = result.value;
            }
            if (fields.hasPreparedMean)
                meanLeftData = {{leftMean, fields.preparedMean(sample.cell, 0)}, fields.preparedMean(sample.cell, 1)};
            else
            {
                const auto result = Acoustic(gas, leftMean);
                if (!result)
                    return result.status.At(sample.cell, sample.point);
                meanLeftData = result.value;
            }
            if (fields.hasPreparedMean && internal)
                meanRightData = {{rightMean, fields.preparedMean(sample.other, 0)}, fields.preparedMean(sample.other, 1)};
            else
            {
                const auto result = Acoustic(gas, rightMean);
                if (!result)
                    return result.status.At(sample.cell, sample.point);
                meanRightData = result.value;
            }
        }
        const auto grid = controls.frame.GridVelocity(sample.ownPoint);
        auto settings = controls.fluxPolicy.settings;
        if (controls.useStencil && controls.fluxPolicy.transverseSignal)
        {
            real jump = fields.waveJump(sample.cell, 0);
            if (internal)
                jump = std::max(jump, fields.waveJump(sample.other, 0));
            settings.dLambda = std::max(settings.dLambda, jump);
        }
        if (controls.useStencil && controls.fluxPolicy.pressureStencil)
        {
            real minimum = fields.pressureMinimum(sample.cell, 0);
            if (internal)
                minimum = std::min(minimum, fields.pressureMinimum(sample.other, 0));
            settings.pressureStencilMin = std::min(settings.pressureStencilMin, minimum);
        }
        Riemann::FluxResult convection;
        if constexpr (std::is_same_v<FluxScalar, real>)
        {
            const auto prepared = PreparePerfectGasRiemann(gas, controls.solver, leftData.value, rightData.value, meanLeftData, meanRightData, sample.normal, grid);
            convection = Riemann::EvaluateRoeDissipationCompute<DissipationScalar>(controls.solver, prepared.Input(leftData.value, rightData.value, sample.normal, grid), settings);
        }
        else
        {
            const auto result = TryPerfectGasRiemannCompute<FluxScalar>(gas, controls.solver, leftFlow, rightFlow,
                                                                        meanLeftData.flow.conservative, meanRightData.flow.conservative, sample.normal, grid, settings);
            if (!result)
                return result.status.At(sample.cell, sample.point);
            convection = result.value;
        }
        std::array<real, Components> flux{}, low{};
        for (int j = 0; j < 5; ++j)
            flux[j] = convection.flux[j];
        if (controls.produceLow)
        {
            const auto value = Riemann::Rusanov(wallFluxState ? leftData.value : meanLeftData, wallFluxState ? rightData.value : meanRightData, sample.normal, grid);
            for (int j = 0; j < 5; ++j)
                low[j] = value[j];
        }
        real speedLeft = 0, speedRight = 0;
        for (int d = 0; d < 3; ++d)
        {
            speedLeft += (leftFlow[1 + d] / leftFlow[0] - grid[d]) * sample.normal[d];
            speedRight += (rightFlow[1 + d] / rightFlow[0] - grid[d]) * sample.normal[d];
        }
        const real alpha = controls.solver == Riemann::Solver::Rusanov ? .5 * (std::abs(speedLeft) + std::abs(speedRight)) : convection.contact;
        for (int j = 5; j < Components; ++j)
            flux[j] = wallFluxState ? face.values[j] * speedLeft : AdvectedScalarFlux(left.values[j], right.values[j], speedLeft, speedRight, alpha);
        // Cauchy omits the former isotropic 2*rho*k/3 momentum flux and energy work.
        if (controls.viscous && (internal || wall))
        {
            if (controls.primitiveGradient)
            {
                const auto diffusion = PrimitiveDiffusion(geometry, parameters, controls, fields, sample, left, right);
                if (!diffusion)
                    return diffusion.status.At(sample.cell, sample.point);
                for (int j = 0; j < 5; ++j)
                {
                    flux[j] -= diffusion.value[j];
                    if (controls.produceLow)
                        low[j] -= diffusion.value[j];
                }
            }
            else
            {
                const real inverseDistance = wall && controls.fluxPolicy.noGRPOnWall ? 0 : 1 / sample.viscousDistance;
                for (int j = 0; j < Components; ++j)
                {
                    face.values[j] = .5 * (left.values[j] + right.values[j]);
                    for (int d = 0; d < 3; ++d)
                        face.gradients[j][d] = .5 * (left.gradients[j][d] + right.gradients[j][d]) +
                                               sample.normal[d] * (right.values[j] - left.values[j]) * inverseDistance;
                }
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                if constexpr (Components == 6)
                    if (wallSAViscousInterior != 0)
                        for (int d = 0; d < 3; ++d)
                            face.gradients[5][d] -= sample.normal[d] * wallSAViscousInterior * inverseDistance;
#endif
                if (wallFluxState)
                {
                    const auto status = WallPoint(geometry, parameters, controls, fields, sample, left, face);
                    if (!status)
                        return status.At(sample.cell, sample.point);
                }
                const real distance = geometry.hasWall ? geometry.wallFaces(sample.face, sample.point - geometry.faceOffsets(sample.face, 0)) : 0;
                const auto diffusion = [&]() -> NumericalResult<std::array<real, Components>>
                {
                    if constexpr (std::is_same_v<FluxScalar, real>)
                        return face.Diffusion(parameters.physics, sample.normal, distance, wall && !condition.isothermal);
                    else
                    {
                        static_assert(Components == 5, "reduced full flux is a flow5 specialization");
                        FlowGradient leftGradient, rightGradient;
                        for (int j = 0; j < 5; ++j)
                        {
                            leftGradient[j] = left.gradients[j];
                            rightGradient[j] = right.gradients[j];
                        }
                        return TryPerfectGasPolynomialViscousCompute<FluxScalar>(gas, left.Flow(), right.Flow(), leftGradient, rightGradient,
                                                                                 sample.normal, inverseDistance, wall && !condition.isothermal, wallFluxState, face.Flow());
                    }
                }();
                if (!diffusion)
                    return diffusion.status.At(sample.cell, sample.point);
                for (int j = 0; j < Components; ++j)
                    flux[j] -= diffusion.value[j];
                if (controls.produceLow)
                    for (int j = 0; j < 5; ++j)
                        low[j] -= diffusion.value[j];
            }
        }
        // Experimental narrower arithmetic has a narrower exponent range
        // too. Report overflow explicitly; never publish a successful
        // face status for a nonfinite reduced-precision flux/gradient.
        if constexpr (!std::is_same_v<PolynomialScalar, real> || !std::is_same_v<FluxScalar, real> || !std::is_same_v<DissipationScalar, real>)
            for (int j = 0; j < Components; ++j)
                if (!std::isfinite(flux[j]) || (controls.produceLow && !std::isfinite(low[j])))
                    return NumericalStatus{NumericalError::InvalidState, 15}.At(sample.cell, sample.point);
        const real weight = geometry.quadrature(sample.point, 6);
        if constexpr (!std::is_same_v<PolynomialScalar, real> || !std::is_same_v<FluxScalar, real> || !std::is_same_v<DissipationScalar, real>)
            for (int j = 0; j < Components; ++j)
                if (!std::isfinite(weight * flux[j]) || (controls.produceLow && !std::isfinite(weight * low[j])))
                    return NumericalStatus{NumericalError::InvalidState, 15}.At(sample.cell, sample.point);
        if (output.wallForce.RowCount() && sample.boundaryPoint != UnInitIndex)
            for (int d = 0; d < 3; ++d)
                output.wallForce(sample.boundaryPoint, d) = controls.captureBoundaryFlux || wall || condition.kind == BoundaryKind::SlipWall ? weight * flux[1 + d] : 0;
        for (int j = 0; j < Components; ++j)
        {
            integrated[j] -= weight * flux[j];
            if (controls.produceLow)
                integratedLow[j] -= weight * low[j];
        }
        return {};
    }

    /** Perfect-gas non-species RHS arithmetic specialized to 5/6/7 bounded
     * scalars, with a face-owned flux and a separate cell gather. No atomics,
     * per-point allocation, EOS callbacks or exception recovery are used.
     * Residuals keep the CPU sign convention: -weight*normalFlux.
     */
    template <int Components, class PolynomialScalar = real, class FluxScalar = real, class DissipationScalar = real,
              class Fields, class Outputs>
    DNDS_DEVICE_CALLABLE NumericalStatus EvaluateFace(const RHSGeometry &geometry, const RHSKernelParameters &parameters,
                                                      const RHSKernelControls &controls, const Fields &fields,
                                                      const Outputs &output, index iFace, index &fallbacks) noexcept
    {
        const auto anchor = geometry.FaceAnchor(iFace);
        const index first = geometry.faceOffsets(anchor.face, 0);
        const index end = first + geometry.faceOffsets(anchor.face + 1, 0) - geometry.faceOffsets(anchor.face, 0);
        std::array<real, Components> integrated{}, integratedLow{};
        // Reduced-precision comparison leaves retain a serial point sum.
        // The production FP64 path uses the separate CTA-segment shell above
        // this common point arithmetic; no warp-mapping variant is retained.
        for (index entry = first; entry < end; ++entry)
        {
            const auto pointStatus = EvaluateFacePoint<Components, PolynomialScalar, FluxScalar, DissipationScalar>(
                geometry, parameters, controls, fields, output, iFace, entry, integrated, integratedLow, fallbacks);
            if (!pointStatus)
                return pointStatus;
        }
        if constexpr (!std::is_same_v<PolynomialScalar, real> || !std::is_same_v<FluxScalar, real> || !std::is_same_v<DissipationScalar, real>)
            for (int j = 0; j < Components; ++j)
                if (!std::isfinite(integrated[j]) || (controls.produceLow && !std::isfinite(integratedLow[j])))
                    return NumericalStatus{NumericalError::InvalidState, 15}.At(anchor.cell, anchor.point);
        for (int j = 0; j < Components; ++j)
        {
            output.high(iFace, j) = integrated[j];
            if (controls.produceLow)
                output.low(iFace, j) = integratedLow[j];
        }
        return {};
    }
}
