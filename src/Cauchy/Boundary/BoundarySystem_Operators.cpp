#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/Input/FlowProfile.hpp"

namespace DNDS::Cauchy
{
    void BoundaryCondition::ValidateEquationProducer() const
    {
        DNDS_check_throw_info(kind == BoundaryKind::Extrapolate || kind == BoundaryKind::Farfield || kind == BoundaryKind::TotalConditionInlet ||
                                  kind == BoundaryKind::SlipWall || kind == BoundaryKind::NoSlipWall ||
                                  kind == BoundaryKind::Interface || kind == BoundaryKind::VelocityInlet || kind == BoundaryKind::PressureOutlet ||
                                  (kind == BoundaryKind::DoubleMachBottom && hasPerfectGasState) ||
                                  (kind == BoundaryKind::Analytic &&
                                   (profile.kind == "constant" || hasPerfectGasState)),
                              "requested equation-state boundary producer is not connected");
    }
    namespace
    {
        NumericalResult<FlowState> EquationValue(const BoundaryCondition &condition, PointEvaluator &evaluator,
                                                 const FlowState &inside, Physics::Chemistry::SpeciesBufferView rhoY,
                                                 const Vector3 &normal, const Vector3 &point, real time,
                                                 OptionalRef<const RigidFrame> frame, bool exterior) noexcept
        {
            const auto kind = condition.KindAt(point, frame);
            if (kind == BoundaryKind::Extrapolate)
                return inside;
            if (kind == BoundaryKind::TotalConditionInlet)
            {
                if (condition.prescribedFractions.size() != size_t(rhoY.nSpecies) || !(inside[0] > 0))
                    return NumericalStatus{NumericalError::InvalidState, 4, -1};
                const auto &total = condition.totalCondition;
                Vector3 direction{total[2], total[3], total[4]};
                if (frame)
                    direction = frame->LocalVector(direction);
                real speedSquared = 0, inward = 0;
                for (int d = 0; d < 3; ++d)
                {
                    speedSquared += sqr(inside[1 + d] / inside[0]);
                    inward += direction[d] * normal[d];
                }
                if (!(inward < -1e-14))
                    return NumericalStatus{NumericalError::InvalidState, 5, -1};
                const auto result = evaluator.TryTotalToStatic(total[0], total[1], speedSquared,
                                                               {condition.prescribedFractions.data(), int(condition.prescribedFractions.size())});
                if (!result)
                    return result.status;
                const auto &state = result.value;
                const real speed = std::sqrt(state.speedSquared);
                for (int k = 0; k < rhoY.nSpecies; ++k)
                    rhoY[k] = state.density * condition.prescribedFractions[k];
                return evaluator.TryFromPrimitive({state.density, direction[0] * speed,
                                                   direction[1] * speed, direction[2] * speed, state.pressure},
                                                  rhoY, state.temperature);
            }
            if (kind == BoundaryKind::Analytic && condition.profile.kind != "constant")
            {
                DNDS_assert_info(condition.hasPerfectGasState && !rhoY.nSpecies,
                                 "nonconstant analytic boundary requires a perfect-gas flow5 profile");
                const auto value = TryProfileValue(condition.profile, condition.prescribedPrimitive,
                                                   evaluator.PerfectGasModel(),
                                                   frame ? frame.Get()->WorldPoint(point) : point, time);
                if (!value)
                    return value.status;
                return frame ? frame.Get()->LocalState(value.value) : value.value;
            }
            if (kind == BoundaryKind::Farfield || kind == BoundaryKind::Analytic)
            {
                if (!(condition.prescribedFractions.size() == size_t(rhoY.nSpecies)))
                    return NumericalStatus{NumericalError::InvalidState, 1, -1};
                auto primitive = condition.prescribedPrimitive;
                if (frame)
                {
                    const auto velocity = frame.Get()->LocalVector({primitive[1], primitive[2], primitive[3]});
                    for (int d = 0; d < 3; ++d)
                        primitive[1 + d] = velocity[d];
                }
                if (kind == BoundaryKind::Farfield)
                {
                    PointProperties properties;
                    if (const auto status = evaluator.TryEvaluate(inside, rhoY, 0, {}, properties); !status)
                        return status;
                    const auto grid = frame ? frame.Get()->GridVelocity(point) : Vector3{};
                    real speed = 0;
                    for (int d = 0; d < 3; ++d)
                        speed += (inside[1 + d] / inside[0] - grid[d]) * normal[d];
                    if (speed > properties.thermo.soundSpeed)
                        return inside;
                    if (speed > 0)
                    {
                        // Subsonic outflow preserves rho, velocity and raw
                        // composition; only pressure is prescribed.
                        primitive[0] = inside[0];
                        for (int d = 0; d < 3; ++d)
                            primitive[1 + d] = inside[1 + d] / inside[0];
                        return evaluator.TryFromPrimitive(primitive, rhoY);
                    }
                    if (speed > -properties.thermo.soundSpeed)
                        primitive[4] = properties.thermo.pressure;
                }
                // Supersonic inflow or a prescribed value replaces all data;
                // subsonic inflow retains the interior pressure above.
                for (int k = 0; k < rhoY.nSpecies; ++k)
                    rhoY[k] = primitive[0] * condition.prescribedFractions[k];
                return evaluator.TryFromPrimitive(primitive, rhoY);
            }
            if (!(kind == BoundaryKind::SlipWall || kind == BoundaryKind::NoSlipWall))
                return NumericalStatus{NumericalError::InvalidState, 2, -1};
            const auto wall = condition.WallVelocityAt(point, frame);
            const real rho = inside[0], factor = exterior ? 2 : 1;
            if (!(rho > 0))
                return NumericalStatus{NumericalError::InvalidState, 3, -1};
            Vector3 velocity{inside[1] / rho, inside[2] / rho, inside[3] / rho}, target = velocity;
            if (kind == BoundaryKind::SlipWall)
            {
                real speed = 0;
                for (int d = 0; d < 3; ++d)
                    speed += (velocity[d] - wall[d]) * normal[d];
                for (int d = 0; d < 3; ++d)
                    target[d] -= factor * speed * normal[d];
            }
            else
                for (int d = 0; d < 3; ++d)
                    target[d] += factor * (wall[d] - velocity[d]);
            if (kind == BoundaryKind::NoSlipWall && condition.isothermal)
            {
                PointProperties properties;
                if (const auto status = evaluator.TryEvaluate(inside, rhoY, 0, {}, properties); !status)
                    return status;
                const real density = properties.thermo.pressure / (properties.thermo.gasConstant * condition.temperature);
                for (int k = 0; k < rhoY.nSpecies; ++k)
                    rhoY[k] *= density / rho;
                return evaluator.TryFromPrimitive({density, target[0], target[1], target[2], properties.thermo.pressure}, rhoY, condition.temperature);
            }
            // Adiabatic reflection retains internal energy and composition.
            auto result = inside;
            for (int d = 0; d < 3; ++d)
            {
                result[1 + d] = rho * target[d];
                result[4] += .5 * rho * (target[d] * target[d] - velocity[d] * velocity[d]);
            }
            return result;
        }
    }
    FlowState BoundaryCondition::FaceValue(PointEvaluator &evaluator, const FlowState &inside,
                                           Physics::Chemistry::SpeciesBufferView rhoY, const Vector3 &normal,
                                           const Vector3 &point, real time,
                                           OptionalRef<const RigidFrame> frame) const
    {
        const auto result = TryFaceValue(evaluator, inside, rhoY, normal, point, time, frame);
        DNDS_check_throw_info(result.status, "boundary point evaluation failed");
        return result.value;
    }
    NumericalResult<FlowState> BoundaryCondition::TryFaceValue(PointEvaluator &evaluator, const FlowState &inside,
                                                               Physics::Chemistry::SpeciesBufferView rhoY, const Vector3 &normal,
                                                               const Vector3 &point, real time,
                                                               OptionalRef<const RigidFrame> frame) const noexcept
    {
        return EquationValue(*this, evaluator, inside, rhoY, normal, point, time, frame, false);
    }

    FlowState BoundaryCondition::Exterior(PointEvaluator &evaluator, const FlowState &inside,
                                          Physics::Chemistry::SpeciesBufferView rhoY, const Vector3 &normal,
                                          const Vector3 &point, real time,
                                          OptionalRef<const RigidFrame> frame) const
    {
        const auto result = TryExterior(evaluator, inside, rhoY, normal, point, time, frame);
        DNDS_check_throw_info(result.status, "boundary point evaluation failed");
        return result.value;
    }
    NumericalResult<FlowState> BoundaryCondition::TryExterior(PointEvaluator &evaluator, const FlowState &inside,
                                                              Physics::Chemistry::SpeciesBufferView rhoY, const Vector3 &normal,
                                                              const Vector3 &point, real time,
                                                              OptionalRef<const RigidFrame> frame) const noexcept
    {
        return EquationValue(*this, evaluator, inside, rhoY, normal, point, time, frame, true);
    }

}
