#pragma once

#include "Cauchy/Riemann/RiemannFlux.hpp"
#include <algorithm>

/** @file RiemannFluxMath.hpp
 * @brief Shared host/device gas-agnostic numerical flux arithmetic.
 * @details The host operator validates the selected scheme and supplies all
 * required EOS/mean/linearization inputs. Optional pointers borrow point-local
 * data in the execution address space; no host pointer is uploaded implicitly.
 * These functions perform no allocation, ownership, validation or fallback
 * through exceptions. Point admissibility belongs to the physical producer.
 */
namespace DNDS::Cauchy::Riemann::Math
{
    /// Preserve the FP64 guard exactly; a reduced-range type cannot represent
    /// 1e-200. A positive normal floor keeps zero-denominator protection active.
    /// This is a compute-range guard, not a change to state admissibility.
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Scalar DenominatorGuard() noexcept
    {
        return DNDS::Math::ScalarMax(Scalar(verySmallReal), DNDS::Math::ScalarMinPositive<Scalar>());
    }
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Scalar ScalarSignTol(Scalar value, Scalar tolerance) noexcept
    {
        return Scalar(value > tolerance ? 1 : (value < -tolerance ? -1 : 0));
    }
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Scalar ScalarSignP(Scalar value) noexcept
    {
        return Scalar(value >= 0 ? 1 : -1);
    }

    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Scalar Dot(const Vector3T<Scalar> &a, const Vector3T<Scalar> &b)
    {
        Scalar result = 0;
        for (int d = 0; d < 3; ++d)
            result += a[d] * b[d];
        return result;
    }
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Vector3T<Scalar> Velocity(const AcousticStateT<Scalar> &state)
    {
        Vector3T<Scalar> result;
        for (int d = 0; d < 3; ++d)
            result[d] = state.flow.conservative[1 + d] / state.flow.conservative[0];
        return result;
    }
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Vector3T<Scalar> Relative(Vector3T<Scalar> velocity, const Vector3T<Scalar> &grid)
    {
        for (int d = 0; d < 3; ++d)
            velocity[d] -= grid[d];
        return velocity;
    }
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Scalar NormalSpeed(const AcousticStateT<Scalar> &state, const FluxInputT<Scalar> &input) { return Dot(Relative(Velocity(state), input.gridVelocity), input.normal); }
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline const AcousticStateT<Scalar> &Mean(const FluxInputT<Scalar> &input, bool left) { return left ? (input.leftMean ? *input.leftMean : input.left) : (input.rightMean ? *input.rightMean : input.right); }
    DNDS_DEVICE_CALLABLE inline bool AUSM(Solver kind) { return kind >= Solver::AUSMPWPlus; }
    DNDS_DEVICE_CALLABLE inline bool Roe(Solver kind) { return kind >= Solver::Roe && kind <= Solver::Roe_M9; }
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline void EntropyFix(int scheme, Scalar aL, Scalar aR, Scalar a, Scalar uL, Scalar uR, Scalar u, Scalar velocityMagnitude, const FluxSettingsT<Scalar> &settings, FluxResultT<Scalar> &result)
    {
        const Scalar hartenYee = Scalar(.05), lowDissipation = Scalar(.2), hCorrection = Scalar(.25);
        const Scalar scaleHY = hartenYee * settings.fixScale, scaleLD = lowDissipation * settings.fixScale, scaleH = hCorrection * settings.fixScale;
        const Scalar scale = settings.incrementScale;
        auto set = [&](Scalar speed, Scalar acoustic)
        {
            result.acousticMinus = DNDS::Math::ScalarAbs(speed - acoustic);
            result.contact = DNDS::Math::ScalarAbs(speed);
            result.acousticPlus = DNDS::Math::ScalarAbs(speed + acoustic);
        };
        if (scheme == 0)
        {
            const auto limit = [&](Scalar value, Scalar lambda, Scalar left, Scalar right)
            {
                const Scalar threshold = DNDS::Math::ScalarMax(DNDS::Math::ScalarMax(Scalar(0), lambda - left), right - lambda) * scaleHY;
                return value < threshold ? Scalar(.5) * (value * value / threshold + threshold) : value;
            };
            result.acousticMinus = limit(result.acousticMinus, u - a, uL - aL, uR - aR);
            result.contact = limit(result.contact * scale, u, uL, uR);
            result.acousticPlus = limit(result.acousticPlus, u + a, uL + aL, uR + aR);
        }
        else if (scheme == 1)
        {
            const Scalar vL = ScalarSignTol(uL, aL * Scalar(smallReal)) * DNDS::Math::ScalarMax(DNDS::Math::ScalarAbs(uL) * scale, aL * scaleLD);
            const Scalar vR = ScalarSignTol(uR, aR * Scalar(smallReal)) * DNDS::Math::ScalarMax(DNDS::Math::ScalarAbs(uR) * scale, aR * scaleLD);
            result.acousticMinus = DNDS::Math::ScalarMax(DNDS::Math::ScalarAbs(vL - aL), DNDS::Math::ScalarAbs(vR - aR));
            result.contact = DNDS::Math::ScalarMax(DNDS::Math::ScalarAbs(vL), DNDS::Math::ScalarAbs(vR));
            result.acousticPlus = DNDS::Math::ScalarMax(DNDS::Math::ScalarAbs(vL + aL), DNDS::Math::ScalarAbs(vR + aR));
        }
        else if (scheme == 3)
        {
            const Scalar star = DNDS::Math::ScalarMin(DNDS::Math::ScalarAbs(u) / scaleLD, a);
            result.acousticMinus = DNDS::Math::ScalarAbs(u * scale - star);
            result.acousticPlus = DNDS::Math::ScalarAbs(u * scale + star);
        }
        else if (scheme == 4)
            set(ScalarSignTol(u, a * Scalar(smallReal)) * DNDS::Math::ScalarMax(a * scaleLD, DNDS::Math::ScalarAbs(u) * scale), a);
        else if (scheme == 5)
        {
            const Scalar vL = uL * scale, vR = uR * scale;
            const Scalar acousticL = DNDS::Math::ScalarMin(aL, DNDS::Math::ScalarAbs(vL) / scaleLD), acousticR = DNDS::Math::ScalarMin(aR, DNDS::Math::ScalarAbs(vR) / scaleLD);
            result.acousticMinus = DNDS::Math::ScalarMax(DNDS::Math::ScalarAbs(vL - acousticL), DNDS::Math::ScalarAbs(vR - acousticR));
            result.contact = DNDS::Math::ScalarMax(DNDS::Math::ScalarAbs(vL), DNDS::Math::ScalarAbs(vR));
            result.acousticPlus = DNDS::Math::ScalarMax(DNDS::Math::ScalarAbs(vL + acousticL), DNDS::Math::ScalarAbs(vR + acousticR));
        }
        else
        {
            const Scalar acoustic = scheme == 9 ? DNDS::Math::ScalarMax(aL, aR) : a;
            set(u * scale, acoustic);
            if (scheme != 7)
            {
                result.acousticMinus = DNDS::Math::ScalarMax(result.acousticMinus, settings.dLambda * scaleH);
                result.contact = DNDS::Math::ScalarMax(result.contact, settings.dLambda * scaleH);
                result.acousticPlus = DNDS::Math::ScalarMax(result.acousticPlus, settings.dLambda * scaleH);
            }
            if (scheme != 6)
            {
                const Scalar threshold = scaleHY * (velocityMagnitude + acoustic);
                auto smooth = [&](Scalar &value)
                {
                    if (value < threshold)
                        value = (value * value + threshold * threshold) / (2 * threshold);
                };
                smooth(result.acousticMinus);
                smooth(result.contact);
                smooth(result.acousticPlus);
            }
        }
    }
    template <class Scalar>
    struct DecompositionT
    {
        Scalar minus, contact, plus;
        Vector3T<Scalar> shear;
    };
    using Decomposition = DecompositionT<real>;
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline DecompositionT<Scalar> Decompose(const FluxInputT<Scalar> &input)
    {
        const auto &roe = *input.roe;
        FlowStateT<Scalar> jump;
        for (int j = 0; j < 5; ++j)
            jump[j] = input.right.flow.conservative[j] - input.left.flow.conservative[j];
        // Pressure-neutral mode data comes from the equation adapter. This
        // numerical decomposition never reconstructs thermodynamics.
        jump[4] -= roe.contactEnergyJump;
        DecompositionT<Scalar> result{};
        for (int j = 0; j < 5; ++j)
            result.contact += roe.contactProjection[j] * jump[j];
        Scalar momentumNormal = 0;
        for (int d = 0; d < 3; ++d)
        {
            momentumNormal += jump[1 + d] * input.normal[d];
            result.shear[d] = jump[1 + d] - jump[0] * roe.velocity[d];
        }
        const Scalar shearNormal = Dot(result.shear, input.normal), un = Dot(roe.velocity, input.normal);
        for (int d = 0; d < 3; ++d)
            result.shear[d] -= shearNormal * input.normal[d];
        result.minus = (jump[0] * (un + roe.soundSpeed) - momentumNormal - roe.soundSpeed * result.contact) / (2 * roe.soundSpeed);
        result.plus = jump[0] - result.minus - result.contact;
        return result;
    }
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline FlowStateT<Scalar> Dissipation(const FluxInputT<Scalar> &input, const DecompositionT<Scalar> &waves, const FluxResultT<Scalar> &magnitudes)
    {
        const auto &roe = *input.roe;
        const Scalar minus = waves.minus * magnitudes.acousticMinus, contact = waves.contact * magnitudes.contact, plus = waves.plus * magnitudes.acousticPlus;
        const Scalar un = Dot(roe.velocity, input.normal);
        FlowStateT<Scalar> result{minus + contact + plus, 0, 0, 0,
                                  (roe.enthalpy - un * roe.soundSpeed) * minus + roe.contactEnergy * contact + (roe.enthalpy + un * roe.soundSpeed) * plus +
                                      Dot(waves.shear, roe.velocity) * magnitudes.contact + roe.contactEnergyJump * magnitudes.contact};
        for (int d = 0; d < 3; ++d)
            result[1 + d] = (roe.velocity[d] - roe.soundSpeed * input.normal[d]) * minus + roe.velocity[d] * contact +
                            (roe.velocity[d] + roe.soundSpeed * input.normal[d]) * plus + waves.shear[d] * magnitudes.contact;
        return result;
    }
    /// Scalar-wave dissipation shared by Rusanov and the Roe_M2 variant.
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline FlowStateT<Scalar> RusanovDissipation(const FluxInputT<Scalar> &input, FluxResultT<Scalar> &magnitudes) noexcept
    {
        const auto &left = Mean(input, true), &right = Mean(input, false);
        const Scalar uL = NormalSpeed(left, input), uR = NormalSpeed(right, input);
        const Scalar speed = DNDS::Math::ScalarMax(DNDS::Math::ScalarAbs(uL) + left.soundSpeed, DNDS::Math::ScalarAbs(uR) + right.soundSpeed);
        magnitudes.acousticMinus = magnitudes.contact = magnitudes.acousticPlus = speed;
        FlowStateT<Scalar> result;
        for (int j = 0; j < 5; ++j)
            result[j] = speed * (input.right.flow.conservative[j] - input.left.flow.conservative[j]);
        return result;
    }

    /** @brief Numerical dissipation of the Roe algorithm family, without its central physical flux.
     * @details The host caller validates a Roe-family selection and the optional
     * physical inputs. Roe_M2 uses its scalar-wave definition and needs no Roe
     * linearization. All other variants use the shared decomposition, entropy
     * fix and eigenvector assembly. This single arithmetic path is also used
     * by mixed-precision dissipation-only kernels; no EOS is recovered here.
     */
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline FlowStateT<Scalar> RoeDissipation(Solver solver, const FluxInputT<Scalar> &input,
                                                                  const FluxSettingsT<Scalar> &settings, FluxResultT<Scalar> &magnitudes) noexcept
    {
        if (solver == Solver::Roe_M2)
            return RusanovDissipation(input, magnitudes);
        const auto &roe = *input.roe;
        const auto &left = Mean(input, true), &right = Mean(input, false);
        const Scalar uL = NormalSpeed(left, input), uR = NormalSpeed(right, input);
        const Scalar un = Dot(roe.velocity, input.normal), u = un - Dot(input.gridVelocity, input.normal), a = roe.soundSpeed;
        magnitudes.acousticMinus = DNDS::Math::ScalarAbs(u - a);
        magnitudes.contact = DNDS::Math::ScalarAbs(u);
        magnitudes.acousticPlus = DNDS::Math::ScalarAbs(u + a);
        const auto waves = Decompose(input);
        EntropyFix(int(solver) - int(Solver::Roe), left.soundSpeed, right.soundSpeed, a, uL, uR, u,
                   DNDS::Math::ScalarSqrt(Dot(Relative(roe.velocity, input.gridVelocity), Relative(roe.velocity, input.gridVelocity))), settings, magnitudes);
        return Dissipation(input, waves, magnitudes);
    }

    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Scalar MachSplit(Scalar mach, bool positive, bool fourth)
    {
        const Scalar s = positive ? 1 : -1;
        if (DNDS::Math::ScalarAbs(mach) >= 1)
            return Scalar(.5) * (mach + s * DNDS::Math::ScalarAbs(mach));
        return s * Scalar(.25) * (mach + s) * (mach + s) + (fourth ? s * Scalar(.125) * DNDS::Math::ScalarSquare(mach * mach - 1) : 0);
    }
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Scalar PressureSplit(Scalar mach, bool positive, Scalar alpha)
    {
        const Scalar s = positive ? 1 : -1;
        if (DNDS::Math::ScalarAbs(mach) >= 1)
            return Scalar(.5) * (1 + s * (mach > 0 ? 1 : -1));
        return Scalar(.25) * DNDS::Math::ScalarSquare(mach + s) * (2 - s * mach) + s * alpha * mach * DNDS::Math::ScalarSquare(mach * mach - 1);
    }
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline FlowStateT<Scalar> AUSMFlux(Solver kind, const FluxInputT<Scalar> &input, const FluxSettingsT<Scalar> &settings)
    {
        const auto &left = input.left.flow, &right = input.right.flow;
        const auto vL = Relative(Velocity(input.left), input.gridVelocity), vR = Relative(Velocity(input.right), input.gridVelocity);
        const Scalar unL = Dot(vL, input.normal), unR = Dot(vR, input.normal), a = input.ausm->faceSoundSpeed;
        const Scalar ml = unL / a, mr = unR / a, pL = left.pressure, pR = right.pressure;
        Scalar cL, cR, pressure;
        if (kind == Solver::AUSMPWPlus)
        {
            const Scalar mp = MachSplit(ml, true, false), mm = MachSplit(mr, false, false);
            pressure = PressureSplit(ml, true, Scalar(0)) * pL + PressureSplit(mr, false, Scalar(0)) * pR;
            const Scalar ratio = DNDS::Math::ScalarMin(pL / pR, pR / pL), weight = 1 - ratio * ratio * ratio;
            const Scalar stencil = DNDS::Math::ScalarSquare(DNDS::Math::ScalarMin(Scalar(1), settings.pressureStencilMin / DNDS::Math::ScalarMin(pL, pR)));
            const Scalar fL = pressure > 0 ? (pL / pressure - 1) * stencil : 0, fR = pressure > 0 ? (pR / pressure - 1) * stencil : 0;
            if (mp + mm >= 0)
            {
                cL = a * (mp + mm * ((1 - weight) * (1 + fR) - fL));
                cR = a * mm * weight * (1 + fR);
            }
            else
            {
                cL = a * mp * weight * (1 + fL);
                cR = a * (mm + mp * ((1 - weight) * (1 + fL) - fR));
            }
        }
        else
        {
            const Scalar meanMach = Scalar(.5) * (ml * ml + mr * mr), m0 = DNDS::Math::ScalarSqrt(DNDS::Math::ScalarMin(Scalar(1), DNDS::Math::ScalarMax(meanMach, DNDS::Math::ScalarSquare(settings.referenceMach)))), fa = m0 * (2 - m0);
            const Scalar density = Scalar(.5) * (left.conservative[0] + right.conservative[0]);
            const Scalar massMach = MachSplit(ml, true, true) + MachSplit(mr, false, true) - Scalar(.25) / fa * DNDS::Math::ScalarMax(1 - meanMach, Scalar(0)) * (pR - pL) / (density * a * a);
            cL = massMach >= 0 ? a * massMach : 0;
            cR = massMach < 0 ? a * massMach : 0;
            if (kind == Solver::AUSMPlusUp)
            {
                const Scalar alpha = Scalar(3.) / 16 * (-4 + 5 * fa * fa), pp = PressureSplit(ml, true, alpha), pm = PressureSplit(mr, false, alpha);
                pressure = pp * pL + pm * pR - Scalar(.75) * pp * pm * (left.conservative[0] + right.conservative[0]) * fa * a * (unR - unL);
            }
            else
            {
                const Scalar pp = PressureSplit(ml, true, Scalar(0)), pm = PressureSplit(mr, false, Scalar(0)), speed = DNDS::Math::ScalarSqrt(Scalar(.5) * (Dot(vL, vL) + Dot(vR, vR)));
                pressure = Scalar(.5) * (pL + pR) + Scalar(.5) * (pp - pm) * (pL - pR) + speed * (pp + pm - 1) * density * a;
            }
        }
        FlowStateT<Scalar> result;
        for (int j = 0; j < 5; ++j)
            result[j] = cL * left.conservative[j] + cR * right.conservative[j];
        for (int d = 0; d < 3; ++d)
            result[1 + d] += pressure * input.normal[d];
        result[4] += cL * pL + cR * pR + pressure * Dot(input.gridVelocity, input.normal);
        return result;
    }

    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline FluxResultT<Scalar> Evaluate(Solver solver, const FluxInputT<Scalar> &input, const FluxSettingsT<Scalar> &settings) noexcept
    {
        const auto &left = input.left.flow.conservative, &right = input.right.flow.conservative;
        const auto fluxLeft = PhysicalFlux(input.left.flow, input.normal, input.gridVelocity), fluxRight = PhysicalFlux(input.right.flow, input.normal, input.gridVelocity);
        const auto &meanL = Mean(input, true), &meanR = Mean(input, false);
        const Scalar uL = NormalSpeed(meanL, input), uR = NormalSpeed(meanR, input);
        FluxResultT<Scalar> result{};
        if (solver == Solver::Rusanov || solver == Solver::Roe_M2)
        {
            const auto dissipation = RusanovDissipation(input, result);
            for (int j = 0; j < 5; ++j)
                result.flux[j] = Scalar(.5) * (fluxLeft[j] + fluxRight[j] - dissipation[j]);
            return result;
        }
        if (AUSM(solver))
        {
            result.flux = AUSMFlux(solver, input, settings);
            const Scalar u = Scalar(.5) * (uL + uR), a = DNDS::Math::ScalarMax(meanL.soundSpeed, meanR.soundSpeed);
            result.acousticMinus = DNDS::Math::ScalarAbs(u - a);
            result.contact = DNDS::Math::ScalarAbs(u);
            result.acousticPlus = DNDS::Math::ScalarAbs(u + a);
            return result;
        }
        if (solver == Solver::HLLC)
        {
            const Scalar sl = input.outerWaves->left, sr = input.outerWaves->right;
            // Diagnostic wave estimates preserve the legacy Roe-mean contract
            // when a linearization is supplied; bounds remain externally set.
            const Scalar u = input.roe ? Dot(Relative(input.roe->velocity, input.gridVelocity), input.normal) : Scalar(.5) * (uL + uR);
            const Scalar a = input.roe ? input.roe->soundSpeed : DNDS::Math::ScalarMax(meanL.soundSpeed, meanR.soundSpeed);
            result.acousticMinus = DNDS::Math::ScalarAbs(u - a);
            result.contact = DNDS::Math::ScalarAbs(u);
            result.acousticPlus = DNDS::Math::ScalarAbs(u + a);
            if (sl >= 0)
            {
                result.flux = fluxLeft;
                return result;
            }
            if (sr <= 0)
            {
                result.flux = fluxRight;
                return result;
            }
            const Scalar denominator = left[0] * (sl - uL) - right[0] * (sr - uR);
            const Scalar faceL = NormalSpeed(input.left, input), faceR = NormalSpeed(input.right, input);
            const Scalar contact = DNDS::Math::ScalarAbs(denominator) > DenominatorGuard<Scalar>() ? (input.right.flow.pressure - input.left.flow.pressure + left[0] * faceL * (sl - uL) - right[0] * faceR * (sr - uR)) / denominator : 0;
            const bool upstream = contact >= 0;
            const auto &state = upstream ? left : right, &physical = upstream ? fluxLeft : fluxRight;
            const Scalar wave = upstream ? sl : sr, velocity = upstream ? uL : uR, pressure = upstream ? input.left.flow.pressure : input.right.flow.pressure;
            const Scalar div = wave - contact;
            if (DNDS::Math::ScalarAbs(div) < DenominatorGuard<Scalar>())
            {
                result.flux = physical;
                return result;
            }
            const Scalar starPressure = pressure + state[0] * (wave - velocity) * (contact - velocity);
            for (int j = 0; j < 5; ++j)
            {
                const Scalar direction = j == 0 ? 0 : j == 4 ? contact
                                                             : input.normal[j - 1];
                result.flux[j] = ((state[j] * wave - physical[j]) * contact + direction * starPressure * wave) / div;
            }
            return result;
        }
        if (Roe(solver))
        {
            const auto dissipation = RoeDissipation(solver, input, settings, result);
            for (int j = 0; j < 5; ++j)
                result.flux[j] = Scalar(.5) * (fluxLeft[j] + fluxRight[j] - dissipation[j]);
            return result;
        }
        const auto &roe = *input.roe;
        const Scalar un = Dot(roe.velocity, input.normal), u = un - Dot(input.gridVelocity, input.normal), a = roe.soundSpeed;
        result.acousticMinus = DNDS::Math::ScalarAbs(u - a);
        result.contact = DNDS::Math::ScalarAbs(u);
        result.acousticPlus = DNDS::Math::ScalarAbs(u + a);
        const auto waves = Decompose(input);
        const Scalar sl = DNDS::Math::ScalarMin(u - a, uL - input.left.soundSpeed), sr = DNDS::Math::ScalarMax(u + a, uR + input.right.soundSpeed);
        const Scalar sp = DNDS::Math::ScalarMax(sr, Scalar(0)), sm = DNDS::Math::ScalarMin(sl, Scalar(0)), dfix = a / (a + DNDS::Math::ScalarMax(DNDS::Math::ScalarAbs(u), settings.dLambda * settings.fixScale));
        if (solver == Solver::HLLEP)
        {
            const Scalar div = sp - sm + ScalarSignP(sp - sm) * DenominatorGuard<Scalar>();
            for (int j = 0; j < 5; ++j)
            {
                const Scalar contactMode = j == 0 ? 1 : j == 4 ? roe.contactEnergy
                                                               : roe.velocity[j - 1];
                result.flux[j] = (sp * fluxLeft[j] - sm * fluxRight[j]) / div + (sp * sm / div) * (right[j] - left[j] - dfix * contactMode * waves.contact);
            }
        }
        else
        {
            result.contact = DNDS::Math::ScalarAbs(((sp + sm) * u - 2 * (1 - dfix) * sp * sm) / (sp - sm));
            result.acousticMinus = DNDS::Math::ScalarAbs(((sp + sm) * (u - a) - 2 * sp * sm) / (sp - sm));
            result.acousticPlus = DNDS::Math::ScalarAbs(((sp + sm) * (u + a) - 2 * sp * sm) / (sp - sm));
            const auto dissipation = Dissipation(input, waves, result);
            for (int j = 0; j < 5; ++j)
                result.flux[j] = Scalar(.5) * (fluxLeft[j] + fluxRight[j] - dissipation[j]);
        }
        return result;
    }
}
