#pragma once

#include "Cauchy/Physics/FlowState.hpp"
#include "DNDS/Core/Errors.hpp"
#include "DNDS/Math/ComputeScalar.hpp"
#include <cmath>
#include <string>
#include <limits>

namespace DNDS::Cauchy::Riemann
{
    /// EOS-independent physical input. Pressure is supplied by the equation
    /// layer; conservative energy retains that layer's energy convention.
    template <class Scalar>
    struct EulerStateT
    {
        FlowStateT<Scalar> conservative;
        Scalar pressure;
    };
    using EulerState = EulerStateT<real>;
    /// Rusanov additionally requires the equation's physical sound speed.
    /// Further algorithms add their own typed enthalpy/derivative/average data;
    /// absent thermodynamics must never be guessed from conservative energy.
    template <class Scalar>
    struct AcousticStateT
    {
        EulerStateT<Scalar> flow;
        Scalar soundSpeed;
    };
    using AcousticState = AcousticStateT<real>;
    struct Requirements
    {
        bool pressure, soundSpeed, enthalpy;
        bool meanStates = false, roe = false, outerWaveBounds = false, ausmFaceSpeed = false;
    };
    enum class Solver
    {
        Rusanov,
        Roe,
        Roe_M1,
        Roe_M2,
        Roe_M3,
        Roe_M4,
        Roe_M5,
        Roe_M6,
        Roe_M7,
        Roe_M8,
        Roe_M9,
        HLLC,
        HLLEP,
        HLLEP_V1,
        AUSMPWPlus,
        AUSMPlusUp,
        AUSMPlusUp2
    };

    /// Pure scheme metadata, shared by host preparation and device point code.
    DNDS_DEVICE_CALLABLE constexpr Requirements RequiredInputs(Solver solver)
    {
        const bool roe = solver >= Solver::Roe && solver <= Solver::Roe_M9;
        const bool linearization = (roe && solver != Solver::Roe_M2) ||
                                   solver == Solver::HLLEP || solver == Solver::HLLEP_V1 || solver == Solver::HLLC;
        return {true, true, linearization, solver != Solver::Rusanov, linearization,
                solver == Solver::HLLC, solver >= Solver::AUSMPWPlus};
    }

    /// Equation-prepared Roe linearization. contactProjection is a row of its
    /// inverse eigenbasis, mapping conservative jumps onto the density/contact
    /// wave. contactEnergy and enthalpy are the energy entries of its right
    /// basis. This describes the supplied model; no gamma or EOS recovery is
    /// required in Roe or HLLEP. Extra pressure-neutral energy can be supplied
    /// explicitly for multi-energy/species profiles with the corresponding mode.
    template <class Scalar>
    struct RoeDataT
    {
        Vector3T<Scalar> velocity;
        Scalar soundSpeed, enthalpy, contactEnergy;
        FlowStateT<Scalar> contactProjection;
        Scalar contactEnergyJump = 0;
    };
    using RoeData = RoeDataT<real>;
    template <class Scalar>
    struct WaveBoundsT
    {
        Scalar left, right;
    };
    using WaveBounds = WaveBoundsT<real>;
    template <class Scalar>
    struct AUSMDataT
    {
        Scalar faceSoundSpeed;
    };
    using AUSMData = AUSMDataT<real>;
    /// Common modular flux interface. Optional data pointers are required only
    /// by RequiredInputs(); their storage is caller-owned through Evaluate.
    /// Face and mean states are separate, preserving legacy implicit estimates.
    template <class Scalar>
    struct FluxInputT
    {
        AcousticStateT<Scalar> left, right;
        Vector3T<Scalar> normal, gridVelocity{};
        const AcousticStateT<Scalar> *leftMean = nullptr, *rightMean = nullptr;
        const RoeDataT<Scalar> *roe = nullptr;
        const WaveBoundsT<Scalar> *outerWaves = nullptr;
        const AUSMDataT<Scalar> *ausm = nullptr;
    };
    using FluxInput = FluxInputT<real>;
    template <class Scalar>
    struct FluxResultT
    {
        FlowStateT<Scalar> flux;
        Scalar acousticMinus, contact, acousticPlus;
    };
    using FluxResult = FluxResultT<real>;
    template <class Scalar>
    struct FluxSettingsT
    {
        Scalar dLambda = 0, fixScale = 1, incrementScale = 1;
        Scalar referenceMach = Scalar(.1), pressureStencilMin = DNDS::Math::ScalarMaxFinite<Scalar>();
    };
    using FluxSettings = FluxSettingsT<real>;

    /// Unit normal and explicit grid velocity (zero by default); only
    /// kinematics and supplied pressure are used. No EOS, heat capacity or
    /// total-enthalpy recovery occurs here.
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline FlowStateT<Scalar> PhysicalFlux(const EulerStateT<Scalar> &input, const Vector3T<Scalar> &normal, const Vector3T<Scalar> &gridVelocity = {})
    {
        const auto &state = input.conservative;
        Scalar speed = 0;
        for (int d = 0; d < 3; ++d)
            speed += (state[1 + d] / state[0] - gridVelocity[d]) * normal[d];
        FlowStateT<Scalar> flux{state[0] * speed, 0, 0, 0, (state[4] + input.pressure) * speed};
        for (int d = 0; d < 3; ++d)
        {
            flux[1 + d] = state[1 + d] * speed + input.pressure * normal[d];
            flux[4] += input.pressure * gridVelocity[d] * normal[d];
        }
        return flux;
    }
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Scalar SignalSpeed(const AcousticStateT<Scalar> &input, const Vector3T<Scalar> &normal, const Vector3T<Scalar> &gridVelocity = {})
    {
        Scalar speed = 0;
        for (int d = 0; d < 3; ++d)
            speed += (input.flow.conservative[1 + d] / input.flow.conservative[0] - gridVelocity[d]) * normal[d];
        return DNDS::Math::ScalarAbs(speed) + input.soundSpeed;
    }
    /// Face contribution to the legacy H-correction stencil. The sound speeds
    /// are equation-provided. This scalar is invariant under swapping sides
    /// and reversing the normal, and under a common constant grid velocity.
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline Scalar TransverseSignalJump(const AcousticStateT<Scalar> &left, const AcousticStateT<Scalar> &right, const Vector3T<Scalar> &normal)
    {
        Scalar jump = 0;
        for (int d = 0; d < 3; ++d)
            jump += (right.flow.conservative[1 + d] / right.flow.conservative[0] -
                     left.flow.conservative[1 + d] / left.flow.conservative[0]) *
                    normal[d];
        // max(|du+da|, |du-da|, |da|) = |du|+|da|.
        return DNDS::Math::ScalarAbs(jump) + DNDS::Math::ScalarAbs(right.soundSpeed - left.soundSpeed);
    }
    template <class Scalar>
    DNDS_DEVICE_CALLABLE inline FlowStateT<Scalar> Rusanov(const AcousticStateT<Scalar> &left, const AcousticStateT<Scalar> &right, const Vector3T<Scalar> &normal, const Vector3T<Scalar> &gridVelocity = {})
    {
        const auto a = PhysicalFlux(left.flow, normal, gridVelocity), b = PhysicalFlux(right.flow, normal, gridVelocity);
        const Scalar speedLeft = SignalSpeed(left, normal, gridVelocity), speedRight = SignalSpeed(right, normal, gridVelocity);
        const Scalar alpha = speedLeft > speedRight ? speedLeft : speedRight;
        FlowStateT<Scalar> result;
        for (int j = 0; j < 5; ++j)
            result[j] = Scalar(.5) * (a[j] + b[j] - alpha * (right.flow.conservative[j] - left.flow.conservative[j]));
        return result;
    }
    /// Coarse selection resolved before face loops. Required inputs are explicit
    /// and kernel-ready. Caller validates state positivity, finite model data
    /// and unit/frame-consistent normals at the equation/operator boundary.
    /// This is a compressible-Euler flux profile, not a universal AC-NS flux.
    class RiemannFlux
    {
        Solver _solver;
        explicit RiemannFlux(Solver solver) : _solver(solver) {}

    public:
        static RiemannFlux Resolve(const std::string &name);
        static bool Available(const std::string &name);
        DNDS_DEVICE_CALLABLE Solver Kind() const { return _solver; }
        Requirements RequiredInputs() const;
        /// These optional spatial inputs are prepared by the FV caller, never
        /// by the pointwise flux. A manual scalar remains valid for unit tests
        /// and one-dimensional users.
        bool UsesTransverseSignal() const;
        bool UsesPressureStencil() const;
        /// Validate optional data once in an outer preparation/operator boundary.
        /// Evaluate assumes that contract; unsupported/missing data never falls
        /// back to another numerical algorithm.
        void Validate(const FluxInput &input, const FluxSettings &settings = {}) const;
        FluxResult Evaluate(const FluxInput &input, const FluxSettings &settings = {}) const noexcept;
    };
}
