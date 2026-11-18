#pragma once

namespace DNDS::Cauchy
{
    class EquationSystem;
    /// Select model semantics independently of logical width. An ACM-SA row
    /// and a perfect-gas laminar row can both have five components.
    enum class EquationPhysicsProfile
    {
        Generic,
        ACM,
        ACMSA,
        ACMTwoEquation,
        PerfectGas,
        PerfectGasSA,
        PerfectGasTwoEquation
    };
    enum class ReconstructionForm
    {
        Generic,
        MeanOnly,
        DirectGradient,
        Modal
    };

    constexpr bool IsACMProfile(EquationPhysicsProfile physics)
    {
        return physics == EquationPhysicsProfile::ACM || physics == EquationPhysicsProfile::ACMSA || physics == EquationPhysicsProfile::ACMTwoEquation;
    }

    EquationPhysicsProfile ResolveEquationPhysicsProfile(const EquationSystem &equation);
}
