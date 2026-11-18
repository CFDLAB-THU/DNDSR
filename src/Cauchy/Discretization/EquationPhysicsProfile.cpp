#include "Cauchy/Discretization/EquationPhysicsProfile.hpp"

#include "Cauchy/Physics/EquationSystem.hpp"
#include "Physics/RANS/Model.hpp"

namespace DNDS::Cauchy
{
    EquationPhysicsProfile ResolveEquationPhysicsProfile(const EquationSystem &equation)
    {
        if (equation.HasMixture())
            return EquationPhysicsProfile::Generic;
        const bool acm = equation.IsACM();
        if (!equation.HasRANS())
            return acm ? EquationPhysicsProfile::ACM : EquationPhysicsProfile::PerfectGas;
        switch (equation.RANS().Kind())
        {
        case Physics::RANS::ModelKind::SA:
            return acm ? EquationPhysicsProfile::ACMSA : EquationPhysicsProfile::PerfectGasSA;
        case Physics::RANS::ModelKind::KOmegaWilcox:
        case Physics::RANS::ModelKind::KOmegaSST:
            return acm ? EquationPhysicsProfile::ACMTwoEquation : EquationPhysicsProfile::PerfectGasTwoEquation;
        default:
            return EquationPhysicsProfile::Generic;
        }
    }

}
