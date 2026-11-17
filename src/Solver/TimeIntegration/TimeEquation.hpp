#pragma once

#include "DNDS/Core/Defines.hpp"

namespace DNDS::ODE
{
    /// Normalized equation r = resOther + rhsScale * F(U,t) - inverseDt * U.
    /// resOther contains no contribution from this unknown or its current RHS.
    /// Other unknowns/history must remain frozen while this equation is used,
    /// including throughout a spatial FAS cycle. No MPI or physics is hidden here.
    struct TimeEquationCoefficients
    {
        real inverseDt = 0;
        real rhsScale = 1;
        real time = 0;

        void Validate() const
        {
            DNDS_check_throw_info(std::isfinite(inverseDt) && inverseDt >= 0 &&
                                      std::isfinite(rhsScale) && rhsScale > 0 && std::isfinite(time),
                                  "invalid normalized time equation coefficients");
        }

        /// Output must have distinct storage. Only owned values are produced;
        /// storage/layout and access synchronization belong to Operations.
        template <class Operations, class State>
        void AssembleResidual(State &out, const State &resOther, const State &state, const State &rate, const Operations &operations = {}) const
        {
            Validate();
            Operations::RequireDistinct(out, resOther);
            Operations::RequireDistinct(out, state);
            Operations::RequireDistinct(out, rate);
            operations.AssembleTimeResidual(out, resOther, state, rate, inverseDt, rhsScale);
        }
    };

    /// Borrowed view of one equation. Its method owner keeps resOther alive.
    /// coupledStage is an unknown within a nonlinear problem, not a serialized
    /// DIRK stage. Finish the update before asking the method for another view.
    template <class State>
    struct CoupledStageEquation
    {
        State &state;
        const State &resOther;
        TimeEquationCoefficients coefficients;
        int coupledStage = 0;
    };
}
