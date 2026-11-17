#pragma once

#include "Solver/TimeIntegration/TimeEquation.hpp"

namespace DNDS::ODE
{
    /// BDF1 startup, variable-step BDF2, or BDF1 steady coefficients normalized
    /// to the same -U/dt direct term. History ownership belongs to the method;
    /// this value object only describes its equation and direct source assembly.
    struct BDFCoefficients
    {
        TimeEquationCoefficients equation;
        real initial = 0, previous = 0;

        BDFCoefficients(int order, real time, real dt, real previousDt = 0, bool steady = false)
        {
            DNDS_check_throw_info((order == 1 || order == 2) && std::isfinite(time) &&
                                      (steady ? order == 1 : std::isfinite(dt) && dt > 0 && std::isfinite(1 / dt) && std::isfinite(time + dt) && time + dt > time),
                                  "invalid BDF order or timestep");
            equation = {steady ? 0 : 1 / dt, 1, steady ? time : time + dt};
            initial = equation.inverseDt;
            if (!steady && order == 2 && previousDt != 0)
            {
                DNDS_check_throw_info(std::isfinite(previousDt) && previousDt > 0, "invalid BDF accepted step interval");
                const real ratio = dt / previousDt;
                DNDS_check_throw_info(std::isfinite(ratio) && ratio > 0 && std::isfinite(ratio * ratio), "invalid BDF step ratio");
                const real denominator = 1 + 2 * ratio;
                initial = (1 + ratio) * (1 + ratio) / (denominator * dt);
                previous = -ratio * ratio / (denominator * dt);
                equation.rhsScale = (1 + ratio) / denominator;
            }
            DNDS_check_throw_info(std::isfinite(initial) && std::isfinite(previous), "nonfinite BDF history coefficients");
        }

        /// prev is required only for a nonzero history coefficient. No current
        /// unknown or RHS appears here. Output/history must not alias.
        template <class Operations, class State>
        void AssembleResOther(State &out, const State &start, const State *prev = nullptr, const Operations &operations = {}) const
        {
            Operations::RequireDistinct(out, start);
            if (previous != 0)
            {
                DNDS_check_throw_info(prev, "BDF2 source requires accepted history");
                Operations::RequireDistinct(out, *prev);
                operations.PhysicalLinearCombination(out, initial, start, previous, *prev);
            }
            else if (initial == 0)
                operations.Fill(out, 0);
            else
            {
                operations.PhysicalCopyScale(out, initial, start);
            }
        }
    };
}
