#pragma once

#include "DNDS/Core/Defines.hpp"
#include <array>

namespace DNDS::ODE
{
    enum class DITRVariant
    {
        U2R2,
        U2R1
    };
    struct DITRSettings
    {
        real fraction = .55, theta = .9146;
        DITRVariant variant = DITRVariant::U2R2;
        void Validate() const
        {
            DNDS_check_throw_info(std::isfinite(fraction) && fraction > 0 && fraction < 1 &&
                                      std::isfinite(theta) && theta > 0 &&
                                      (variant == DITRVariant::U2R2 || variant == DITRVariant::U2R1),
                                  "invalid DITR fraction, transformed-equation weight or variant");
        }
    };
    struct DITRCoefficients
    {
        std::array<real, 4> interpolation;
        std::array<real, 3> integration;
        /// Reciprocal infinity-norm condition number of the simultaneous
        /// algebraic-rate system [H+theta I, I] for (g_mid, g_end).
        /// Its determinant is -w1*c3 (1/6 for U2R1, a/6 for U2R2).
        /// Use this product directly; subtracting two theta-scaled products
        /// loses the determinant precisely for the poorly conditioned cases.
        real AlgebraicReciprocalCondition(real theta) const
        {
            const real w1 = integration[1], w2 = integration[2], c3 = interpolation[3];
            const real a = theta * w1, b = c3 + theta * w2;
            const real norm = std::max(std::abs(a) + std::abs(b), std::abs(w1) + std::abs(w2));
            const real adjugateNorm = std::max(std::abs(w2) + std::abs(b), std::abs(w1) + std::abs(a));
            return std::abs(w1 * c3) / norm / adjugateNorm;
        }
        explicit DITRCoefficients(real a, DITRVariant variant = DITRVariant::U2R2)
        {
            DNDS_check_throw_info(std::isfinite(a) && a > 0 && a < 1, "DITR fraction must be between zero and one");
            DNDS_check_throw_info(variant == DITRVariant::U2R2 || variant == DITRVariant::U2R1, "unknown DITR variant");
            // Same interpolation masks as Euler ODE.hpp. Integration weights
            // are shared; U2R1 drops the initial-rate interpolation constraint.
            if (variant == DITRVariant::U2R1)
                interpolation = {1 - 2 * a + a * a, 2 * a - a * a, 0, -a + a * a};
            else
                interpolation = {1 - 3 * a * a + 2 * a * a * a, 3 * a * a - 2 * a * a * a,
                                 a - 2 * a * a + a * a * a, -a * a + a * a * a};
            integration = {.5 - 1 / (6 * a), -1 / (6 * a * (a - 1)), .5 + 1 / (6 * (a - 1))};
            for (real value : integration)
                DNDS_check_throw_info(std::isfinite(value), "nonfinite DITR integration coefficient");
        }
    };
}
