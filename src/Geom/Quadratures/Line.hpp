#pragma once

#include <array>
#include "QuadratureBase.hpp"

namespace DNDS::Geom::Elem
{
    // ===================================================================
    // Gauss-Legendre Quadrature on [-1, 1]
    // ===================================================================
    // These are the standard Gauss-Legendre quadrature points and weights
    // for integration over the interval [-1, 1].
    //
    // A rule with n points integrates polynomials up to degree 2n-1 exactly.
    //
    // Data format: [2][n] array where:
    //   [0][i] = quadrature point (xi)
    //   [1][i] = quadrature weight (wi)
    // ===================================================================

    /// Gauss-Legendre with 1 point: exact for degree 1
    /// Point: 0, Weight: 2
    namespace detail
    {
        static constexpr std::array<std::array<t_real, 1>, 2> GaussLegendre_1{{{{0}},
                                                                               {{2}}}};

        /// Gauss-Legendre with 2 points: exact for degree 3
        /// Points: ±1/√3 ≈ ±0.577350269189626
        /// Weights: 1, 1
        static constexpr std::array<std::array<t_real, 2>, 2> GaussLegendre_2{{{{-0.577350269189626, 0.577350269189626}},
                                                                               {{1, 1}}}};

        /// Gauss-Legendre with 3 points: exact for degree 5
        /// Points: 0, ±√(3/5) ≈ ±0.774596669241483
        /// Weights: 8/9, 5/9, 5/9
        static constexpr std::array<std::array<t_real, 3>, 2> GaussLegendre_3{{{{-0.774596669241483, 0, 0.774596669241483}},
                                                                               {{0.555555555555555, 0.888888888888889, 0.555555555555555}}}};

        /// Gauss-Legendre with 4 points: exact for degree 7
        static constexpr std::array<std::array<t_real, 4>, 2> GaussLegendre_4{{{{-0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053}},
                                                                               {{0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454}}}};

        /// Gauss-Legendre with 5 points: exact for degree 9
        static constexpr std::array<std::array<t_real, 5>, 2> GaussLegendre_5{{{{-0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664}},
                                                                               {{0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189}}}};

    } // namespace detail
} // namespace DNDS::Geom::Elem