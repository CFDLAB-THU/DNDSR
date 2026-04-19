#pragma once

#include "QuadratureBase.hpp"

namespace DNDS::Geom::Elem
{
    // ===================================================================
    // Gauss-Jacobi Quadrature on [0, 1] with weight (1-z)^2
    // ===================================================================
    // These are used for pyramid quadrature in the z-direction.
    // The pyramid is parameterized with z in [0,1] and (x,y) in the base
    // scaled by (1-z). The Jacobian factor introduces a (1-z)^2 weight.
    //
    // Parameters: alpha = 2, beta = 0 (Jacobi polynomials)
    // Reference interval: [0, 1]
    //
    // Data format: [2][n] array where:
    //   [0][i] = quadrature point (zi) in [0, 1]
    //   [1][i] = quadrature weight (wi)
    // ===================================================================

    /// Gauss-Jacobi (alpha=2, beta=0) with 1 point
    static const t_real __GaussJacobi_01A2B0_1[2][1]{
        {0.250000000000000},
        {0.333333333333333}};

    /// Gauss-Jacobi (alpha=2, beta=0) with 2 points
    static const t_real __GaussJacobi_01A2B0_2[2][2]{
        {0.122514822655441, 0.544151844011225},
        {0.232547451253508, 0.100785882079825}};

    /// Gauss-Jacobi (alpha=2, beta=0) with 3 points
    static const t_real __GaussJacobi_01A2B0_3[2][3]{
        {0.072994024073150, 0.347003766038352, 0.705002209888499},
        {0.157136361064887, 0.146246269259866, 0.029950703008581}};

    /// Gauss-Jacobi (alpha=2, beta=0) with 4 points
    static const t_real __GaussJacobi_01A2B0_4[2][4]{
        {0.048500549446997, 0.238600737551862, 0.517047295104367, 0.795851417896773},
        {0.110888415611278, 0.143458789799214, 0.068633887172923, 0.010352240749918}};

    /// Gauss-Jacobi (alpha=2, beta=0) with 5 points
    static const t_real __GaussJacobi_01A2B0_5[2][5]{
        {0.034578939918215, 0.173480320771696, 0.389886387065519, 0.634333472630887, 0.851054212947016},
        {0.081764784285771, 0.126198961899911, 0.089200161221590, 0.032055600722962, 0.004113825203099}};

} // namespace DNDS::Geom::Elem