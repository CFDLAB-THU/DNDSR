#pragma once

#include "Geom/Geometric.hpp"

namespace DNDS::Geom::Elem
{
    // ===================================================================
    // Integration Scheme Constants
    // ===================================================================
    // These constants define the number of quadrature points for each
    // integration scheme. The naming convention is:
    //   INT_SCHEME_<Shape>_<N> = N points
    //
    // Expected orders of exactness (polynomial degree):
    //   Line:   Gauss-Legendre, exact for degree 2n-1
    //   Quad:   Product of 1D Gauss-Legendre
    //   Hex:    Product of 1D Gauss-Legendre
    //   Tri:    Hammer rules, exactness varies
    //   Tet:    Hammer rules, exactness varies
    //   Prism:  Triangle x Line product
    //   Pyramid: Gauss-Legendre x Gauss-Jacobi product
    // ===================================================================

    // Line [-1, 1] - Gauss-Legendre quadrature
    static const t_index INT_SCHEME_Line_1 = 1; // O1 exact
    static const t_index INT_SCHEME_Line_2 = 2; // O3 exact
    static const t_index INT_SCHEME_Line_3 = 3; // O5 exact
    static const t_index INT_SCHEME_Line_4 = 4; // O7 exact

    // Quad [-1, 1]^2 - Product of 1D Gauss-Legendre
    static const t_index INT_SCHEME_Quad_1 = 1;   // O1 exact (1x1)
    static const t_index INT_SCHEME_Quad_4 = 4;   // O3 exact (2x2)
    static const t_index INT_SCHEME_Quad_9 = 9;   // O5 exact (3x3)
    static const t_index INT_SCHEME_Quad_16 = 16; // O7 exact (4x4)

    // Triangle (reference: vertices at (0,0), (1,0), (0,1)) - Hammer rules
    static const t_index INT_SCHEME_Tri_1 = 1;   // O1 exact (centroid)
    static const t_index INT_SCHEME_Tri_3 = 3;   // O2 exact
    static const t_index INT_SCHEME_Tri_6 = 6;   // O4 exact
    static const t_index INT_SCHEME_Tri_7 = 7;   // O5 exact
    static const t_index INT_SCHEME_Tri_12 = 12; // O6 exact

    // Tetrahedron (reference: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)) - Hammer rules
    static const t_index INT_SCHEME_Tet_1 = 1;   // O1 exact (centroid)
    static const t_index INT_SCHEME_Tet_4 = 4;   // O2 exact
    static const t_index INT_SCHEME_Tet_8 = 8;   // O3 exact
    static const t_index INT_SCHEME_Tet_14 = 14; // O5 exact
    static const t_index INT_SCHEME_Tet_24 = 24; // O6 exact

    // Hex [-1, 1]^3 - Product of 1D Gauss-Legendre
    static const t_index INT_SCHEME_Hex_1 = 1;   // O1 exact (1x1x1)
    static const t_index INT_SCHEME_Hex_8 = 8;   // O3 exact (2x2x2)
    static const t_index INT_SCHEME_Hex_27 = 27; // O5 exact (3x3x3)
    static const t_index INT_SCHEME_Hex_64 = 64; // O7 exact (4x4x4)

    // Prism (Triangle x Line) - Product of Tri and Line rules
    static const t_index INT_SCHEME_Prism_1 = 1 * 1;   // O1 exact (Tri_1 x Line_1)
    static const t_index INT_SCHEME_Prism_6 = 3 * 2;   // O2 exact (Tri_3 x Line_2)
    static const t_index INT_SCHEME_Prism_18 = 6 * 3;  // O4 exact (Tri_6 x Line_3)
    static const t_index INT_SCHEME_Prism_21 = 7 * 3;  // O5 exact (Tri_7 x Line_3)
    static const t_index INT_SCHEME_Prism_48 = 12 * 4; // O6 exact (Tri_12 x Line_4)

    // Pyramid - Product of Quad (in scaled coords) and Gauss-Jacobi (in z)
    static const t_index INT_SCHEME_Pyramid_1 = 1;   // O1 exact
    static const t_index INT_SCHEME_Pyramid_8 = 8;   // O3 exact
    static const t_index INT_SCHEME_Pyramid_27 = 27; // O5 exact
    static const t_index INT_SCHEME_Pyramid_64 = 64; // O7 exact

    // Maximum integration order supported
    static const int INT_ORDER_MAX = 6;

} // namespace DNDS::Geom::Elem