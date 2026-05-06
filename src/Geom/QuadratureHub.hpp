#pragma once

// =============================================================================
// QuadratureHub.hpp - Master include for all quadrature rules
// =============================================================================
// This file includes all individual quadrature data files and provides:
//   1. GetQuadratureScheme() - Select appropriate scheme for given order
//   2. GetQuadraturePoint() - Retrieve quadrature points and weights
//
// The design follows the modular pattern used for Elements:
//   - QuadratureBase.hpp: Shared constants and type aliases
//   - Line.hpp, Tri.hpp, Tet.hpp, etc.: Individual parametric space data
//   - GaussJacobi.hpp: Special quadrature for pyramid elements
//   - QuadratureHub.hpp: Unified interface (this file)
// =============================================================================

#include "Elements.hpp"
#include "Quadratures/QuadratureBase.hpp"
#include "Quadratures/Line.hpp"
#include "Quadratures/Tri.hpp"
#include "Quadratures/Tet.hpp"
#include "Quadratures/GaussJacobi.hpp"

namespace DNDS::Geom::Elem
{
    using namespace detail;

    // =========================================================================
    // Scheme Selection
    // =========================================================================
    // Returns the number of quadrature points for the given parametric space
    // and integration order. Returns 0 if the order is not supported.
    //
    // The returned value is both:
    //   - The scheme identifier (INT_SCHEME_*)
    //   - The number of quadrature points
    // =========================================================================

    constexpr t_index GetQuadratureScheme(ParamSpace ps, int int_order)
    {
        if (ps == LineSpace)
            switch (int_order)
            {
            case 0:
            case 1:
                return INT_SCHEME_Line_1;
            case 2:
            case 3:
                return INT_SCHEME_Line_2;
            case 4:
            case 5:
                return INT_SCHEME_Line_3;
            case 6:
                return INT_SCHEME_Line_4;
            default:
                return 0;
            }

        if (ps == TriSpace)
            switch (int_order)
            {
            case 0:
            case 1:
                return INT_SCHEME_Tri_1;
            case 2:
                return INT_SCHEME_Tri_3;
            case 3:
            case 4:
                return INT_SCHEME_Tri_6;
            case 5:
                return INT_SCHEME_Tri_7;
            case 6:
                return INT_SCHEME_Tri_12;
            default:
                return 0;
            }

        if (ps == QuadSpace)
            switch (int_order)
            {
            case 0:
            case 1:
                return INT_SCHEME_Quad_1;
            case 2:
            case 3:
                return INT_SCHEME_Quad_4;
            case 4:
            case 5:
                return INT_SCHEME_Quad_9;
            case 6:
                return INT_SCHEME_Quad_16;
            default:
                return 0;
            }

        if (ps == TetSpace)
            switch (int_order)
            {
            case 0:
            case 1:
                return INT_SCHEME_Tet_1;
            case 2:
                return INT_SCHEME_Tet_4;
            case 3:
                return INT_SCHEME_Tet_8;
            case 4:
            case 5:
                return INT_SCHEME_Tet_14;
            case 6:
                return INT_SCHEME_Tet_24;
            default:
                return 0;
            }

        if (ps == HexSpace)
            switch (int_order)
            {
            case 0:
            case 1:
                return INT_SCHEME_Hex_1;
            case 2:
            case 3:
                return INT_SCHEME_Hex_8;
            case 4:
            case 5:
                return INT_SCHEME_Hex_27;
            case 6:
                return INT_SCHEME_Hex_64;
            default:
                return 0;
            }

        if (ps == PyramidSpace)
            switch (int_order)
            {
            case 0:
            case 1:
                return INT_SCHEME_Pyramid_1;
            case 2:
            case 3:
                return INT_SCHEME_Pyramid_8;
            case 4:
            case 5:
                return INT_SCHEME_Pyramid_27;
            case 6:
                return INT_SCHEME_Pyramid_64;
            default:
                return 0;
            }

        if (ps == PrismSpace)
            switch (int_order)
            {
            case 0:
            case 1:
                return INT_SCHEME_Prism_1;
            case 2:
                return INT_SCHEME_Prism_6;
            case 3:
            case 4:
                return INT_SCHEME_Prism_18;
            case 5:
                return INT_SCHEME_Prism_21;
            case 6:
                return INT_SCHEME_Prism_48;
            default:
                return 0;
            }
        return 0;
    }

    // =========================================================================
    // Quadrature Point Access
    // =========================================================================
    // Retrieves the iG-th quadrature point (pParam) and weight (w) for the
    // given parametric space and scheme.
    //
    // @param ps        Parametric space (LineSpace, TriSpace, etc.)
    // @param scheme    Integration scheme (from GetQuadratureScheme)
    // @param iG        Quadrature point index (0 <= iG < scheme)
    // @param pParam    Output: parametric coordinates [xi, eta, zeta]
    // @param w         Output: quadrature weight
    //
    // @warning pParam should be initialized (with 0) before calling
    // =========================================================================

    template <class TPoint>
    inline void GetQuadraturePoint(ParamSpace ps, t_index scheme, int iG, TPoint &pParam, t_real &w)
    {
        int scheme_size = scheme;
        DNDS_assert(iG < scheme_size);

        // -----------------------------------------------------------------
        // Line Space - Gauss-Legendre
        // -----------------------------------------------------------------
        if (ps == LineSpace)
        {
            switch (scheme)
            {
            case INT_SCHEME_Line_1:
                pParam[0] = GaussLegendre_1[0][iG];
                w = GaussLegendre_1[1][iG];
                return;
            case INT_SCHEME_Line_2:
                pParam[0] = GaussLegendre_2[0][iG];
                w = GaussLegendre_2[1][iG];
                return;
            case INT_SCHEME_Line_3:
                pParam[0] = GaussLegendre_3[0][iG];
                w = GaussLegendre_3[1][iG];
                return;
            case INT_SCHEME_Line_4:
                pParam[0] = GaussLegendre_4[0][iG];
                w = GaussLegendre_4[1][iG];
                return;
            default:
                w = 1e100;
            }
        }

        // -----------------------------------------------------------------
        // Triangle Space - Hammer Rules
        // -----------------------------------------------------------------
        if (ps == TriSpace)
        {
            switch (scheme)
            {
            case INT_SCHEME_Tri_1:
                pParam[0] = HammerTri_1[0][iG];
                pParam[1] = HammerTri_1[1][iG];
                w = HammerTri_1[2][iG];
                return;
            case INT_SCHEME_Tri_3:
                pParam[0] = HammerTri_3[0][iG];
                pParam[1] = HammerTri_3[1][iG];
                w = HammerTri_3[2][iG];
                return;
            case INT_SCHEME_Tri_6:
                pParam[0] = HammerTri_6[0][iG];
                pParam[1] = HammerTri_6[1][iG];
                w = HammerTri_6[2][iG];
                return;
            case INT_SCHEME_Tri_7:
                pParam[0] = HammerTri_7[0][iG];
                pParam[1] = HammerTri_7[1][iG];
                w = HammerTri_7[2][iG];
                return;
            case INT_SCHEME_Tri_12:
                pParam[0] = HammerTri_12[0][iG];
                pParam[1] = HammerTri_12[1][iG];
                w = HammerTri_12[2][iG];
                return;
            default:
                w = 1e100;
            }
        }

        // -----------------------------------------------------------------
        // Tetrahedron Space - Hammer Rules
        // -----------------------------------------------------------------
        if (ps == TetSpace)
        {
            switch (scheme)
            {
            case INT_SCHEME_Tet_1:
                pParam[0] = HammerTet_1[0][iG];
                pParam[1] = HammerTet_1[1][iG];
                pParam[2] = HammerTet_1[2][iG];
                w = HammerTet_1[3][iG];
                return;
            case INT_SCHEME_Tet_4:
                pParam[0] = HammerTet_4[0][iG];
                pParam[1] = HammerTet_4[1][iG];
                pParam[2] = HammerTet_4[2][iG];
                w = HammerTet_4[3][iG];
                return;
            case INT_SCHEME_Tet_8:
                pParam[0] = HammerTet_8[0][iG];
                pParam[1] = HammerTet_8[1][iG];
                pParam[2] = HammerTet_8[2][iG];
                w = HammerTet_8[3][iG];
                return;
            case INT_SCHEME_Tet_14:
                pParam[0] = HammerTet_14[0][iG];
                pParam[1] = HammerTet_14[1][iG];
                pParam[2] = HammerTet_14[2][iG];
                w = HammerTet_14[3][iG];
                return;
            case INT_SCHEME_Tet_24:
                pParam[0] = HammerTet_24[0][iG];
                pParam[1] = HammerTet_24[1][iG];
                pParam[2] = HammerTet_24[2][iG];
                w = HammerTet_24[3][iG];
                return;
            default:
                w = 1e100;
            }
        }

        // -----------------------------------------------------------------
        // Quadrilateral Space - Product of 1D Gauss-Legendre
        // -----------------------------------------------------------------
        if (ps == QuadSpace)
        {
            const t_real *GLData = nullptr;
            int GLSize = 0;
            switch (scheme)
            {
            case INT_SCHEME_Quad_1:
                GLData = GaussLegendre_1.front().data();
                GLSize = 1;
                break;
            case INT_SCHEME_Quad_4:
                GLData = GaussLegendre_2.front().data();
                GLSize = 2;
                break;
            case INT_SCHEME_Quad_9:
                GLData = GaussLegendre_3.front().data();
                GLSize = 3;
                break;
            case INT_SCHEME_Quad_16:
                GLData = GaussLegendre_4.front().data();
                GLSize = 4;
                break;
            default:
                w = 1e100;
                return;
            }
            int iGi = iG % GLSize;
            int iGj = iG / GLSize;
            pParam[0] = GLData[iGi];
            pParam[1] = GLData[iGj];
            w = GLData[GLSize + iGi] * GLData[GLSize + iGj];
            return;
        }

        // -----------------------------------------------------------------
        // Hexahedron Space - Product of 1D Gauss-Legendre
        // -----------------------------------------------------------------
        if (ps == HexSpace)
        {
            const t_real *GLData = nullptr;
            int GLSize = 0;
            switch (scheme)
            {
            case INT_SCHEME_Hex_1:
                GLData = GaussLegendre_1.front().data();
                GLSize = 1;
                break;
            case INT_SCHEME_Hex_8:
                GLData = GaussLegendre_2.front().data();
                GLSize = 2;
                break;
            case INT_SCHEME_Hex_27:
                GLData = GaussLegendre_3.front().data();
                GLSize = 3;
                break;
            case INT_SCHEME_Hex_64:
                GLData = GaussLegendre_4.front().data();
                GLSize = 4;
                break;
            default:
                w = 1e100;
                return;
            }
            int iGi = iG % GLSize;
            int iGj = (iG / GLSize) % GLSize;
            int iGk = (iG / (GLSize * GLSize));

            pParam[0] = GLData[iGi];
            pParam[1] = GLData[iGj];
            pParam[2] = GLData[iGk];
            w = GLData[GLSize + iGi] * GLData[GLSize + iGj] * GLData[GLSize + iGk];
            return;
        }

        // -----------------------------------------------------------------
        // Pyramid Space - Product of Gauss-Legendre and Gauss-Jacobi
        // -----------------------------------------------------------------
        if (ps == PyramidSpace)
        {
            const t_real *GLData = nullptr;
            const t_real *GJData = nullptr;
            int GLSize = 0; // == GJSize
            switch (scheme)
            {
            case INT_SCHEME_Pyramid_1:
                GLData = GaussLegendre_1.front().data();
                GJData = GaussJacobi_01A2B0_1.front().data();
                GLSize = 1;
                break;
            case INT_SCHEME_Pyramid_8:
                GLData = GaussLegendre_2.front().data();
                GJData = GaussJacobi_01A2B0_2.front().data();
                GLSize = 2;
                break;
            case INT_SCHEME_Pyramid_27:
                GLData = GaussLegendre_3.front().data();
                GJData = GaussJacobi_01A2B0_3.front().data();
                GLSize = 3;
                break;
            case INT_SCHEME_Pyramid_64:
                GLData = GaussLegendre_4.front().data();
                GJData = GaussJacobi_01A2B0_4.front().data();
                GLSize = 4;
                break;
            default:
                w = 1e100;
                return;
            }
            int iGi = iG % GLSize;
            int iGj = (iG / GLSize) % GLSize;
            int iGk = (iG / (GLSize * GLSize));

            pParam[0] = GLData[iGi] * (1 - GJData[iGk]);
            pParam[1] = GLData[iGj] * (1 - GJData[iGk]);
            pParam[2] = GJData[iGk];
            w = GLData[GLSize + iGi] * GLData[GLSize + iGj] * GJData[GLSize + iGk];
            return;
        }

        // -----------------------------------------------------------------
        // Prism Space - Product of Triangle and Line
        // -----------------------------------------------------------------
        if (ps == PrismSpace)
        {
            const t_real *GLData = nullptr;
            const t_real *HammerData = nullptr;
            int GLSize = 0;
            int HammerSize = 0;
            switch (scheme)
            {
            case INT_SCHEME_Prism_1:
                GLData = GaussLegendre_1.front().data();
                GLSize = 1;
                HammerData = HammerTri_1.front().data();
                HammerSize = 1;
                break;
            case INT_SCHEME_Prism_6:
                GLData = GaussLegendre_2.front().data();
                GLSize = 2;
                HammerData = HammerTri_3.front().data();
                HammerSize = 3;
                break;
            case INT_SCHEME_Prism_18:
                GLData = GaussLegendre_3.front().data();
                GLSize = 3;
                HammerData = HammerTri_6.front().data();
                HammerSize = 6;
                break;
            case INT_SCHEME_Prism_21:
                GLData = GaussLegendre_3.front().data();
                GLSize = 3;
                HammerData = HammerTri_7.front().data();
                HammerSize = 7;
                break;
            case INT_SCHEME_Prism_48:
                GLData = GaussLegendre_4.front().data();
                GLSize = 4;
                HammerData = HammerTri_12.front().data();
                HammerSize = 12;
                break;
            default:
                w = 1e100;
                return;
            }
            int iGi = iG % GLSize;
            int iGj = iG / GLSize;

            pParam[0] = HammerData[0 * HammerSize + iGj];
            pParam[1] = HammerData[1 * HammerSize + iGj];
            pParam[2] = GLData[iGi];

            w = GLData[GLSize + iGi] * HammerData[2 * HammerSize + iGj];
            return;
        }

        w = 1e100;
    }

} // namespace DNDS::Geom::Elem