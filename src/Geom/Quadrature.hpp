#pragma once

#include "Elements.hpp"

namespace DNDS::Geom::Elem
{
    static const t_index INT_SCHEME_Line_1 = 1; // O1
    static const t_index INT_SCHEME_Line_2 = 2; // O3
    static const t_index INT_SCHEME_Line_3 = 3; // O5
    static const t_index INT_SCHEME_Line_4 = 4; // O7

    static const t_index INT_SCHEME_Quad_1 = 1;   // O1
    static const t_index INT_SCHEME_Quad_4 = 4;   // O3
    static const t_index INT_SCHEME_Quad_9 = 9;   // O5
    static const t_index INT_SCHEME_Quad_16 = 16; // O7

    static const t_index INT_SCHEME_Tri_1 = 1;   // O1
    static const t_index INT_SCHEME_Tri_3 = 3;   // O2
    static const t_index INT_SCHEME_Tri_6 = 6;   // O4
    static const t_index INT_SCHEME_Tri_7 = 7;   // O5
    static const t_index INT_SCHEME_Tri_12 = 12; // O6

    static const t_index INT_SCHEME_Tet_1 = 1;   // O1
    static const t_index INT_SCHEME_Tet_4 = 4;   // O2
    static const t_index INT_SCHEME_Tet_8 = 8;   // O3
    static const t_index INT_SCHEME_Tet_14 = 14; // O5
    static const t_index INT_SCHEME_Tet_24 = 24; // O6

    static const t_index INT_SCHEME_Hex_1 = 1;   // O1
    static const t_index INT_SCHEME_Hex_8 = 8;   // O3
    static const t_index INT_SCHEME_Hex_27 = 27; // O5
    static const t_index INT_SCHEME_Hex_64 = 64; // O7

    static const t_index INT_SCHEME_Prism_1 = 1 * 1;   // O1
    static const t_index INT_SCHEME_Prism_6 = 3 * 2;   // O2
    static const t_index INT_SCHEME_Prism_18 = 6 * 3;  // O4
    static const t_index INT_SCHEME_Prism_21 = 7 * 3;  // O5
    static const t_index INT_SCHEME_Prism_48 = 12 * 4; // O6

    static const t_index INT_SCHEME_Pyramid_1 = 1;   // O1
    static const t_index INT_SCHEME_Pyramid_8 = 8;   // O3
    static const t_index INT_SCHEME_Pyramid_27 = 27; // O5
    static const t_index INT_SCHEME_Pyramid_64 = 64; // O7

    static const int INT_ORDER_MAX = 6;

    static const t_real __GaussLegendre_1[2][1]{
        {0},
        {2}};

    static const t_real __GaussLegendre_2[2][2]{
        {-0.577350269189626, 0.577350269189626},
        {1, 1}};

    static const t_real __GaussLegendre_3[2][3]{
        {-0.774596669241483, 0, 0.774596669241483},
        {0.555555555555555, 0.888888888888889, 0.555555555555555}};

    static const t_real __GaussLegendre_4[2][4]{
        {-0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053},
        {0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454}};

    static const t_real __GaussLegendre_5[2][5]{
        {-0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664},
        {0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189}};

    static const t_real __GaussJacobi_01A2B0_1[2][1]{
        {0.250000000000000},
        {0.333333333333333}};

    static const t_real __GaussJacobi_01A2B0_2[2][2]{
        {0.122514822655441, 0.544151844011225},
        {0.232547451253508, 0.100785882079825}};

    static const t_real __GaussJacobi_01A2B0_3[2][3]{
        {0.072994024073150, 0.347003766038352, 0.705002209888499},
        {0.157136361064887, 0.146246269259866, 0.029950703008581}};

    static const t_real __GaussJacobi_01A2B0_4[2][4]{
        {0.048500549446997, 0.238600737551862, 0.517047295104367, 0.795851417896773},
        {0.110888415611278, 0.143458789799214, 0.068633887172923, 0.010352240749918}};

    static const t_real __GaussJacobi_01A2B0_5[2][5]{
        {0.034578939918215, 0.173480320771696, 0.389886387065519, 0.634333472630887, 0.851054212947016},
        {0.081764784285771, 0.126198961899911, 0.089200161221590, 0.032055600722962, 0.004113825203099}};

    static const t_real __HammerTri_1[3][1]{
        // order 1
        {1. / 3.},
        {1. / 3.},
        {1. / 2.},
    };

    static const t_real __HammerTri_3[3][3]{
        // order 2
        {2. / 3., 1. / 6., 1. / 6.},
        {1. / 6., 2. / 3., 1. / 6.},
        {1. / 6., 1. / 6., 1. / 6.}};

    // static const t_real __HammerTri_4[3][4]{ //
    //     {1. / 3., 0.6, 0.2, 0.2},
    //     {1. / 3., 0.2, 0.6, 0.2},
    //     {-27. / 96., 25. / 96., 25. / 96., 25. / 96.}};

    static const t_real __HammerTri_6[3][6]{
        // order 4
        {0.09157621350977102, 0.8168475729804585, 0.09157621350977052, 0.445948490915965, 0.10810301816807, 0.445948490915965},
        {0.09157621350977102, 0.09157621350977052, 0.8168475729804585, 0.445948490915965, 0.445948490915965, 0.10810301816807},
        {0.054975871827661, 0.054975871827661, 0.054975871827661, 0.1116907948390057, 0.1116907948390057, 0.1116907948390057}};

    static const t_real __HammerTri_7[3][7]{
        // order 5
        {0.3333333333333335, 0.470142064105115, 0.05971587178977, 0.470142064105115, 0.1012865073234565, 0.7974269853530875, 0.1012865073234565},
        {0.3333333333333335, 0.470142064105115, 0.470142064105115, 0.05971587178977, 0.1012865073234565, 0.1012865073234565, 0.7974269853530875},
        {0.1125, 0.066197076394253, 0.066197076394253, 0.066197076394253, 0.0629695902724135, 0.0629695902724135, 0.0629695902724135}};
    ;

    static const t_real __HammerTri_12[3][12]{
        // order 6
        {0.2492867451709105, 0.501426509658179, 0.2492867451709105, 0.063089014491502, 0.8738219710169954, 0.063089014491502, 0.3103524510337845, 0.05314504984481699, 0.6365024991213986, 0.05314504984481699, 0.6365024991213986, 0.3103524510337845},
        {0.2492867451709105, 0.2492867451709105, 0.501426509658179, 0.063089014491502, 0.063089014491502, 0.8738219710169954, 0.05314504984481699, 0.3103524510337845, 0.05314504984481699, 0.6365024991213986, 0.3103524510337845, 0.6365024991213986},
        {0.05839313786318975, 0.05839313786318975, 0.05839313786318975, 0.0254224531851035, 0.0254224531851035, 0.0254224531851035, 0.04142553780918675, 0.04142553780918675, 0.04142553780918675, 0.04142553780918675, 0.04142553780918675, 0.04142553780918675}};

    static const t_real __HammerTet_1[4][1]{
        // order 1
        {0.25},
        {0.25},
        {0.25},
        {0.1666666666666667}};

    static const t_real __HammerTet_4[4][4]{
        // order 2
        {0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105},
        {0.1381966011250105, 0.5854101966249684, 0.1381966011250105, 0.1381966011250105},
        {0.5854101966249684, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105},
        {0.04166666666666666, 0.04166666666666666, 0.04166666666666666, 0.04166666666666666}};

    static const t_real __HammerTet_8[4][8]{
        // order 3
        {0.1069940147705369, 0.1069940147705369, 0.6790179556883893, 0.1069940147705369, 0.3280585716625127, 0.3280585716625127, 0.01582428501246191, 0.3280585716625127},
        {0.1069940147705369, 0.6790179556883893, 0.1069940147705369, 0.1069940147705369, 0.3280585716625127, 0.01582428501246191, 0.3280585716625127, 0.3280585716625127},
        {0.6790179556883893, 0.1069940147705369, 0.1069940147705369, 0.1069940147705369, 0.01582428501246191, 0.3280585716625127, 0.3280585716625127, 0.3280585716625127},
        {0.01859314997209119, 0.01859314997209119, 0.01859314997209119, 0.01859314997209119, 0.02307351669457548, 0.02307351669457548, 0.02307351669457548, 0.02307351669457548}};

    static const t_real __HammerTet_14[4][14]{
        // order 5
        {0.09273525031089125, 0.09273525031089125, 0.7217942490673263, 0.09273525031089125, 0.3108859192633008, 0.3108859192633008, 0.06734224221009777, 0.3108859192633008, 0.04550370412564952, 0.4544962958743505, 0.04550370412564952, 0.04550370412564952, 0.4544962958743505, 0.4544962958743505},
        {0.09273525031089125, 0.7217942490673263, 0.09273525031089125, 0.09273525031089125, 0.3108859192633008, 0.06734224221009777, 0.3108859192633008, 0.3108859192633008, 0.4544962958743505, 0.04550370412564952, 0.04550370412564952, 0.4544962958743505, 0.04550370412564952, 0.4544962958743505},
        {0.7217942490673263, 0.09273525031089125, 0.09273525031089125, 0.09273525031089125, 0.06734224221009777, 0.3108859192633008, 0.3108859192633008, 0.3108859192633008, 0.4544962958743505, 0.4544962958743505, 0.4544962958743505, 0.04550370412564952, 0.04550370412564952, 0.04550370412564952},
        {0.01224884051939367, 0.01224884051939367, 0.01224884051939367, 0.01224884051939367, 0.01878132095300264, 0.01878132095300264, 0.01878132095300264, 0.01878132095300264, 0.0070910034628469, 0.0070910034628469, 0.0070910034628469, 0.0070910034628469, 0.0070910034628469, 0.0070910034628469}};

    static const t_real __HammerTet_24[4][24]{
        // order 6
        {0.2146028712591523, 0.2146028712591523, 0.3561913862225431, 0.2146028712591523, 0.3223378901422754, 0.3223378901422754, 0.0329863295731736, 0.3223378901422754, 0.04067395853461131, 0.04067395853461131, 0.8779781243961661, 0.04067395853461131, 0.6030056647916491, 0.6030056647916491, 0.06366100187501755, 0.2696723314583158, 0.06366100187501755, 0.06366100187501755, 0.2696723314583158, 0.06366100187501755, 0.06366100187501755, 0.06366100187501755, 0.2696723314583158, 0.6030056647916491},
        {0.2146028712591523, 0.3561913862225431, 0.2146028712591523, 0.2146028712591523, 0.3223378901422754, 0.0329863295731736, 0.3223378901422754, 0.3223378901422754, 0.04067395853461131, 0.8779781243961661, 0.04067395853461131, 0.04067395853461131, 0.06366100187501755, 0.06366100187501755, 0.06366100187501755, 0.6030056647916491, 0.2696723314583158, 0.6030056647916491, 0.06366100187501755, 0.2696723314583158, 0.06366100187501755, 0.6030056647916491, 0.06366100187501755, 0.2696723314583158},
        {0.3561913862225431, 0.2146028712591523, 0.2146028712591523, 0.2146028712591523, 0.0329863295731736, 0.3223378901422754, 0.3223378901422754, 0.3223378901422754, 0.8779781243961661, 0.04067395853461131, 0.04067395853461131, 0.04067395853461131, 0.2696723314583158, 0.06366100187501755, 0.6030056647916491, 0.06366100187501755, 0.6030056647916491, 0.06366100187501755, 0.6030056647916491, 0.06366100187501755, 0.2696723314583158, 0.2696723314583158, 0.06366100187501755, 0.06366100187501755},
        {0.006653791709694545, 0.006653791709694545, 0.006653791709694545, 0.006653791709694545, 0.009226196923942479, 0.009226196923942479, 0.009226196923942479, 0.009226196923942479, 0.001679535175886773, 0.001679535175886773, 0.001679535175886773, 0.001679535175886773, 0.008035714285714285, 0.008035714285714285, 0.008035714285714285, 0.008035714285714285, 0.008035714285714285, 0.008035714285714285, 0.008035714285714285, 0.008035714285714285, 0.008035714285714285, 0.008035714285714285, 0.008035714285714285, 0.008035714285714285}};

    inline constexpr t_index GetQuadratureScheme(ParamSpace ps, int int_order)
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

    /**
     *
     * @warning pParam should be initialized (with 0)
     */
    template <class TPoint>
    inline void GetQuadraturePoint(ParamSpace ps, t_index scheme, int iG, TPoint &pParam, t_real &w)
    {
        int scheme_size = scheme;
        DNDS_assert(iG < scheme_size);
        if (ps == LineSpace)
        {
            switch (scheme)
            {
            case INT_SCHEME_Line_1:
            {
                pParam[0] = __GaussLegendre_1[0][iG];
                w = __GaussLegendre_1[1][iG];
                return;
            }
            case INT_SCHEME_Line_2:
            {
                pParam[0] = __GaussLegendre_2[0][iG];
                w = __GaussLegendre_2[1][iG];
                return;
            }
            case INT_SCHEME_Line_3:
            {
                pParam[0] = __GaussLegendre_3[0][iG];
                w = __GaussLegendre_3[1][iG];
                return;
            }
            case INT_SCHEME_Line_4:
            {
                pParam[0] = __GaussLegendre_4[0][iG];
                w = __GaussLegendre_4[1][iG];
                return;
            }
            default:
                w = 1e100;
            }
        }
        if (ps == TriSpace)
        {
            switch (scheme)
            {
            case INT_SCHEME_Tri_1:
            {
                pParam[0] = __HammerTri_1[0][iG];
                pParam[1] = __HammerTri_1[1][iG];
                w = __HammerTri_1[2][iG];
                return;
            }
            case INT_SCHEME_Tri_3:
            {
                pParam[0] = __HammerTri_3[0][iG];
                pParam[1] = __HammerTri_3[1][iG];
                w = __HammerTri_3[2][iG];
                return;
            }
            case INT_SCHEME_Tri_6:
            {
                pParam[0] = __HammerTri_6[0][iG];
                pParam[1] = __HammerTri_6[1][iG];
                w = __HammerTri_6[2][iG];
                return;
            }
            case INT_SCHEME_Tri_7:
            {
                pParam[0] = __HammerTri_7[0][iG];
                pParam[1] = __HammerTri_7[1][iG];
                w = __HammerTri_7[2][iG];
                return;
            }
            case INT_SCHEME_Tri_12:
            {
                pParam[0] = __HammerTri_12[0][iG];
                pParam[1] = __HammerTri_12[1][iG];
                w = __HammerTri_12[2][iG];
                return;
            }
            default:
                w = 1e100;
            }
        }

        if (ps == TetSpace)
        {
            switch (scheme)
            {
            case INT_SCHEME_Tet_1:
            {
                pParam[0] = __HammerTet_1[0][iG];
                pParam[1] = __HammerTet_1[1][iG];
                pParam[2] = __HammerTet_1[2][iG];
                w = __HammerTet_1[3][iG];
                return;
            }
            case INT_SCHEME_Tet_4:
            {
                pParam[0] = __HammerTet_4[0][iG];
                pParam[1] = __HammerTet_4[1][iG];
                pParam[2] = __HammerTet_4[2][iG];
                w = __HammerTet_4[3][iG];
                return;
            }
            case INT_SCHEME_Tet_8:
            {
                pParam[0] = __HammerTet_8[0][iG];
                pParam[1] = __HammerTet_8[1][iG];
                pParam[2] = __HammerTet_8[2][iG];
                w = __HammerTet_8[3][iG];
                return;
            }
            case INT_SCHEME_Tet_14:
            {
                pParam[0] = __HammerTet_14[0][iG];
                pParam[1] = __HammerTet_14[1][iG];
                pParam[2] = __HammerTet_14[2][iG];
                w = __HammerTet_14[3][iG];
                return;
            }
            case INT_SCHEME_Tet_24:
            {
                pParam[0] = __HammerTet_24[0][iG];
                pParam[1] = __HammerTet_24[1][iG];
                pParam[2] = __HammerTet_24[2][iG];
                w = __HammerTet_24[3][iG];
                return;
            }
            default:
                w = 1e100;
            }
        }

        if (ps == QuadSpace)
        {
            const t_real *GLData = nullptr;
            int GLSize = 0;
            switch (scheme)
            {
            case INT_SCHEME_Quad_1:
            {
                GLData = &(__GaussLegendre_1[0][0]);
                GLSize = 1;
                break;
            }
            case INT_SCHEME_Quad_4:
            {
                GLData = &(__GaussLegendre_2[0][0]);
                GLSize = 2;
                break;
            }
            case INT_SCHEME_Quad_9:
            {
                GLData = &(__GaussLegendre_3[0][0]);
                GLSize = 3;
                break;
            }
            case INT_SCHEME_Quad_16:
            {
                GLData = &(__GaussLegendre_4[0][0]);
                GLSize = 4;
                break;
            }
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

        if (ps == HexSpace)
        {
            const t_real *GLData = nullptr;
            int GLSize = 0;
            switch (scheme)
            {
            case INT_SCHEME_Hex_1:
            {
                GLData = &(__GaussLegendre_1[0][0]);
                GLSize = 1;
                break;
            }
            case INT_SCHEME_Hex_8:
            {
                GLData = &(__GaussLegendre_2[0][0]);
                GLSize = 2;
                break;
            }
            case INT_SCHEME_Hex_27:
            {
                GLData = &(__GaussLegendre_3[0][0]);
                GLSize = 3;
                break;
            }
            case INT_SCHEME_Hex_64:
            {
                GLData = &(__GaussLegendre_4[0][0]);
                GLSize = 4;
                break;
            }
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

        if (ps == PyramidSpace)
        {
            const t_real *GLData = nullptr;
            const t_real *GJData = nullptr;
            int GLSize = 0; // == GJSize
            switch (scheme)
            {
            case INT_SCHEME_Pyramid_1:
            {
                GLData = &(__GaussLegendre_1[0][0]);
                GJData = &(__GaussJacobi_01A2B0_1[0][0]);
                GLSize = 1;
                break;
            }
            case INT_SCHEME_Pyramid_8:
            {
                GLData = &(__GaussLegendre_2[0][0]);
                GJData = &(__GaussJacobi_01A2B0_2[0][0]);
                GLSize = 2;
                break;
            }
            case INT_SCHEME_Pyramid_27:
            {
                GLData = &(__GaussLegendre_3[0][0]);
                GJData = &(__GaussJacobi_01A2B0_3[0][0]);
                GLSize = 3;
                break;
            }
            case INT_SCHEME_Pyramid_64:
            {
                GLData = &(__GaussLegendre_4[0][0]);
                GJData = &(__GaussJacobi_01A2B0_4[0][0]);
                GLSize = 4;
                break;
            }
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

        if (ps == PrismSpace)
        {
            const t_real *GLData = nullptr;
            const t_real *HammerData = nullptr;
            int GLSize = 0;
            int HammerSize = 0;
            switch (scheme)
            {
            case INT_SCHEME_Prism_1:
            {
                GLData = &(__GaussLegendre_1[0][0]);
                GLSize = 1;
                HammerData = &(__HammerTri_1[0][0]);
                HammerSize = 1;
                break;
            }
            case INT_SCHEME_Prism_6:
            {
                GLData = &(__GaussLegendre_2[0][0]);
                GLSize = 2;
                HammerData = &(__HammerTri_3[0][0]);
                HammerSize = 3;
                break;
            }
            case INT_SCHEME_Prism_18:
            {
                GLData = &(__GaussLegendre_3[0][0]);
                GLSize = 3;
                HammerData = &(__HammerTri_6[0][0]);
                HammerSize = 6;
                break;
            }
            case INT_SCHEME_Prism_21:
            {
                GLData = &(__GaussLegendre_3[0][0]);
                GLSize = 3;
                HammerData = &(__HammerTri_7[0][0]);
                HammerSize = 7;
                break;
            }
            case INT_SCHEME_Prism_48:
            {
                GLData = &(__GaussLegendre_4[0][0]);
                GLSize = 4;
                HammerData = &(__HammerTri_12[0][0]);
                HammerSize = 12;
                break;
            }
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

}

namespace DNDS::Geom::Elem
{
    class SummationNoOp
    {
    public:
        void operator+=(const SummationNoOp &R)
        {
            return;
        }

        SummationNoOp operator*(const SummationNoOp &R) const
        {
            return SummationNoOp();
        }

        friend SummationNoOp operator*(t_real L, const SummationNoOp &R)
        {
            return SummationNoOp();
        }

        friend SummationNoOp operator*(const SummationNoOp &L, t_real R)
        {
            return SummationNoOp();
        }
    };

    static struct __TNBufferAtQuadrature
    {
        std::array<std::array<std::vector<tD01Nj>, INT_ORDER_MAX + 1>, ElemType_NUM> buf;

        __TNBufferAtQuadrature()
        {
            for (t_index i = 1; i < ElemType_NUM; i++)
            {
                Element c_elem{ElemType(i)};
                for (int order = 0; order <= INT_ORDER_MAX; order++)
                {
                    auto int_scheme = GetQuadratureScheme(c_elem.GetParamSpace(), order);
                    buf.at(i).at(order).resize(int_scheme);
                    for (auto &m : buf.at(i).at(order))
                        m.resize(4, c_elem.GetNumNodes());
                    for (int iG = 0; iG < int_scheme; iG++)
                    {
                        tPoint pParam{0, 0, 0};
                        t_real w;
                        GetQuadraturePoint(c_elem.GetParamSpace(), int_scheme, iG, pParam, w);
                        c_elem.GetD01Nj(pParam, buf.at(i).at(order).at(iG));
                    }
                }
            }
        }

    } __NBufferAtQuadrature{};

    struct Quadrature
    {

        Element elem;
        int int_order;
        ParamSpace ps = UnknownPSpace;
        t_index int_scheme = 0;

        Quadrature(Element n_elem, int n_int_order)
            : elem(n_elem), int_order(n_int_order), ps(elem.GetParamSpace())
        {
            int_scheme = GetQuadratureScheme(ps, int_order);
        }

        /**
         * @param f  f(TAcc& inc, int iG, tPoint pParam, tD01Nj D01Nj)
         */
        template <class TAcc, class TFunc>
        void Integration(TAcc &buf, TFunc &&f)
        {
            for (t_index iG = 0; iG < int_scheme; iG++)
            {
                tPoint pParam{0, 0, 0};
                t_real w;
                GetQuadraturePoint(ps, int_scheme, iG, pParam, w);
                TAcc acc;
                f(acc, iG, pParam, __NBufferAtQuadrature.buf.at(elem.type).at(int_order).at(iG));
                buf += acc * w;
            }
        }

        /**
         * @param f  f(TAcc& inc, int iG)
         */
        template <class TAcc, class TFunc>
        void IntegrationSimple(TAcc &buf, TFunc &&f)
        {
            for (t_index iG = 0; iG < int_scheme; iG++)
            {
                tPoint pParam{0, 0, 0};
                t_real w;
                GetQuadraturePoint(ps, int_scheme, iG, pParam, w);
                TAcc acc;
                f(acc, iG);
                buf += acc * w;
            }
        }

        t_index GetNumPoints() { return int_scheme; }
    };
}
