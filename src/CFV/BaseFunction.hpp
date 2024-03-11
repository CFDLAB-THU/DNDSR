#pragma once

#include "DNDS/Defines.hpp"
#include "Geom/EigenTensor.hpp"
#include "Geom/PeriodicInfo.hpp"

namespace DNDS::CFV
{
    /// including up to 3 orders or diffs
    static const int ndiff = 3;
    static const int ndiffSiz = 20;
    static const int ndiffSiz2D = 10;
    static const int diffOperatorOrderList[ndiffSiz][3] =
        {
            //{diffOrderX_0, diffOrderX_1, diffOrder_X2} // indexPlace, diffSeq
            {0, 0, 0}, // 00
            {1, 0, 0}, // 01
            {0, 1, 0}, // 02
            {0, 0, 1}, // 03
            {2, 0, 0}, // 04  00
            {0, 2, 0}, // 05  11
            {0, 0, 2}, // 06  22
            {1, 1, 0}, // 07  01
            {0, 1, 1}, // 08  12
            {1, 0, 1}, // 09  02
            {3, 0, 0}, // 10  000
            {0, 3, 0}, // 11  111
            {0, 0, 3}, // 12  222
            {2, 1, 0}, // 13  001
            {1, 2, 0}, // 14  011
            {0, 2, 1}, // 15  112
            {0, 1, 2}, // 16  122
            {1, 0, 2}, // 17  022
            {2, 0, 1}, // 18  002
            {1, 1, 1}, // 19  012
    };

    static const int diffOperatorOrderList2D[ndiffSiz2D][3] =
        {
            {0, 0, 0}, // 00 00
            {1, 0, 0}, // 01 01 0
            {0, 1, 0}, // 02 02 1
            {2, 0, 0}, // 03 04 00
            {1, 1, 0}, // 04 05 01
            {0, 2, 0}, // 05 07 11
            {3, 0, 0}, // 06 10 000
            {2, 1, 0}, // 07 11 001
            {1, 2, 0}, // 08 13 011
            {0, 3, 0}, // 09 16 111
    };
    static const int dFactorials[ndiff + 1][ndiff + 1] = {
        {1, 0, 0, 0},
        {1, 1, 0, 0},
        {1, 2, 2, 0},
        {1, 3, 6, 6}};

    static const int factorials[ndiff * 3 + 1] = {
        1,
        1,
        1 * 2,
        1 * 2 * 3,
        1 * 2 * 3 * 4,
        1 * 2 * 3 * 4 * 5,
        1 * 2 * 3 * 4 * 5 * 6,
        1 * 2 * 3 * 4 * 5 * 6 * 7,
        1 * 2 * 3 * 4 * 5 * 6 * 7 * 8,
        1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9,
    };
    static const int diffNCombs2D[ndiffSiz2D]{
        1, 1, 1, 1, 2, 1, 1, 3, 3, 1};

    static const int diffNCombs[ndiffSiz]{
        1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 3, 3, 3, 3, 3, 3, 6};

    inline real iPow(int p, real x)
    {
        switch (p)
        {
        case 0:
            return 1.;
        case 1:
            return x;
        case 2:
            return x * x;
        case 3:
            return x * x * x;
        default:
            return 1e300;
            break;
        }
    }

    template <int dim, int order>
    inline constexpr int PolynomialNDOF() //  2-d specific
    {
        switch (dim)
        {
        case 2:
        {
            switch (order)
            {
            case 0:
                return 1;
            case 1:
                return 3;
            case 2:
                return 6;
            case 3:
                return 10;
            default:
                return -1;
            }
        }
        break;
        case 3:
        {
            switch (order)
            {
            case 0:
                return 1;
            case 1:
                return 4;
            case 2:
                return 10;
            case 3:
                return 20;
            default:
                return -1;
            }
        }
        break;
        default:
            return -1;
        }
    }

    inline static real FPolynomial3D(int px, int py, int pz, int dx, int dy, int dz, real x, real y, real z)
    {
        int c = dFactorials[px][dx] * dFactorials[py][dy] * dFactorials[pz][dz];
        return c ? c * iPow(px - dx, x) * iPow(py - dy, y) * iPow(pz - dz, z) : 0.;
        // return c ? c * std::pow(x, px - dx) * std::pow(y, py - dy) * std::pow(z, pz - dz) : 0.;
    }

    template <class TDIBJ>
    void FPolynomialFill2D(TDIBJ &T, real x, real y, real z, real lx, real ly, real lz, int rows, int cols)
    {
        T.setZero();
        if (rows == 10 && cols == 10)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = x * x;
            T(0, 4) = x * y;
            T(0, 5) = y * y;
            T(0, 6) = x * x * x;
            T(0, 7) = (x * x) * y;
            T(0, 8) = x * (y * y);
            T(0, 9) = y * y * y;
            T(1, 1) = 1.0 / lx;
            T(1, 3) = (x * 2.0) / lx;
            T(1, 4) = y / lx;
            T(1, 6) = ((x * x) * 3.0) / lx;
            T(1, 7) = (x * y * 2.0) / lx;
            T(1, 8) = (y * y) / lx;
            T(2, 2) = 1.0 / ly;
            T(2, 4) = x / ly;
            T(2, 5) = (y * 2.0) / ly;
            T(2, 7) = (x * x) / ly;
            T(2, 8) = (x * y * 2.0) / ly;
            T(2, 9) = ((y * y) * 3.0) / ly;
            T(3, 3) = 1.0 / (lx * lx) * 2.0;
            T(3, 6) = 1.0 / (lx * lx) * x * 6.0;
            T(3, 7) = 1.0 / (lx * lx) * y * 2.0;
            T(4, 4) = 1.0 / (lx * ly);
            T(4, 7) = (x * 2.0) / (lx * ly);
            T(4, 8) = (y * 2.0) / (lx * ly);
            T(5, 5) = 1.0 / (ly * ly) * 2.0;
            T(5, 8) = 1.0 / (ly * ly) * x * 2.0;
            T(5, 9) = 1.0 / (ly * ly) * y * 6.0;
            T(6, 6) = 1.0 / (lx * lx * lx) * 6.0;
            T(7, 7) = (1.0 / (lx * lx) * 2.0) / ly;
            T(8, 8) = (1.0 / (ly * ly) * 2.0) / lx;
            T(9, 9) = 1.0 / (ly * ly * ly) * 6.0;
        }
        else if (rows == 3 && cols == 10)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = x * x;
            T(0, 4) = x * y;
            T(0, 5) = y * y;
            T(0, 6) = x * x * x;
            T(0, 7) = (x * x) * y;
            T(0, 8) = x * (y * y);
            T(0, 9) = y * y * y;
            T(1, 1) = 1.0 / lx;
            T(1, 3) = (x * 2.0) / lx;
            T(1, 4) = y / lx;
            T(1, 6) = ((x * x) * 3.0) / lx;
            T(1, 7) = (x * y * 2.0) / lx;
            T(1, 8) = (y * y) / lx;
            T(2, 2) = 1.0 / ly;
            T(2, 4) = x / ly;
            T(2, 5) = (y * 2.0) / ly;
            T(2, 7) = (x * x) / ly;
            T(2, 8) = (x * y * 2.0) / ly;
            T(2, 9) = ((y * y) * 3.0) / ly;
        }
        else if (rows == 1 && cols == 10)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = x * x;
            T(0, 4) = x * y;
            T(0, 5) = y * y;
            T(0, 6) = x * x * x;
            T(0, 7) = (x * x) * y;
            T(0, 8) = x * (y * y);
            T(0, 9) = y * y * y;
        }
        else if (rows == 6 && cols == 6)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = x * x;
            T(0, 4) = x * y;
            T(0, 5) = y * y;
            T(1, 1) = 1.0 / lx;
            T(1, 3) = (x * 2.0) / lx;
            T(1, 4) = y / lx;
            T(2, 2) = 1.0 / ly;
            T(2, 4) = x / ly;
            T(2, 5) = (y * 2.0) / ly;
            T(3, 3) = 1.0 / (lx * lx) * 2.0;
            T(4, 4) = 1.0 / (lx * ly);
            T(5, 5) = 1.0 / (ly * ly) * 2.0;
        }
        else if (rows == 3 && cols == 6)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = x * x;
            T(0, 4) = x * y;
            T(0, 5) = y * y;
            T(1, 1) = 1.0 / lx;
            T(1, 3) = (x * 2.0) / lx;
            T(1, 4) = y / lx;
            T(2, 2) = 1.0 / ly;
            T(2, 4) = x / ly;
            T(2, 5) = (y * 2.0) / ly;
        }
        else if (rows == 1 && cols == 6)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = x * x;
            T(0, 4) = x * y;
            T(0, 5) = y * y;
        }
        else if (rows == 3 && cols == 3)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(1, 1) = 1.0 / lx;
            T(2, 2) = 1.0 / ly;
        }
        else if (rows == 1 && cols == 3)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
        }
        else
        {
            for (int idiff = 0; idiff < rows; idiff++)
                for (int ibase = 0; ibase < cols; ibase++)
                {

                    int px = diffOperatorOrderList2D[ibase][0];
                    int py = diffOperatorOrderList2D[ibase][1];
                    int pz = diffOperatorOrderList2D[ibase][2];
                    int ndx = diffOperatorOrderList2D[idiff][0];
                    int ndy = diffOperatorOrderList2D[idiff][1];
                    int ndz = diffOperatorOrderList2D[idiff][2];
                    T(idiff, ibase) =
                        FPolynomial3D(px, py, pz, ndx, ndy, ndz,
                                      x, y, z / 1.) /
                        (iPow(ndx, lx) * iPow(ndy, ly) * iPow(ndz, 1.0));
                }
        }
    }

    template <class TDIBJ>
    void FPolynomialFill3D(TDIBJ &T, real x, real y, real z, real lx, real ly, real lz, int rows, int cols)
    {
        T.setZero();
        if (rows == 20 && cols == 20)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = z;
            T(0, 4) = x * x;
            T(0, 5) = y * y;
            T(0, 6) = z * z;
            T(0, 7) = x * y;
            T(0, 8) = y * z;
            T(0, 9) = x * z;
            T(0, 10) = x * x * x;
            T(0, 11) = y * y * y;
            T(0, 12) = z * z * z;
            T(0, 13) = (x * x) * y;
            T(0, 14) = x * (y * y);
            T(0, 15) = (y * y) * z;
            T(0, 16) = y * (z * z);
            T(0, 17) = x * (z * z);
            T(0, 18) = (x * x) * z;
            T(0, 19) = x * y * z;
            T(1, 1) = 1.0 / lx;
            T(1, 4) = (x * 2.0) / lx;
            T(1, 7) = y / lx;
            T(1, 9) = z / lx;
            T(1, 10) = ((x * x) * 3.0) / lx;
            T(1, 13) = (x * y * 2.0) / lx;
            T(1, 14) = (y * y) / lx;
            T(1, 17) = (z * z) / lx;
            T(1, 18) = (x * z * 2.0) / lx;
            T(1, 19) = (y * z) / lx;
            T(2, 2) = 1.0 / ly;
            T(2, 5) = (y * 2.0) / ly;
            T(2, 7) = x / ly;
            T(2, 8) = z / ly;
            T(2, 11) = ((y * y) * 3.0) / ly;
            T(2, 13) = (x * x) / ly;
            T(2, 14) = (x * y * 2.0) / ly;
            T(2, 15) = (y * z * 2.0) / ly;
            T(2, 16) = (z * z) / ly;
            T(2, 19) = (x * z) / ly;
            T(3, 3) = 1.0 / lz;
            T(3, 6) = (z * 2.0) / lz;
            T(3, 8) = y / lz;
            T(3, 9) = x / lz;
            T(3, 12) = ((z * z) * 3.0) / lz;
            T(3, 15) = (y * y) / lz;
            T(3, 16) = (y * z * 2.0) / lz;
            T(3, 17) = (x * z * 2.0) / lz;
            T(3, 18) = (x * x) / lz;
            T(3, 19) = (x * y) / lz;
            T(4, 4) = 1.0 / (lx * lx) * 2.0;
            T(4, 10) = 1.0 / (lx * lx) * x * 6.0;
            T(4, 13) = 1.0 / (lx * lx) * y * 2.0;
            T(4, 18) = 1.0 / (lx * lx) * z * 2.0;
            T(5, 5) = 1.0 / (ly * ly) * 2.0;
            T(5, 11) = 1.0 / (ly * ly) * y * 6.0;
            T(5, 14) = 1.0 / (ly * ly) * x * 2.0;
            T(5, 15) = 1.0 / (ly * ly) * z * 2.0;
            T(6, 6) = 1.0 / (lz * lz) * 2.0;
            T(6, 12) = 1.0 / (lz * lz) * z * 6.0;
            T(6, 16) = 1.0 / (lz * lz) * y * 2.0;
            T(6, 17) = 1.0 / (lz * lz) * x * 2.0;
            T(7, 7) = 1.0 / (lx * ly);
            T(7, 13) = (x * 2.0) / (lx * ly);
            T(7, 14) = (y * 2.0) / (lx * ly);
            T(7, 19) = z / (lx * ly);
            T(8, 8) = 1.0 / (ly * lz);
            T(8, 15) = (y * 2.0) / (ly * lz);
            T(8, 16) = (z * 2.0) / (ly * lz);
            T(8, 19) = x / (ly * lz);
            T(9, 9) = 1.0 / (lx * lz);
            T(9, 17) = (z * 2.0) / (lx * lz);
            T(9, 18) = (x * 2.0) / (lx * lz);
            T(9, 19) = y / (lx * lz);
            T(10, 10) = 1.0 / (lx * lx * lx) * 6.0;
            T(11, 11) = 1.0 / (ly * ly * ly) * 6.0;
            T(12, 12) = 1.0 / (lz * lz * lz) * 6.0;
            T(13, 13) = (1.0 / (lx * lx) * 2.0) / ly;
            T(14, 14) = (1.0 / (ly * ly) * 2.0) / lx;
            T(15, 15) = (1.0 / (ly * ly) * 2.0) / lz;
            T(16, 16) = (1.0 / (lz * lz) * 2.0) / ly;
            T(17, 17) = (1.0 / (lz * lz) * 2.0) / lx;
            T(18, 18) = (1.0 / (lx * lx) * 2.0) / lz;
            T(19, 19) = 1.0 / (lx * ly * lz);
        }
        else if (rows == 4 && cols == 20)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = z;
            T(0, 4) = x * x;
            T(0, 5) = y * y;
            T(0, 6) = z * z;
            T(0, 7) = x * y;
            T(0, 8) = y * z;
            T(0, 9) = x * z;
            T(0, 10) = x * x * x;
            T(0, 11) = y * y * y;
            T(0, 12) = z * z * z;
            T(0, 13) = (x * x) * y;
            T(0, 14) = x * (y * y);
            T(0, 15) = (y * y) * z;
            T(0, 16) = y * (z * z);
            T(0, 17) = x * (z * z);
            T(0, 18) = (x * x) * z;
            T(0, 19) = x * y * z;
            T(1, 1) = 1.0 / lx;
            T(1, 4) = (x * 2.0) / lx;
            T(1, 7) = y / lx;
            T(1, 9) = z / lx;
            T(1, 10) = ((x * x) * 3.0) / lx;
            T(1, 13) = (x * y * 2.0) / lx;
            T(1, 14) = (y * y) / lx;
            T(1, 17) = (z * z) / lx;
            T(1, 18) = (x * z * 2.0) / lx;
            T(1, 19) = (y * z) / lx;
            T(2, 2) = 1.0 / ly;
            T(2, 5) = (y * 2.0) / ly;
            T(2, 7) = x / ly;
            T(2, 8) = z / ly;
            T(2, 11) = ((y * y) * 3.0) / ly;
            T(2, 13) = (x * x) / ly;
            T(2, 14) = (x * y * 2.0) / ly;
            T(2, 15) = (y * z * 2.0) / ly;
            T(2, 16) = (z * z) / ly;
            T(2, 19) = (x * z) / ly;
            T(3, 3) = 1.0 / lz;
            T(3, 6) = (z * 2.0) / lz;
            T(3, 8) = y / lz;
            T(3, 9) = x / lz;
            T(3, 12) = ((z * z) * 3.0) / lz;
            T(3, 15) = (y * y) / lz;
            T(3, 16) = (y * z * 2.0) / lz;
            T(3, 17) = (x * z * 2.0) / lz;
            T(3, 18) = (x * x) / lz;
            T(3, 19) = (x * y) / lz;
        }
        else if (rows == 1 && cols == 20)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = z;
            T(0, 4) = x * x;
            T(0, 5) = y * y;
            T(0, 6) = z * z;
            T(0, 7) = x * y;
            T(0, 8) = y * z;
            T(0, 9) = x * z;
            T(0, 10) = x * x * x;
            T(0, 11) = y * y * y;
            T(0, 12) = z * z * z;
            T(0, 13) = (x * x) * y;
            T(0, 14) = x * (y * y);
            T(0, 15) = (y * y) * z;
            T(0, 16) = y * (z * z);
            T(0, 17) = x * (z * z);
            T(0, 18) = (x * x) * z;
            T(0, 19) = x * y * z;
        }
        else if (rows == 10 && cols == 10)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = z;
            T(0, 4) = x * x;
            T(0, 5) = y * y;
            T(0, 6) = z * z;
            T(0, 7) = x * y;
            T(0, 8) = y * z;
            T(0, 9) = x * z;
            T(1, 1) = 1.0 / lx;
            T(1, 4) = (x * 2.0) / lx;
            T(1, 7) = y / lx;
            T(1, 9) = z / lx;
            T(2, 2) = 1.0 / ly;
            T(2, 5) = (y * 2.0) / ly;
            T(2, 7) = x / ly;
            T(2, 8) = z / ly;
            T(3, 3) = 1.0 / lz;
            T(3, 6) = (z * 2.0) / lz;
            T(3, 8) = y / lz;
            T(3, 9) = x / lz;
            T(4, 4) = 1.0 / (lx * lx) * 2.0;
            T(5, 5) = 1.0 / (ly * ly) * 2.0;
            T(6, 6) = 1.0 / (lz * lz) * 2.0;
            T(7, 7) = 1.0 / (lx * ly);
            T(8, 8) = 1.0 / (ly * lz);
            T(9, 9) = 1.0 / (lx * lz);
        }
        else if (rows == 4 && cols == 10)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = z;
            T(0, 4) = x * x;
            T(0, 5) = y * y;
            T(0, 6) = z * z;
            T(0, 7) = x * y;
            T(0, 8) = y * z;
            T(0, 9) = x * z;
            T(1, 1) = 1.0 / lx;
            T(1, 4) = (x * 2.0) / lx;
            T(1, 7) = y / lx;
            T(1, 9) = z / lx;
            T(2, 2) = 1.0 / ly;
            T(2, 5) = (y * 2.0) / ly;
            T(2, 7) = x / ly;
            T(2, 8) = z / ly;
            T(3, 3) = 1.0 / lz;
            T(3, 6) = (z * 2.0) / lz;
            T(3, 8) = y / lz;
            T(3, 9) = x / lz;
        }
        else if (rows == 1 && cols == 10)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = z;
            T(0, 4) = x * x;
            T(0, 5) = y * y;
            T(0, 6) = z * z;
            T(0, 7) = x * y;
            T(0, 8) = y * z;
            T(0, 9) = x * z;
        }
        else if (rows == 4 && cols == 4)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = z;
            T(1, 1) = 1.0 / lx;
            T(2, 2) = 1.0 / ly;
            T(3, 3) = 1.0 / lz;
        }
        else if (rows == 1 && cols == 4)
        {
            T(0, 0) = 1.0;
            T(0, 1) = x;
            T(0, 2) = y;
            T(0, 3) = z;
        }
        else
        {
            for (int idiff = 0; idiff < rows; idiff++)
                for (int ibase = 0; ibase < cols; ibase++)
                {
                    int px = diffOperatorOrderList[ibase][0];
                    int py = diffOperatorOrderList[ibase][1];
                    int pz = diffOperatorOrderList[ibase][2];
                    int ndx = diffOperatorOrderList[idiff][0];
                    int ndy = diffOperatorOrderList[idiff][1];
                    int ndz = diffOperatorOrderList[idiff][2];
                    T(idiff, ibase) =
                        FPolynomial3D(px, py, pz, ndx, ndy, ndz,
                                      x, y, z) /
                        (iPow(ndx, lx) * iPow(ndy, ly) * iPow(ndz, lz));
                }
        }
    }

    template <int dim = 3, int rank = 0, int powV = 1, class VLe, class VRi>
    real NormSymDiffOrderTensorV(VLe &&Le, VRi &&Ri)
    {
        real ret = 0;
        if constexpr (dim == 3)
        {
            if constexpr (powV == 1)
            {
                if constexpr (rank == 0)
                {
                    ret += Le(0, 0) * Ri(0, 0) * diffNCombs[0];
                }
                else if constexpr (rank == 1)
                {
                    ret += Le(0, 0) * Ri(0, 0) * diffNCombs[0 + 1];
                    ret += Le(1, 0) * Ri(1, 0) * diffNCombs[1 + 1];
                    ret += Le(2, 0) * Ri(2, 0) * diffNCombs[2 + 1];
                }
                else if constexpr (rank == 2)
                {
                    ret += Le(0, 0) * Ri(0, 0) * diffNCombs[0 + 4];
                    ret += Le(1, 0) * Ri(1, 0) * diffNCombs[1 + 4];
                    ret += Le(2, 0) * Ri(2, 0) * diffNCombs[2 + 4];
                    ret += Le(3, 0) * Ri(3, 0) * diffNCombs[3 + 4];
                    ret += Le(4, 0) * Ri(4, 0) * diffNCombs[4 + 4];
                    ret += Le(5, 0) * Ri(5, 0) * diffNCombs[5 + 4];
                }
                else if constexpr (rank == 3)
                {
                    ret += Le(0, 0) * Ri(0, 0) * diffNCombs[0 + 10];
                    ret += Le(1, 0) * Ri(1, 0) * diffNCombs[1 + 10];
                    ret += Le(2, 0) * Ri(2, 0) * diffNCombs[2 + 10];
                    ret += Le(3, 0) * Ri(3, 0) * diffNCombs[3 + 10];
                    ret += Le(4, 0) * Ri(4, 0) * diffNCombs[4 + 10];
                    ret += Le(5, 0) * Ri(5, 0) * diffNCombs[5 + 10];
                    ret += Le(6, 0) * Ri(6, 0) * diffNCombs[6 + 10];
                    ret += Le(7, 0) * Ri(7, 0) * diffNCombs[7 + 10];
                    ret += Le(8, 0) * Ri(8, 0) * diffNCombs[8 + 10];
                    ret += Le(9, 0) * Ri(9, 0) * diffNCombs[9 + 10];
                }
                else
                {
                    DNDS_assert(false);
                }
            }
            else
            {
                if constexpr (rank == 0)
                {
                    ret += Le(0, 0) * Ri(0, 0) * std::pow(diffNCombs[0], powV);
                }
                else if constexpr (rank == 1)
                {
                    ret += Le(0, 0) * Ri(0, 0) * std::pow(diffNCombs[0 + 1], powV);
                    ret += Le(1, 0) * Ri(1, 0) * std::pow(diffNCombs[1 + 1], powV);
                    ret += Le(2, 0) * Ri(2, 0) * std::pow(diffNCombs[2 + 1], powV);
                }
                else if constexpr (rank == 2)
                {
                    ret += Le(0, 0) * Ri(0, 0) * std::pow(diffNCombs[0 + 4], powV);
                    ret += Le(1, 0) * Ri(1, 0) * std::pow(diffNCombs[1 + 4], powV);
                    ret += Le(2, 0) * Ri(2, 0) * std::pow(diffNCombs[2 + 4], powV);
                    ret += Le(3, 0) * Ri(3, 0) * std::pow(diffNCombs[3 + 4], powV);
                    ret += Le(4, 0) * Ri(4, 0) * std::pow(diffNCombs[4 + 4], powV);
                    ret += Le(5, 0) * Ri(5, 0) * std::pow(diffNCombs[5 + 4], powV);
                }
                else if constexpr (rank == 3)
                {
                    ret += Le(0, 0) * Ri(0, 0) * std::pow(diffNCombs[0 + 10], powV);
                    ret += Le(1, 0) * Ri(1, 0) * std::pow(diffNCombs[1 + 10], powV);
                    ret += Le(2, 0) * Ri(2, 0) * std::pow(diffNCombs[2 + 10], powV);
                    ret += Le(3, 0) * Ri(3, 0) * std::pow(diffNCombs[3 + 10], powV);
                    ret += Le(4, 0) * Ri(4, 0) * std::pow(diffNCombs[4 + 10], powV);
                    ret += Le(5, 0) * Ri(5, 0) * std::pow(diffNCombs[5 + 10], powV);
                    ret += Le(6, 0) * Ri(6, 0) * std::pow(diffNCombs[6 + 10], powV);
                    ret += Le(7, 0) * Ri(7, 0) * std::pow(diffNCombs[7 + 10], powV);
                    ret += Le(8, 0) * Ri(8, 0) * std::pow(diffNCombs[8 + 10], powV);
                    ret += Le(9, 0) * Ri(9, 0) * std::pow(diffNCombs[9 + 10], powV);
                }
                else
                {
                    DNDS_assert(false);
                }
            }
        }
        else
        {
            if constexpr (powV == 1)
            {
                if constexpr (rank == 0)
                {
                    ret += Le(0) * Ri(0) * diffNCombs2D[0];
                }
                else if constexpr (rank == 1)
                {
                    ret += Le(0) * Ri(0) * diffNCombs2D[0 + 1];
                    ret += Le(1) * Ri(1) * diffNCombs2D[1 + 1];
                }
                else if constexpr (rank == 2)
                {
                    ret += Le(0) * Ri(0) * diffNCombs2D[0 + 3];
                    ret += Le(1) * Ri(1) * diffNCombs2D[1 + 3];
                    ret += Le(2) * Ri(2) * diffNCombs2D[2 + 3];
                }
                else if constexpr (rank == 3)
                {
                    ret += Le(0) * Ri(0) * diffNCombs2D[0 + 6];
                    ret += Le(1) * Ri(1) * diffNCombs2D[1 + 6];
                    ret += Le(2) * Ri(2) * diffNCombs2D[2 + 6];
                    ret += Le(3) * Ri(3) * diffNCombs2D[3 + 6];
                }
                else
                {
                    DNDS_assert(false);
                }
            }
            else
            {
                if constexpr (rank == 0)
                {
                    ret += Le(0) * Ri(0) * std::pow(diffNCombs2D[0], powV);
                }
                else if constexpr (rank == 1)
                {
                    ret += Le(0) * Ri(0) * std::pow(diffNCombs2D[0 + 1], powV);
                    ret += Le(1) * Ri(1) * std::pow(diffNCombs2D[1 + 1], powV);
                }
                else if constexpr (rank == 2)
                {
                    ret += Le(0) * Ri(0) * std::pow(diffNCombs2D[0 + 3], powV);
                    ret += Le(1) * Ri(1) * std::pow(diffNCombs2D[1 + 3], powV);
                    ret += Le(2) * Ri(2) * std::pow(diffNCombs2D[2 + 3], powV);
                }
                else if constexpr (rank == 3)
                {
                    ret += Le(0) * Ri(0) * std::pow(diffNCombs2D[0 + 6], powV);
                    ret += Le(1) * Ri(1) * std::pow(diffNCombs2D[1 + 6], powV);
                    ret += Le(2) * Ri(2) * std::pow(diffNCombs2D[2 + 6], powV);
                    ret += Le(3) * Ri(3) * std::pow(diffNCombs2D[3 + 6], powV);
                }
                else
                {
                    DNDS_assert(false);
                }
            }
        }
        return ret;
    }

    // #include <unsupported/Eigen/CXX11/TensorSymmetry>

    template <int dim, int rank, class VLe, class Trans>
    void TransSymDiffOrderTensorV(VLe &&Le, Trans &&trans)
    {
        if constexpr (dim == 3)
        {
            if constexpr (rank == 0)
            {
            }
            else if constexpr (rank == 1)
            {
                Le = trans * Le;
            }
            else if constexpr (rank == 2)
            {
                /*
                00
                11
                22
                01
                12
                02*/
                Eigen::Matrix3d symTensorR2{
                    {Le(0), Le(3), Le(5)},
                    {Le(3), Le(1), Le(4)},
                    {Le(5), Le(4), Le(2)}};
                symTensorR2 = trans * symTensorR2 * trans.transpose();
                Le(0) = symTensorR2(0, 0), Le(1) = symTensorR2(1, 1), Le(2) = symTensorR2(2, 2);
                Le(3) = symTensorR2(0, 1), Le(4) = symTensorR2(1, 2), Le(5) = symTensorR2(0, 2);
            }
            else if constexpr (rank == 3)
            {

                /*
                000
                111
                222
                001
                011
                112
                122
                022
                002
                012*/
                DNDS::ETensor::ETensorR3<real, 3, 3, 3> symTensorR3;
                symTensorR3(0, 0, 0) = Le(0);
                symTensorR3(1, 1, 1) = Le(1);
                symTensorR3(2, 2, 2) = Le(2);

                symTensorR3(0, 0, 1) = symTensorR3(0, 1, 0) = symTensorR3(1, 0, 0) = Le(3);
                symTensorR3(0, 1, 1) = symTensorR3(1, 1, 0) = symTensorR3(1, 0, 1) = Le(4);
                symTensorR3(1, 1, 2) = symTensorR3(1, 2, 1) = symTensorR3(2, 1, 1) = Le(5);
                symTensorR3(1, 2, 2) = symTensorR3(2, 2, 1) = symTensorR3(2, 1, 2) = Le(6);
                symTensorR3(0, 2, 2) = symTensorR3(2, 2, 0) = symTensorR3(2, 0, 2) = Le(7);
                symTensorR3(0, 0, 2) = symTensorR3(0, 2, 0) = symTensorR3(2, 0, 0) = Le(8);

                symTensorR3(0, 1, 2) = symTensorR3(1, 2, 0) = symTensorR3(2, 0, 1) =
                    symTensorR3(0, 2, 1) = symTensorR3(2, 1, 0) = symTensorR3(1, 0, 2) = Le(9);

                symTensorR3.MatTransform0(trans.transpose());
                symTensorR3.MatTransform1(trans.transpose());
                symTensorR3.MatTransform2(trans.transpose());
                Le(0) = symTensorR3(0, 0, 0);
                Le(1) = symTensorR3(1, 1, 1);
                Le(2) = symTensorR3(2, 2, 2);
                Le(3) = symTensorR3(0, 0, 1);
                Le(4) = symTensorR3(0, 1, 1);
                Le(5) = symTensorR3(1, 1, 2);
                Le(6) = symTensorR3(1, 2, 2);
                Le(7) = symTensorR3(0, 2, 2);
                Le(8) = symTensorR3(0, 0, 2);
                Le(9) = symTensorR3(0, 1, 2);
            }
            else
            {
                DNDS_assert(false);
            }
        }
        else // 2-d tensor
        {
            if constexpr (rank == 0)
            {
            }
            else if constexpr (rank == 1)
            {
                Le = trans * Le;
            }
            else if constexpr (rank == 2)
            {
                Eigen::Matrix2d symTensorR2{{Le(0), Le(1)},
                                            {Le(1), Le(2)}};
                symTensorR2 = trans * symTensorR2 * trans.transpose();
                Le(0) = symTensorR2(0, 0), Le(1) = symTensorR2(0, 1), Le(2) = symTensorR2(1, 1);
            }
            else if constexpr (rank == 3)
            {

                DNDS::ETensor::ETensorR3<real, 2, 2, 2> symTensorR3;
                symTensorR3(0, 0, 0) = Le(0);
                symTensorR3(0, 0, 1) = symTensorR3(0, 1, 0) = symTensorR3(1, 0, 0) = Le(1);
                symTensorR3(0, 1, 1) = symTensorR3(1, 0, 1) = symTensorR3(1, 1, 0) = Le(2);
                symTensorR3(1, 1, 1) = Le(3);
                symTensorR3.MatTransform0(trans.transpose());
                symTensorR3.MatTransform1(trans.transpose());
                symTensorR3.MatTransform2(trans.transpose());
                Le(0) = symTensorR3(0, 0, 0);
                Le(1) = symTensorR3(0, 0, 1);
                Le(2) = symTensorR3(0, 1, 1);
                Le(3) = symTensorR3(1, 1, 1);
            }
            else
            {
                DNDS_assert(false);
            }
        }
    }

    template <int dim, class TMat>
    inline void ConvertDiffsLinMap(TMat &&mat, const Geom::tGPoint &dXijdXi)
    {
        int rows = mat.rows();
        if constexpr (dim == 2)
            switch (rows)
            {
            case 10:
                for (int iB = 0; iB < mat.cols(); iB++)
                {
                    TransSymDiffOrderTensorV<2, 3>(
                        mat(Eigen::seq(Eigen::fix<6>, Eigen::fix<9>), iB),
                        dXijdXi({0, 1}, {0, 1}));
                }
            case 6:
                for (int iB = 0; iB < mat.cols(); iB++)
                    TransSymDiffOrderTensorV<2, 2>(
                        mat(Eigen::seq(Eigen::fix<3>, Eigen::fix<5>), iB),
                        dXijdXi({0, 1}, {0, 1}));

            case 3:
                mat({1, 2}, Eigen::all) = dXijdXi({0, 1}, {0, 1}) * mat({1, 2}, Eigen::all);
            case 1:
                break;

            default:
                std::cerr << mat.rows() << std::endl;
                DNDS_assert(false);
                break;
            }
        else // dim ==3
            switch (rows)
            {
            case 20:
                for (int iB = 0; iB < mat.cols(); iB++)
                {
                    TransSymDiffOrderTensorV<3, 3>(
                        mat(Eigen::seq(Eigen::fix<10>, Eigen::fix<19>), iB),
                        dXijdXi);
                }
            case 10:
                for (int iB = 0; iB < mat.cols(); iB++)
                    TransSymDiffOrderTensorV<3, 2>(
                        mat(Eigen::seq(Eigen::fix<4>, Eigen::fix<9>), iB),
                        dXijdXi);

            case 4:
                mat({1, 2, 3}, Eigen::all) = dXijdXi * mat({1, 2, 3}, Eigen::all);
            case 1:
                break;

            default:
                std::cerr << mat.rows() << std::endl;
                DNDS_assert(false);
                break;
            }
    }

    template <int dim>
    inline int GetNDof(int maxOrder)
    {
        int maxNDOF = -1;
        switch (maxOrder)
        {
        case 0:
            maxNDOF = PolynomialNDOF<dim, 0>();
            break;
        case 1:
            maxNDOF = PolynomialNDOF<dim, 1>();
            break;
        case 2:
            maxNDOF = PolynomialNDOF<dim, 2>();
            break;
        case 3:
            maxNDOF = PolynomialNDOF<dim, 3>();
            break;
        default:
        {
            DNDS_assert_info(false, "maxNDOF invalid");
        }
        }
        return maxNDOF;
    }

    class CFVPeriodicity : public Geom::Periodicity
    {
    public:
        using tBase = Geom::Periodicity;
        using tBase::tBase;

        CFVPeriodicity(const tBase &vBase) : tBase(vBase) {} // copy from tBase

        template <int dim, class TU>
        void TransDiValueInplace(TU &u, Geom::t_index id)
        {
            using namespace Geom;
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            ConvertDiffsLinMap<dim>(u, rotation[i]);
        }

        template <int dim, class TU>
        void TransDiValueBackInplace(TU &u, Geom::t_index id)
        {
            using namespace Geom;
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            ConvertDiffsLinMap<dim>(u, rotation[i].transpose());
        }
    };
}
