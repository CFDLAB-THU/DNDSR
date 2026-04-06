/**
 * @file test_Elements.cpp
 * @brief Doctest-based unit tests for the auto-generated per-element
 *        shape functions (ShapeFuncImpl<T>).
 *
 * Tests:
 *   1. Partition of unity:  sum_j N_j(p) == 1 at random interior points.
 *   2. Nodal interpolation: N_i(node_j) == delta_ij at reference nodes.
 *   3. First derivative consistency: dN/dxi finite-difference vs analytic.
 *
 * This is a serial test (no MPI required).
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Geom/Elements.hpp"

#include <cmath>
#include <random>
#include <array>
#include <iostream>

using namespace DNDS::Geom;
using namespace DNDS::Geom::Elem;

// Convenience: 3D point from 1–3 coords, rest zeroed.
static tPoint MakePoint(t_real x, t_real y = 0, t_real z = 0)
{
    tPoint p;
    p << x, y, z;
    return p;
}

// Get reference node coordinates (columns of a 3×N matrix).
static Eigen::Matrix<t_real, 3, Eigen::Dynamic> NodeCoords(ElemType t)
{
    return GetStandardCoord(t);
}

// Random interior point for each param-space type.
// Uses a fixed seed for reproducibility.
static tPoint RandomInteriorPoint(ElemType t, std::mt19937 &rng)
{
    std::uniform_real_distribution<t_real> u01(0.05, 0.95);
    std::uniform_real_distribution<t_real> um11(-0.8, 0.8);

    Element e{t};
    auto ps = e.GetParamSpace();

    switch (ps)
    {
    case LineSpace:
        return MakePoint(um11(rng));
    case TriSpace:
    {
        // Ensure xi + et < 1
        t_real xi = u01(rng) * 0.5;
        t_real et = u01(rng) * (1.0 - xi) * 0.8;
        return MakePoint(xi, et);
    }
    case QuadSpace:
        return MakePoint(um11(rng), um11(rng));
    case TetSpace:
    {
        t_real xi = u01(rng) * 0.3;
        t_real et = u01(rng) * (1.0 - xi) * 0.3;
        t_real zt = u01(rng) * (1.0 - xi - et) * 0.8;
        return MakePoint(xi, et, zt);
    }
    case HexSpace:
        return MakePoint(um11(rng), um11(rng), um11(rng));
    case PrismSpace:
    {
        t_real xi = u01(rng) * 0.5;
        t_real et = u01(rng) * (1.0 - xi) * 0.8;
        t_real zt = um11(rng);
        return MakePoint(xi, et, zt);
    }
    case PyramidSpace:
    {
        // zt in (0, 1), base in (-1,1)^2 scaled by (1-zt)
        t_real zt = u01(rng) * 0.8;
        t_real scale = (1.0 - zt) * 0.8;
        t_real xi = um11(rng) * scale;
        t_real et = um11(rng) * scale;
        return MakePoint(xi, et, zt);
    }
    default:
        return MakePoint(0);
    }
}

// All valid element types.
static constexpr ElemType AllTypes[] = {
    Line2, Line3, Tri3, Tri6, Quad4, Quad9,
    Tet4, Tet10, Hex8, Hex27, Prism6, Prism18,
    Pyramid5, Pyramid14};

// ===================================================================
// Test: Partition of unity — sum_j N_j(p) == 1
// ===================================================================
TEST_CASE("Shape functions: partition of unity")
{
    constexpr int nTrials = 10;
    std::mt19937 rng(42);

    for (auto t : AllTypes)
    {
        Element e{t};
        CAPTURE(t);
        for (int trial = 0; trial < nTrials; trial++)
        {
            auto p = RandomInteriorPoint(t, rng);
            tNj Nj;
            e.GetNj(p, Nj);

            t_real sum = Nj.sum();
            CHECK(sum == doctest::Approx(1.0).epsilon(1e-12));
        }
    }
}

// ===================================================================
// Test: Nodal interpolation — N_i(node_j) == delta_ij
// ===================================================================
TEST_CASE("Shape functions: nodal interpolation (Kronecker delta)")
{
    for (auto t : AllTypes)
    {
        Element e{t};
        auto coords = NodeCoords(t);
        DNDS::index nn = e.GetNumNodes();
        CAPTURE(t);

        for (DNDS::index j = 0; j < nn; j++)
        {
            tPoint p = coords.col(j);
            tNj Nj;
            e.GetNj(p, Nj);

            for (DNDS::index i = 0; i < nn; i++)
            {
                t_real expected = (i == j) ? 1.0 : 0.0;
                CAPTURE(i);
                CAPTURE(j);
                CHECK(Nj(0, i) == doctest::Approx(expected).epsilon(1e-10));
            }
        }
    }
}

// ===================================================================
// Test: First derivative via finite difference
// ===================================================================
TEST_CASE("Shape functions: D1 derivatives vs finite difference")
{
    constexpr int nTrials = 5;
    constexpr t_real h = 1e-7;
    constexpr t_real tol = 1e-5;
    std::mt19937 rng(123);

    for (auto t : AllTypes)
    {
        Element e{t};
        DNDS::index nn = e.GetNumNodes();
        int dim = e.GetDim();
        CAPTURE(t);

        for (int trial = 0; trial < nTrials; trial++)
        {
            auto p = RandomInteriorPoint(t, rng);

            tD1Nj D1Nj;
            e.GetD1Nj(p, D1Nj);

            // Finite difference for each parametric direction
            for (int d = 0; d < dim; d++)
            {
                tPoint pp = p, pm = p;
                pp[d] += h;
                pm[d] -= h;

                tNj Np, Nm;
                e.GetNj(pp, Np);
                e.GetNj(pm, Nm);

                for (DNDS::index j = 0; j < nn; j++)
                {
                    t_real fd = (Np(0, j) - Nm(0, j)) / (2 * h);
                    t_real an = D1Nj(d, j);
                    CAPTURE(d);
                    CAPTURE(j);
                    CHECK(an == doctest::Approx(fd).epsilon(tol));
                }
            }
        }
    }
}

// ===================================================================
// Test: Second derivative via finite difference of first derivative
// ===================================================================
TEST_CASE("Shape functions: D2 derivatives vs finite difference of D1")
{
    constexpr t_real h = 1e-6;
    constexpr t_real tol = 1e-3;
    std::mt19937 rng(456);

    for (auto t : AllTypes)
    {
        Element e{t};
        DNDS::index nn = e.GetNumNodes();
        int dim = e.GetDim();
        CAPTURE(t);

        auto p = RandomInteriorPoint(t, rng);

        tDiNj DiNj;
        e.GetDiNj(p, DiNj, 2);

        // D2 rows: dim==1 -> 1 row; dim==2 -> 3 rows; dim==3 -> 6 rows
        // We only test the pure second derivatives (d^2/dxi^2, d^2/det^2, ...)
        // by finite-differencing D1.
        for (int d = 0; d < dim; d++)
        {
            tPoint pp = p, pm = p;
            pp[d] += h;
            pm[d] -= h;

            tD1Nj D1p, D1m;
            e.GetD1Nj(pp, D1p);
            e.GetD1Nj(pm, D1m);

            for (DNDS::index j = 0; j < nn; j++)
            {
                t_real fd_d2 = (D1p(d, j) - D1m(d, j)) / (2 * h);
                // Row index for d^2/dxi_d^2 in the DiNj layout:
                // dim==1: diffOrder2 row 0 = d^2/dxi^2, but overall offset = ndiffSizC2D[1]=2 (for 2D) or ndiffSizC[1]=4 (for 3D)
                // Actually we need the row offsets from the DiNj layout.
                // The DiNj matrix is stacked: rows [0] = Nj, [1..dim] = D1,
                // then D2 rows follow.
                // For dim==2: offsets are ndiffSizC2D = {1, 3, 6, 10}, so
                //   diffOrder 0: 1 row (row 0)
                //   diffOrder 1: 2 rows (rows 1,2)
                //   diffOrder 2: 3 rows (rows 3,4,5)
                //   d^2/dxi^2 = row 3, d^2/dxidet = row 4, d^2/det^2 = row 5
                // For dim==3: ndiffSizC = {1, 4, 10, 20}
                //   d^2/dxi^2 = row 4, d^2/det^2 = row 5, d^2/dzt^2 = row 6
                //   d^2/dxidet = row 7, d^2/detdzt = row 8, d^2/dxidzt = row 9
                // For dim==1: ndiffSizC for 1D is {1,2,3,4} conceptually
                //   but actually the DiNj for 1D is built with ndiffSizC2D
                //   (since the code uses dim==2 layout for 2D elements).

                // Simpler: the DiNj row for d^2/dxi_d^2 is:
                // For dim==1: row offset for D2 = 1(D0) + 1(D1) = 2, then row 2 = d^2/dxi^2
                // For dim==2: row offset for D2 = 1 + 2 = 3, pure diags at rows 3 (d=0), 5 (d=1)
                // For dim==3: row offset for D2 = 1 + 3 = 4, pure diags at rows 4 (d=0), 5 (d=1), 6 (d=2)

                DNDS::index row;
                if (dim == 1)
                    row = 2; // d^2/dxi^2
                else if (dim == 2)
                    row = 3 + (d == 0 ? 0 : 2); // row 3=d2/dxi2, row 5=d2/det2
                else
                    row = 4 + d; // row 4=d2/dxi2, row 5=d2/det2, row 6=d2/dzt2

                t_real an = DiNj(row, j);
                CAPTURE(d);
                CAPTURE(j);
                CHECK(an == doctest::Approx(fd_d2).epsilon(tol));
            }
        }
    }
}
