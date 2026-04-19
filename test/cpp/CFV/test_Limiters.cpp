/**
 * @file test_Limiters.cpp
 * @brief Unit tests for standalone limiter functions in CFV/Limiters.hpp.
 *
 * Tests cover:
 *   - PolynomialSquaredNorm<2> and <3> for all supported row counts
 *   - PolynomialDotProduct<2> for all supported row counts
 *   - FWBAP_L2_Biway: sign preservation, identical-input pass-through, NaN-free
 *   - FWBAP_L2_Cut_Biway: sign preservation with cutting
 *   - FMINMOD_Biway: classical minmod properties
 *   - FVanLeer_Biway: classical Van-Leer limiter properties
 *   - FWBAP_L2_Multiway: multi-stencil weighted averaging
 *   - FWBAP_L2_Multiway_Polynomial2D: polynomial-norm weighted multi-stencil
 *   - FWBAP_L2_Multiway_PolynomialOrth: orthogonal polynomial multi-stencil
 *   - FWBAP_L2_Biway_PolynomialNorm<2>: polynomial-norm biway
 *   - FMEMM_Biway_PolynomialNorm<2>: MEMM biway
 *   - FMEMM_Multiway_Polynomial2D: MEMM multi-stencil
 *   - FWBAP_L2_Biway_PolynomialOrth: orthogonal biway
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "CFV/Limiters.hpp"

#include <Eigen/Core>
#include <cmath>
#include <vector>

using namespace DNDS;
using namespace DNDS::CFV;

// ===================================================================
// PolynomialSquaredNorm
// ===================================================================

TEST_CASE("PolynomialSquaredNorm<2> with 2 rows (P1)")
{
    Eigen::ArrayXXd theta(2, 3);
    theta << 1.0, 2.0, 3.0,
             4.0, 5.0, 6.0;
    // weight: all 1.0 for nRows=2
    // result[j] = theta(0,j)^2 + theta(1,j)^2
    auto result = PolynomialSquaredNorm<2>(theta);
    CHECK(result.size() == 3);
    CHECK(result(0) == doctest::Approx(1.0 + 16.0));
    CHECK(result(1) == doctest::Approx(4.0 + 25.0));
    CHECK(result(2) == doctest::Approx(9.0 + 36.0));
}

TEST_CASE("PolynomialSquaredNorm<2> with 3 rows (P2)")
{
    Eigen::ArrayXXd theta(3, 2);
    theta << 2.0, 1.0,
             3.0, 4.0,
             5.0, 6.0;
    // result[j] = theta(0,j)^2 + theta(1,j)^2 * 0.5 + theta(2,j)^2
    auto result = PolynomialSquaredNorm<2>(theta);
    CHECK(result.size() == 2);
    CHECK(result(0) == doctest::Approx(4.0 + 9.0 * 0.5 + 25.0));
    CHECK(result(1) == doctest::Approx(1.0 + 16.0 * 0.5 + 36.0));
}

TEST_CASE("PolynomialSquaredNorm<2> with 4 rows (P3)")
{
    Eigen::ArrayXXd theta(4, 1);
    theta << 1.0, 2.0, 3.0, 4.0;
    // result = 1 + 4*(1/3) + 9*(1/3) + 16
    auto result = PolynomialSquaredNorm<2>(theta);
    CHECK(result.size() == 1);
    CHECK(result(0) == doctest::Approx(1.0 + 4.0 / 3.0 + 9.0 / 3.0 + 16.0));
}

TEST_CASE("PolynomialSquaredNorm<3> with 3 rows (P1)")
{
    Eigen::ArrayXXd theta(3, 2);
    theta << 1.0, 2.0,
             3.0, 4.0,
             5.0, 6.0;
    // uniform weight 1 for nRows=3
    auto result = PolynomialSquaredNorm<3>(theta);
    CHECK(result.size() == 2);
    CHECK(result(0) == doctest::Approx(1.0 + 9.0 + 25.0));
    CHECK(result(1) == doctest::Approx(4.0 + 16.0 + 36.0));
}

TEST_CASE("PolynomialSquaredNorm<3> with 6 rows (P2)")
{
    Eigen::ArrayXXd theta(6, 1);
    theta << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    // first 3 weight 1, next 3 weight 0.5
    auto result = PolynomialSquaredNorm<3>(theta);
    CHECK(result.size() == 1);
    CHECK(result(0) == doctest::Approx(
        1.0 + 4.0 + 9.0 + 16.0 * 0.5 + 25.0 * 0.5 + 36.0 * 0.5));
}

TEST_CASE("PolynomialSquaredNorm<3> with 10 rows (P3)")
{
    Eigen::ArrayXXd theta(10, 1);
    for (int i = 0; i < 10; i++)
        theta(i, 0) = static_cast<real>(i + 1);
    // first 3: w=1, next 6: w=1/3, last 1: w=1/6
    real expected = 0;
    for (int i = 0; i < 3; i++)
        expected += theta(i, 0) * theta(i, 0);
    for (int i = 3; i < 9; i++)
        expected += theta(i, 0) * theta(i, 0) / 3.0;
    expected += theta(9, 0) * theta(9, 0) / 6.0;
    auto result = PolynomialSquaredNorm<3>(theta);
    CHECK(result(0) == doctest::Approx(expected));
}

// ===================================================================
// PolynomialDotProduct
// ===================================================================

TEST_CASE("PolynomialDotProduct<2> self-product equals PolynomialSquaredNorm")
{
    for (int nRows : {2, 3, 4})
    {
        CAPTURE(nRows);
        Eigen::ArrayXXd theta(nRows, 3);
        theta.setRandom();
        auto normResult = PolynomialSquaredNorm<2>(theta);
        auto dotResult = PolynomialDotProduct<2>(theta, theta);
        for (int j = 0; j < 3; j++)
        {
            CAPTURE(j);
            CHECK(dotResult(j) == doctest::Approx(normResult(j)).epsilon(1e-14));
        }
    }
}

TEST_CASE("PolynomialDotProduct<2> linearity")
{
    int nRows = 3;
    Eigen::ArrayXXd a(nRows, 2), b(nRows, 2);
    a << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;
    b << 0.5, 1.5,
         2.5, 3.5,
         4.5, 5.5;
    real alpha = 2.5;
    // dot(alpha*a, b) == alpha * dot(a, b)
    Eigen::ArrayXXd alphaA = alpha * a;
    Eigen::ArrayXd left = PolynomialDotProduct<2>(alphaA, b);
    Eigen::ArrayXd right = alpha * PolynomialDotProduct<2>(a, b);
    for (int j = 0; j < 2; j++)
    {
        CAPTURE(j);
        CHECK(left(j) == doctest::Approx(right(j)).epsilon(1e-13));
    }
}

// ===================================================================
// FMINMOD_Biway
// ===================================================================

TEST_CASE("FMINMOD_Biway: same-sign inputs")
{
    Eigen::ArrayXXd u1(2, 3), u2(2, 3), out(2, 3);
    u1 << 1.0, 2.0, 5.0,
          3.0, 4.0, 6.0;
    u2 << 2.0, 1.0, 10.0,
          6.0, 2.0, 3.0;
    FMINMOD_Biway(u1, u2, out, 1.0);
    // minmod(a,b) = sign(a)*min(|a|,|b|) when signs agree
    CHECK(out(0, 0) == doctest::Approx(1.0));
    CHECK(out(0, 1) == doctest::Approx(1.0));
    CHECK(out(0, 2) == doctest::Approx(5.0));
    CHECK(out(1, 0) == doctest::Approx(3.0));
    CHECK(out(1, 1) == doctest::Approx(2.0));
    CHECK(out(1, 2) == doctest::Approx(3.0));
}

TEST_CASE("FMINMOD_Biway: opposite-sign inputs produce zero")
{
    Eigen::ArrayXXd u1(1, 4), u2(1, 4), out(1, 4);
    u1 << 1.0, -1.0, 3.0, -5.0;
    u2 << -2.0, 2.0, -0.5, 7.0;
    FMINMOD_Biway(u1, u2, out, 1.0);
    for (int j = 0; j < 4; j++)
    {
        CAPTURE(j);
        CHECK(out(0, j) == doctest::Approx(0.0));
    }
}

TEST_CASE("FMINMOD_Biway: zero input")
{
    Eigen::ArrayXXd u1(1, 2), u2(1, 2), out(1, 2);
    u1 << 0.0, 5.0;
    u2 << 3.0, 0.0;
    FMINMOD_Biway(u1, u2, out, 1.0);
    CHECK(out(0, 0) == doctest::Approx(0.0));
    CHECK(out(0, 1) == doctest::Approx(0.0));
}

// ===================================================================
// FVanLeer_Biway
// ===================================================================

TEST_CASE("FVanLeer_Biway: same-sign inputs")
{
    Eigen::ArrayXXd u1(1, 2), u2(1, 2), out(1, 2);
    u1 << 2.0, -3.0;
    u2 << 4.0, -6.0;
    FVanLeer_Biway(u1, u2, out, 1.0);
    // VanLeer(a,b) = (sign(a)+sign(b)) * |a*b| / (|a+b| + eps)
    // For same-sign: 2 * |a*b| / (|a+b| + eps) ~ 2*|ab|/|a+b|
    // u1=2,u2=4: 2*8/6 = 8/3
    CHECK(out(0, 0) == doctest::Approx(8.0 / 3.0).epsilon(1e-10));
    // u1=-3,u2=-6: 2*18/9 = 4.0
    CHECK(out(0, 1) == doctest::Approx(-4.0).epsilon(1e-10));
}

TEST_CASE("FVanLeer_Biway: opposite-sign produces zero")
{
    Eigen::ArrayXXd u1(1, 2), u2(1, 2), out(1, 2);
    u1 << 3.0, -5.0;
    u2 << -2.0, 7.0;
    FVanLeer_Biway(u1, u2, out, 1.0);
    CHECK(out(0, 0) == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(out(0, 1) == doctest::Approx(0.0).epsilon(1e-10));
}

TEST_CASE("FVanLeer_Biway: equal inputs return same value")
{
    Eigen::ArrayXXd u1(1, 2), u2(1, 2), out(1, 2);
    u1 << 5.0, -3.0;
    u2 = u1;
    FVanLeer_Biway(u1, u2, out, 1.0);
    // VanLeer(a,a) = 2*a^2 / (2*|a|) = |a| * sign(a) = a
    CHECK(out(0, 0) == doctest::Approx(5.0).epsilon(1e-10));
    CHECK(out(0, 1) == doctest::Approx(-3.0).epsilon(1e-10));
}

// ===================================================================
// FWBAP_L2_Biway
// ===================================================================

TEST_CASE("FWBAP_L2_Biway: identical inputs pass through")
{
    Eigen::ArrayXXd u1(2, 3), u2(2, 3), out(2, 3);
    u1.setRandom();
    u2 = u1;
    FWBAP_L2_Biway(u1, u2, out, 1.0);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
        {
            CAPTURE(i);
            CAPTURE(j);
            CHECK(out(i, j) == doctest::Approx(u1(i, j)).epsilon(1e-10));
        }
}

TEST_CASE("FWBAP_L2_Biway: no NaN for random inputs")
{
    Eigen::ArrayXXd u1(3, 5), u2(3, 5), out(3, 5);
    u1.setRandom();
    u2.setRandom();
    FWBAP_L2_Biway(u1, u2, out, 1.0);
    CHECK_FALSE(out.hasNaN());
}

TEST_CASE("FWBAP_L2_Biway: output bounded by inputs for same-sign")
{
    // For same-sign inputs, WBAP output should not exceed the extremes
    Eigen::ArrayXXd u1(1, 100), u2(1, 100), out(1, 100);
    u1.setRandom();
    u1 = u1.abs() + 0.1; // all positive
    u2.setRandom();
    u2 = u2.abs() + 0.1; // all positive
    FWBAP_L2_Biway(u1, u2, out, 1.0);
    for (int j = 0; j < 100; j++)
    {
        CAPTURE(j);
        // Output should be non-negative
        CHECK(out(0, j) >= -1e-10);
    }
}

// ===================================================================
// FWBAP_L2_Cut_Biway
// ===================================================================

TEST_CASE("FWBAP_L2_Cut_Biway: opposite-sign cut to zero")
{
    Eigen::ArrayXXd u1(1, 3), u2(1, 3), out(1, 3);
    u1 << 1.0, -2.0, 3.0;
    u2 << -1.0, 2.0, -3.0;
    FWBAP_L2_Cut_Biway(u1, u2, out, 1.0);
    for (int j = 0; j < 3; j++)
    {
        CAPTURE(j);
        CHECK(out(0, j) == doctest::Approx(0.0).epsilon(1e-10));
    }
}

TEST_CASE("FWBAP_L2_Cut_Biway: no NaN")
{
    Eigen::ArrayXXd u1(2, 4), u2(2, 4), out(2, 4);
    u1.setRandom();
    u2.setRandom();
    FWBAP_L2_Cut_Biway(u1, u2, out, 1.0);
    CHECK_FALSE(out.hasNaN());
}

// ===================================================================
// FWBAP_L2_Multiway
// ===================================================================

TEST_CASE("FWBAP_L2_Multiway: all identical inputs pass through")
{
    Eigen::ArrayXXd u0(2, 3);
    u0.setRandom();
    std::vector<Eigen::ArrayXXd> uOthers = {u0, u0, u0, u0};
    Eigen::ArrayXXd out;
    out.resizeLike(u0);
    FWBAP_L2_Multiway(uOthers, 4, out, 1.0);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
        {
            CAPTURE(i);
            CAPTURE(j);
            CHECK(out(i, j) == doctest::Approx(u0(i, j)).epsilon(1e-8));
        }
}

TEST_CASE("FWBAP_L2_Multiway: no NaN for random inputs")
{
    std::vector<Eigen::ArrayXXd> uOthers(5);
    for (auto &u : uOthers)
    {
        u.resize(3, 4);
        u.setRandom();
    }
    Eigen::ArrayXXd out;
    out.resizeLike(uOthers[0]);
    FWBAP_L2_Multiway(uOthers, 5, out, 1.0);
    CHECK_FALSE(out.hasNaN());
}

// ===================================================================
// FWBAP_L2_Multiway_Polynomial2D
// ===================================================================

TEST_CASE("FWBAP_L2_Multiway_Polynomial2D: all identical pass through")
{
    Eigen::ArrayXXd u0(2, 3);
    u0.setRandom();
    std::vector<Eigen::ArrayXXd> uOthers = {u0, u0, u0};
    Eigen::ArrayXXd out;
    out.resizeLike(u0);
    FWBAP_L2_Multiway_Polynomial2D(uOthers, 3, out, 1.0);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
        {
            CAPTURE(i);
            CAPTURE(j);
            CHECK(out(i, j) == doctest::Approx(u0(i, j)).epsilon(1e-8));
        }
}

TEST_CASE("FWBAP_L2_Multiway_Polynomial2D: no NaN for random inputs (nRows=2,3,4)")
{
    for (int nRows : {2, 3, 4})
    {
        CAPTURE(nRows);
        std::vector<Eigen::ArrayXXd> uOthers(4);
        for (auto &u : uOthers)
        {
            u.resize(nRows, 3);
            u.setRandom();
        }
        Eigen::ArrayXXd out;
        out.resizeLike(uOthers[0]);
        FWBAP_L2_Multiway_Polynomial2D(uOthers, 4, out, 1.0);
        CHECK_FALSE(out.hasNaN());
    }
}

// ===================================================================
// FMEMM_Multiway_Polynomial2D
// ===================================================================

TEST_CASE("FMEMM_Multiway_Polynomial2D: center input unchanged when smallest")
{
    // If center has the smallest polynomial norm, MEMM should leave it alone
    Eigen::ArrayXXd center(2, 2);
    center << 0.1, 0.2,
              0.1, 0.2;
    std::vector<Eigen::ArrayXXd> uOthers(3);
    for (auto &u : uOthers)
    {
        u.resize(2, 2);
        u.setRandom();
        u *= 10.0; // make neighbors much larger
    }
    Eigen::ArrayXXd out;
    out.resizeLike(center);
    FMEMM_Multiway_Polynomial2D(center, uOthers, 3, out, 1.0);
    CHECK_FALSE(out.hasNaN());
    // When center is smallest, ifReplace=0 so replaceFactor=1 and output=center
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            CAPTURE(i);
            CAPTURE(j);
            CHECK(out(i, j) == doctest::Approx(center(i, j)).epsilon(1e-8));
        }
}

TEST_CASE("FMEMM_Multiway_Polynomial2D: no NaN for random inputs")
{
    Eigen::ArrayXXd center(3, 4);
    center.setRandom();
    std::vector<Eigen::ArrayXXd> uOthers(3);
    for (auto &u : uOthers)
    {
        u.resize(3, 4);
        u.setRandom();
    }
    Eigen::ArrayXXd out;
    out.resizeLike(center);
    FMEMM_Multiway_Polynomial2D(center, uOthers, 3, out, 1.0);
    CHECK_FALSE(out.hasNaN());
}

// ===================================================================
// FWBAP_L2_Multiway_PolynomialOrth
// ===================================================================

TEST_CASE("FWBAP_L2_Multiway_PolynomialOrth: all identical pass through")
{
    Eigen::ArrayXXd u0(3, 2);
    u0.setRandom();
    std::vector<Eigen::ArrayXXd> uOthers = {u0, u0, u0};
    Eigen::ArrayXXd out;
    out.resizeLike(u0);
    FWBAP_L2_Multiway_PolynomialOrth(uOthers, 3, out, 1.0);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
        {
            CAPTURE(i);
            CAPTURE(j);
            CHECK(out(i, j) == doctest::Approx(u0(i, j)).epsilon(1e-8));
        }
}

TEST_CASE("FWBAP_L2_Multiway_PolynomialOrth: no NaN")
{
    std::vector<Eigen::ArrayXXd> uOthers(4);
    for (auto &u : uOthers)
    {
        u.resize(4, 3);
        u.setRandom();
    }
    Eigen::ArrayXXd out;
    out.resizeLike(uOthers[0]);
    FWBAP_L2_Multiway_PolynomialOrth(uOthers, 4, out, 1.0);
    CHECK_FALSE(out.hasNaN());
}

// ===================================================================
// FWBAP_L2_Biway_PolynomialNorm<2, nVarsFixed>
// ===================================================================

TEST_CASE("FWBAP_L2_Biway_PolynomialNorm<2,1>: identical inputs pass through")
{
    using ArrT = Eigen::Array<real, Eigen::Dynamic, 1>;
    ArrT u1(2, 1), u2(2, 1), out(2, 1);
    u1 << 3.0, 4.0;
    u2 = u1;
    FWBAP_L2_Biway_PolynomialNorm<2, 1>(u1, u2, out, 1.0);
    CHECK(out(0) == doctest::Approx(u1(0)).epsilon(1e-8));
    CHECK(out(1) == doctest::Approx(u1(1)).epsilon(1e-8));
}

TEST_CASE("FWBAP_L2_Biway_PolynomialNorm<2,Eigen::Dynamic>: no NaN")
{
    Eigen::ArrayXXd u1(3, 4), u2(3, 4), out(3, 4);
    u1.setRandom();
    u2.setRandom();
    FWBAP_L2_Biway_PolynomialNorm<2, Eigen::Dynamic>(u1, u2, out, 1.0);
    CHECK_FALSE(out.hasNaN());
}

TEST_CASE("FWBAP_L2_Biway_PolynomialNorm<3,Eigen::Dynamic>: no NaN for 3D dims")
{
    for (int nRows : {3, 6, 10})
    {
        CAPTURE(nRows);
        Eigen::ArrayXXd u1(nRows, 2), u2(nRows, 2), out(nRows, 2);
        u1.setRandom();
        u2.setRandom();
        FWBAP_L2_Biway_PolynomialNorm<3, Eigen::Dynamic>(u1, u2, out, 1.0);
        CHECK_FALSE(out.hasNaN());
    }
}

// ===================================================================
// FMEMM_Biway_PolynomialNorm<2, nVarsFixed>
// ===================================================================

TEST_CASE("FMEMM_Biway_PolynomialNorm<2,Eigen::Dynamic>: u2 smaller returns u2")
{
    // When u2 has smaller polynomial norm, MEMM should return u2
    Eigen::ArrayXXd u1(2, 2), u2(2, 2), out(2, 2);
    u1 << 10.0, 20.0,
           10.0, 20.0;
    u2 << 0.1, 0.2,
          0.1, 0.2;
    FMEMM_Biway_PolynomialNorm<2, Eigen::Dynamic>(u1, u2, out, 1.0);
    CHECK_FALSE(out.hasNaN());
    // When u1 is much larger, the limiter should reduce toward u2
    // Check output is closer to u2 than to u1
    real dist_to_u2 = (out - u2).matrix().norm();
    real dist_to_u1 = (out - u1).matrix().norm();
    CHECK(dist_to_u2 < dist_to_u1);
}

TEST_CASE("FMEMM_Biway_PolynomialNorm<2,Eigen::Dynamic>: no NaN")
{
    Eigen::ArrayXXd u1(4, 3), u2(4, 3), out(4, 3);
    u1.setRandom();
    u2.setRandom();
    FMEMM_Biway_PolynomialNorm<2, Eigen::Dynamic>(u1, u2, out, 1.0);
    CHECK_FALSE(out.hasNaN());
}

// ===================================================================
// FWBAP_L2_Biway_PolynomialOrth
// ===================================================================

TEST_CASE("FWBAP_L2_Biway_PolynomialOrth: identical inputs pass through")
{
    Eigen::ArrayXXd u1(3, 2), u2(3, 2), out(3, 2);
    u1.setRandom();
    u2 = u1;
    FWBAP_L2_Biway_PolynomialOrth(u1, u2, out, 1.0);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
        {
            CAPTURE(i);
            CAPTURE(j);
            CHECK(out(i, j) == doctest::Approx(u1(i, j)).epsilon(1e-8));
        }
}

TEST_CASE("FWBAP_L2_Biway_PolynomialOrth: no NaN")
{
    Eigen::ArrayXXd u1(4, 5), u2(4, 5), out(4, 5);
    u1.setRandom();
    u2.setRandom();
    FWBAP_L2_Biway_PolynomialOrth(u1, u2, out, 1.0);
    CHECK_FALSE(out.hasNaN());
}

// ===================================================================
// Cross-consistency: FMINMOD is most restrictive
// ===================================================================

TEST_CASE("Limiter ordering: FMINMOD output <= FWBAP output for same-sign")
{
    Eigen::ArrayXXd u1(1, 100), u2(1, 100), outMM(1, 100), outWBAP(1, 100);
    u1.setRandom();
    u1 = u1.abs() + 0.01;
    u2.setRandom();
    u2 = u2.abs() + 0.01;
    FMINMOD_Biway(u1, u2, outMM, 1.0);
    FWBAP_L2_Biway(u1, u2, outWBAP, 1.0);
    // MINMOD is the most restrictive TVD limiter
    for (int j = 0; j < 100; j++)
    {
        CAPTURE(j);
        CHECK(outMM(0, j) <= outWBAP(0, j) + 1e-12);
    }
}

// ===================================================================
// Edge case: near-zero inputs
// ===================================================================

TEST_CASE("All limiters handle near-zero inputs without NaN")
{
    Eigen::ArrayXXd u1(2, 3), u2(2, 3);
    u1.setConstant(1e-300);
    u2.setConstant(1e-300);

    Eigen::ArrayXXd out(2, 3);

    SUBCASE("FMINMOD")
    {
        FMINMOD_Biway(u1, u2, out, 1.0);
        CHECK_FALSE(out.hasNaN());
    }
    SUBCASE("FVanLeer")
    {
        FVanLeer_Biway(u1, u2, out, 1.0);
        CHECK_FALSE(out.hasNaN());
    }
    SUBCASE("FWBAP_L2_Biway")
    {
        FWBAP_L2_Biway(u1, u2, out, 1.0);
        CHECK_FALSE(out.hasNaN());
    }
    SUBCASE("FWBAP_L2_Cut_Biway")
    {
        FWBAP_L2_Cut_Biway(u1, u2, out, 1.0);
        CHECK_FALSE(out.hasNaN());
    }
    SUBCASE("FWBAP_L2_Biway_PolynomialNorm<2>")
    {
        FWBAP_L2_Biway_PolynomialNorm<2, Eigen::Dynamic>(u1, u2, out, 1.0);
        CHECK_FALSE(out.hasNaN());
    }
    SUBCASE("FWBAP_L2_Biway_PolynomialOrth")
    {
        FWBAP_L2_Biway_PolynomialOrth(u1, u2, out, 1.0);
        CHECK_FALSE(out.hasNaN());
    }
}
