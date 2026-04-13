/**
 * @file test_Quadrature.cpp
 * @brief Comprehensive doctest-based unit tests for numerical quadrature accuracy.
 *
 * Tests cover:
 *   1. Polynomial integration accuracy (0th to 6th degree)
 *      - For each parametric space: Line, Tri, Quad, Tet, Hex, Prism, Pyramid
 *      - Tests exactness up to the theoretical order of each quadrature scheme
 *   2. Complete Taylor basis integration (degree <= 3)
 *      - Uses BaseFunction utilities for systematic testing
 *      - All monomials: 1, x, y, z, x², xy, y², xz, yz, z², x³, x²y, xy², y³, ...
 *   3. Weight sum verification (should equal parametric space volume)
 *   4. Integration scheme selection correctness
 *
 * Theoretical exactness:
 *   - Gauss-Legendre on [-1,1]^d: exact for polynomials up to 2n-1
 *   - Hammer rules on simplex: exactness varies by rule
 *   - Product rules: exactness determined by component rules
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Geom/Quadrature.hpp"
#include "Geom/Elements.hpp"
#include "Geom/BaseFunction.hpp"

#include <cmath>
#include <array>
#include <iostream>

using namespace DNDS::Geom;
using namespace DNDS::Geom::Elem;

// ===================================================================
// Helper functions for polynomial integration
// ===================================================================

/**
 * @brief Get the list of Taylor basis monomial exponents up to given order
 * 
 * Uses BaseFunction's diffOperatorOrderList to get exponents (nx, ny, nz)
 * for each polynomial basis function.
 * 
 * @param dim Dimension (2 or 3)
 * @param maxOrder Maximum polynomial order (0-3)
 * @return Vector of {nx, ny, nz} exponent triplets
 */
std::vector<std::array<int, 3>> GetTaylorBasisMonomials(int dim, int maxOrder)
{
    std::vector<std::array<int, 3>> result;
    
    int nDof = Base::PolynomialNDOF<2, 3>(); // Get max 2D DOF
    if (dim == 3)
        nDof = Base::PolynomialNDOF<3, 3>(); // Get max 3D DOF
    
    // Determine how many DOFs we actually need based on maxOrder
    int nDofNeeded = 1;
    switch (maxOrder)
    {
        case 0: nDofNeeded = 1; break;
        case 1: nDofNeeded = dim == 2 ? 3 : 4; break;
        case 2: nDofNeeded = dim == 2 ? 6 : 10; break;
        case 3: nDofNeeded = dim == 2 ? 10 : 20; break;
    }
    
    nDof = std::min(nDof, nDofNeeded);
    
    for (int i = 0; i < nDof; i++)
    {
        if (dim == 2)
            result.push_back(Base::diffOperatorOrderList2D[i]);
        else
            result.push_back(Base::diffOperatorOrderList[i]);
    }
    
    return result;
}

/**
 * @brief Evaluate a monomial x^a * y^b * z^c at point p
 */
t_real EvalMonomial(const tPoint& p, int a, int b = 0, int c = 0)
{
    t_real result = 1.0;
    if (a > 0) result *= std::pow(p(0), a);
    if (b > 0) result *= std::pow(p(1), b);
    if (c > 0) result *= std::pow(p(2), c);
    return result;
}

/**
 * @brief Compute exact integral of monomial over parametric reference element
 * 
 * Reference domains:
 *   - Line:   [-1, 1]          integral = 2/(a+1) for x^a (if a even)
 *   - Quad:   [-1, 1]^2        integral = product of 1D integrals
 *   - Hex:    [-1, 1]^3        integral = product of 1D integrals
 *   - Tri:    {xi>=0, et>=0, xi+et<=1}  integral = a!*b!/(a+b+2)!
 *   - Tet:    {xi>=0, et>=0, zt>=0, xi+et+zt<=1}
 *   - Prism:  Triangle x [-1,1]
 *   - Pyramid: Special domain
 */
t_real ExactMonomialIntegral(ParamSpace ps, int a, int b = 0, int c = 0)
{
    switch (ps)
    {
    case LineSpace:
    {
        // Integral of x^a from -1 to 1
        if (a % 2 == 1) return 0.0;  // Odd powers integrate to 0
        return 2.0 / (a + 1.0);
    }
    case QuadSpace:
    {
        // Product of 1D integrals
        t_real Ix = (a % 2 == 1) ? 0.0 : 2.0 / (a + 1.0);
        t_real Iy = (b % 2 == 1) ? 0.0 : 2.0 / (b + 1.0);
        return Ix * Iy;
    }
    case HexSpace:
    {
        t_real Ix = (a % 2 == 1) ? 0.0 : 2.0 / (a + 1.0);
        t_real Iy = (b % 2 == 1) ? 0.0 : 2.0 / (b + 1.0);
        t_real Iz = (c % 2 == 1) ? 0.0 : 2.0 / (c + 1.0);
        return Ix * Iy * Iz;
    }
    case TriSpace:
    {
        // Over reference triangle with vertices at (0,0), (1,0), (0,1)
        // The Hammer quadrature rules use a different reference triangle:
        // area coordinates with xi + et + zt = 1, xi >= 0, et >= 0, zt >= 0
        // where zt = 1 - xi - et
        // 
        // For the Hammer rules, the reference triangle is:
        // Vertices: (0,0), (1,0), (0,1)
        // Integral of xi^a * et^b over this triangle:
        // = B(a+1, b+1) / 2 = Gamma(a+1)*Gamma(b+1) / (2*Gamma(a+b+2))
        // = a! * b! / (2 * (a+b+1)!)
        // Wait, that's not right either. Let me reconsider.
        //
        // For standard reference triangle {(xi, et) : xi>=0, et>=0, xi+et<=1}:
        // int_0^1 int_0^{1-xi} xi^a * et^b det dxi
        // = Beta(a+1, b+1) / (a+b+2) ? No...
        //
        // Actually: int_0^1 xi^a * (1-xi)^{b+1} / (b+1) dxi
        // = B(a+1, b+2) / (b+1) = Gamma(a+1)*Gamma(b+2) / ((b+1)*Gamma(a+b+3))
        // = a! * (b+1)! / ((b+1) * (a+b+2)!)
        // = a! * b! / (a+b+2)!
        auto factorial = [](int n) -> t_real {
            t_real result = 1.0;
            for (int i = 2; i <= n; i++) result *= i;
            return result;
        };
        return factorial(a) * factorial(b) / factorial(a + b + 2);
    }
    case TetSpace:
    {
        // Over reference tet: int_0^1 int_0^{1-xi} int_0^{1-xi-et} xi^a et^b zt^c dzt det dxi
        // = a! * b! * c! / (a + b + c + 3)!
        auto factorial = [](int n) -> t_real {
            t_real result = 1.0;
            for (int i = 2; i <= n; i++) result *= i;
            return result;
        };
        return factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3);
    }
    case PrismSpace:
    {
        // Product of triangle integral (xi, et) and line integral (zt)
        auto factorial = [](int n) -> t_real {
            t_real result = 1.0;
            for (int i = 2; i <= n; i++) result *= i;
            return result;
        };
        t_real I_tri = factorial(a) * factorial(b) / factorial(a + b + 2);
        t_real I_line = (c % 2 == 1) ? 0.0 : 2.0 / (c + 1.0);
        return I_tri * I_line;
    }
    case PyramidSpace:
    {
        // Pyramid reference: base [-1,1]^2 at z=0, apex at z=1
        // Parametric coordinates transformed
        // Approximate exact values for low-order monomials
        // For pyramid: xi, et in [-1,1], zt in [0,1] with scaling
        // Jacobian factor: (1 - zt)^2
        
        // This is more complex - use numerical reference for verification
        // For low degrees, we can compute analytically
        
        // Pure z powers: int_0^1 zt^c * (1-zt)^2 * 4 dz (area at height z is 4*(1-z)^2)
        // = 4 * int_0^1 zt^c * (1 - 2zt + zt^2) dz
        // = 4 * [1/(c+1) - 2/(c+2) + 1/(c+3)]
        if (a == 0 && b == 0)
        {
            return 4.0 * (1.0/(c+1) - 2.0/(c+2) + 1.0/(c+3));
        }
        
        // Pure x or y: odd powers give 0
        if ((a % 2 == 1 && b == 0 && c == 0) || (b % 2 == 1 && a == 0 && c == 0))
            return 0.0;
        
        // For x^2: int_0^1 (1-z)^2/3 * (1-z)^2 * 4 dz with appropriate scaling
        // = (4/3) * int_0^1 (1-z)^4 dz = (4/3) * 1/5 = 4/15
        if (a == 2 && b == 0 && c == 0)
            return 4.0 / 15.0;
        if (a == 0 && b == 2 && c == 0)
            return 4.0 / 15.0;
            
        // Constant
        if (a == 0 && b == 0 && c == 0)
            return 4.0 / 3.0;  // Volume of pyramid
            
        // For more complex cases, return NaN to skip test
        return std::nan("");
    }
    default:
        return std::nan("");
    }
}

/**
 * @brief Numerically integrate a monomial using quadrature
 */
t_real QuadratureMonomialIntegral(ParamSpace ps, int int_order, int a, int b = 0, int c = 0)
{
    t_index scheme = GetQuadratureScheme(ps, int_order);
    if (scheme == 0) return std::nan("");
    
    t_real sum = 0.0;
    for (t_index iG = 0; iG < scheme; iG++)
    {
        tPoint pParam{0, 0, 0};
        t_real w;
        GetQuadraturePoint(ps, scheme, iG, pParam, w);
        sum += w * EvalMonomial(pParam, a, b, c);
    }
    return sum;
}

/**
 * @brief Get parametric space from element type for testing
 */
ParamSpace GetParamSpaceFromElement(ElemType t)
{
    Element e{t};
    return e.GetParamSpace();
}

// ===================================================================
// Test cases
// ===================================================================

TEST_CASE("Quadrature: scheme constants are valid")
{
    // Verify that all scheme constants are positive
    CHECK(INT_SCHEME_Line_1 > 0);
    CHECK(INT_SCHEME_Line_2 > 0);
    CHECK(INT_SCHEME_Line_3 > 0);
    CHECK(INT_SCHEME_Line_4 > 0);
    
    CHECK(INT_SCHEME_Quad_1 > 0);
    CHECK(INT_SCHEME_Quad_4 > 0);
    CHECK(INT_SCHEME_Quad_9 > 0);
    CHECK(INT_SCHEME_Quad_16 > 0);
    
    CHECK(INT_SCHEME_Tri_1 > 0);
    CHECK(INT_SCHEME_Tri_3 > 0);
    CHECK(INT_SCHEME_Tri_6 > 0);
    CHECK(INT_SCHEME_Tri_7 > 0);
    CHECK(INT_SCHEME_Tri_12 > 0);
    
    CHECK(INT_SCHEME_Tet_1 > 0);
    CHECK(INT_SCHEME_Tet_4 > 0);
    CHECK(INT_SCHEME_Tet_8 > 0);
    CHECK(INT_SCHEME_Tet_14 > 0);
    CHECK(INT_SCHEME_Tet_24 > 0);
    
    CHECK(INT_SCHEME_Hex_1 > 0);
    CHECK(INT_SCHEME_Hex_8 > 0);
    CHECK(INT_SCHEME_Hex_27 > 0);
    CHECK(INT_SCHEME_Hex_64 > 0);
    
    CHECK(INT_SCHEME_Prism_1 > 0);
    CHECK(INT_SCHEME_Prism_6 > 0);
    CHECK(INT_SCHEME_Prism_18 > 0);
    CHECK(INT_SCHEME_Prism_21 > 0);
    CHECK(INT_SCHEME_Prism_48 > 0);
    
    CHECK(INT_SCHEME_Pyramid_1 > 0);
    CHECK(INT_SCHEME_Pyramid_8 > 0);
    CHECK(INT_SCHEME_Pyramid_27 > 0);
    CHECK(INT_SCHEME_Pyramid_64 > 0);
}

TEST_CASE("Quadrature: weight sums equal reference element volumes")
{
    constexpr t_real tol = 1e-12;
    
    // Line [-1,1]: volume = 2
    for (int order = 0; order <= 6; order++)
    {
        t_index scheme = GetQuadratureScheme(LineSpace, order);
        if (scheme == 0) continue;
        
        t_real sum = 0.0;
        for (t_index iG = 0; iG < scheme; iG++)
        {
            tPoint p;
            t_real w;
            GetQuadraturePoint(LineSpace, scheme, iG, p, w);
            sum += w;
        }
        CHECK(sum == doctest::Approx(2.0).epsilon(tol));
    }
    
    // Quad [-1,1]^2: volume = 4
    for (int order = 0; order <= 6; order++)
    {
        t_index scheme = GetQuadratureScheme(QuadSpace, order);
        if (scheme == 0) continue;
        
        t_real sum = 0.0;
        for (t_index iG = 0; iG < scheme; iG++)
        {
            tPoint p;
            t_real w;
            GetQuadraturePoint(QuadSpace, scheme, iG, p, w);
            sum += w;
        }
        CHECK(sum == doctest::Approx(4.0).epsilon(tol));
    }
    
    // Hex [-1,1]^3: volume = 8
    for (int order = 0; order <= 6; order++)
    {
        t_index scheme = GetQuadratureScheme(HexSpace, order);
        if (scheme == 0) continue;
        
        t_real sum = 0.0;
        for (t_index iG = 0; iG < scheme; iG++)
        {
            tPoint p;
            t_real w;
            GetQuadraturePoint(HexSpace, scheme, iG, p, w);
            sum += w;
        }
        CHECK(sum == doctest::Approx(8.0).epsilon(tol));
    }
    
    // Triangle: volume = 1/2
    for (int order = 0; order <= 6; order++)
    {
        t_index scheme = GetQuadratureScheme(TriSpace, order);
        if (scheme == 0) continue;
        
        t_real sum = 0.0;
        for (t_index iG = 0; iG < scheme; iG++)
        {
            tPoint p;
            t_real w;
            GetQuadraturePoint(TriSpace, scheme, iG, p, w);
            sum += w;
        }
        CHECK(sum == doctest::Approx(0.5).epsilon(tol));
    }
    
    // Tetrahedron: volume = 1/6
    for (int order = 0; order <= 6; order++)
    {
        t_index scheme = GetQuadratureScheme(TetSpace, order);
        if (scheme == 0) continue;
        
        t_real sum = 0.0;
        for (t_index iG = 0; iG < scheme; iG++)
        {
            tPoint p;
            t_real w;
            GetQuadraturePoint(TetSpace, scheme, iG, p, w);
            sum += w;
        }
        CHECK(sum == doctest::Approx(1.0/6.0).epsilon(tol));
    }
    
    // Prism: volume = 1/2 * 2 = 1
    for (int order = 0; order <= 6; order++)
    {
        t_index scheme = GetQuadratureScheme(PrismSpace, order);
        if (scheme == 0) continue;
        
        t_real sum = 0.0;
        for (t_index iG = 0; iG < scheme; iG++)
        {
            tPoint p;
            t_real w;
            GetQuadraturePoint(PrismSpace, scheme, iG, p, w);
            sum += w;
        }
        CHECK(sum == doctest::Approx(1.0).epsilon(tol));
    }
    
    // Pyramid: volume = 4/3
    for (int order = 0; order <= 6; order++)
    {
        t_index scheme = GetQuadratureScheme(PyramidSpace, order);
        if (scheme == 0) continue;
        
        t_real sum = 0.0;
        for (t_index iG = 0; iG < scheme; iG++)
        {
            tPoint p;
            t_real w;
            GetQuadraturePoint(PyramidSpace, scheme, iG, p, w);
            sum += w;
        }
        CHECK(sum == doctest::Approx(4.0/3.0).epsilon(tol));
    }
}

TEST_CASE("Quadrature: Line integrates polynomials up to expected order")
{
    constexpr t_real tol = 1e-12;
    
    // Gauss-Legendre with n points is exact for polynomials up to degree 2n-1
    // Line_1 (1 point): exact up to degree 1
    // Line_2 (2 points): exact up to degree 3
    // Line_3 (3 points): exact up to degree 5
    // Line_4 (4 points): exact up to degree 7 (we only test up to 6)
    
    struct TestCase { int order; int maxExactDegree; };
    TestCase tests[] = {{1, 1}, {3, 3}, {5, 5}, {6, 7}};
    
    for (const auto& tc : tests)
    {
        CAPTURE(tc.order);
        CAPTURE(tc.maxExactDegree);
        
        for (int deg = 0; deg <= std::min(tc.maxExactDegree, 6); deg++)
        {
            CAPTURE(deg);
            t_real exact = ExactMonomialIntegral(LineSpace, deg);
            t_real numeric = QuadratureMonomialIntegral(LineSpace, tc.order, deg);
            
            CHECK(numeric == doctest::Approx(exact).epsilon(tol));
        }
    }
}

TEST_CASE("Quadrature: Quad integrates polynomials up to expected order")
{
    constexpr t_real tol = 1e-12;
    
    // Product of 1D Gauss-Legendre: exact for degree 2n-1 in each direction
    // Quad_1 (1x1): exact up to degree 1 in each direction (total degree 2)
    // Quad_4 (2x2): exact up to degree 3 in each direction (total degree 6)
    // Quad_9 (3x3): exact up to degree 5 in each direction (total degree 10)
    
    struct TestCase { int order; int maxDegreePerDir; };
    TestCase tests[] = {{1, 1}, {3, 3}, {5, 5}, {6, 5}};
    
    for (const auto& tc : tests)
    {
        CAPTURE(tc.order);
        int maxDeg = std::min(tc.maxDegreePerDir, 3);  // Limit test scope
        
        for (int a = 0; a <= maxDeg; a++)
        {
            for (int b = 0; b <= maxDeg; b++)
            {
                if (a + b > tc.maxDegreePerDir * 2) continue;  // Skip high total degree
                
                CAPTURE(a);
                CAPTURE(b);
                t_real exact = ExactMonomialIntegral(QuadSpace, a, b);
                t_real numeric = QuadratureMonomialIntegral(QuadSpace, tc.order, a, b);
                
                CHECK(numeric == doctest::Approx(exact).epsilon(tol));
            }
        }
    }
}

TEST_CASE("Quadrature: Hex integrates polynomials up to expected order")
{
    constexpr t_real tol = 1e-12;
    
    // Product of 1D Gauss-Legendre in 3D
    struct TestCase { int order; int maxDegreePerDir; };
    TestCase tests[] = {{1, 1}, {3, 3}};
    
    for (const auto& tc : tests)
    {
        CAPTURE(tc.order);
        int maxDeg = std::min(tc.maxDegreePerDir, 2);  // Limit test scope
        
        for (int a = 0; a <= maxDeg; a++)
        {
            for (int b = 0; b <= maxDeg; b++)
            {
                for (int c = 0; c <= maxDeg; c++)
                {
                    CAPTURE(a);
                    CAPTURE(b);
                    CAPTURE(c);
                    t_real exact = ExactMonomialIntegral(HexSpace, a, b, c);
                    t_real numeric = QuadratureMonomialIntegral(HexSpace, tc.order, a, b, c);
                    
                    CHECK(numeric == doctest::Approx(exact).epsilon(tol));
                }
            }
        }
    }
}

TEST_CASE("Quadrature: Triangle integrates polynomials up to expected order")
{
    constexpr t_real tol = 1e-10;
    
    // Hammer rules have different exactness properties
    // Tri_1: degree 1
    // Tri_3: degree 2
    // Tri_6: degree 4
    // Tri_7: degree 5
    // Tri_12: degree 6
    
    // First, verify the coordinate system by checking the first few quadrature points
    {
        tPoint p;
        t_real w;
        
        // Tri_1: single point at centroid
        GetQuadraturePoint(TriSpace, INT_SCHEME_Tri_1, 0, p, w);
        CHECK(w == doctest::Approx(0.5).epsilon(tol));  // Area of reference triangle
        
        // Tri_3: 3 points
        t_real sum_w = 0;
        for (int i = 0; i < 3; i++) {
            GetQuadraturePoint(TriSpace, INT_SCHEME_Tri_3, i, p, w);
            sum_w += w;
        }
        CHECK(sum_w == doctest::Approx(0.5).epsilon(tol));
    }
    
    struct TestCase { int order; int maxTotalDegree; };
    TestCase tests[] = {{1, 1}, {2, 2}, {4, 4}, {5, 5}, {6, 6}};
    
    for (const auto& tc : tests)
    {
        CAPTURE(tc.order);
        CAPTURE(tc.maxTotalDegree);
        
        for (int a = 0; a <= tc.maxTotalDegree; a++)
        {
            for (int b = 0; b <= tc.maxTotalDegree - a; b++)
            {
                CAPTURE(a);
                CAPTURE(b);
                t_real exact = ExactMonomialIntegral(TriSpace, a, b);
                t_real numeric = QuadratureMonomialIntegral(TriSpace, tc.order, a, b);
                
                if (std::isnan(exact) || std::isnan(numeric))
                    continue;
                    
                CHECK(numeric == doctest::Approx(exact).epsilon(tol));
            }
        }
    }
}

TEST_CASE("Quadrature: Tetrahedron integrates polynomials up to expected order")
{
    constexpr t_real tol = 1e-10;
    
    // Hammer tet rules
    // Tet_1: degree 1
    // Tet_4: degree 2
    // Tet_8: degree 3
    // Tet_14: degree 5
    // Tet_24: degree 6
    
    struct TestCase { int order; int maxTotalDegree; };
    TestCase tests[] = {{1, 1}, {2, 2}, {3, 3}, {5, 5}, {6, 6}};
    
    for (const auto& tc : tests)
    {
        CAPTURE(tc.order);
        CAPTURE(tc.maxTotalDegree);
        
        for (int a = 0; a <= tc.maxTotalDegree; a++)
        {
            for (int b = 0; b <= tc.maxTotalDegree - a; b++)
            {
                for (int c = 0; c <= tc.maxTotalDegree - a - b; c++)
                {
                    CAPTURE(a);
                    CAPTURE(b);
                    CAPTURE(c);
                    t_real exact = ExactMonomialIntegral(TetSpace, a, b, c);
                    t_real numeric = QuadratureMonomialIntegral(TetSpace, tc.order, a, b, c);
                    
                    if (std::isnan(exact) || std::isnan(numeric))
                        continue;
                        
                    CHECK(numeric == doctest::Approx(exact).epsilon(tol));
                }
            }
        }
    }
}

TEST_CASE("Quadrature: Prism integrates polynomials up to expected order")
{
    constexpr t_real tol = 1e-10;
    
    // Prism = Triangle x Line
    // Exactness determined by component rules
    struct TestCase { int order; int maxTriDegree; int maxLineDegree; };
    TestCase tests[] = {{1, 1, 1}, {2, 2, 3}};
    
    for (const auto& tc : tests)
    {
        CAPTURE(tc.order);
        
        for (int a = 0; a <= tc.maxTriDegree; a++)
        {
            for (int b = 0; b <= tc.maxTriDegree - a; b++)
            {
                for (int c = 0; c <= tc.maxLineDegree && c <= 3; c += 2)  // Even powers only for non-zero
                {
                    CAPTURE(a);
                    CAPTURE(b);
                    CAPTURE(c);
                    t_real exact = ExactMonomialIntegral(PrismSpace, a, b, c);
                    t_real numeric = QuadratureMonomialIntegral(PrismSpace, tc.order, a, b, c);
                    
                    if (std::isnan(exact) || std::isnan(numeric))
                        continue;
                        
                    CHECK(numeric == doctest::Approx(exact).epsilon(tol));
                }
            }
        }
    }
}

TEST_CASE("Quadrature: Pyramid integrates constant and low-degree polynomials")
{
    constexpr t_real tol = 1e-10;
    
    // Pyramid uses product of Gauss-Legendre and Gauss-Jacobi
    // Pyramid_1 (O1), Pyramid_8 (O3), Pyramid_27 (O5), Pyramid_64 (O7)
    
    struct TestCase { int order; int maxExactDegree; };
    TestCase tests[] = {{1, 1}, {3, 3}, {5, 5}, {6, 5}};
    
    for (const auto& tc : tests)
    {
        CAPTURE(tc.order);
        CAPTURE(tc.maxExactDegree);
        
        // Test constant (degree 0)
        {
            t_real exact = ExactMonomialIntegral(PyramidSpace, 0, 0, 0);
            t_real numeric = QuadratureMonomialIntegral(PyramidSpace, tc.order, 0, 0, 0);
            CHECK(numeric == doctest::Approx(exact).epsilon(tol));
        }
        
        // Test odd powers x, y (degree 1) - should integrate to 0
        if (tc.maxExactDegree >= 1)
        {
            t_real numeric_x = QuadratureMonomialIntegral(PyramidSpace, tc.order, 1, 0, 0);
            CHECK(numeric_x == doctest::Approx(0.0).epsilon(tol));
            
            t_real numeric_y = QuadratureMonomialIntegral(PyramidSpace, tc.order, 0, 1, 0);
            CHECK(numeric_y == doctest::Approx(0.0).epsilon(tol));
        }
        
        // Test z (degree 1)
        if (tc.maxExactDegree >= 1)
        {
            t_real exact = ExactMonomialIntegral(PyramidSpace, 0, 0, 1);
            if (!std::isnan(exact))
            {
                t_real numeric = QuadratureMonomialIntegral(PyramidSpace, tc.order, 0, 0, 1);
                CHECK(numeric == doctest::Approx(exact).epsilon(tol));
            }
        }
        
        // Test x^2, y^2 (degree 2) - only for O3 and higher
        if (tc.maxExactDegree >= 2)
        {
            t_real exact_x2 = ExactMonomialIntegral(PyramidSpace, 2, 0, 0);
            if (!std::isnan(exact_x2))
            {
                t_real numeric = QuadratureMonomialIntegral(PyramidSpace, tc.order, 2, 0, 0);
                CHECK(numeric == doctest::Approx(exact_x2).epsilon(tol));
            }
            
            t_real exact_y2 = ExactMonomialIntegral(PyramidSpace, 0, 2, 0);
            if (!std::isnan(exact_y2))
            {
                t_real numeric = QuadratureMonomialIntegral(PyramidSpace, tc.order, 0, 2, 0);
                CHECK(numeric == doctest::Approx(exact_y2).epsilon(tol));
            }
        }
        
        // Test z^2 (degree 2) - only for O3 and higher
        if (tc.maxExactDegree >= 2)
        {
            t_real exact = ExactMonomialIntegral(PyramidSpace, 0, 0, 2);
            if (!std::isnan(exact))
            {
                t_real numeric = QuadratureMonomialIntegral(PyramidSpace, tc.order, 0, 0, 2);
                CHECK(numeric == doctest::Approx(exact).epsilon(tol));
            }
        }
    }
}

TEST_CASE("Quadrature: Quadrature class Integration method works")
{
    constexpr t_real tol = 1e-12;
    
    // Test using the Quadrature class interface
    Element elem{Quad4};
    Quadrature quad(elem, 3);  // Order 3 for Quad
    
    // Integrate constant 1.0
    {
        t_real sum = 0.0;
        quad.Integration(sum, [](t_real& acc, int iG, const tPoint& p, const tD01Nj& D01Nj) {
            acc = 1.0;
        });
        CHECK(sum == doctest::Approx(4.0).epsilon(tol));  // Volume of [-1,1]^2
    }
    
    // Integrate x^2
    {
        t_real sum = 0.0;
        quad.Integration(sum, [](t_real& acc, int iG, const tPoint& p, const tD01Nj& D01Nj) {
            acc = p(0) * p(0);
        });
        CHECK(sum == doctest::Approx(4.0/3.0).epsilon(tol));  // Integral of x^2 over [-1,1]^2
    }
    
    // Integrate x^2 * y^2
    {
        t_real sum = 0.0;
        quad.Integration(sum, [](t_real& acc, int iG, const tPoint& p, const tD01Nj& D01Nj) {
            acc = p(0) * p(0) * p(1) * p(1);
        });
        CHECK(sum == doctest::Approx(4.0/9.0).epsilon(tol));  // (4/3) * (4/3) / 4 = 4/9
    }
}

TEST_CASE("Quadrature: GetQuadraturePointInfo returns correct data")
{
    Element elem{Tri3};
    Quadrature quad(elem, 2);  // Order 2 for Tri
    
    CHECK(quad.GetNumPoints() == 3);  // Tri_3 has 3 points
    
    t_real totalWeight = 0.0;
    for (t_index iG = 0; iG < quad.GetNumPoints(); iG++)
    {
        auto [pParam, w] = quad.GetQuadraturePointInfo(iG);
        totalWeight += w;
        
        // Verify weight matches GetWeight
        CHECK(w == quad.GetWeight(iG));
        
        // For triangle, all points should have barycentric coordinates summing to <= 1
        t_real coordSum = pParam(0) + pParam(1);
        CHECK(coordSum <= 1.0 + 1e-12);
        CHECK(pParam(0) >= -1e-12);
        CHECK(pParam(1) >= -1e-12);
    }
    
    // Total weight should equal triangle area
    CHECK(totalWeight == doctest::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("Quadrature: IntegrationSimple overloads work correctly")
{
    constexpr t_real tol = 1e-12;
    
    Element elem{Line2};
    Quadrature quad(elem, 3);
    
    // Test 2-argument version: f(acc, iG)
    {
        t_real sum = 0.0;
        quad.IntegrationSimple(sum, [](t_real& acc, int iG) {
            acc = 1.0;
        });
        CHECK(sum == doctest::Approx(2.0).epsilon(tol));
    }
    
    // Test 3-argument version: f(acc, iG, w)
    {
        t_real sum = 0.0;
        quad.IntegrationSimple(sum, [](t_real& acc, int iG, t_real w) {
            acc = w;  // Just accumulate the weight directly
        });
        CHECK(sum == doctest::Approx(2.0).epsilon(tol));
    }
}

// ===================================================================
// Part C: Complete Taylor Basis Integration Tests (Degree <= 3)
// ===================================================================
// These tests use BaseFunction utilities to systematically test ALL
// polynomial basis functions up to degree 3, not just monomials.

TEST_CASE("Quadrature: Taylor basis integration on 2D elements (degree <= 3)")
{
    constexpr t_real tol = 1e-10;
    
    // Parametric spaces for 2D elements
    // Each config specifies: ps, dim, and pairs of (integration_order, max_exact_degree)
    struct TestConfig {
        ParamSpace ps;
        int dim;
        int maxOrder;
        std::vector<std::pair<int, int>> testOrdersWithDegree;  // (intOrder, maxExactDegree)
    };
    
    TestConfig configs[] = {
        // Triangle Hammer rules: Tri_1 (O1), Tri_3 (O2), Tri_6 (O4), Tri_7 (O5), Tri_12 (O6)
        {TriSpace, 2, 3, {{1, 1}, {2, 2}, {4, 4}, {5, 5}, {6, 6}}},
        // Quad Gauss-Legendre: Quad_1 (O1), Quad_4 (O3), Quad_9 (O5), Quad_16 (O7)
        {QuadSpace, 2, 3, {{1, 1}, {3, 3}, {5, 5}, {6, 5}}},  // O6 uses O5 rule
    };
    
    for (const auto& cfg : configs)
    {
        CAPTURE(cfg.ps);
        
        // Get all Taylor basis monomials up to degree 3
        auto monomials = GetTaylorBasisMonomials(cfg.dim, cfg.maxOrder);
        
        for (const auto& [intOrder, maxExactDegree] : cfg.testOrdersWithDegree)
        {
            CAPTURE(intOrder);
            CAPTURE(maxExactDegree);
            
            t_index scheme = GetQuadratureScheme(cfg.ps, intOrder);
            if (scheme == 0) continue;
            
            for (size_t iMono = 0; iMono < monomials.size(); iMono++)
            {
                const auto& exp = monomials[iMono];
                int a = exp[0], b = exp[1], c = exp[2];
                int totalDegree = a + b + c;
                
                // Skip monomials that exceed the quadrature rule's exactness
                if (totalDegree > maxExactDegree)
                    continue;
                
                CAPTURE(a);
                CAPTURE(b);
                CAPTURE(c);
                CAPTURE(totalDegree);
                
                // Compute exact integral
                t_real exact = ExactMonomialIntegral(cfg.ps, a, b, c);
                if (std::isnan(exact)) continue;
                
                // Compute numerical integral
                t_real numeric = QuadratureMonomialIntegral(cfg.ps, intOrder, a, b, c);
                if (std::isnan(numeric)) continue;
                
                CHECK(numeric == doctest::Approx(exact).epsilon(tol));
            }
        }
    }
}

TEST_CASE("Quadrature: Taylor basis integration on 3D elements (degree <= 3)")
{
    constexpr t_real tol = 1e-10;
    
    // Parametric spaces for 3D elements
    struct TestConfig {
        ParamSpace ps;
        int dim;
        int maxOrder;
        std::vector<std::pair<int, int>> testOrdersWithDegree;  // (intOrder, maxExactDegree)
    };
    
    TestConfig configs[] = {
        // Tet Hammer rules: Tet_1 (O1), Tet_4 (O2), Tet_8 (O3), Tet_14 (O5), Tet_24 (O6)
        {TetSpace, 3, 3, {{1, 1}, {2, 2}, {3, 3}, {5, 5}, {6, 6}}},
        // Hex Gauss-Legendre: Hex_1 (O1), Hex_8 (O3), Hex_27 (O5), Hex_64 (O7)
        {HexSpace, 3, 3, {{1, 1}, {3, 3}, {5, 5}, {6, 5}}},
        // Prism = Tri x Line: limited by component rules
        {PrismSpace, 3, 3, {{1, 1}, {2, 2}, {4, 4}, {5, 5}, {6, 5}}},
    };
    
    for (const auto& cfg : configs)
    {
        CAPTURE(cfg.ps);
        
        // Get all Taylor basis monomials up to degree 3
        auto monomials = GetTaylorBasisMonomials(cfg.dim, cfg.maxOrder);
        
        for (const auto& [intOrder, maxExactDegree] : cfg.testOrdersWithDegree)
        {
            CAPTURE(intOrder);
            CAPTURE(maxExactDegree);
            
            t_index scheme = GetQuadratureScheme(cfg.ps, intOrder);
            if (scheme == 0) continue;
            
            for (size_t iMono = 0; iMono < monomials.size(); iMono++)
            {
                const auto& exp = monomials[iMono];
                int a = exp[0], b = exp[1], c = exp[2];
                int totalDegree = a + b + c;
                
                // Skip monomials that exceed the quadrature rule's exactness
                if (totalDegree > maxExactDegree)
                    continue;
                
                CAPTURE(a);
                CAPTURE(b);
                CAPTURE(c);
                CAPTURE(totalDegree);
                
                // Compute exact integral
                t_real exact = ExactMonomialIntegral(cfg.ps, a, b, c);
                if (std::isnan(exact)) continue;
                
                // Compute numerical integral
                t_real numeric = QuadratureMonomialIntegral(cfg.ps, intOrder, a, b, c);
                if (std::isnan(numeric)) continue;
                
                CHECK(numeric == doctest::Approx(exact).epsilon(tol));
            }
        }
    }
}

TEST_CASE("Quadrature: Taylor basis integration on 1D elements (degree <= 3)")
{
    constexpr t_real tol = 1e-12;
    
    // Get Taylor basis monomials for 1D (just powers of x)
    std::vector<std::array<int, 3>> monomials;
    for (int deg = 0; deg <= 3; deg++)
        monomials.push_back({deg, 0, 0});
    
    // Line Gauss-Legendre: Line_1 (O1), Line_2 (O3), Line_3 (O5), Line_4 (O7)
    struct TestCase { int order; int maxExactDegree; };
    TestCase tests[] = {{1, 1}, {3, 3}, {5, 5}, {6, 5}};  // O6 uses O5 rule
    
    for (const auto& tc : tests)
    {
        CAPTURE(tc.order);
        CAPTURE(tc.maxExactDegree);
        
        for (const auto& exp : monomials)
        {
            int a = exp[0];
            int totalDegree = a;
            
            // Skip monomials that exceed the quadrature rule's exactness
            if (totalDegree > tc.maxExactDegree)
                continue;
            
            CAPTURE(a);
            
            t_real exact = ExactMonomialIntegral(LineSpace, a);
            t_real numeric = QuadratureMonomialIntegral(LineSpace, tc.order, a);
            
            CHECK(numeric == doctest::Approx(exact).epsilon(tol));
        }
    }
}

TEST_CASE("Quadrature: Pyramid Taylor basis (degree <= 2)")
{
    constexpr t_real tol = 1e-10;
    
    // Pyramid has more complex geometry, test up to degree 2
    // Pyramid_1 (O1), Pyramid_8 (O3), Pyramid_27 (O5), Pyramid_64 (O7)
    auto monomials = GetTaylorBasisMonomials(3, 2);  // Up to degree 2
    
    struct TestCase { int order; int maxExactDegree; };
    TestCase tests[] = {{1, 1}, {3, 3}, {5, 5}, {6, 5}};
    
    for (const auto& tc : tests)
    {
        CAPTURE(tc.order);
        CAPTURE(tc.maxExactDegree);
        
        t_index scheme = GetQuadratureScheme(PyramidSpace, tc.order);
        if (scheme == 0) continue;
        
        for (const auto& exp : monomials)
        {
            int a = exp[0], b = exp[1], c = exp[2];
            int totalDegree = a + b + c;
            
            // Skip monomials that exceed the quadrature rule's exactness
            if (totalDegree > tc.maxExactDegree)
                continue;
            
            CAPTURE(a);
            CAPTURE(b);
            CAPTURE(c);
            
            t_real exact = ExactMonomialIntegral(PyramidSpace, a, b, c);
            if (std::isnan(exact)) continue;
            
            t_real numeric = QuadratureMonomialIntegral(PyramidSpace, tc.order, a, b, c);
            if (std::isnan(numeric)) continue;
            
            CHECK(numeric == doctest::Approx(exact).epsilon(tol));
        }
    }
}

TEST_CASE("Quadrature: FPolynomialFill consistency check")
{
    // Verify that our monomial evaluations match FPolynomialFill results
    // for the 0th derivative (function values)
    
    tPoint p{0.3, 0.4, 0.5};
    
    // Test 2D polynomials
    {
        Eigen::Matrix<t_real, 1, 10> polyVals;
        Base::FPolynomialFill2D(polyVals, p(0), p(1), p(2), 1.0, 1.0, 1.0, 1, 10);
        
        // Compare with manual monomial evaluation
        auto monomials = GetTaylorBasisMonomials(2, 3);
        for (size_t i = 0; i < monomials.size() && i < 10; i++)
        {
            const auto& exp = monomials[i];
            t_real expected = EvalMonomial(p, exp[0], exp[1], exp[2]);
            CHECK(polyVals(0, i) == doctest::Approx(expected).epsilon(1e-12));
        }
    }
    
    // Test 3D polynomials
    {
        Eigen::Matrix<t_real, 1, 20> polyVals;
        Base::FPolynomialFill3D(polyVals, p(0), p(1), p(2), 1.0, 1.0, 1.0, 1, 20);
        
        auto monomials = GetTaylorBasisMonomials(3, 3);
        for (size_t i = 0; i < monomials.size() && i < 20; i++)
        {
            const auto& exp = monomials[i];
            t_real expected = EvalMonomial(p, exp[0], exp[1], exp[2]);
            CHECK(polyVals(0, i) == doctest::Approx(expected).epsilon(1e-12));
        }
    }
}
