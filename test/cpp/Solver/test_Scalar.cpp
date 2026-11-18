/**
 * @file test_Scalar.cpp
 * @brief Unit tests for BisectSolveLower in Solver/Scalar.hpp.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Solver/Scalar.hpp"

#include <cmath>

using namespace DNDS;

TEST_CASE("BisectSolveLower: x^2 = 4 => x = 2")
{
    auto F = [](double x)
    { return x * x; };
    double v = Scalar::BisectSolveLower(F, 0.0, 3.0, 4.0, 100);
    CHECK(v == doctest::Approx(2.0).epsilon(1e-10));
}

TEST_CASE("BisectSolveLower: few iterations gives lower accuracy")
{
    auto F = [](double x)
    { return x * x; };
    double v = Scalar::BisectSolveLower(F, 0.0, 3.0, 4.0, 5);
    // 5 iterations: interval halves 5 times: 3/2^5 = 0.09375
    CHECK(std::abs(v - 2.0) < 0.1);
    CHECK(std::abs(v - 2.0) > 1e-6); // should not be exact with only 5 iters
}
