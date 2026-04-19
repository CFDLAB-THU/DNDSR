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
    auto F = [](double x) { return x * x; };
    double v = Scalar::BisectSolveLower(F, 0.0, 3.0, 4.0, 100);
    CHECK(v == doctest::Approx(2.0).epsilon(1e-10));
}

TEST_CASE("BisectSolveLower: sin(x) = 0.5 => x ~ 0.5236")
{
    auto F = [](double x) { return std::sin(x); };
    double v = Scalar::BisectSolveLower(F, 0.0, 1.5, 0.5, 100);
    double expected = std::asin(0.5);
    CHECK(v == doctest::Approx(expected).epsilon(1e-10));
}

TEST_CASE("BisectSolveLower: linear f(x)=x, target=0.75")
{
    auto F = [](double x) { return x; };
    double v = Scalar::BisectSolveLower(F, 0.0, 1.0, 0.75, 100);
    CHECK(v == doctest::Approx(0.75).epsilon(1e-10));
}

TEST_CASE("BisectSolveLower: exp(x) = e => x = 1")
{
    auto F = [](double x) { return std::exp(x); };
    double v = Scalar::BisectSolveLower(F, 0.0, 2.0, std::exp(1.0), 100);
    CHECK(v == doctest::Approx(1.0).epsilon(1e-10));
}

TEST_CASE("BisectSolveLower: few iterations gives lower accuracy")
{
    auto F = [](double x) { return x * x; };
    double v = Scalar::BisectSolveLower(F, 0.0, 3.0, 4.0, 5);
    // 5 iterations: interval halves 5 times: 3/2^5 = 0.09375
    CHECK(std::abs(v - 2.0) < 0.1);
    CHECK(std::abs(v - 2.0) > 1e-6); // should not be exact with only 5 iters
}
