#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "CFV/Math/TaylorBasis.hpp"
#include "CFV/Math/ReconstructionFunctional.hpp"

namespace M = DNDS::CFV::Math;
using S = DNDS::CFV::ReconstructionFunctionalSettings;
using Point = DNDS::Geom::tPoint;

namespace
{
    template <class Scalar>
    void CheckTypedTaylor(double tolerance)
    {
        namespace A = DNDS::Math;
        const std::array<double, 3> point{.21, -.37, .63}, center{-.19, .13, -.09};
        const double scale = 1.7;
        const M::TaylorBasis basis(3, 3);
        const auto parameters = basis.KernelParameters();
        const auto prepared = M::TaylorPointEvaluationT<Scalar>::FromPoint(point, center, scale);
        for (int derivative = 0; derivative < basis.Size(); ++derivative)
            for (int mode = 0; mode < basis.Size(); ++mode)
            {
                double expected = 1;
                const auto &alpha = basis.MultiIndex(mode), &beta = basis.MultiIndex(derivative);
                for (int d = 0; d < 3; ++d)
                {
                    if (beta[d] > alpha[d])
                    {
                        expected = 0;
                        break;
                    }
                    for (int k = 0; k < beta[d]; ++k)
                        expected *= alpha[d] - k;
                    expected *= std::pow(point[d] - center[d], alpha[d] - beta[d]) / std::pow(scale, alpha[d]);
                }
                CHECK(A::ScalarCast<double>(parameters.Derivative(derivative, mode, prepared)) == doctest::Approx(expected).epsilon(tolerance));
            }
        const Scalar two(2), three(3);
        CHECK(A::ScalarCast<double>(A::ScalarMin(two, three)) == 2);
        CHECK(A::ScalarCast<double>(A::ScalarMax(two, three)) == 3);
        CHECK(A::ScalarCast<double>(A::ScalarAbs(-two)) == 2);
        CHECK(A::ScalarCast<double>(A::ScalarSqrt(two)) == doctest::Approx(std::sqrt(2.)).epsilon(tolerance));
        CHECK(A::ScalarFinite(two));
        CHECK(A::ScalarEpsilon<Scalar>() > Scalar(0));
        CHECK(A::ScalarMinPositive<Scalar>() > Scalar(0));
    }
}

TEST_CASE("Compute scalar Taylor contexts preserve derivatives across explicit arithmetic choices")
{
    CheckTypedTaylor<double>(2e-13);
    CheckTypedTaylor<DNDS::Math::real_f32>(3e-6);
#ifdef DNDS_USE_CCCL_COMPUTE
    CheckTypedTaylor<DNDS::Math::real_f32f32>(2e-12);
    using Pair = DNDS::Math::real_f32f32;
    const auto increment = Pair(1.0000000001) - Pair(1.0);
    CHECK(DNDS::Math::ScalarCast<double>(increment) == doctest::Approx(1.000000082740371e-10).epsilon(1e-12));
#endif
}

TEST_CASE("Prepared Taylor points preserve nonunit-scale physical derivatives and zero-mean modes")
{
    const std::array<DNDS::real, 3> point{0.19, -0.37, 0.61}, center{-0.23, 0.11, -0.17};
    for (int dim : {2, 3})
        for (int degree : {1, 2, 3})
            for (DNDS::real scale : {0.7, 1.9})
            {
                const M::TaylorBasis basis(dim, degree);
                const auto parameters = basis.KernelParameters();
                const auto prepared = M::TaylorPointEvaluation::FromPoint(point, center, scale);
                std::array<DNDS::real, 3> normalized{};
                for (int d = 0; d < 3; ++d)
                    normalized[d] = (point[d] - center[d]) / scale;
                M::SmallMatrix legacy;
                basis.Derivatives(legacy, {point[0], point[1], point[2]}, {center[0], center[1], center[2]}, scale);
                for (int derivative = 0; derivative < basis.Size(); ++derivative)
                    for (int mode = 0; mode < basis.Size(); ++mode)
                    {
                        // Independent physical polynomial oracle: differentiate
                        // product_d (point_d-center_d)^alpha_d / scale^|alpha|.
                        // This does not use the prepared coordinates or powers.
                        DNDS::real expected = 1;
                        const auto &alpha = basis.MultiIndex(mode), &beta = basis.MultiIndex(derivative);
                        for (int d = 0; d < 3; ++d)
                        {
                            if (beta[d] > alpha[d])
                            {
                                expected = 0;
                                break;
                            }
                            for (int k = 0; k < beta[d]; ++k)
                                expected *= alpha[d] - k;
                            expected *= std::pow(point[d] - center[d], alpha[d] - beta[d]) / std::pow(scale, alpha[d]);
                        }
                        CAPTURE(dim);
                        CAPTURE(degree);
                        CAPTURE(scale);
                        CAPTURE(derivative);
                        CAPTURE(mode);
                        CHECK(parameters.Derivative(derivative, mode, prepared) == doctest::Approx(expected).epsilon(2e-13));
                        CHECK(parameters.Derivative(derivative, mode, normalized, scale) == doctest::Approx(expected).epsilon(2e-13));
                        CHECK(legacy(derivative, mode) == doctest::Approx(expected).epsilon(2e-13));
                    }
                // Cell moments affect values only, never physical derivatives.
                std::array<DNDS::real, 20> moments{};
                for (int mode = 0; mode < basis.Size(); ++mode)
                    moments[mode] = 0.03 * (mode + 1);
                M::SmallMatrix momentRows(1, basis.Size());
                for (int mode = 0; mode < basis.Size(); ++mode)
                    momentRows(0, mode) = moments[mode];
                M::SmallMatrix reconstruction;
                const M::SmallMatrix noMoments;
                basis.Reconstruction(reconstruction, {point[0], point[1], point[2]}, {center[0], center[1], center[2]}, scale, moments, degree);
                for (int derivative = 0; derivative < basis.Size(); ++derivative)
                    for (int mode = 1; mode < basis.Size(); ++mode)
                    {
                        CHECK(reconstruction(derivative, mode - 1) == doctest::Approx(
                                                                          parameters.Derivative(derivative, mode, prepared) - (derivative == 0 ? moments[mode] : 0))
                                                                          .epsilon(2e-13));
                        CHECK(parameters.ReconstructionDerivative(derivative, mode - 1, prepared, momentRows, 0) ==
                              doctest::Approx(reconstruction(derivative, mode - 1)).epsilon(2e-13));
                        if (derivative != 0)
                            CHECK(parameters.ReconstructionDerivative(derivative, mode - 1, prepared, noMoments, 0) ==
                                  doctest::Approx(reconstruction(derivative, mode - 1)).epsilon(2e-13));
                    }
            }
}

TEST_CASE("Taylor derivative pullback preserves Frobenius tensor norms through degree three")
{
    for (int dim : {2, 3})
    {
        M::TaylorBasis basis(dim, 3);
        M::SmallMatrix values;
        const auto rotation = Eigen::AngleAxisd(0.53, dim == 2 ? Point::UnitZ().eval() : Point(1, 2, 3).normalized()).toRotationMatrix();
        basis.Derivatives(values, {0.2, -0.3, 0.4}, {0, 0, 0}, 0.7);
        auto rotated = values;
        basis.Pullback(rotated, rotation);
        int squareMode = -1;
        for (int mode = 0; mode < basis.Size(); ++mode)
            if (basis.MultiIndex(mode) == std::array<int, 3>{2, 0, 0})
                squareMode = mode;
        REQUIRE(squareMode >= 0);
        for (int direction = 0; direction < dim; ++direction)
            CHECK(rotated(1 + direction, squareMode) == doctest::Approx(values(1, squareMode) * rotation(0, direction)).epsilon(1e-12));
        const double orderWeights[]{1, 1, 1, 1};
        double weights[20];
        M::FunctionalWeights(weights, dim, 3, 1, orderWeights);
        for (int degree = 0; degree <= 3; ++degree)
            for (int mode = 0; mode < basis.Size(); ++mode)
            {
                double before = 0, after = 0;
                for (int derivative = 0; derivative < basis.Size(); ++derivative)
                    if (basis.Degree(derivative) == degree)
                    {
                        before += weights[derivative] * DNDS::sqr(values(derivative, mode));
                        after += weights[derivative] * DNDS::sqr(rotated(derivative, mode));
                    }
                CHECK(after == doctest::Approx(before).epsilon(1e-12));
            }
        basis.Pullback(rotated, rotation.transpose());
        CHECK(rotated.isApprox(values, 1e-12));
    }
}

TEST_CASE("Central isotropic weight schemes preserve configured legacy formulas")
{
    S settings;
    auto values = M::ReconstructionOrderWeights(settings, 2, 3, 2, 4, {2, 1, 0});
    CHECK((values == std::array<double, 4>{1, 1, 0.5, 1. / 6}));
    settings.dirWeightScheme = S::DirectionalWeight::HQM_OPT;
    settings.geomWeightScheme = S::GeometricWeight::HQM_SD;
    settings.geomWeightBias = 1;
    values = M::ReconstructionOrderWeights(settings, 2, 2, 2, 4, {2, 1, 0});
    CHECK(values[0] == doctest::Approx(1 + std::pow(0.5, 0.25)));
    CHECK(values[1] / values[0] == doctest::Approx(0.4643));
    settings.dirWeightScheme = S::DirectionalWeight::ManualDirWeight;
    settings.manualDirWeights = {2, 3, 4, 5};
    settings.geomWeightScheme = S::GeometricWeight::SD_Power;
    settings.geomWeightBias = 0.1;
    settings.geomWeightPower1 = 2;
    settings.geomWeightPower2 = -2;
    values = M::ReconstructionOrderWeights(settings, 3, 3, 2, 4, {2, 1, 0});
    CHECK(values[2] == doctest::Approx(4 * (0.1 + 2. / 4)));
    settings.dirWeightScheme = S::DirectionalWeight::TEST_OPT;
    auto first = M::ReconstructionOrderWeights(settings, 3, 3, 2, 4, {2, 1, 0});
    settings.geomWeightBias = 100;
    CHECK(first == M::ReconstructionOrderWeights(settings, 3, 3, 2, 4, {2, 1, 0}));
    settings.dirWeightScheme = static_cast<S::DirectionalWeight>(999);
    CHECK_THROWS(settings.Validate());
}
