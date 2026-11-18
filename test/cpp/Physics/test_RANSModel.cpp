#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Physics/RANS/Model.hpp"
#include "Physics/RANS/PointKernels.hpp"
#include <Eigen/Geometry>

namespace R = DNDS::Physics::RANS;
namespace
{
    R::Point Fixture()
    {
        R::Point point;
        point.state << 1.3, .4, -.2, .1, 3.2, .018, 2.7;
        point.gradient << .1, .12, -.2, .08, .2, .005, -.02,
            -.07, .32, -.08, .15, -.1, -.012, .01,
            .02, -.05, .11, -.03, .01, .003, .005;
        point.molecularViscosity = .0017;
        point.wallDistance = .09;
        point.cellLength = .025;
        point.cp = 3.5;
        return point;
    }
    void Near(double a, double b) { CHECK(a == doctest::Approx(b).epsilon(2e-13).scale(1e-10)); }
}

TEST_CASE("Shared SA property and diffusion extraction preserves the Euler formulas")
{
    auto point = Fixture();
    R::ModelSettings settings;
    settings.model = "SA";
    settings.viscosityReference = .004;
    R::Model model(settings);
    const Eigen::Vector3d normal = Eigen::Vector3d(1, 2, -1).normalized();
    for (double transported : {-.02, .018, 12.})
    {
        point.state[5] = transported;
        point.state[6] = 0;
        double chi = transported * settings.viscosityReference / point.molecularViscosity;
        double smoothed = chi;
#ifdef USE_NS_SA_NEGATIVE_MODEL
        if (smoothed < 10)
            smoothed = .05 * std::log(1 + std::exp(20 * smoothed));
#endif
        const double chi3 = smoothed * smoothed * smoothed;
        const double mut = point.molecularViscosity * std::max(smoothed * chi3 / (chi3 + std::pow(7.1, 3)), 0.);
        const auto properties = model.Transport(point);
        Near(properties.viscosity, mut);
        Near(properties.conductivity, point.cp * mut / settings.turbulentPrandtl);
        Near(properties.speciesDiffusivity, mut / (point.state[0] * settings.turbulentSchmidt));
        const auto gradient = (point.gradient.col(5) - point.state[5] / point.state[0] * point.gradient.col(0)) / point.state[0];
        double fn = 1;
#ifdef USE_NS_SA_NEGATIVE_MODEL
        if (transported < 0)
            fn = (16 + std::pow(chi, 3)) / (16 - std::pow(chi, 3));
#endif
        Near(model.DiffusionFlux(point, normal, mut)[0], gradient.dot(normal) *
                                                             (point.molecularViscosity + transported * settings.viscosityReference * fn) / (2. / 3.));
        CHECK(model.DiffusionFlux(point, normal, mut)[1] == 0);
    }
}

TEST_CASE("RANS adapter dispatch retains source and destruction diagonal conventions")
{
    const auto point = Fixture();
    for (const char *name : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
    {
        R::ModelSettings settings;
        settings.model = name;
        settings.viscosityReference = .003;
        settings.desScale = 1e10;
        settings.sa.useFt2 = false;
        settings.sa.productionLimit = 30;
        settings.kOmega.productionLimit = 15;
        settings.sst.productionLimit = 10;
        const R::Model model(settings);
        if (model.Kind() == R::ModelKind::KOmegaSST)
        {
            auto missingLength = point;
            missingLength.cellLength = 0;
            CHECK_THROWS(model.Source(missingLength));
        }
        for (int mode = 0; mode < 2; ++mode)
        {
            Eigen::Matrix<double, 7, 1> expected = Eigen::Matrix<double, 7, 1>::Zero();
            if (model.Kind() == R::ModelKind::SA)
                R::GetSource_SA<3>(point.state, point.gradient, settings.viscosityReference, point.molecularViscosity, point.gamma,
                                   point.wallDistance, settings.desScale, point.cellLength, settings.desMode, expected,
                                   settings.saRotationCorrection, mode, settings.sa, settings.saVersion);
            else if (model.Kind() == R::ModelKind::KOmegaWilcox)
                R::GetSource_KOWilcox<3>(point.state, point.gradient, point.molecularViscosity, point.wallDistance, expected, mode, settings.kOmega);
            else if (model.Kind() == R::ModelKind::KOmegaSST)
                R::GetSource_SST<3>(point.state, point.gradient, point.molecularViscosity, point.wallDistance,
                                    point.cellLength * settings.desScale, expected, mode, settings.sst);
            else
                R::GetSource_RealizableKe<3>(point.state, point.gradient, point.molecularViscosity, point.wallDistance, expected, mode, settings.realizableKE);
            const auto actual = mode ? model.ImplicitSourceDiagonal(point) : model.Source(point);
            for (int j = 0; j < model.ComponentNumber(); ++j)
                Near(actual[j], expected[5 + j]);
        }
    }
}

TEST_CASE("RANS transport source and diffusion are independent of Cartesian frame orientation")
{
    const auto point = Fixture();
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(.63, Eigen::Vector3d(1, -2, 3).normalized()).toRotationMatrix();
    auto rotated = point;
    rotated.state.segment<3>(1) = rotation * point.state.segment<3>(1);
    rotated.gradient = rotation * point.gradient;
    rotated.gradient.block<3, 3>(0, 1) = rotation * point.gradient.block<3, 3>(0, 1) * rotation.transpose();
    const Eigen::Vector3d normal = Eigen::Vector3d(1, 2, 4).normalized();
    for (const char *name : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
    {
        R::ModelSettings settings;
        settings.model = name;
        const R::Model model(settings);
        const auto a = model.Transport(point), b = model.Transport(rotated);
        Near(a.viscosity, b.viscosity);
        const auto f = model.DiffusionFlux(point, normal, a.viscosity), g = model.DiffusionFlux(rotated, rotation * normal, b.viscosity);
        const auto s = model.Source(point), t = model.Source(rotated);
        for (int j = 0; j < model.ComponentNumber(); ++j)
        {
            Near(f[j], g[j]);
            Near(s[j], t[j]);
        }
    }
    R::ModelSettings invalid;
    invalid.model = "unknown";
    CHECK_THROWS(R::Model{invalid});
    invalid.model = "SA";
    invalid.viscosityReference = 0;
    CHECK_THROWS(R::Model{invalid});
}

TEST_CASE("SST-2003 source blends inner and outer production coefficients")
{
    R::ModelSettings settings;
    settings.model = "KOmegaSST";
    settings.desScale = 1e100;
    const R::Model model(settings);
    auto point = Fixture();
    point.state << 1, 1, 0, 0, 3, 1e-4, 2;
    point.gradient.setZero();
    point.gradient(1, 1) = 1; // Below the production cap in both wall-distance limits.
    point.molecularViscosity = 1e-5;
    constexpr double innerGamma = 5. / 9.;
    constexpr double outerGamma = .44;
    for (bool inner : {true, false})
    {
        point.wallDistance = inner ? 1e-5 : 1e6;
        const double expected = (inner ? innerGamma : outerGamma) - (inner ? .075 : .0828) * 4;
        CHECK(model.Source(point)[1] == doctest::Approx(expected).epsilon(1e-12));
    }
}

TEST_CASE("Status-returning RANS point math preserves the CPU closure and reports missing source geometry")
{
    const Eigen::Vector3d normal(.6, .8, 0);
    for (const char *name : {"SA", "KOmegaWilcox", "KOmegaSST"})
        for (int desMode = 0; desMode < 4; ++desMode)
        {
            R::ModelSettings settings;
            settings.model = name;
            settings.viscosityReference = .004;
            settings.desScale = desMode == 0 ? DNDS::veryLargeReal : .65;
            settings.desMode = desMode;
            settings.sst.implicitAdditionalSinks = true;
            const R::Model model(settings);
            auto point = Fixture();
            const auto turbulent = R::TryTransport(model.KernelParameters(), point);
            REQUIRE(turbulent);
            const auto reference = model.Transport(point);
            Near(turbulent.value.viscosity, reference.viscosity);
            Near(turbulent.value.conductivity, reference.conductivity);
            Near(turbulent.value.speciesDiffusivity, reference.speciesDiffusivity);
            const auto diffusion = model.TryDiffusionFlux(point, normal, reference.viscosity);
            REQUIRE(diffusion);
            const auto expectedDiffusion = model.DiffusionFlux(point, normal, reference.viscosity);
            for (int j = 0; j < 2; ++j)
                Near(diffusion.value[j], expectedDiffusion[j]);
            for (int mode : {0, 1})
            {
                const auto actual = model.TrySource(point, mode);
                REQUIRE(actual);
                const auto expected = mode ? model.ImplicitSourceDiagonal(point) : model.Source(point);
                for (int j = 0; j < 2; ++j)
                    Near(actual.value[j], expected[j]);
            }
            // Transport needs no source cell length. Only enabled DES and SST
            // consume it, and the kernel-facing API returns an error, not throw.
            point.cellLength = 0;
            CHECK(model.TryTransport(point));
            if ((model.Kind() == R::ModelKind::SA && settings.desScale <= 100) || model.Kind() == R::ModelKind::KOmegaSST)
                CHECK(model.TrySource(point).status.code == DNDS::NumericalError::InvalidInput);
        }
    R::PointParameters unsupported;
    unsupported.kind = R::ModelKind::RealizableKE;
    CHECK(R::TryTransport(unsupported, Fixture()).status.code == DNDS::NumericalError::UnsupportedRequest);
}
