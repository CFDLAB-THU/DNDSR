#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Plans/LimitedVRPlan.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "Euler/LimitedVR.hpp" // Independent retained reference; no Euler solver/storage.
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include <limits>

namespace C = DNDS::CFV;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Native v2/v3 gates preserve legacy thresholds, contact response and expansion admission")
{
    for (const std::string variant : {"v2", "v3"})
        for (bool expansion : {false, true})
        {
            auto settings = C::LimitedVRSettings::ForVariant(variant);
            settings.detectExpansion = expansion;
            settings.Validate();
            DNDS::Euler::LimitedVRSettings old;
            old.alphaMax = settings.alphaMax;
            old.contactAlphaMax = settings.contactAlphaMax;
            old.pressureJumpStart = settings.pressureJump[0];
            old.pressureJumpFull = settings.pressureJump[1];
            old.compressionStart = settings.compression[0];
            old.compressionFull = settings.compression[1];
            old.contactDensityJumpStart = settings.densityJump[0];
            old.contactDensityJumpFull = settings.densityJump[1];
            old.detectExpansion = expansion;
            for (double scale : {1e-12, 1., 1e12})
                for (double pressure : {1., 1.05, 1.2, 2.})
                    for (double velocity : {-.5, -.07, 0., .07, .5})
                    {
                        const auto current = C::Math::EvaluateLimitedVRFaceSensor(scale, 2 * scale, scale, pressure * scale, velocity, 0., 1., 1., settings);
                        const auto shock = DNDS::Euler::EvaluateLimitedVRFaceSensor(scale, pressure * scale, velocity, 0., 1., 1., old);
                        const auto contact = DNDS::Euler::EvaluateLimitedVRContactFaceSensor(scale, 2 * scale, old);
                        CHECK(current.pressureJump == shock.pressureJump);
                        CHECK(current.compression == shock.compression);
                        CHECK(current.densityJump == contact.densityJump);
                        CHECK(current.shock == shock.alpha);
                        CHECK(current.contact == contact.alpha);
                        if (pressure == 1 || (!expansion && velocity <= 0))
                            CHECK(current.shock == 0);
                    }
        }
    auto settings = C::LimitedVRSettings::ForVariant("v2");
    CHECK(settings.alphaMax == .5);
    CHECK(settings.contactAlphaMax == 0);
    CHECK_THROWS_AS(C::LimitedVRSettings::ForVariant("v4"), std::exception);
    settings.neighborLayers = 5;
    CHECK_THROWS_AS(settings.Validate(), std::exception);
    settings = C::LimitedVRSettings{};
    settings.pressureJump[1] = settings.pressureJump[0];
    CHECK_THROWS_AS(settings.Validate(), std::exception);
}

TEST_CASE("Supported extremum seeds preserve incidence counts, retention and boundary exclusion")
{
    const auto settings = C::LimitedVRSettings::ForVariant("v3");
    C::Math::LimitedVRExtremumStencil maximum;
    maximum.Observe(1.1, 1., 1., settings.equalityTolerance);
    maximum.Observe(1.1, 1.05, 0., settings.equalityTolerance);
    maximum.Observe(1.1, 1.1, 0., settings.equalityTolerance);
    maximum.Observe(1.1, 1.1, 0., settings.equalityTolerance);
    CHECK(maximum.Correction(1.1, 0, 1, settings.pressureSupport, settings) == 1);
    maximum.boundary = true;
    CHECK(maximum.Correction(1.1, 1, 1, settings.pressureSupport, settings) == 0);
    maximum.boundary = false;
    maximum.equal = 1;
    CHECK(maximum.Correction(1.1, 1, 1, settings.pressureSupport, settings) == 0);
    C::Math::LimitedVRExtremumStencil plateau;
    for (int i = 0; i < 4; ++i)
        plateau.Observe(1., 1., .2, settings.equalityTolerance);
    CHECK(plateau.Correction(1., .8, 1., settings.pressureSupport, settings) == doctest::Approx(.76));
    plateau.support = 0;
    CHECK(plateau.Correction(1., .8, 1., settings.pressureSupport, settings) == 0);
    CHECK(plateau.Correction(1., .8, 0., settings.pressureSupport, settings) == 0);
}

TEST_CASE("LVR neighbor rounds cross rank boundaries with synchronous Cartesian distance and saturation oracles")
{
    constexpr I width = 24;
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, width).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    const auto plan = C::LimitedVRPlan::Prepare(metrics);
    const int active = mpi.size > 2 ? mpi.size - 1 : mpi.size;
    const I seedX = 12, seedY = active > 1 ? DNDS::EvenSplitRange(0, active, width).second - 1 : 12;
    for (int layout = 0; layout < 3; ++layout)
        for (bool excludeBoundary : {false, true})
            for (bool saturatedOnly : {false, true})
            {
                const auto shape = layout == 0 ? R::RowLayout::Dense(0, 3, 5) : layout == 1 ? R::RowLayout::ColMajor(0, 3, {16, 1})
                                                                                            : R::RowLayout::TiledColMajor(0, 3, 16);
                auto gates = metrics.BuildField(shape), all = metrics.BuildField(shape), split = metrics.BuildField(shape);
                {
                    auto write = gates.Owned().Write();
                    const auto v = write.View<R::UniformTableRows>();
                    for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    {
                        const I id = mesh->Entities(G::EntityKind::Cell).LocalToGlobal(iCell);
                        const bool seed = id == seedX + width * seedY;
                        v(iCell, 0) = seed ? 1 : 0;
                        v(iCell, 1) = seed ? .3 : 0;
                        v(iCell, 2) = .001 * id;
                    }
                }
                auto halo = metrics.BindHalo(gates);
                for (int layer = 1; layer <= 2; ++layer)
                {
                    auto transfer = halo.Begin();
                    plan.NeighborMaximum(gates, split, {true, true, false}, excludeBoundary, saturatedOnly, {1, 1, 1}, C::ReconstructionRegion::Core);
                    if (metrics.Cells().NumGhost())
                    {
                        const auto before = split.Owned().Version();
                        CHECK_THROWS_AS(plan.NeighborMaximum(gates, split, {true, true, false}, excludeBoundary, saturatedOnly), std::exception);
                        CHECK(split.Owned().Version() == before);
                    }
                    transfer.Wait();
                    plan.NeighborMaximum(gates, split, {true, true, false}, excludeBoundary, saturatedOnly, {1, 1, 1}, C::ReconstructionRegion::HaloDependent);
                    plan.NeighborMaximum(gates, all, {true, true, false}, excludeBoundary, saturatedOnly);
                    {
                        auto a = all.Owned().Read(), b = split.Owned().Read();
                        const auto av = a.View<R::UniformTableRows>(), bv = b.View<R::UniformTableRows>();
                        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                        {
                            const I id = mesh->Entities(G::EntityKind::Cell).LocalToGlobal(iCell), x = id % width, y = id / width;
                            const bool boundary = x == 0 || x == width - 1 || y == 0 || y == width - 1;
                            const bool reached = std::abs(x - seedX) + std::abs(y - seedY) <= layer && (!excludeBoundary || !boundary);
                            CHECK(av(iCell, 0) == (reached ? 1 : 0));
                            CHECK(av(iCell, 1) == ((saturatedOnly ? x == seedX && y == seedY : reached) ? .3 : 0));
                            CHECK(av(iCell, 2) == .001 * id);
                            for (I j = 0; j < 3; ++j)
                                CHECK(av(iCell, j) == bv(iCell, j));
                        }
                    }
                    A::Copy(gates, split);
                }
                if (metrics.NumCell())
                    CHECK_THROWS_WITH_AS(plan.NeighborMaximum(gates, gates, {true, true, false}, false), doctest::Contains("aliases input"), std::exception);
            }
}

TEST_CASE("Distributed v2/v3 correction seeds distinguish mean contact support, transverse equality and history")
{
    constexpr I width = 24;
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, width).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    const auto plan = C::LimitedVRPlan::Prepare(metrics);
    for (const std::string variant : {"v2", "v3"})
        for (int layout = 0; layout < 3; ++layout)
        {
            const auto settings = C::LimitedVRSettings::ForVariant(variant);
            const auto shape = [&](I columns)
            {
                return layout == 0 ? R::RowLayout::Dense(0, columns, columns + 2) : layout == 1 ? R::RowLayout::ColMajor(0, columns, {16, 1})
                                                                                                : R::RowLayout::TiledColMajor(0, columns, 16);
            };
            auto scalar = metrics.BuildField(shape(2)), gates = metrics.BuildField(shape(3)), all = metrics.BuildField(shape(3)), split = metrics.BuildField(shape(3));
            R::Array<double> previous(R::RowLayout::Dense(metrics.NumCell(), 2));
            {
                auto s = scalar.Owned().Write(), g = gates.Owned().Write(), p = previous.Write();
                const auto sv = s.View<R::UniformTableRows>(), gv = g.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                {
                    const I x = mesh->Entities(G::EntityKind::Cell).LocalToGlobal(iCell) % width;
                    sv(iCell, 0) = x < 8 ? 1 : x == 8 ? 1.1
                                                      : 1.05;
                    sv(iCell, 1) = x < 14 ? 1 : x == 14 ? 1.2
                                                        : 1.1;
                    gv(iCell, 0) = x == 7 || x == 9 ? .2 * settings.alphaMax : 0;
                    gv(iCell, 1) = 0; // The support must come from column2.
                    gv(iCell, 2) = x == 13 ? .2 * settings.contactAlphaMax : 0;
                    p.View()(iCell, 0) = x == 10 ? .8 * settings.alphaMax : 0;
                    p.View()(iCell, 1) = 0;
                }
            }
            auto scalarHalo = metrics.BindHalo(scalar), gateHalo = metrics.BindHalo(gates);
            auto scalarTransfer = scalarHalo.Begin(), gateTransfer = gateHalo.Begin();
            plan.BuildExtremumSeeds(scalar, gates, previous, split, settings, C::ReconstructionRegion::Core);
            if (metrics.Cells().NumGhost())
            {
                const auto before = split.Owned().Version();
                CHECK_THROWS_AS(plan.BuildExtremumSeeds(scalar, gates, previous, split, settings), std::exception);
                CHECK(split.Owned().Version() == before);
            }
            scalarTransfer.Wait();
            gateTransfer.Wait();
            plan.BuildExtremumSeeds(scalar, gates, previous, split, settings, C::ReconstructionRegion::HaloDependent);
            plan.BuildExtremumSeeds(scalar, gates, previous, all, settings);
            auto a = all.Owned().Read(), b = split.Owned().Read();
            const auto av = a.View<R::UniformTableRows>(), bv = b.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                const I id = mesh->Entities(G::EntityKind::Cell).LocalToGlobal(iCell), x = id % width, y = id / width;
                const bool boundary = x == 0 || x == width - 1 || y == 0 || y == width - 1;
                for (I j = 0; j < 3; ++j)
                    CHECK(av(iCell, j) == bv(iCell, j));
                if (boundary)
                    CHECK(av(iCell, 0) == 0);
                else if (x == 8)
                    CHECK(av(iCell, 0) == settings.alphaMax);
                else if (x == 10)
                    CHECK(av(iCell, 0) == doctest::Approx(.76 * settings.alphaMax));
                if (variant == "v2" || boundary)
                    CHECK(av(iCell, 1) == 0);
                else if (x == 14)
                    CHECK(av(iCell, 1) == settings.contactAlphaMax);
                CHECK(av(iCell, 2) == 0);
            }
        }
}
TEST_CASE("A-weighted limited VR preserves generic components, relaxation and split SOR semantics")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 24).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    for (bool sor : {false, true})
        for (int layout = 0; layout < 3; ++layout)
        {
            C::ReconstructionSettings settings;
            settings.maxOrder = 3;
            settings.jacobiRelax = .7;
            settings.SORInstead = sor;
            settings.coreFirst = true;
            auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
            const I width = 4 * plan.NRecDOF();
            const auto shape = [&](I columns)
            {
                return layout == 0 ? R::RowLayout::Dense(0, columns, columns + 2) : layout == 1 ? R::RowLayout::ColMajor(0, columns, {16, 1})
                                                                                                : R::RowLayout::TiledColMajor(0, columns, 16);
            };
            auto mean = metrics.BuildField(shape(4)), input = metrics.BuildField(shape(width)), target = metrics.BuildField(shape(width));
            auto ordinary = plan.BuildWorkspace(4, shape(width), {1}), full = plan.BuildWorkspace(4, shape(width), {1}), split = plan.BuildWorkspace(4, shape(width), {1});
            auto boundary = plan.BuildBoundaryValues(4);
            R::Array<double> alpha(R::RowLayout::Dense(metrics.NumCell(), 1));
            {
                auto u = mean.Owned().Write(), x = input.Owned().Write(), t = target.Owned().Write();
                const auto uv = u.View<R::UniformTableRows>(), xv = x.View<R::UniformTableRows>(), tv = t.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                {
                    const double id = mesh->Entities(G::EntityKind::Cell).LocalToGlobal(iCell);
                    for (I j = 0; j < 4; ++j)
                        uv(iCell, j) = 1 + .002 * id + .1 * j;
                    for (I j = 0; j < width; ++j)
                    {
                        xv(iCell, j) = .001 * std::sin(.1 * id + j);
                        tv(iCell, j) = .003 * std::cos(.2 * id - j);
                    }
                }
                auto b = boundary.values.Write();
                for (I iPoint = 0; iPoint < boundary.values.RowCount(); ++iPoint)
                    for (I j = 0; j < 4; ++j)
                        b.View()(iPoint, j) = 1 + .1 * j;
            }
            auto meanHalo = metrics.BindHalo(mean), inputHalo = metrics.BindHalo(input);
            meanHalo.Begin().Wait();
            inputHalo.Begin().Wait();
            const auto inputVersion = input.Owned().Version(), meanVersion = mean.Owned().Version(), targetVersion = target.Owned().Version();
            plan.DoReconstructionIterTo(mean, input, ordinary, boundary);
            for (double fraction : {0., .4, 1.})
            {
                {
                    auto a = alpha.Write();
                    for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                        a.View()(iCell, 0) = fraction;
                }
                plan.DoReconstructionIterLimitedTo(mean, input, full, boundary, target, alpha);
                auto meanTransfer = meanHalo.Begin(), coefficientTransfer = inputHalo.Begin();
                CHECK_NOTHROW(plan.DoReconstructionIterLimitedTo(mean, input, split, boundary, target, alpha, C::ReconstructionRegion::Core));
                if (metrics.Cells().NumGhost())
                {
                    const auto before = split.temporary.Owned().Version();
                    CHECK_THROWS_AS(plan.DoReconstructionIterLimitedTo(mean, input, split, boundary, target, alpha, C::ReconstructionRegion::HaloDependent), std::exception);
                    CHECK(split.temporary.Owned().Version() == before);
                }
                meanTransfer.Wait();
                coefficientTransfer.Wait();
                plan.DoReconstructionIterLimitedTo(mean, input, split, boundary, target, alpha, C::ReconstructionRegion::HaloDependent);
                const auto old = input.Owned().Read(), desired = target.Owned().Read(), raw = ordinary.temporary.Owned().Read();
                const auto all = full.temporary.Owned().Read(), parts = split.temporary.Owned().Read();
                const auto x = old.View<R::UniformTableRows>(), t = desired.View<R::UniformTableRows>(), r = raw.View<R::UniformTableRows>();
                const auto a = all.View<R::UniformTableRows>(), b = parts.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I j = 0; j < width; ++j)
                    {
                        CHECK(a(iCell, j) == b(iCell, j));
                        if (fraction == 0)
                            CHECK(a(iCell, j) == r(iCell, j));
                        else if (fraction == 1 || !sor)
                        {
                            // Independent A-weighted-penalty update, before
                            // relaxation. SOR's intermediate predecessors depend
                            // on alpha, so its mixed case uses split equivalence.
                            const double expected = (1 - settings.jacobiRelax) * x(iCell, j) + fraction * settings.jacobiRelax * t(iCell, j) +
                                                    (1 - fraction) * (r(iCell, j) - (1 - settings.jacobiRelax) * x(iCell, j));
                            CHECK(a(iCell, j) == doctest::Approx(expected).epsilon(1e-11));
                        }
                    }
            }
            CHECK(input.Owned().Version() == inputVersion);
            CHECK(mean.Owned().Version() == meanVersion);
            CHECK(target.Owned().Version() == targetVersion);
            if (metrics.NumCell())
            {
                auto aliased = target;
                aliased.Owned() = full.temporary.Owned();
                const auto before = full.temporary.Owned().Version();
                CHECK_THROWS_WITH_AS(plan.DoReconstructionIterLimitedTo(mean, input, full, boundary, aliased, alpha),
                                     doctest::Contains("output aliases penalty"), std::exception);
                CHECK(full.temporary.Owned().Version() == before);
                CHECK(full.temporary.Owned().IsValid());

                {
                    // Spread the three guards across layouts instead of
                    // failing the same output three times per layout.
                    const double invalid[] = {std::numeric_limits<double>::quiet_NaN(), -1., 1.1};
                    auto a = alpha.Write();
                    a.View()(0, 0) = invalid[layout];
                }
                CHECK_THROWS_WITH_AS(plan.DoReconstructionIterLimitedTo(mean, input, full, boundary, target, alpha),
                                     doctest::Contains("alpha must be finite"), std::exception);
                // A numerical failure after the write lease was issued
                // quarantines the output. Treating it as current is unsafe.
                CHECK_FALSE(full.temporary.Owned().IsValid());
            }
        }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    MPI_Finalize();
    return result;
}
