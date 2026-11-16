#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/DistributedInterfaceSamplingPlan.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include "../Geom/InterfaceFixture.hpp"

namespace C = DNDS::CFV;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Distributed polynomial sampling evaluates on cell owners and returns query order")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, true);
    auto mesh = fixture.mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.maxOrder = 1;
    auto reconstruction = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto mean = metrics.BuildField(2);
    auto constantMean = metrics.BuildField(2);
    auto gradients = metrics.BuildField(2 * metrics.Dimension());
    auto coefficients = metrics.BuildField(2 * reconstruction.NRecDOF());
    {
        auto geometry = metrics.AcquireRead();
        const auto view = geometry.HostView();
        auto u = mean.Owned().Write();
        auto c = constantMean.Owned().Write();
        auto g = gradients.Owned().Write();
        auto r = coefficients.Owned().Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto center = view.GetCellCent(iCell);
            u.View()(iCell, 0) = 1 + 2 * center[0] - 0.5 * center[1] + 10 * mpi.rank;
            u.View()(iCell, 1) = -3 + 0.7 * center[0] + 4 * center[1] + 3 * mpi.rank;
            c.View()(iCell, 0) = 100 + mpi.rank;
            c.View()(iCell, 1) = -200 + 3 * mpi.rank;
            g.View()(iCell, 0) = 2;
            g.View()(iCell, 1) = -.5;
            g.View()(iCell, 2) = .7;
            g.View()(iCell, 3) = 4;
            const double scale = view.cells(iCell, 7);
            r.View()(iCell, 0) = 2 * scale;
            r.View()(iCell, 1) = -0.5 * scale;
            r.View()(iCell, 2) = 0.7 * scale;
            r.View()(iCell, 3) = 4 * scale;
        }
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(constantMean).Begin().Wait();
    metrics.BindHalo(gradients).Begin().Wait();
    metrics.BindHalo(coefficients).Begin().Wait();
    std::vector<I> selected(size_t(mesh->Entities(G::EntityKind::Bnd).NumOwned()));
    std::iota(selected.begin(), selected.end(), I(0));
    auto reference = G::BoundaryPatch::Create(mesh, selected);
    auto unchecked = std::make_shared<R::CommunicationContext>(mpi.comm, false);
    // The empty geometry owner also submits queries to remote cell owners.
    R::Array<double> points(R::RowLayout::Dense(3, 3));
    R::Array<I> ids(R::RowLayout::Dense(3, 1));
    for (int step = 0; step < 2; ++step)
    {
        G::RigidPose pose;
        pose.rotation = Eigen::AngleAxisd(0.4 * step, G::tPoint::UnitZ()).toRotationMatrix();
        pose.translation = {0.2 * step, 0.1 * step, 0};
        auto patch = reference.WithPose(pose);
        {
            auto p = points.Write();
            auto key = ids.Write();
            for (I iQuery = 0; iQuery < 3; ++iQuery)
            {
                const auto query = pose.Apply({0, (double((mpi.rank + 3 * iQuery + step) % 10) + 0.31) * 0.1, 0});
                for (int d = 0; d < 3; ++d)
                    p.View()(iQuery, d) = query[d];
                key.View()(iQuery, 0) = 40 - 11 * iQuery;
            }
        }
        for (auto mode : {G::InterfaceRoutingMode::CoarseBoxes, G::InterfaceRoutingMode::AllGatherBoundary})
        {
            const auto map = mode == G::InterfaceRoutingMode::CoarseBoxes
                                 ? G::DistributedInterfacePointMap::Match(unchecked, patch, points, ids, mode)
                                 : G::DistributedInterfacePointMap::Match(patch, points, ids, mode);
            C::DistributedInterfaceSamplingPlan sampler(reconstruction, map);
            auto work = sampler.BuildWorkspace(2);
            R::Array<double> values(R::RowLayout::Dense(3, 2));
            sampler.Evaluate(mean, coefficients, work, values);
            {
                auto read = values.Read();
                for (I iQuery = 0; iQuery < 3; ++iQuery)
                {
                    const auto match = map.Get(iQuery);
                    const auto p = match.geometry.donorPoint;
                    CHECK(read.View()(iQuery, 0) == doctest::Approx(1 + 2 * p[0] - 0.5 * p[1] + 10 * match.donorRank).epsilon(1e-12));
                    CHECK(read.View()(iQuery, 1) == doctest::Approx(-3 + 0.7 * p[0] + 4 * p[1] + 3 * match.donorRank).epsilon(1e-12));
                }
            }
            // Reuse a persistent result binding; only donor evaluation changes
            // the scratch output between communication operations.
            auto binding = sampler.BindResult(work, values);
            sampler.EvaluateDonors(mean, coefficients, work);
            {
                auto write = work.donorValues.Write();
                for (I iRow = 0; iRow < work.donorValues.RowCount(); ++iRow)
                    for (I component = 0; component < 2; ++component)
                        write.View()(iRow, component) += 0.25;
            }
            binding.Begin().Wait();
            auto read = values.Read();
            for (I iQuery = 0; iQuery < 3; ++iQuery)
            {
                const auto match = map.Get(iQuery);
                const auto p = match.geometry.donorPoint;
                CHECK(read.View()(iQuery, 0) == doctest::Approx(1.25 + 2 * p[0] - 0.5 * p[1] + 10 * match.donorRank).epsilon(1e-12));
                CHECK(read.View()(iQuery, 1) == doctest::Approx(-2.75 + 0.7 * p[0] + 4 * p[1] + 3 * match.donorRank).epsilon(1e-12));
            }
            // The same geometric/sparse-return plan supports a metrics-only
            // P0 donor and a direct physical-gradient P1 donor. Neither path
            // allocates or fabricates variational coefficients.
            R::Array<double> directValues(R::RowLayout::Dense(3, 2));
            auto directWork = sampler.BuildWorkspace(2);
            auto directBinding = sampler.BindResult(
                directWork, directValues, R::TransferMode::Persistent, 1);
            sampler.EvaluateDonors(
                std::make_optional(C::DonorInputs(constantMean)), directWork);
            directBinding.Begin().Wait();
            {
                auto direct = directValues.Read();
                for (I iQuery = 0; iQuery < 3; ++iQuery)
                {
                    const auto match = map.Get(iQuery);
                    CHECK(direct.View()(iQuery, 0) == 100 + match.donorRank);
                    CHECK(direct.View()(iQuery, 1) == -200 + 3 * match.donorRank);
                }
            }
            sampler.EvaluateDonors(
                std::make_optional(C::DonorInputs::Gradient(mean, gradients)),
                directWork);
            directBinding.Begin().Wait();
            {
                auto direct = directValues.Read();
                for (I iQuery = 0; iQuery < 3; ++iQuery)
                {
                    const auto match = map.Get(iQuery);
                    const auto p = match.geometry.donorPoint;
                    CHECK(direct.View()(iQuery, 0) ==
                          doctest::Approx(1 + 2 * p[0] - .5 * p[1] +
                                          10 * match.donorRank)
                              .epsilon(1e-12));
                    CHECK(direct.View()(iQuery, 1) ==
                          doctest::Approx(-3 + .7 * p[0] + 4 * p[1] +
                                          3 * match.donorRank)
                              .epsilon(1e-12));
                }
            }
        }
    }
}

TEST_CASE("Warped quad interface projects offset queries before distributed polynomial sampling")
{
    auto mesh = InterfaceFixture::WarpedHexes(mpi);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.maxOrder = 1;
    auto reconstruction = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    REQUIRE(reconstruction.NRecDOF() == 3);
    auto mean = metrics.BuildField(1);
    auto coefficients = metrics.BuildField(3);
    const G::tPoint gradient{0.3, -0.7, 2};
    const auto field = [&](const G::tPoint &p, int owner)
    { return 1 + gradient.dot(p) + 10 * owner; };
    {
        auto geometry = metrics.AcquireRead();
        const auto view = geometry.HostView();
        auto u = mean.Owned().Write();
        auto r = coefficients.Owned().Write();
        // Linear coefficients in the central, isotropically scaled Taylor
        // basis. Means use volume barycenters, independently of the matcher.
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            u.View()(iCell, 0) = field(view.GetCellBary(iCell), mpi.rank);
            for (int d = 0; d < 3; ++d)
                r.View()(iCell, d) = gradient[d] * view.cells(iCell, 7);
        }
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(coefficients).Begin().Wait();
    // Hex8 face 5 is its bilinear top, z = 1 + 0.25*a*b.
    const auto reference = G::BoundaryPatch::Create(mesh, {5});
    R::Array<double> points(R::RowLayout::Dense(2, 3));
    R::Array<I> ids(R::RowLayout::Dense(2, 1));
    for (int step = 0; step < 2; ++step)
    {
        G::RigidPose pose;
        pose.rotation = Eigen::AngleAxisd(0.3 * step, G::tPoint::UnitY()).toRotationMatrix();
        pose.translation = {0.2 * step, -0.1 * step, 0.3 * step};
        const auto patch = reference.WithPose(pose);
        std::array<G::tPoint, 2> expected;
        std::array<int, 2> owners;
        {
            auto p = points.Write();
            auto k = ids.Write();
            for (I iQuery = 0; iQuery < 2; ++iQuery)
            {
                owners[iQuery] = (mpi.rank + 1 + int(iQuery)) % mpi.size;
                const double a = 0.23 + 0.13 * iQuery + 0.03 * step, b = 0.67 - 0.11 * iQuery;
                expected[iQuery] = {2. * owners[iQuery] + a, b, 1 + 0.25 * a * b};
                const auto normal = G::tPoint(-0.25 * b, -0.25 * a, 1).normalized();
                const G::tPoint offSurface = expected[iQuery] + 0.02 * normal;
                const auto query = pose.Apply(offSurface);
                std::copy_n(query.data(), 3, p.View()[iQuery].data());
                k.View()(iQuery, 0) = 17 - 9 * iQuery;
                // This fixture would detect evaluating at the unprojected
                // point even though that point lies within the allowed gap.
                CHECK(std::abs(field(offSurface, owners[iQuery]) - field(expected[iQuery], owners[iQuery])) > 0.02);
            }
        }
        std::optional<G::DistributedInterfacePointMap> baseline;
        G::InterfaceSearchSettings search;
        search.maxGap = 0.03;
        search.coarseBoxesPerRank = 3;
        for (auto backend : {G::BoundingBoxBackend::Native, G::BoundingBoxBackend::CGAL})
            for (auto mode : {G::InterfaceRoutingMode::AllGatherBoundary, G::InterfaceRoutingMode::CoarseBoxes})
            {
                const auto activePatch = patch.WithBackend(backend);
                CHECK(activePatch.Backend() == backend);
                const auto map = G::DistributedInterfacePointMap::Match(activePatch, points, ids, mode, search);
                C::DistributedInterfaceSamplingPlan sampler(reconstruction, map);
                auto work = sampler.BuildWorkspace(1);
                R::Array<double> values(R::RowLayout::Dense(2, 1));
                sampler.Evaluate(mean, coefficients, work, values);
                auto read = values.Read();
                for (I iQuery = 0; iQuery < 2; ++iQuery)
                {
                    const auto match = map.Get(iQuery);
                    const int geometryOwner = (owners[iQuery] + mpi.size - 1) % mpi.size;
                    CHECK(match.donorRank == owners[iQuery]);
                    CHECK(match.geometryRank == geometryOwner);
                    CHECK(match.donorOwnedCell == 0);
                    CHECK(match.geometry.donorOriginalId == 6 * geometryOwner + 5);
                    CHECK(match.geometry.queryId == 17 - 9 * iQuery);
                    CHECK((match.geometry.donorPoint - expected[iQuery]).norm() < 1e-11);
                    CHECK((match.geometry.projectedPoint - pose.Apply(expected[iQuery])).norm() < 1e-11);
                    CHECK(match.geometry.distance == doctest::Approx(0.02).epsilon(1e-9));
                    CHECK(read.View()(iQuery, 0) == doctest::Approx(field(expected[iQuery], owners[iQuery])).epsilon(1e-11));
                    if (baseline)
                    {
                        const auto other = baseline->Get(iQuery);
                        CHECK(match.geometry.donorOriginalId == other.geometry.donorOriginalId);
                        CHECK((match.geometry.reference - other.geometry.reference).norm() < 1e-12);
                        CHECK((match.geometry.projectedPoint - other.geometry.projectedPoint).norm() < 1e-12);
                        CHECK(match.geometryRank == other.geometryRank);
                        CHECK(match.donorRank == other.donorRank);
                    }
                }
                if (!baseline)
                    baseline = map;
            }
    }
}

TEST_CASE("Coupling participants without a donor mesh sample a reversed donor subgroup in query order")
{
    const bool member = mpi.size == 1 || mpi.rank > 0;
    const int members = mpi.size == 1 ? 1 : mpi.size - 1;
    MPI_Comm donorComm = MPI_COMM_NULL;
    MPI_Comm_split(mpi.comm, member ? 0 : MPI_UNDEFINED, mpi.size - 1 - mpi.rank, &donorComm);
    {
        auto coupling = std::make_shared<R::CommunicationContext>(mpi.comm);
        std::optional<G::BoundaryPatch> patch;
        std::optional<C::FiniteVolumePlan> metrics;
        std::optional<C::VariationalReconstructionPlan> reconstruction;
        std::optional<C::FVField> mean, coefficients;
        G::RigidPose pose;
        pose.rotation = Eigen::AngleAxisd(.21, G::tPoint::UnitZ()).toRotationMatrix();
        pose.translation = {.2, -.1, .03};
        if (member)
        {
            const DNDS::MPIInfo donorMPI(donorComm);
            auto mesh = G::OverlapTest::BuildNativeFixture(donorMPI, false).mesh;
            metrics.emplace(C::FiniteVolumePlan::ConstructMetrics(mesh));
            C::ReconstructionSettings settings;
            settings.maxOrder = 1;
            reconstruction.emplace(C::VariationalReconstructionPlan::ConstructRecCoeff(*metrics, settings));
            mean.emplace(metrics->BuildField(1));
            coefficients.emplace(metrics->BuildField(reconstruction->NRecDOF()));
            {
                auto geometry = metrics->AcquireRead();
                const auto view = geometry.HostView();
                auto u = mean->Owned().Write(), r = coefficients->Owned().Write();
                for (I iCell = 0; iCell < metrics->NumCell(); ++iCell)
                {
                    const auto p = view.GetCellCent(iCell);
                    u.View()(iCell, 0) = 1 + 2 * p[0] - .5 * p[1] + 10 * donorMPI.rank;
                    r.View()(iCell, 0) = 2 * view.cells(iCell, 7);
                    r.View()(iCell, 1) = -.5 * view.cells(iCell, 7);
                }
            }
            metrics->BindHalo(*mean).Begin().Wait();
            metrics->BindHalo(*coefficients).Begin().Wait();
            std::vector<I> boundaries(size_t(mesh->NumBnd()));
            std::iota(boundaries.begin(), boundaries.end(), I(0));
            patch.emplace(G::BoundaryPatch::Create(mesh, boundaries, pose));
        }
        R::Array<double> points(R::RowLayout::Dense(2, 3)), values(R::RowLayout::Dense(2, 1));
        R::Array<I> ids(R::RowLayout::Dense(2, 1));
        {
            auto p = points.Write();
            auto key = ids.Write();
            for (I iQuery = 0; iQuery < 2; ++iQuery)
            {
                const I row = (3 * mpi.rank + 2 * iQuery) % 10;
                const auto location = pose.Apply({0, (double(row) + .31) * .1, 0});
                for (int d = 0; d < 3; ++d)
                    p.View()(iQuery, d) = location[d];
                key.View()(iQuery, 0) = 19 - 7 * iQuery;
            }
        }
        for (auto mode : {G::InterfaceRoutingMode::CoarseBoxes, G::InterfaceRoutingMode::AllGatherBoundary})
        {
            const auto map = G::DistributedInterfacePointMap::Match(coupling, patch, points, ids, mode);
            CHECK(map.HasLocalDonor() == member);
            C::DistributedInterfaceSamplingPlan sampler(reconstruction, map);
            auto workspace = sampler.BuildWorkspace(1);
            auto binding = sampler.BindResult(workspace, values, R::TransferMode::Persistent, 1);
            for (int repetition = 0; repetition < 2; ++repetition)
            {
                if (repetition == 0)
                    sampler.EvaluateCoupled(mean ? std::make_optional(DNDS::CFV::DonorInputs(*mean, *coefficients)) : std::nullopt, workspace, values);
                else
                {
                    sampler.EvaluateDonors(mean ? std::make_optional(DNDS::CFV::DonorInputs(*mean, *coefficients)) : std::nullopt, workspace);
                    binding.Begin().Wait();
                }
                const auto read = values.Read();
                for (I iQuery = 0; iQuery < 2; ++iQuery)
                {
                    const I row = (3 * mpi.rank + 2 * iQuery) % 10;
                    int owner = -1, geometryOwner = -1;
                    for (int rank = 0; rank < members; ++rank)
                    {
                        const auto cells = DNDS::EvenSplitRange(rank, members, I(10));
                        const auto boundaries = DNDS::EvenSplitRange(rank, members, I(40));
                        if (row >= cells.first && row < cells.second)
                            owner = rank;
                        if (39 - row >= boundaries.first && 39 - row < boundaries.second)
                            geometryOwner = rank;
                    }
                    const auto translate = [&](int rank)
                    { return mpi.size == 1 ? 0 : mpi.size - 1 - rank; };
                    const auto match = map.Get(iQuery);
                    CHECK(match.donorRank == translate(owner));
                    CHECK(match.geometryRank == translate(geometryOwner));
                    CHECK(match.geometry.queryId == 19 - 7 * iQuery);
                    CHECK(match.geometry.donorOriginalId == 39 - row);
                    CHECK(read.View()(iQuery, 0) == doctest::Approx(1 - .5 * (double(row) + .31) * .1 + 10 * owner).epsilon(1e-12));
                }
            }
        }
    }
    if (donorComm != MPI_COMM_NULL)
        MPI_Comm_free(&donorComm);
}
int main(int argc, char **argv)
{
    // Native row kernels may create OMP/CPU workers even with controller MPI.
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    mpi.setWorld();
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    DNDS::MPI::Finalize();
    return result;
}
