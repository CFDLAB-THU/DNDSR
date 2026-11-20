#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include <limits>

namespace C = DNDS::Cauchy;
namespace G = DNDS::Geom;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Nodal wall sampling retains P1 and P2 shapes on distributed cells and faces")
{
    for (bool quadratic : {false, true})
    {
        auto mesh = CFVTest::CartesianHexes(mpi, quadratic ? 1 : 2);
        if (quadratic)
        {
            auto elevated = DNDS::make_ssp<G::UnstructuredMesh>(mpi, 3);
            elevated->BuildO2FromO1Elevation(*mesh);
            elevated->PrepareNativeTopology();
            mesh = std::move(elevated);
        }
        G::UnstructuredMesh::WallDistOptions options;
        options.minWallDist = 0;
        mesh->BuildNodeWallDist([](auto zone)
                                { return zone == 7; }, options);
        const int order = 4;
        const auto samples = C::WallDistanceSamples::Construct(mesh, order);
        CHECK(samples.IntegrationOrder() == order);
        // Both nodal sets have only one nonzero distance, at the box center.
        // Tensor-product interpolation gives an independent closed-form oracle;
        // this deliberately differs from the exact nearest-wall distance inside.
        const auto expected = [&](const G::tPoint &point)
        {
            double value = .5;
            for (int d = 0; d < 3; ++d)
                value *= quadratic ? 4 * point[d] * (1 - point[d]) : 1 - std::abs(2 * point[d] - 1);
            return std::max(0., value);
        };
        {
            auto access = samples.AcquireRead();
            const auto values = access.HostView();
            auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
            const auto view = read.HostView();
            for (I iCell = 0; iCell < view.NumCellProc(); ++iCell)
            {
                const auto nodes = view.CellNodes(iCell);
                G::tPoint low = view.Coord(nodes[0]), high = low;
                for (I iNode = 1; iNode < I(nodes.size()); ++iNode)
                {
                    low = low.cwiseMin(view.Coord(nodes[iNode]));
                    high = high.cwiseMax(view.Coord(nodes[iNode]));
                }
                G::Elem::Quadrature quadrature(view.GetCellElement(iCell), order);
                REQUIRE(values.cells[iCell].size() == size_t(quadrature.GetNumPoints()));
                double sum = 0;
                for (int iPoint = 0; iPoint < quadrature.GetNumPoints(); ++iPoint)
                {
                    const auto param = std::get<0>(quadrature.GetQuadraturePointInfo(iPoint));
                    const G::tPoint point = (high + low) * .5 + (high - low).cwiseProduct(param) * .5;
                    const double value = expected(point);
                    CHECK(values.cells(iCell, iPoint) == doctest::Approx(value).epsilon(1e-12));
                    sum += value;
                }
                CHECK(values.cellMean(iCell, 0) == doctest::Approx(sum / quadrature.GetNumPoints()).epsilon(1e-12));
            }
            for (I iFace = 0; iFace < view.NumFaceProc(); ++iFace)
            {
                const auto nodes = view.FaceNodes(iFace);
                const G::tPoint origin = view.Coord(nodes[0]);
                const G::tPoint dx = view.Coord(nodes[1]) - origin, dy = view.Coord(nodes[3]) - origin;
                G::Elem::Quadrature quadrature(view.GetFaceElement(iFace), order);
                REQUIRE(values.faces[iFace].size() == size_t(quadrature.GetNumPoints()));
                for (int iPoint = 0; iPoint < quadrature.GetNumPoints(); ++iPoint)
                {
                    const auto param = std::get<0>(quadrature.GetQuadraturePointInfo(iPoint));
                    const G::tPoint point = origin + .5 * (1 + param[0]) * dx + .5 * (1 + param[1]) * dy;
                    CHECK(values.faces(iFace, iPoint) == doctest::Approx(expected(point)).epsilon(1e-12));
                }
            }
            CHECK_THROWS(mesh->versions.BeginMutation(DNDS::Runtime::Change::Geometry));
        }
        // Cache identity includes the wall publication, not just coordinates.
        mesh->BuildNodeWallDist([](auto)
                                { return false; }, options);
        CHECK_THROWS(samples.AcquireRead());
        const auto rebuilt = C::WallDistanceSamples::Construct(mesh, order);
        const auto access = rebuilt.AcquireRead();
        for (I iCell = 0; iCell < mesh->NumCellProc(); ++iCell)
            CHECK(access.HostView().cellMean(iCell, 0) == doctest::Approx(std::pow(DNDS::veryLargeReal, .25)));
    }
}

TEST_CASE("First-layer wall distance selects a cell value for volume and mean samples")
{
    namespace R = DNDS::Runtime;
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    mesh->BuildNodeWallDist([](int zone)
                            { return zone == 7; });
    const auto communication = std::make_shared<R::CommunicationContext>(mpi.comm);
    const auto mask = C::WallDistanceSamples::BuildFirstLayerMask(*mesh, communication, [](int zone)
                                                                  { return zone == 7; });
    const auto sampled = C::WallDistanceSamples::Construct(mesh, 4);
    const auto centered = C::WallDistanceSamples::Construct(mesh, 4, 0., std::cref(mask));
    const auto halfMax = C::WallDistanceSamples::Construct(mesh, 4, 0., std::cref(mask), C::FirstLayerWallDistance::HalfMaxNodal);
    const auto floored = C::WallDistanceSamples::Construct(mesh, 4, .4, std::cref(mask), C::FirstLayerWallDistance::HalfMaxNodal);
    const auto midpoint = C::WallDistanceSamples::Construct(mesh, 1);
    const auto a = sampled.AcquireRead(), b = centered.AcquireRead(), c = midpoint.AcquireRead(), d = halfMax.AcquireRead();
    const auto av = a.HostView(), bv = b.HostView(), cv = c.HostView(), dv = d.HostView();
    const auto selected = C::WallDistanceSamples::Construct(mesh, 4, 0., std::cref(mask), C::FirstLayerWallDistance::HalfMaxNodal, true, false);
    const auto selectedAccess = selected.AcquireRead();
    const auto uniform = C::WallDistanceSamples::Construct(mesh, 4, .1, std::cref(mask),
                                                           C::FirstLayerWallDistance::HalfMaxNodal, false, true, true);
    const auto uniformAccess = uniform.AcquireRead();
    const auto uniformView = uniformAccess.HostView();
    const auto list = selected.FirstLayerCells().Read();
    const auto rows = list.View<R::FixedRows<1>>();
    I nextOwned = 0;
    const auto floorAccess = floored.AcquireRead();
    const auto floorView = floorAccess.HostView();
    auto requirements = G::MeshReadRequirements::CellFlux();
    requirements.wallDistance = true;
    auto access = mesh->AcquireRead(requirements);
    const auto view = access.HostView();
    double localMinimum = std::numeric_limits<double>::infinity();
    double minimumHalfMax = std::numeric_limits<double>::infinity();
    for (I iCell = 0; iCell < view.NumCellProc(); ++iCell)
    {
        const auto cellNodes = view.CellNodes(iCell);
        G::tPoint point = G::tPoint::Zero();
        double maxNodal = 0;
        for (const auto iNode : cellNodes)
        {
            point += view.Coord(iNode) / double(cellNodes.size());
            maxNodal = std::max(maxNodal, view.NodeWallDist(iNode).norm());
        }
        const bool firstLayer = point.minCoeff() < .2 || point.maxCoeff() > .8;
        const double cellCenter = cv.cells(iCell, 0);
        if (iCell < view.NumCell())
        {
            localMinimum = std::min(localMinimum, cellCenter);
            if (firstLayer)
            {
                REQUIRE(nextOwned < rows.RowCount());
                CHECK(rows(nextOwned, 0).cell == iCell);
                CHECK(rows(nextOwned, 0).distance == .5 * maxNodal);
                ++nextOwned;
            }
            if (firstLayer)
                minimumHalfMax = std::min(minimumHalfMax, .5 * maxNodal);
        }
        const double effectiveCenter = std::max(.1, cellCenter);
        CHECK(uniformView.cellMean(iCell, 0) ==
              (firstLayer ? std::max(.1, .5 * maxNodal) : effectiveCenter));
        CHECK(bv.cellMean(iCell, 0) == (firstLayer ? cellCenter : av.cellMean(iCell, 0)));
        CHECK(dv.cellMean(iCell, 0) == (firstLayer ? .5 * maxNodal : av.cellMean(iCell, 0)));
        CHECK(floorView.cellMean(iCell, 0) == doctest::Approx(.4));
        for (I iPoint = 0; iPoint < I(av.cells[iCell].size()); ++iPoint)
        {
            CHECK(uniformView.cells(iCell, iPoint) == effectiveCenter);
            CHECK(selectedAccess.HostView().cells(iCell, iPoint) == av.cells(iCell, iPoint));
            CHECK(bv.cells(iCell, iPoint) == (firstLayer ? cellCenter : av.cells(iCell, iPoint)));
            CHECK(dv.cells(iCell, iPoint) == (firstLayer ? .5 * maxNodal : av.cells(iCell, iPoint)));
            CHECK(floorView.cells(iCell, iPoint) == .4);
        }
    }
    CHECK(nextOwned == rows.RowCount());
    CHECK(centered.LocalMinimumCellCenterDistance() == localMinimum);
    CHECK(sampled.LocalMinimumCellCenterDistance() == localMinimum);
    CHECK(halfMax.LocalMinimumCellCenterDistance() == localMinimum);
    CHECK(halfMax.LocalMinimumFirstLayerDistance() == minimumHalfMax);
    for (I iFace = 0; iFace < view.NumFaceProc(); ++iFace)
        for (I iPoint = 0; iPoint < I(av.faces[iFace].size()); ++iPoint)
        {
            CHECK(bv.faces(iFace, iPoint) == av.faces(iFace, iPoint));
            CHECK(dv.faces(iFace, iPoint) == av.faces(iFace, iPoint));
            CHECK(floorView.faces(iFace, iPoint) == .4);
            const auto parents = view.FaceCells(iFace);
            const auto center = [&](I cell)
            { return std::max(.1, cv.cells(cell, 0)); };
            double expected = std::max(.1, av.faces(iFace, iPoint));
            if (parents[0] >= 0 && parents[1] == DNDS::UnInitIndex)
                expected = center(parents[0]);
            else if (parents[0] >= 0 && parents[1] >= 0)
                expected = .5 * (center(parents[0]) + center(parents[1]));
            CHECK(uniformView.faces(iFace, iPoint) == expected);
        }
}

int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    mpi.setWorld();
    int status;
    {
        doctest::Context context(argc, argv);
        status = context.run();
    }
    DNDS::MPI::Finalize();
    return status;
}
