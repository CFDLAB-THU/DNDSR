#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "DNDS/Core/OMP.hpp"

#define CGAL_DISABLE_ROUNDING_MATH_CHECK // for valgrind
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#undef CGAL_DISABLE_ROUNDING_MATH_CHECK
namespace DNDS::Geom
{
    void UnstructuredMesh::BuildNodeWallDist(const std::function<bool(Geom::t_index)> &fBndIsWall, WallDistOptions options)
    {
        const auto communication = PreparationCommunication();
        const bool checks = !communication || communication->ChecksCollectivePreflight();
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, checks);
        CheckCollectiveMutationAvailable(checks);
        std::unique_ptr<MeshReadAccess> retained;
        std::optional<UnstructuredMeshView> nativeView;
        Runtime::VersionStamp stamp;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            DNDS_check_throw_info(HasNativeState(), "wall distance requires native mesh authority");
            DNDS_check_throw_info((options.method == 0 || options.method == 20 || options.method == 1) &&
                                      options.subdivide_quad >= 0 && std::isfinite(options.minWallDist) && options.minWallDist >= 0 && options.wallDistExecution >= 0,
                                  "invalid wall distance options");
            auto requirements = CurrentNativeRequirements();
            requirements.faces = true;
            retained = std::make_unique<MeshReadAccess>(AcquireRead(requirements));
            nativeView = retained->HostView();
            stamp = versions.Stamp(); });
        std::array<int, 3> parameters{options.method, options.subdivide_quad, options.wallDistExecution};
        auto expectedParameters = parameters;
        auto expectedMinimum = options.minWallDist;
        MPI_Bcast(expectedParameters.data(), int(expectedParameters.size()), MPI_INT, 0, mpi.comm);
        MPI_Bcast(&expectedMinimum, 1, MPI_DOUBLE, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        { DNDS_check_throw_info(parameters == expectedParameters && options.minWallDist == expectedMinimum, "wall distance options differ across ranks"); });
        auto coordinate = [&](index node) -> tPoint
        { return tPoint(nativeView->Coord(node)); };
        auto boundaryNodes = [&](index boundary)
        {
            std::vector<index> result;
            auto row = nativeView->BoundaryNodes(boundary);
            result.assign(row.begin(), row.end());
            return result;
        };
        auto faceCoordinates = [&](index face, tSmallCoords &output)
        { nativeView->GetCoordsOnFace(face, output); };
        std::vector<Eigen::Matrix<real, 3, 3>> Triangles;
        std::vector<index> wallNodes;
        Runtime::Array<real> wallOwned;
        std::optional<Runtime::Array<real>::WriteLease> wallWrite;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            wallOwned = Runtime::Array<real>(Runtime::RowLayout::Dense(size_t(NumNode()), 3));
            wallWrite.emplace(wallOwned.Write()); });
        auto wall = [&](index row)
        { return Eigen::Map<tPoint>(wallWrite->View().Row(size_t(row))); };
        OMP::ForSetupRows(0, NumNode(), [&](index iNode)
                          { wall(iNode) = tPoint{std::pow(veryLargeReal, 0.25), 0, 0}; });
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            // The user zone predicate has no concurrent-callback contract.
            // Keep its invocation and shared triangle/node appends ordered;
            // only immutable search queries below run on worker teams.
            for (index iBnd = 0; iBnd < NumBnd(); iBnd++)
            {
                // skip if not wall
                if (!fBndIsWall(nativeView->GetBndInfo(iBnd).zone))
                    continue;

                index iFace = nativeView->BoundaryFaces(iBnd)[0];
                DNDS_check_throw_info(iFace >= 0 && iFace < NumFaceProc(), "wall boundary face is unavailable");
                auto elem = nativeView->GetFaceElement(iFace);
                auto quad = Geom::Elem::Quadrature{elem, options.subdivide_quad};
                DNDS_check_throw_info(options.method != 1 || quad.GetNumPoints() > 0, "unsupported wall quadrature order");

                for (auto node : boundaryNodes(iBnd))
                {
                    DNDS_check_throw_info(node >= 0 && node < NumNodeProc(), "wall boundary node is unavailable");
                    wallNodes.push_back(node);
                }

                if (options.method == 0 || options.method == 20)
                {
                    if (elem.type == Geom::Elem::ElemType::Line2 || elem.type == Geom::Elem::ElemType::Line3) //!
                    {
                        Geom::tSmallCoords coords;
                        faceCoordinates(iFace, coords);
                        Eigen::Matrix<real, 3, 3> tri;
                        tri(EigenAll, 0) = coords(EigenAll, 0);
                        tri(EigenAll, 1) = coords(EigenAll, 1);
                        tri(EigenAll, 2) = coords(EigenAll, 1) + Geom::tPoint{0., 0., 1.0};
                        Triangles.push_back(tri);
                    }
                    else if (elem.type == Geom::Elem::ElemType::Tri3 || elem.type == Geom::Elem::ElemType::Tri6) //! TODO
                    {
                        Geom::tSmallCoords coords;
                        faceCoordinates(iFace, coords);
                        Eigen::Matrix<real, 3, 3> tri;
                        tri(EigenAll, 0) = coords(EigenAll, 0);
                        tri(EigenAll, 1) = coords(EigenAll, 1);
                        tri(EigenAll, 2) = coords(EigenAll, 2);
                        Triangles.push_back(tri);
                    }
                    else if (elem.type == Geom::Elem::ElemType::Quad4 || elem.type == Geom::Elem::ElemType::Quad9)
                    {
                        Geom::tSmallCoords coords;
                        faceCoordinates(iFace, coords);
                        Eigen::Matrix<real, 3, 3> tri;
                        tri(EigenAll, 0) = coords(EigenAll, 0);
                        tri(EigenAll, 1) = coords(EigenAll, 1);
                        tri(EigenAll, 2) = coords(EigenAll, 2);
                        Triangles.push_back(tri);
                        tri(EigenAll, 0) = coords(EigenAll, 0);
                        tri(EigenAll, 1) = coords(EigenAll, 2);
                        tri(EigenAll, 2) = coords(EigenAll, 3);
                        Triangles.push_back(tri);
                    }
                    else
                    {
                        DNDS_check_throw_info(false, "unsupported wall distance face element");
                    }
                }
                else if (options.method == 1)
                {
                    Geom::tSmallCoords coords;
                    faceCoordinates(iFace, coords);
                    std::vector<tPoint> faceQuadraturePPhysics;
                    faceQuadraturePPhysics.reserve(quad.GetNumPoints());
                    real v{0};
                    quad.Integration(
                        v,
                        [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                        {
                            vInc = 0; // This quadrature only collects points.
                            tPoint pPhy = Elem::PPhysicsCoordD01Nj(coords, DiNj);
                            faceQuadraturePPhysics.emplace_back(pPhy);
                        });

                    auto qPatches = Geom::Elem::GetQuadPatches(quad);
                    for (auto &qPatch : qPatches)
                    {
                        Eigen::Matrix<real, 3, 3> tri;

                        for (int iV = 0; iV < 3; iV++)
                            if (qPatch[iV] > 0)
                                tri(EigenAll, iV) = coords(EigenAll, qPatch[iV] - 1);
                            else if (qPatch[iV] < 0)
                                tri(EigenAll, iV) = faceQuadraturePPhysics.at(-qPatch[iV] - 1);
                            else
                                tri(EigenAll, iV) = coords(EigenAll, 1) + Geom::tPoint{0., 0., 1.0};
                        Triangles.push_back(tri);
                    }
                }
            }
            OMP::ForSetupRows(0, index(Triangles.size()), [&](index iTriangle)
            {
                const auto &triangle = Triangles[size_t(iTriangle)];
                DNDS_check_throw_info(triangle.allFinite() && (triangle.col(1) - triangle.col(0)).cross(triangle.col(2) - triangle.col(0)).squaredNorm() > 0,
                                      "wall search patch is nonfinite or degenerate"); }); });
        // Exact replicated surface search: scratch on each rank scales with
        // ALL selected wall patches, independently of sparse mesh ghost storage.
        auto gather = [&](const auto &local, MPI_Datatype datatype, int components)
        {
            using T = typename std::decay_t<decltype(local)>::value_type;
            int count = 0;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            { DNDS_check_throw_info(local.size() <= size_t(INT_MAX / components), "wall surface exceeds MPI count range"); count = int(local.size()) * components; });
            std::vector<int> counts(size_t(mpi.size)), offsets(size_t(mpi.size) + 1, 0);
            MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                for (int rank = 0; rank < mpi.size; ++rank)
                { DNDS_check_throw_info(counts[size_t(rank)] >= 0 && counts[size_t(rank)] <= INT_MAX - offsets[size_t(rank)], "wall gather exceeds MPI count range"); offsets[size_t(rank) + 1] = offsets[size_t(rank)] + counts[size_t(rank)]; } });
            std::vector<T> result;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            { result.resize(size_t(offsets.back() / components)); });
            MPI_Allgatherv(local.data(), count, datatype, result.data(), counts.data(), offsets.data(), datatype, mpi.comm);
            return result;
        };
        static_assert(sizeof(Eigen::Matrix<real, 3, 3>) == 9 * sizeof(real), "wall MPI payload requires packed coordinates");
        auto trianglesFull = gather(Triangles, MPI_DOUBLE, 9);
        const auto nodeSpace = Entities(EntityKind::Node);
        Runtime::Array<uint8_t> selectedOwned, selectedGhost;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            selectedOwned = Runtime::Array<uint8_t>(Runtime::RowLayout::Dense(size_t(NumNode()), 1));
            selectedGhost = Runtime::Array<uint8_t>(Runtime::RowLayout::Dense(size_t(NumNodeGhost()), 1));
            auto local = selectedOwned.Write(), remote = selectedGhost.Write();
            OMP::ForSetupRows(0, selectedOwned.RowCount(), [&](index iRow)
                              { local.View()(iRow, 0) = 0; });
            OMP::ForSetupRows(0, selectedGhost.RowCount(), [&](index iRow)
                              { remote.View()(iRow, 0) = 0; });
            // Several selected boundaries can name the same node. This small
            // scatter is ordered; parallel byte stores would still race.
            for (auto node : wallNodes)
                (node < NumNode() ? local.View()(size_t(node), 0) : remote.View()(size_t(node - NumNode()), 0)) = 1; });
        const auto route = nodeSpace.BuildRowTransferPlan();
        Runtime::PayloadPlan reverse(route.Reverse(), selectedGhost.Layout(), selectedOwned.Layout(), Runtime::ElementSchema::Arithmetic<uint8_t>());
        reverse.Bind(selectedGhost, selectedOwned, Runtime::Reducer::Builtin<uint8_t>(Runtime::Reduction::BitOr),
                     Runtime::TransferTransport::Direct, {Runtime::TransferMode::Transient})
            .Begin()
            .Wait();
        {
            auto selected = selectedOwned.Read();
            OMP::ForSetupRows(0, NumNode(), [&](index iNode)
                              {
                if (selected.View()(iNode, 0))
                    wall(iNode).setZero(); });
        }
        if (options.verbose >= 1 && mpi.rank == 0)
            log() << fmt::format("=== UnstructuredMesh::BuildNodeWallDist() with minWallDist = {:.4e}, To search in {}", options.minWallDist, trianglesFull.size()) << std::endl;

        auto executeSearch = [&]()
        {
            if (options.verbose >= 3)
                log() << fmt::format("Start search rank [{}]", getMPI().rank) << std::endl;
            using K = CGAL::Simple_cartesian<double>;
            // using FT = K::FT;
            // typedef K::Ray_3 Ray;
            // typedef K::Line_3 Line;
            using Point = K::Point_3;
            using Triangle = K::Triangle_3;
            using Iterator = std::vector<Triangle>::iterator;
            using Primitive = CGAL::AABB_triangle_primitive_3<K, Iterator>;
            using AABB_triangle_traits = CGAL::AABB_traits_3<K, Primitive>;
            using Tree = CGAL::AABB_tree<AABB_triangle_traits>;

            std::vector<Triangle> triangles;
            triangles.reserve(trianglesFull.size());

            for (index i = 0; i < trianglesFull.size(); i++)
            {
                Point p0(trianglesFull[size_t(i)](0, 0), trianglesFull[size_t(i)](1, 0), trianglesFull[size_t(i)](2, 0));
                Point p1(trianglesFull[size_t(i)](0, 1), trianglesFull[size_t(i)](1, 1), trianglesFull[size_t(i)](2, 1));
                Point p2(trianglesFull[size_t(i)](0, 2), trianglesFull[size_t(i)](1, 2), trianglesFull[size_t(i)](2, 2));
                triangles.emplace_back(p0, p1, p2);
            }
            trianglesFull.clear();
            double minDist = veryLargeReal;

            if (!triangles.empty())
            {
                // std::cout << "tree building" << std::endl;
                Tree tree(triangles.begin(), triangles.end());
                // CGAL otherwise lazily builds both structures in const
                // distance queries. Finish them on the controller so all
                // concurrent queries read immutable tree/hint state.
                tree.build();
                tree.accelerate_distance_queries();
                const auto &search = tree;
                // std::cout << "tree built" << std::endl;
                // search
                const index parts = std::min<index>(OMP::DistributedThreads(), std::max<index>(NumNode(), 1));
                std::vector<real> minima(size_t(parts), veryLargeReal);
                OMP::ForSetupRows(0, parts, [&](index part)
                                  {
                    const index begin = part * (NumNode() / parts) + std::min(part, NumNode() % parts);
                    const index end = begin + NumNode() / parts + (part < NumNode() % parts);
                    real localMinimum = veryLargeReal;
                    for (index iNode = begin; iNode < end; ++iNode)
                    {
                    // skip already on-wall data
                    if (wall(iNode).squaredNorm() == 0.0)
                        continue;
                    // std::cout << "iCell " << iCell << std::endl;
                    // std::cout << "iG " << ig << std::endl;
                    auto p = coordinate(iNode);
                    Point pQ(p[0], p[1], p[2]);
                    // std::cout << "pQ " << pQ << std::endl;
                    // Point closest_point = tree.closest_point(pQ);
                    // FT sqd = tree.squared_distance(pQ);
                    auto closest_point = search.closest_point_and_primitive(pQ);
                    // std::cout << "sqd" << sqd << std::endl;
                    auto cp = tPoint{closest_point.first.x(), closest_point.first.y(), closest_point.first.z()};

                    real dist = (p - cp).norm();
                    localMinimum = std::min(dist, localMinimum);
                    dist = std::max(options.minWallDist, dist);

                    wall(iNode) = dist * (p - cp).stableNormalized();
                    }
                    minima[size_t(part)] = localMinimum; });
                for (const auto value : minima)
                    minDist = std::min(minDist, value);
            }
            else
            {
                OMP::ForSetupRows(0, NumNode(), [&](index iNode)
                                  { wall(iNode) = tPoint{std::pow(veryLargeReal, 1. / 4.), 0, 0}; });
            }
            if (options.verbose >= 2)
                log() << fmt::format("[{}] MinDist: ", getMPI().rank) << minDist << std::endl;
        };
        // Every execution batch coordinates failures before peers advance.
        const int batches = options.wallDistExecution == 1 ? mpi.size : std::max(1, options.wallDistExecution);
        for (int batch = 0; batch < batches; ++batch)
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                if (options.wallDistExecution == 0 || mpi.rank % batches == batch) executeSearch(); });

        // Release the writer before transfer leases are acquired.
        wallWrite.reset();
        EntityGhostMapping nodeRoute(nodeSpace);
        auto wallGhost = nodeRoute.TransferRows(nodeSpace, wallOwned, Runtime::ElementSchema::Arithmetic<real>());
        std::shared_ptr<const UnstructuredMeshState> prepared;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            wallGhost = Runtime::Array<real>::FromStorage(Runtime::RowLayout::Dense(wallGhost.RowCount(), 3), wallGhost.Storage());
            prepared = _nativeState->WithWallDistance({wallOwned, wallGhost}); });
        retained.reset();
        PublishNativeGeometry(std::move(prepared), stamp, Runtime::Change::Values | Runtime::Change::Interface | Runtime::Change::Allocation);
    }
}
