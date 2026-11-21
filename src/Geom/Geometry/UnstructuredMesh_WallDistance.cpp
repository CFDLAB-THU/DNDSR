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
#include <CGAL/AABB_segment_primitive_3.h>
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
        using SegmentCoords = Eigen::Matrix<real, 3, 2>;
        using TriangleCoords = Eigen::Matrix<real, 3, 3>;
        const bool useSegments = dim == 2 && options.method == 1;
        std::vector<SegmentCoords> segments;
        std::vector<TriangleCoords> triangles;
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
                        triangles.push_back(tri);
                    }
                    else if (elem.type == Geom::Elem::ElemType::Tri3 || elem.type == Geom::Elem::ElemType::Tri6) //! TODO
                    {
                        Geom::tSmallCoords coords;
                        faceCoordinates(iFace, coords);
                        Eigen::Matrix<real, 3, 3> tri;
                        tri(EigenAll, 0) = coords(EigenAll, 0);
                        tri(EigenAll, 1) = coords(EigenAll, 1);
                        tri(EigenAll, 2) = coords(EigenAll, 2);
                        triangles.push_back(tri);
                    }
                    else if (elem.type == Geom::Elem::ElemType::Quad4 || elem.type == Geom::Elem::ElemType::Quad9)
                    {
                        Geom::tSmallCoords coords;
                        faceCoordinates(iFace, coords);
                        Eigen::Matrix<real, 3, 3> tri;
                        tri(EigenAll, 0) = coords(EigenAll, 0);
                        tri(EigenAll, 1) = coords(EigenAll, 1);
                        tri(EigenAll, 2) = coords(EigenAll, 2);
                        triangles.push_back(tri);
                        tri(EigenAll, 0) = coords(EigenAll, 0);
                        tri(EigenAll, 1) = coords(EigenAll, 2);
                        tri(EigenAll, 2) = coords(EigenAll, 3);
                        triangles.push_back(tri);
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

                    const auto vertex = [&](int code) -> tPoint
                    {
                        DNDS_check_throw_info(code != 0, "missing wall patch vertex");
                        return code > 0 ? tPoint(coords.col(code - 1)) : faceQuadraturePPhysics.at(-code - 1);
                    };
                    for (const auto &patch : Geom::Elem::GetQuadPatches(quad))
                    {
                        if (useSegments)
                        {
                            // Line patches encode (begin, end, 0). The absent
                            // third vertex is not part of the physical wall.
                            DNDS_check_throw_info(patch[2] == 0, "planar wall requires a line patch");
                            auto &edge = segments.emplace_back();
                            edge.col(0) = vertex(patch[0]);
                            edge.col(1) = vertex(patch[1]);
                            DNDS_check_throw_info(edge.allFinite() && (edge.col(1) - edge.col(0)).squaredNorm() > 0,
                                                  "wall search segment is nonfinite or degenerate");
                        }
                        else
                        {
                            auto &triangle = triangles.emplace_back();
                            for (int i = 0; i < 3; ++i)
                                triangle.col(i) = vertex(patch[i]);
                        }
                    }
                }
            }
            OMP::ForSetupRows(0, index(triangles.size()), [&](index iTriangle)
            {
                const auto &triangle = triangles[size_t(iTriangle)];
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
        static_assert(sizeof(SegmentCoords) == 6 * sizeof(real));
        std::vector<TriangleCoords> trianglesFull;
        std::vector<SegmentCoords> segmentsFull;
        if (useSegments)
            segmentsFull = gather(segments, MPI_DOUBLE, 6);
        else
            trianglesFull = gather(triangles, MPI_DOUBLE, 9);
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
            log() << fmt::format("=== UnstructuredMesh::BuildNodeWallDist() with minWallDist = {:.4e}, To search in {}", options.minWallDist, trianglesFull.size() + segmentsFull.size()) << std::endl;

        auto executeSearch = [&](auto &patches)
        {
            if (options.verbose >= 3)
                log() << fmt::format("Start search rank [{}]", getMPI().rank) << std::endl;
            using K = CGAL::Simple_cartesian<double>;
            using Point = K::Point_3;
            using Coordinates = typename std::decay_t<decltype(patches)>::value_type;
            constexpr bool line = Coordinates::ColsAtCompileTime == 2;
            using Shape = std::conditional_t<line, K::Segment_3, K::Triangle_3>;
            using Iterator = typename std::vector<Shape>::iterator;
            using Primitive = std::conditional_t<line, CGAL::AABB_segment_primitive_3<K, Iterator>,
                                                 CGAL::AABB_triangle_primitive_3<K, Iterator>>;
            using Tree = CGAL::AABB_tree<CGAL::AABB_traits_3<K, Primitive>>;
            std::vector<Shape> shapes;
            shapes.reserve(patches.size());
            for (const auto &patch : patches)
            {
                const auto point = [&](int j)
                { return Point(patch(0, j), patch(1, j), patch(2, j)); };
                if constexpr (line)
                    shapes.emplace_back(point(0), point(1));
                else
                    shapes.emplace_back(point(0), point(1), point(2));
            }
            patches.clear();
            double minDist = veryLargeReal;

            if (!shapes.empty())
            {
                Tree tree(shapes.begin(), shapes.end());
                // CGAL otherwise lazily builds both structures in const
                // distance queries. Finish them on the controller so all
                // concurrent queries read immutable tree/hint state.
                tree.build();
                tree.accelerate_distance_queries();
                const auto &search = tree;
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
                    auto p = coordinate(iNode);
                    Point pQ(p[0], p[1], p[2]);
                    auto closest_point = search.closest_point_and_primitive(pQ);
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
                if (options.wallDistExecution != 0 && mpi.rank % batches != batch) return;
                if (useSegments) executeSearch(segmentsFull);
                else executeSearch(trianglesFull); });

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
