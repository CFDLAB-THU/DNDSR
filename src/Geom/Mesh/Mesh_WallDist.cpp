#include "Mesh.hpp"
#include "Geom/Quadrature.hpp"

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
        /* ************************** */
        // Here implements direct search method
        using TriArray = ArrayEigenMatrix<3, 3>;
        ssp<TriArray> TrianglesLocal, TrianglesFull;
        TrianglesLocal = make_ssp<decltype(TrianglesLocal)::element_type>(ObjName{"BuildNodeWallDist::TrianglesLocal"}, getMPI());
        TrianglesFull = make_ssp<decltype(TrianglesFull)::element_type>(ObjName{"BuildNodeWallDist::TrianglesFull"}, getMPI());
        std::vector<Eigen::Matrix<real, 3, 3>> Triangles;

        nodeWallDist.InitPair("BuildNodeWallDist::nodeWallDist", mpi);
        nodeWallDist.father->Resize(NumNode());
        for (index iNode = 0; iNode < NumNode(); iNode++)
            nodeWallDist[iNode] = tPoint{std::pow(veryLargeReal, 1. / 4.), 0, 0};

        for (index iBnd = 0; iBnd < NumBnd(); iBnd++)
        {
            // skip if not wall
            if (!fBndIsWall(GetBndZone(iBnd)))
                continue;

            index iFace = bnd2faceV[iBnd];
            auto elem = GetFaceElement(iFace);
            auto quad = Geom::Elem::Quadrature{elem, options.subdivide_quad};

            // set all wall dist on wall nodes to be exactly 0.0
            for (auto iNode : bnd2node[iBnd])
                if (iNode < NumNode()) // && iNode >= 0; guaranteed for we assume all NumBnd() bnd faces have valid local position
                {
                    DNDS_assert(iNode >= 0);
                    nodeWallDist[iNode].setConstant(0.0);
                }

            if (options.method == 0 || options.method == 20)
            {
                if (elem.type == Geom::Elem::ElemType::Line2 || elem.type == Geom::Elem::ElemType::Line3) //!
                {
                    Geom::tSmallCoords coords;
                    GetCoordsOnFace(iFace, coords);
                    Eigen::Matrix<real, 3, 3> tri;
                    GetCoordsOnFace(iFace, coords);
                    tri(EigenAll, 0) = coords(EigenAll, 0);
                    tri(EigenAll, 1) = coords(EigenAll, 1);
                    tri(EigenAll, 2) = coords(EigenAll, 1) + Geom::tPoint{0., 0., 1.0};
                    Triangles.push_back(tri);
                }
                else if (elem.type == Geom::Elem::ElemType::Tri3 || elem.type == Geom::Elem::ElemType::Tri6) //! TODO
                {
                    Geom::tSmallCoords coords;
                    GetCoordsOnFace(iFace, coords);
                    Eigen::Matrix<real, 3, 3> tri;
                    tri(EigenAll, 0) = coords(EigenAll, 0);
                    tri(EigenAll, 1) = coords(EigenAll, 1);
                    tri(EigenAll, 2) = coords(EigenAll, 2);
                    Triangles.push_back(tri);
                }
                else if (elem.type == Geom::Elem::ElemType::Quad4 || elem.type == Geom::Elem::ElemType::Quad9)
                {
                    Geom::tSmallCoords coords;
                    GetCoordsOnFace(iFace, coords);
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
                    DNDS_assert_info(false, "This elem not implemented yet for BuildNodeWallDist()");
                }
            }
            else if (options.method == 1)
            {
                Geom::tSmallCoords coords;
                GetCoordsOnFace(iFace, coords);
                std::vector<tPoint> faceQuadraturePPhysics;
                faceQuadraturePPhysics.reserve(quad.GetNumPoints());
                real v{0};
                quad.Integration(
                    v,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
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
        TrianglesLocal->Resize(Triangles.size(), 3, 3);
        for (index i = 0; i < TrianglesLocal->Size(); i++)
            (*TrianglesLocal)[i] = Triangles[i];
        Triangles.clear();
        ArrayTransformerType<TriArray>::Type TrianglesTransformer;
        TrianglesTransformer.setFatherSon(TrianglesLocal, TrianglesFull);
        TrianglesTransformer.createFatherGlobalMapping();

        std::vector<index> pullingSet;
        pullingSet.resize(TrianglesTransformer.pLGlobalMapping->globalSize());
        for (index i = 0; i < pullingSet.size(); i++)
            pullingSet[i] = i;
        TrianglesTransformer.createGhostMapping(pullingSet);
        TrianglesTransformer.createMPITypes();
        TrianglesTransformer.pullOnce();
        if (options.verbose >= 1)
            if (coords.father->getMPI().rank == 0)
                log() << fmt::format("=== UnstructuredMesh::BuildNodeWallDist() with minWallDist = {:.4e}, ", options.minWallDist)
                      << " To search in " << TrianglesFull->Size() << std::endl;

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
            triangles.reserve(TrianglesFull->Size());

            for (index i = 0; i < TrianglesFull->Size(); i++)
            {
                Point p0((*TrianglesFull)[i](0, 0), (*TrianglesFull)[i](1, 0), (*TrianglesFull)[i](2, 0));
                Point p1((*TrianglesFull)[i](0, 1), (*TrianglesFull)[i](1, 1), (*TrianglesFull)[i](2, 1));
                Point p2((*TrianglesFull)[i](0, 2), (*TrianglesFull)[i](1, 2), (*TrianglesFull)[i](2, 2));
                triangles.emplace_back(p0, p1, p2);
            }
            TrianglesLocal->Resize(0, 3, 3);
            TrianglesFull->Resize(0, 3, 3);
            double minDist = veryLargeReal;

            if (!triangles.empty())
            {
                // std::cout << "tree building" << std::endl;
                Tree tree(triangles.begin(), triangles.end());
                // std::cout << "tree built" << std::endl;
                // search
                for (index iNode = 0; iNode < NumNode(); iNode++)
                {
                    // skip already on-wall data
                    if (nodeWallDist[iNode].squaredNorm() == 0.0)
                        continue;
                    // std::cout << "iCell " << iCell << std::endl;
                    // std::cout << "iG " << ig << std::endl;
                    auto p = coords[iNode];
                    Point pQ(p[0], p[1], p[2]);
                    // std::cout << "pQ " << pQ << std::endl;
                    // Point closest_point = tree.closest_point(pQ);
                    // FT sqd = tree.squared_distance(pQ);
                    auto closest_point = tree.closest_point_and_primitive(pQ);
                    // std::cout << "sqd" << sqd << std::endl;
                    auto cp = tPoint{closest_point.first.x(), closest_point.first.y(), closest_point.first.z()};

                    real dist = (p - cp).norm();
                    minDist = std::min(dist, minDist);
                    dist = std::max(options.minWallDist, dist);

                    nodeWallDist[iNode] = dist * (p - cp).stableNormalized();
                }
            }
            else
            {
                for (index iNode = 0; iNode < NumNode(); iNode++)
                {
                    nodeWallDist[iNode] = tPoint{std::pow(veryLargeReal, 1. / 4.), 0, 0};
                }
            }
            if (options.verbose >= 2)
                log() << fmt::format("[{}] MinDist: ", getMPI().rank) << minDist << std::endl;
        };
        if (options.wallDistExecution == 1)
            MPISerialDo(getMPI(), [&]()
                        { executeSearch(); });
        else if (options.wallDistExecution > 1)
            for (int i = 0; i < options.wallDistExecution; i++)
            {
                if (getMPI().rank % options.wallDistExecution == i)
                    executeSearch();
                MPI::Barrier(getMPI().comm);
            }
        else
            executeSearch();

        nodeWallDist.TransAttach();
        nodeWallDist.trans.BorrowGGIndexing(coords.trans);
        nodeWallDist.trans.createMPITypes();
        nodeWallDist.trans.pullOnce();
    }
}
