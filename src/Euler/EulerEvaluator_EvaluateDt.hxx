#pragma once
#include "EulerEvaluator.hpp"
#include "RANS_ke.hpp"
#include "Solver/Linear.hpp"

#define CGAL_DISABLE_ROUNDING_MATH_CHECK // for valgrind
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#undef CGAL_DISABLE_ROUNDING_MATH_CHECK

namespace DNDS::Euler
{

    static const auto model = NS_SA;
    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    void EulerEvaluator<model>::GetWallDist()
    {
        if (settings.wallDistScheme == 0 || settings.wallDistScheme == 1 || settings.wallDistScheme == 20)
        {
            using TriArray = ArrayEigenMatrix<3, 3>;
            ssp<TriArray> TrianglesLocal, TrianglesFull;
            DNDS_MAKE_SSP(TrianglesLocal, mesh->getMPI());
            DNDS_MAKE_SSP(TrianglesFull, mesh->getMPI());
            std::vector<Eigen::Matrix<real, 3, 3>> Triangles;
            for (index iBnd = 0; iBnd < mesh->NumBnd(); iBnd++)
            {
                if (pBCHandler->GetTypeFromID(mesh->GetBndZone(iBnd)) == EulerBCType::BCWall)
                {
                    index iFace = mesh->bnd2face[iBnd];
                    auto elem = mesh->GetFaceElement(iFace);
                    auto quad = vfv->GetFaceQuad(iFace);
                    if (settings.wallDistScheme == 0 || settings.wallDistScheme == 20)
                    {
                        if (elem.type == Geom::Elem::ElemType::Line2 || elem.type == Geom::Elem::ElemType::Line3) //!
                        {
                            Geom::tSmallCoords coords;
                            mesh->GetCoordsOnFace(iFace, coords);
                            Eigen::Matrix<real, 3, 3> tri;
                            mesh->GetCoordsOnFace(iFace, coords);
                            tri(Eigen::all, 0) = coords(Eigen::all, 0);
                            tri(Eigen::all, 1) = coords(Eigen::all, 1);
                            tri(Eigen::all, 2) = coords(Eigen::all, 1) + Geom::tPoint{0., 0., vfv->GetFaceArea(iFace)};
                            Triangles.push_back(tri);
                        }
                        else if (elem.type == Geom::Elem::ElemType::Tri3 || elem.type == Geom::Elem::ElemType::Tri6) //! TODO
                        {
                            Geom::tSmallCoords coords;
                            mesh->GetCoordsOnFace(iFace, coords);
                            Eigen::Matrix<real, 3, 3> tri;
                            tri(Eigen::all, 0) = coords(Eigen::all, 0);
                            tri(Eigen::all, 1) = coords(Eigen::all, 1);
                            tri(Eigen::all, 2) = coords(Eigen::all, 2);
                            Triangles.push_back(tri);
                        }
                        else if (elem.type == Geom::Elem::ElemType::Quad4 || elem.type == Geom::Elem::ElemType::Quad9)
                        {
                            Geom::tSmallCoords coords;
                            mesh->GetCoordsOnFace(iFace, coords);
                            Eigen::Matrix<real, 3, 3> tri;
                            tri(Eigen::all, 0) = coords(Eigen::all, 0);
                            tri(Eigen::all, 1) = coords(Eigen::all, 1);
                            tri(Eigen::all, 2) = coords(Eigen::all, 2);
                            Triangles.push_back(tri);
                            tri(Eigen::all, 0) = coords(Eigen::all, 0);
                            tri(Eigen::all, 1) = coords(Eigen::all, 2);
                            tri(Eigen::all, 2) = coords(Eigen::all, 3);
                            Triangles.push_back(tri);
                        }
                        else
                        {
                            DNDS_assert_info(false, "This elem not implemented yet for GetWallDist()");
                        }
                    }
                    else if (settings.wallDistScheme == 1)
                    {
                        auto qPatches = Geom::Elem::GetQuadPatches(quad);
                        for (auto &qPatch : qPatches)
                        {
                            Eigen::Matrix<real, 3, 3> tri;
                            Geom::tSmallCoords coords;
                            mesh->GetCoordsOnFace(iFace, coords);
                            for (int iV = 0; iV < 3; iV++)
                                if (qPatch[iV] > 0)
                                    tri(Eigen::all, iV) = coords(Eigen::all, qPatch[iV] - 1);
                                else if (qPatch[iV] < 0)
                                    tri(Eigen::all, iV) = vfv->GetFaceQuadraturePPhys(iFace, -qPatch[iV] - 1);
                                else
                                    tri(Eigen::all, iV) = coords(Eigen::all, 1) + Geom::tPoint{0., 0., vfv->GetFaceArea(iFace)};
                            Triangles.push_back(tri);
                        }
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
            if (mesh->coords.father->getMPI().rank == 0)
                log() << fmt::format("=== EulerEvaluator<model>::GetWallDist() with minWallDist = {:.4e}, ", settings.minWallDist)
                      << " To search in " << TrianglesFull->Size() << std::endl;

            auto executeSearch = [&]()
            {
                log() << fmt::format("Start search rank [{}]", mesh->getMPI().rank) << std::endl;
                typedef CGAL::Simple_cartesian<double> K;
                typedef K::FT FT;
                // typedef K::Ray_3 Ray;
                // typedef K::Line_3 Line;
                typedef K::Point_3 Point;
                typedef K::Triangle_3 Triangle;
                typedef std::vector<Triangle>::iterator Iterator;
                typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
                typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
                typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

                std::vector<Triangle> triangles;
                triangles.reserve(TrianglesFull->Size());

                for (index i = 0; i < TrianglesFull->Size(); i++)
                {
                    Point p0((*TrianglesFull)[i](0, 0), (*TrianglesFull)[i](1, 0), (*TrianglesFull)[i](2, 0));
                    Point p1((*TrianglesFull)[i](0, 1), (*TrianglesFull)[i](1, 1), (*TrianglesFull)[i](2, 1));
                    Point p2((*TrianglesFull)[i](0, 2), (*TrianglesFull)[i](1, 2), (*TrianglesFull)[i](2, 2));
                    triangles.push_back(Triangle(p0, p1, p2));
                }
                TrianglesLocal->Resize(0, 3, 3);
                TrianglesFull->Resize(0, 3, 3);
                double minDist = veryLargeReal;
                this->dWall.resize(mesh->NumCellProc());

                if (!triangles.empty())
                {
                    // std::cout << "tree building" << std::endl;
                    Tree tree(triangles.begin(), triangles.end());

                    // std::cout << "tree built" << std::endl;
                    // search

                    for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
                    {
                        // std::cout << "iCell " << iCell << std::endl;
                        auto quadCell = vfv->GetCellQuad(iCell);
                        dWall[iCell].resize(quadCell.GetNumPoints());
                        for (int ig = 0; ig < quadCell.GetNumPoints(); ig++)
                        {
                            // std::cout << "iG " << ig << std::endl;
                            auto p = vfv->GetCellQuadraturePPhys(iCell, ig);
                            Point pQ(p[0], p[1], p[2]);
                            // std::cout << "pQ " << pQ << std::endl;
                            // Point closest_point = tree.closest_point(pQ);
                            FT sqd = tree.squared_distance(pQ);
                            // std::cout << "sqd" << sqd << std::endl;
                            dWall[iCell][ig] = std::sqrt(sqd);
                            // dWall[iCell][ig] = p(0) < 0 ? p({0, 1}).norm() : p(1); // test for plate BL
                            if (dWall[iCell][ig] < minDist)
                                minDist = dWall[iCell][ig];
                            dWall[iCell][ig] = std::max(settings.minWallDist, dWall[iCell][ig]);
                        }
                    }
                }
                else
                {
                    for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
                    {
                        // std::cout << "iCell " << iCell << std::endl;
                        auto quadCell = vfv->GetCellQuad(iCell);
                        dWall[iCell].setConstant(quadCell.GetNumPoints(), std::pow(veryLargeReal, 1. / 4.));
                    }
                }
                log() << fmt::format("[{}] MinDist: ", mesh->getMPI().rank) << minDist << std::endl;
            };
            if (settings.wallDistExection == 1)
                MPISerialDo(mesh->getMPI(), [&]()
                            { executeSearch(); });
            else if (settings.wallDistExection > 1)
                for (int i = 0; i < settings.wallDistExection; i++)
                {
                    if (mesh->getMPI().rank % settings.wallDistExection == i)
                        executeSearch();
                    MPI::Barrier(mesh->getMPI().comm);
                }
            else
                executeSearch();
        }

        if (settings.wallDistScheme == 2 || settings.wallDistScheme == 20)
        {
            typedef CGAL::Simple_cartesian<double> K;
            typedef K::FT FT;
            // typedef K::Ray_3 Ray;
            // typedef K::Line_3 Line;
            typedef K::Point_3 Point;
            typedef K::Triangle_3 Triangle;
            typedef std::vector<Triangle>::iterator Iterator;
            typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
            typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
            typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
            std::vector<Triangle> triangles;
            triangles.reserve(mesh->NumBnd() * 8 + 8);
            for (index iBnd = 0; iBnd < mesh->NumBnd(); iBnd++)
            {
                if (pBCHandler->GetTypeFromID(mesh->GetBndZone(iBnd)) == EulerBCType::BCWall)
                {
                    index iFace = mesh->bnd2face[iBnd];
                    auto elem = mesh->GetFaceElement(iFace);
                    auto quad = vfv->GetFaceQuad(iFace);
                    {
                        auto qPatches = Geom::Elem::GetQuadPatches(quad);
                        for (auto &qPatch : qPatches)
                        {
                            Eigen::Matrix<real, 3, 3> tri;
                            Geom::tSmallCoords coords;
                            mesh->GetCoordsOnFace(iFace, coords);
                            for (int iV = 0; iV < 3; iV++)
                                if (qPatch[iV] > 0)
                                    tri(Eigen::all, iV) = coords(Eigen::all, qPatch[iV] - 1);
                                else if (qPatch[iV] < 0)
                                    tri(Eigen::all, iV) = vfv->GetFaceQuadraturePPhys(iFace, -qPatch[iV] - 1);
                                else
                                    tri(Eigen::all, iV) = coords(Eigen::all, 1) + Geom::tPoint{0., 0., vfv->GetFaceArea(iFace)};

                            Point p0(tri(0, 0), tri(1, 0), tri(2, 0));
                            Point p1(tri(0, 1), tri(1, 1), tri(2, 1));
                            Point p2(tri(0, 2), tri(1, 2), tri(2, 2));
                            triangles.push_back(Triangle(p0, p1, p2));
                        }
                    }
                }
            }

            if (triangles.empty())
            {
                triangles.push_back(Triangle(
                    Point(veryLargeReal, veryLargeReal, veryLargeReal),
                    Point(veryLargeReal, veryLargeReal, veryLargeReal),
                    Point(veryLargeReal, veryLargeReal, veryLargeReal)));
            }
            MPISerialDo(mesh->getMPI(),
                        [&]()
                        {
                            log() << fmt::format("[{},{}] ", mesh->getMPI().rank, triangles.size());
                            if (mesh->getMPI().rank % 10 == 0)
                                log() << "\n";
                            log().flush();
                        });
            if (mesh->getMPI().rank == 0)
                log() << std::endl;
            Tree tree(triangles.begin(), triangles.end());

            if (settings.wallDistScheme == 2)
                dWall.resize(mesh->NumCellProc());
            index iCellBase = 0;
            index nProcessed = 0;
            int cellLoadNum = std::max(1, static_cast<int>(std::ceil(settings.wallDistCellLoadSize / real(mesh->getMPI().size))));
            if (mesh->coords.father->getMPI().rank == 0)
                log() << fmt::format("=== EulerEvaluator<model>::GetWallDist() using cellLoadNum = {}, ", cellLoadNum)
                      << std::endl;

            if (settings.wallDistScheme == 20)
            {
                index nRefine = 0;
                for (auto &ds : dWall)
                    for (auto d : ds)
                        if (d <= settings.wallDistRefineMax)
                            nRefine++;
                MPI::AllreduceOneIndex(nRefine, MPI_SUM, mesh->getMPI());
                if (mesh->coords.father->getMPI().rank == 0)
                    log() << fmt::format("=== EulerEvaluator<model>::GetWallDist() to refine {}, ", nRefine)
                          << std::endl;
            }

            real t0 = MPI_Wtime();
            for (index iIter = 0;; iIter++)
            {
                std::vector<Geom::tPoint> pnts;
                pnts.reserve(cellLoadNum * 64);
                for (int iCLoad = 0; iCLoad < cellLoadNum; iCLoad++)
                {
                    index iCell = iCellBase + iCLoad;
                    if (iCell < mesh->NumCellProc())
                    {
                        auto quadCell = vfv->GetCellQuad(iCell);
                        for (int ig = 0; ig < quadCell.GetNumPoints(); ig++)
                        {
                            if (settings.wallDistScheme == 20)
                                if (dWall[iCell][ig] > settings.wallDistRefineMax)
                                    continue;
                            auto p = vfv->GetCellQuadraturePPhys(iCell, ig);
                            pnts.push_back(p);
                        }
                    }
                }

                using PntArray = ArrayEigenMatrix<3, 1>;
                ssp<PntArray> PntArrayLocal, PntArrayFull;
                DNDS_MAKE_SSP(PntArrayLocal, mesh->getMPI());
                DNDS_MAKE_SSP(PntArrayFull, mesh->getMPI());
                PntArrayLocal->Resize(pnts.size(), 3, 1);
                for (size_t i = 0; i < pnts.size(); i++)
                    (*PntArrayLocal)[i] = pnts[i];
                ArrayTransformerType<PntArray>::Type PntTransformer;
                PntTransformer.setFatherSon(PntArrayLocal, PntArrayFull);
                PntTransformer.createFatherGlobalMapping();
                std::vector<index> pullingSet;
                pullingSet.resize(PntTransformer.pLGlobalMapping->globalSize());
                // std::cout << "Here1 " << iIter << std::endl;
                if (!pullingSet.size())
                    break;
                if (mesh->getMPI().rank == 0)
                    log() << fmt::format("=== EulerEvaluator<model>::GetWallDist() iter [{}], nProcessed [{}], t [{:.6g}] ",
                                         iIter, nProcessed, MPI_Wtime() - t0)
                          << std::endl;
                for (index i = 0; i < pullingSet.size(); i++)
                    pullingSet[i] = i;
                PntTransformer.createGhostMapping(pullingSet);
                PntTransformer.createMPITypes();
                PntTransformer.pullOnce();
                if (mesh->getMPI().rank == 0)
                    log() << fmt::format("=== EulerEvaluator<model>::GetWallDist() iter [{}], pullOnce done, t [{:.6g}] ",
                                         iIter, MPI_Wtime() - t0)
                          << std::endl;

                std::vector<real> distQueryFull(PntArrayFull->Size(), veryLargeReal);
                for (int iQ = 0; iQ < PntArrayFull->Size(); iQ++)
                {
                    Point pQ((*PntArrayFull)[iQ][0], (*PntArrayFull)[iQ][1], (*PntArrayFull)[iQ][2]);
                    FT sqd = tree.squared_distance(pQ);
                    distQueryFull[iQ] = std::sqrt(sqd);
                }
                if (mesh->getMPI().rank == 0)
                    log() << fmt::format("=== EulerEvaluator<model>::GetWallDist() iter [{}], query done, t [{:.6g}]  ",
                                         iIter, MPI_Wtime() - t0)
                          << std::endl;

                std::vector<real> distQueryFullReduced(PntArrayFull->Size(), veryLargeReal);

                {
                    // index reduceBatch = 1024;
                    // MPIReqHolder reqs;
                    // reqs.reserve(mesh->getMPI().size + PntArrayFull->Size() / reduceBatch);
                    // for (MPI_int i = 0; i < mesh->getMPI().size; i++)
                    // {
                    //     index cstart = PntTransformer.pLGhostMapping->ghostStart.at(i);
                    //     index csize = PntTransformer.pLGhostMapping->ghostSizes.at(i);
                    //     if (csize)
                    //     {
                    //         for (index ic = 0; ic < csize; ic += reduceBatch)
                    //         {
                    //             reqs.push_back(MPI_REQUEST_NULL);
                    //             MPI_Ireduce(distQueryFull.data() + cstart + ic,
                    //                         distQueryFullReduced.data() + cstart + ic,
                    //                         std::min(csize - ic, reduceBatch), DNDS_MPI_REAL, MPI_MIN,
                    //                         i, mesh->getMPI().comm, &reqs.back());
                    //         }
                    //     }
                    // }
                    // MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
                }

                {
                    // std::vector<real> distQueryFullGathered(pnts.size() * mesh->getMPI().size, veryLargeReal);
                    // MPIReqHolder reqs;
                    // size_t iReqCurrent = -1;
                    // for (MPI_int i = 0; i < mesh->getMPI().size; i++)
                    // {
                    //     index cstart = PntTransformer.pLGhostMapping->ghostStart.at(i);
                    //     index csize = PntTransformer.pLGhostMapping->ghostSizes.at(i);
                    //     DNDS_assert(i == mesh->getMPI().rank ? (csize == pnts.size()) : true);
                    //     if (csize)
                    //     {
                    //         reqs.push_back(MPI_REQUEST_NULL);
                    //         MPI_Igather(distQueryFull.data() + cstart, csize, DNDS_MPI_REAL,
                    //                     distQueryFullGathered.data(), csize, DNDS_MPI_REAL,
                    //                     i, mesh->getMPI().comm, &reqs.back());
                    //         if (i == mesh->getMPI().rank)
                    //             iReqCurrent = reqs.size() - 1;
                    //     }
                    // }
                    // MPI_Wait(&reqs[iReqCurrent], MPI_STATUS_IGNORE);
                    // index curStart = PntTransformer.pLGhostMapping->ghostStart.at(mesh->getMPI().rank);
                    // for (index iR = 0; iR < mesh->getMPI().size; iR++)
                    //     for (index i = 0; i < pnts.size(); ++i)
                    //         distQueryFullReduced[curStart + i] =
                    //             std::min(distQueryFullReduced[curStart + i],
                    //                      distQueryFullGathered[i + iR * pnts.size()]);
                    // MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
                }

                MPI::Allreduce(distQueryFull.data(), distQueryFullReduced.data(), distQueryFull.size(), DNDS_MPI_REAL, MPI_MIN, mesh->getMPI().comm);

                if (mesh->getMPI().rank == 0)
                    log() << fmt::format("=== EulerEvaluator<model>::GetWallDist() iter [{}], reduce done, t [{:.6g}] ",
                                         iIter, MPI_Wtime() - t0)
                          << std::endl;
                index iQLoad = 0;
                for (int iCLoad = 0; iCLoad < cellLoadNum; iCLoad++)
                {
                    index iCell = iCellBase + iCLoad;
                    if (iCell < mesh->NumCellProc())
                    {
                        auto quadCell = vfv->GetCellQuad(iCell);
                        if (settings.wallDistScheme == 2)
                            dWall[iCell].resize(quadCell.GetNumPoints());
                        for (int ig = 0; ig < quadCell.GetNumPoints(); ig++)
                        {

                            if (settings.wallDistScheme == 20)
                                if (dWall[iCell][ig] > settings.wallDistRefineMax)
                                    continue;
                            dWall[iCell][ig] = distQueryFullReduced.at(
                                PntTransformer.pLGhostMapping->ghostStart.at(mesh->getMPI().rank) +
                                iQLoad);
                            iQLoad++;
                        }
                    }
                }

                iCellBase += cellLoadNum;
                nProcessed += pullingSet.size();
            }
        }

        if (settings.wallDistScheme == 3)
        {
            int nSweep = settings.wallDistNJacobiSweep;
            int nIter = settings.wallDistIter;
            int nIterSee = 10;
            real smoothBias = 0.0;
            int nGmresSubspace = 10;
            int nGmresRestart = 1;
            int useGmres = settings.wallDistLinSolver;
            real resTh = settings.wallDistResTol;

            int nPPoissonStartIter = settings.wallDistIterStart;
            int nPPoisson = settings.wallDistPoissonP;

            real dTauScale = settings.wallDistDTauScale;

            ArrayDOFV<1> phi, rPhi, dPhi, dPhiNew, phiTmp, dTauInv;
            ArrayDOFV<3> diffPhi;
            vfv->BuildUDof(phi, 1);
            vfv->BuildUDof(rPhi, 1);
            vfv->BuildUDof(dPhi, 1);
            vfv->BuildUDof(dPhiNew, 1);
            vfv->BuildUDof(phiTmp, 1);
            vfv->BuildUDof(dTauInv, 1);
            vfv->BuildUDof(diffPhi, 3);
            phi.setConstant(0.0);
            dTauInv.setConstant(0.0);

            std::vector<std::vector<real>> coefs;
            coefs.resize(mesh->NumCell());
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                coefs.at(iCell).resize(mesh->cell2face[iCell].size() + 1);

            std::vector<MatrixXR> mGGs;
            mGGs.resize(mesh->NumCell());

            // get mGG coefs
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                Geom::tPoint bary = vfv->GetCellQuadraturePPhys(iCell, -1);
                auto &mGG = mGGs.at(iCell);
                mGG.setZero(3 + mesh->cell2face[iCell].size(), 3 + mesh->cell2face[iCell].size());
                mGG({0, 1, 2}, {0, 1, 2}) = Eigen::Matrix3d::Identity();
                real sumFaceArea = 0.;
                for (int ic2f = 0; ic2f < mesh->cell2face[iCell].size(); ic2f++)
                {
                    index iFace = mesh->cell2face[iCell][ic2f];
                    index iCellOther = mesh->CellFaceOther(iCell, iFace);
                    int if2c = mesh->CellIsFaceBack(iCell, iFace) ? 0 : 1;
                    Geom::tPoint uNormOut = vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1) * (if2c ? -1 : 1);
                    auto faceBndID = mesh->GetFaceZone(iFace);
                    auto faceBCType = pBCHandler->GetTypeFromID(faceBndID);
                    Geom::tPoint baryOther = bary;
                    Geom::tPoint bFace = vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1);
                    if (iCellOther != UnInitIndex)
                    {
                        baryOther = vfv->GetOtherCellPointFromCell(
                            iCell, iCellOther, iFace,
                            vfv->GetCellQuadraturePPhys(iCellOther, -1));
                    }
                    else
                    {
                        DNDS_assert(faceBCType != BCUnknown);
                        DNDS_assert(!Geom::FaceIDIsInternal(faceBndID));
                        baryOther = bFace * 2 - bary;
                    }
                    mGG({0, 1, 2}, 3 + ic2f) = -uNormOut * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell);
                    mGG(3 + ic2f, {0, 1, 2}) = (baryOther + bary - 2 * bFace).transpose();
                    mGG(3 + ic2f, 3 + ic2f) = 2;
                    sumFaceArea += vfv->GetFaceArea(iFace);
                }
                diffPhi[iCell] /= vfv->GetCellVol(iCell);
                auto mGGLU = mGG.fullPivLu();
                DNDS_assert_info(mGGLU.isInvertible(), fmt::format("[[{}]]\n", Eigen::MatrixFMTSafe<real, -1, -1>(mGG)));
                MatrixXR mGGInv = mGGLU.inverse();
                mGG = mGGInv;

                real LCell = vfv->GetCellVol(iCell) / sumFaceArea;
                dTauInv[iCell](0) = 1. / (1. / std::pow(LCell, 2) * dTauScale);
            }
            dTauInv.trans.startPersistentPull();
            dTauInv.trans.waitPersistentPull();

            auto getDiffPhi = [&](ArrayDOFV<1> &phi, ArrayDOFV<3> &diffPhi)
            {
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    Geom::tPoint bary = vfv->GetCellQuadraturePPhys(iCell, -1);
                    diffPhi[iCell].setZero();
                    Eigen::Vector<real, Eigen::Dynamic> bGG;
                    bGG.setZero(3 + mesh->cell2face[iCell].size());
                    for (int ic2f = 0; ic2f < mesh->cell2face[iCell].size(); ic2f++)
                    {
                        index iFace = mesh->cell2face[iCell][ic2f];
                        index iCellOther = mesh->CellFaceOther(iCell, iFace);
                        int if2c = mesh->CellIsFaceBack(iCell, iFace) ? 0 : 1;
                        Geom::tPoint uNormOut = vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1) * (if2c ? -1 : 1);
                        auto faceBndID = mesh->GetFaceZone(iFace);
                        auto faceBCType = pBCHandler->GetTypeFromID(faceBndID);
                        real phiOther = phi[iCell](0);
                        Geom::tPoint baryOther = bary;
                        Geom::tPoint bFace = vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1);
                        if (iCellOther != UnInitIndex)
                        {
                            phiOther = phi[iCellOther](0);
                            baryOther = vfv->GetOtherCellPointFromCell(
                                iCell, iCellOther, iFace,
                                vfv->GetCellQuadraturePPhys(iCellOther, -1));
                        }
                        else
                        {
                            DNDS_assert(faceBCType != BCUnknown);
                            DNDS_assert(!Geom::FaceIDIsInternal(faceBndID));
                            if (faceBCType == BCWall)
                                phiOther = -phi[iCell](0);
                            baryOther = bFace * 2 - bary;
                        }
                        real distThis = (bFace - bary).norm();
                        real distOther = (bFace - baryOther).norm();
                        real phiFace = (distOther * phi[iCell](0) + distThis * phiOther) / (distOther + distThis) - phi[iCell](0);
                        diffPhi[iCell] += uNormOut * vfv->GetFaceArea(iFace) * phiFace;
                        bGG(3 + ic2f) = phiOther + phi[iCell](0);
                    }
                    diffPhi[iCell] /= vfv->GetCellVol(iCell);
                    diffPhi[iCell] = (mGGs.at(iCell) * bGG)({0, 1, 2}); // comment to use traditional GG
                }
            };

            int pPoissonCur = 2;

            auto rhsPhi = [&](ArrayDOFV<1> &phi, ArrayDOFV<3> &diffPhi, ArrayDOFV<1> &rhs, std::vector<std::vector<real>> &coefs, bool updateCoefs)
            {
                const real supressRec = 1.0;
                getDiffPhi(phi, diffPhi);
                diffPhi.trans.startPersistentPull();
                diffPhi.trans.waitPersistentPull();
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    rhs[iCell](0) = 1.;
                    Geom::tPoint bary = vfv->GetCellQuadraturePPhys(iCell, -1);
                    if (updateCoefs)
                        coefs.at(iCell).at(0) = 0.;
                    for (int ic2f = 0; ic2f < mesh->cell2face[iCell].size(); ic2f++)
                    {
                        index iFace = mesh->cell2face[iCell][ic2f];
                        index iCellOther = mesh->CellFaceOther(iCell, iFace);
                        int if2c = mesh->CellIsFaceBack(iCell, iFace) ? 0 : 1;
                        Geom::tPoint uNormOut = vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1) * (if2c ? -1 : 1);
                        auto faceBndID = mesh->GetFaceZone(iFace);
                        auto faceBCType = pBCHandler->GetTypeFromID(faceBndID);
                        real phiOther = phi[iCell](0);
                        Geom::tPoint baryOther = bary;
                        Geom::tPoint bFace = vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1);
                        real phiThisFace = phi[iCell](0) + (bFace - bary).dot(diffPhi[iCell]) * supressRec;
                        real phiOtherFace = phiThisFace;
                        real diffPhiNormThis = diffPhi[iCell].dot(uNormOut) * supressRec;
                        real diffPhiNormOther = diffPhiNormThis;
                        Geom::tPoint diffPhiFace = diffPhi[iCell] * supressRec;
                        if (iCellOther != UnInitIndex)
                        {
                            phiOther = phi[iCellOther](0);
                            baryOther = vfv->GetOtherCellPointFromCell(
                                iCell, iCellOther, iFace,
                                vfv->GetCellQuadraturePPhys(iCellOther, -1));
                            phiOtherFace = phiOther + (bFace - baryOther).dot(diffPhi[iCellOther]) * supressRec;
                            diffPhiNormOther = diffPhi[iCellOther].dot(uNormOut) * supressRec; //! todo: periodic!!
                            diffPhiFace = 0.5 * (diffPhiFace + diffPhi[iCellOther] * supressRec);
                        }
                        else
                        {
                            DNDS_assert(faceBCType != BCUnknown);
                            DNDS_assert(!Geom::FaceIDIsInternal(faceBndID));
                            if (faceBCType == BCWall)
                                phiOther = -phi[iCell](0), phiOtherFace = -phiThisFace;
                            else
                            {
                                diffPhiNormOther = -diffPhiNormThis;
                                diffPhiFace.setZero();
                            }
                            baryOther = bFace * 2 - bary;
                        }
                        // real dist = (baryOther - bary).norm();
                        real dist = std::abs((baryOther - bary).dot(uNormOut));

                        real diffPhiNorm = ((phiOtherFace - phiThisFace) * 1.0 / dist + 0.5 * (diffPhiNormOther + diffPhiNormThis));
                        diffPhiFace += ((phiOtherFace - phiThisFace) * 1.0 / dist) * uNormOut;
                        real diffPhiFaceMag = diffPhiFace.norm();

                        // rhs[iCell](0) += (phiOther - phi[iCell](0)) * 1.0 / dist * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell);
                        rhs[iCell](0) += std::pow(diffPhiFaceMag, pPoissonCur - 2) * diffPhiNorm * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell);
                        if (updateCoefs)
                        {
                            coefs.at(iCell).at(0) -= 1.0 / dist * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) * std::pow(diffPhiFaceMag, pPoissonCur - 2) * (pPoissonCur - 1);
                            coefs.at(iCell).at(1 + ic2f) = 1.0 / dist * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) * std::pow(diffPhiFaceMag, pPoissonCur - 2) * (pPoissonCur - 1);
                        }
                    }
                }
            };

            auto solveDphi = [&](ArrayDOFV<1> &rhsPhi, ArrayDOFV<1> &dphi, ArrayDOFV<1> &dphiNew, std::vector<std::vector<real>> &coefs, ArrayDOFV<1> &dTauInv)
            {
                dphi.setConstant(0.0);
                for (int iSweep = 1; iSweep <= nSweep; iSweep++)
                {
                    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                    {
                        dphiNew[iCell] = rhsPhi[iCell];
                        for (int ic2f = 0; ic2f < mesh->cell2face[iCell].size(); ic2f++)
                        {
                            index iFace = mesh->cell2face[iCell][ic2f];
                            index iCellOther = mesh->CellFaceOther(iCell, iFace);
                            if (iCellOther != UnInitIndex)
                                dphiNew[iCell] += coefs[iCell][ic2f + 1] * dphi[iCellOther];
                        }
                        dphiNew[iCell] /= -coefs[iCell][0] + dTauInv[iCell](0);
                    }
                    dphiNew.trans.startPersistentPull();
                    dphiNew.trans.waitPersistentPull();
                    dphi = dphiNew;
                }
            };
            Linear::GMRES_LeftPreconditioned<ArrayDOFV<1>> gmres(
                nGmresSubspace,
                [&](ArrayDOFV<1> &v)
                {
                    vfv->BuildUDof(v, 1);
                });
            for (; pPoissonCur <= nPPoisson; pPoissonCur += 2)
            {

                real incNormBase{0};
                real resNormBase{0};
                for (int iIter = 1; iIter <= nIter; iIter++)
                {
                    rhsPhi(phi, diffPhi, rPhi, coefs, true);
                    real normR = rPhi.norm2();
                    real normPhi = phi.norm2();
                    dPhi.setConstant(0.0);

                    if (useGmres == 0)
                    {
                        solveDphi(rPhi, dPhi, dPhiNew, coefs, dTauInv);
                    }
                    else
                    {
                        gmres.solve(
                            [&](ArrayDOFV<1> &x, ArrayDOFV<1> &Ax)
                            {
                                real normX = x.norm2();
                                real ratio = normPhi * 1e-7 / (normX + normPhi * 1e-7 + verySmallReal) + 1e-12;
                                dPhiNew = phi;
                                rhsPhi(dPhiNew, diffPhi, phiTmp, coefs, false);
                                dPhiNew.addTo(x, ratio);
                                rhsPhi(dPhiNew, diffPhi, Ax, coefs, false);
                                Ax.addTo(phiTmp, -1.0);
                                Ax *= -1. / ratio;
                                phiTmp = x;
                                phiTmp *= dTauInv;
                                Ax += phiTmp;
                                Ax.trans.startPersistentPull();
                                Ax.trans.waitPersistentPull();

                                // for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                                // {
                                //     Ax[iCell] = -coefs[iCell][0] * x[iCell];
                                //     for (int ic2f = 0; ic2f < mesh->cell2face[iCell].size(); ic2f++)
                                //     {
                                //         index iFace = mesh->cell2face[iCell][ic2f];
                                //         index iCellOther = mesh->CellFaceOther(iCell, iFace);
                                //         if (iCellOther != UnInitIndex)
                                //             Ax[iCell] += -coefs[iCell][ic2f + 1] * x[iCellOther];
                                //     }
                                // }
                                // Ax.trans.startPersistentPull();
                                // Ax.trans.waitPersistentPull();
                            },
                            [&](ArrayDOFV<1> &x, ArrayDOFV<1> &Mx)
                            {
                                Mx.setConstant(0.0);
                                solveDphi(x, Mx, dPhiNew, coefs, dTauInv);
                                // Mx.trans.startPersistentPull();
                                // Mx.trans.waitPersistentPull();
                            },
                            [&](ArrayDOFV<1> &a, ArrayDOFV<1> &b) -> real
                            {
                                return a.dot(b);
                            },
                            rPhi,
                            dPhi,
                            nGmresRestart,
                            [&](int iRestart, real res, real resB)
                            {
                                // if (res < resB * 1e-6)
                                //     return true;
                                return false;
                            });
                    }

                    real incNorm = dPhi.norm2();
                    DNDS_assert(std::isfinite(incNorm));
                    phi += dPhi;
                    phi.trans.startPersistentPull();
                    phi.trans.waitPersistentPull();
                    if (iIter == 1)
                        incNormBase = incNorm;
                    resNormBase = std::max(resNormBase, normR);
                    bool nowExit = false;
                    if ((normR < resNormBase * resTh * std::pow(0.1, pPoissonCur - 2)) && iIter > nPPoissonStartIter)
                        nowExit = true;
                    if (iIter % nIterSee == 0 || iIter == nIter || nowExit)
                        if (phi.father->getMPI().rank == 0)
                            log() << fmt::format("EulerEvaluator<model>::GetWallDist(): poisson [p={}] [{}] inc:  [{:.4e}] -> [{:.4e}],  res: [{:.4e}] -> [{:.4e}]",
                                                 pPoissonCur, iIter, incNormBase, incNorm, resNormBase, normR)
                                  << std::endl;
                    if (nowExit)
                        break;
                }
                getDiffPhi(phi, diffPhi);
                diffPhi.trans.startPersistentPull();
                diffPhi.trans.waitPersistentPull();
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    Geom::tPoint gradPhi;
                    gradPhi.setZero();
                    gradPhi = diffPhi[iCell];
                    real nD = 1;
                    for (int ic2f = 0; ic2f < mesh->cell2face[iCell].size(); ic2f++)
                    {
                        index iFace = mesh->cell2face[iCell][ic2f];
                        index iCellOther = mesh->CellFaceOther(iCell, iFace);
                        if (iCellOther != UnInitIndex)
                        {
                            gradPhi += diffPhi[iCell];
                            gradPhi += diffPhi[iCellOther] * smoothBias;
                            nD += 1. + smoothBias;
                        }
                    }
                    gradPhi /= nD;
                    real gradPhiNorm = gradPhi.norm();
                    real dEst = std::pow(pPoissonCur / real(pPoissonCur - 1) * phi[iCell](0) + std::pow(gradPhiNorm, pPoissonCur), real(pPoissonCur - 1) / pPoissonCur) - std::pow(gradPhiNorm, pPoissonCur - 1);
                    // dPhi[iCell](0) = phi[iCell](0);
                    dPhi[iCell](0) = dEst;
                }
                dPhi.trans.startPersistentPull();
                dPhi.trans.waitPersistentPull();
                phi = dPhi;
            }

            auto minval = dPhi.min();

            dWall.resize(mesh->NumCellProc());
            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
            {
                auto quadCell = vfv->GetCellQuad(iCell);
                dWall[iCell].resize(quadCell.GetNumPoints());
            }

            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
                dWall.at(iCell).setConstant(std::max(dPhi[iCell](0), settings.minWallDist));
            if (phi.father->getMPI().rank == 0)
                log() << fmt::format("EulerEvaluator<model>::GetWallDist(): poisson min dist [{}]", minval(0)) << std::endl;
        }

        dWallFace.resize(mesh->NumFaceProc());
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto f2c = mesh->face2cell[iFace];
            real facialDist = dWall.at(f2c[0]).mean();
            if (f2c[1] != UnInitIndex)
                facialDist = 0.5 * (facialDist + dWall.at(f2c[1]).mean());
            dWallFace[iFace] = facialDist;
        }
    }

    // Eigen::Vector<real, -1> EulerEvaluator::CompressRecPart(
    //     const Eigen::Vector<real, -1> &umean,
    //     const Eigen::Vector<real, -1> &uRecInc)1

    //! evaluates dt and facial spectral radius
    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    void EulerEvaluator<model>::EvaluateDt(
        ArrayDOFV<1> &dt,
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        real CFL, real &dtMinall, real MaxDt,
        bool UseLocaldt)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "EvaluateDt 1");
        for (auto &i : lambdaCell)
            i = 0.0;

#if defined(DNDS_DIST_MT_USE_OMP)
#pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto f2c = mesh->face2cell[iFace];
            TVec unitNorm = vfv->GetFaceNorm(iFace, -1)(Seq012);

            index iCellL = f2c[0];
            TU UL = u[iCellL];
            this->UFromCell2Face(UL, iFace, f2c[0], 0);
            TU uMean = UL;
            real pL, asqrL, HL, pR, asqrR, HR;
            TVec vL = UL(Seq123) / UL(0);
            TVec vR = vL;
            Gas::IdealGasThermal(UL(I4), UL(0), vL.squaredNorm(),
                                 settings.idealGasProperty.gamma,
                                 pL, asqrL, HL);
            pR = pL, HR = HL, asqrR = asqrL;
            if (f2c[1] != UnInitIndex)
            {
                TU UR = u[f2c[1]];
                this->UFromCell2Face(UR, iFace, f2c[1], 1);
                uMean = (uMean + UR) * 0.5;
                vR = UR(Seq123) / UR(0);
                Gas::IdealGasThermal(UR(I4), UR(0), vR.squaredNorm(),
                                     settings.idealGasProperty.gamma,
                                     pR, asqrR, HR);
            }
            TDiffU GradULxy, GradURxy;
            GradULxy.resize(Eigen::NoChange, nVars);
            GradURxy.resize(Eigen::NoChange, nVars);
            GradULxy.setZero(), GradURxy.setZero();
            if constexpr (gDim == 2)
                GradULxy({0, 1}, Eigen::all) =
                    vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, -1, std::array<int, 2>{1, 2}, 3) *
                    uRec[f2c[0]]; // 2d here
            else
                GradULxy({0, 1, 2}, Eigen::all) =
                    vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, -1, std::array<int, 3>{1, 2, 3}, 4) *
                    uRec[f2c[0]]; // 3d here
            this->DiffUFromCell2Face(GradULxy, iFace, f2c[0], 0);
            GradURxy = GradULxy;
            if (f2c[1] != UnInitIndex)
            {
                if constexpr (gDim == 2)
                    GradURxy({0, 1}, Eigen::all) =
                        vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, -1, std::array<int, 2>{1, 2}, 3) *
                        uRec[f2c[1]]; // 2d here
                else
                    GradURxy({0, 1, 2}, Eigen::all) =
                        vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, -1, std::array<int, 3>{1, 2, 3}, 4) *
                        uRec[f2c[1]]; // 3d here
                this->DiffUFromCell2Face(GradURxy, iFace, f2c[1], 1);
            }
            TDiffU GradUMeanXY = (GradURxy + GradULxy) / 2;

            DNDS_assert(uMean(0) > 0);
            TVec veloMean = (uMean(Seq123).array() / uMean(0)).matrix();
            // real veloNMean = veloMean.dot(unitNorm); // original
            real veloNMean = 0.5 * (vL + vR).dot(unitNorm); // paper
            real veloNL = vL.dot(unitNorm);
            real veloNR = vR.dot(unitNorm);
            real vgN = this->GetFaceVGrid(iFace, -1).dot(unitNorm);

            // real ekFixRatio = 0.001;
            // Eigen::Vector3d velo = uMean({1, 2, 3}) / uMean(0);
            // real vsqr = velo.squaredNorm();
            // real Ek = vsqr * 0.5 * uMean(0);
            // real Efix = Ek * ekFixRatio;
            // real e = uMean(4) - Ek;
            // if (e < 0)
            //     e = 0.5 * Efix;
            // else if (e < Efix)
            //     e = (e * e + Efix * Efix) / (2 * Efix);
            // uMean(4) = Ek + e;

            real pMean, asqrMean, HMean;
            Gas::IdealGasThermal(uMean(I4), uMean(0), veloMean.squaredNorm(),
                                 settings.idealGasProperty.gamma,
                                 pMean, asqrMean, HMean);

            pMean = (pL + pR) * 0.5;
            real aMean = sqrt(settings.idealGasProperty.gamma * pMean / uMean(0)); // paper

            // DNDS_assert(asqrMean >= 0);
            // real aMean = std::sqrt(asqrMean); // original
            real lambdaConvection = std::abs(veloNMean - vgN) + aMean;
            lambdaConvection = std::max(std::sqrt(asqrL) + std::abs(veloNL - vgN), std::sqrt(asqrR) + std::abs(veloNR - vgN));
            DNDS_assert_info(
                asqrL >= 0 && asqrR >= 0,
                fmt::format(" mean value violates PP! asqr: [{} {}]", asqrL, asqrR));

            // ! refvalue:
            real muRef = settings.idealGasProperty.muGas;

            real gamma = settings.idealGasProperty.gamma;
            real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * uMean(0));
            real muf = muEff(uMean, T);
            real muPhy = muf;
            real muTurb = this->getMuTur(uMean, GradUMeanXY, muRef, muf, iFace);
            muf += muTurb;

            real lamVis = muf / uMean(0) *
                          std::max(4. / 3., gamma / settings.idealGasProperty.prGas);

            real lamFace = lambdaConvection * vfv->GetFaceArea(iFace);

            real area = vfv->GetFaceArea(iFace);
            real areaSqr = area * area;
            real volR = vfv->GetCellVol(iCellL);
            // lambdaCell[iCellL] += lamFace + 2 * lamVis * areaSqr / fv->GetCellVol(iCellL);
            if (f2c[1] != UnInitIndex) // can't be non local
                                       // lambdaCell[f2c[1]] += lamFace + 2 * lamVis * areaSqr / fv->volumeLocal[f2c[1]],
                volR = vfv->GetCellVol(f2c[1]);

            lambdaFace[iFace] = lambdaConvection + lamVis * area * (1. / vfv->GetCellVol(iCellL) + 1. / volR);
            lambdaFaceC[iFace] = std::abs(veloNMean - vgN) + lamVis * area * (1. / vfv->GetCellVol(iCellL) + 1. / volR); // passive part
            lambdaFaceVis[iFace] = lamVis * area * (1. / vfv->GetCellVol(iCellL) + 1. / volR);
            lambdaFace0[iFace] = lambdaFace123[iFace] = lambdaFace4[iFace] = lambdaConvection;

            // if (f2c[0] == 10756)
            // {
            //     std::cout << "----Lambdas" << std::setprecision(16) << iFace << std::endl;
            //     std::cout << lambdaConvection << std::endl;
            //     std::cout << lambdaFaceVis[iFace] << std::endl;
            //     std::cout << veloNMean << " " << aMean << std::endl;
            //     std::cout << gamma << " " << pMean << " " << uMean(0) << std::endl;
            // }

            lambdaCell[iCellL] += lambdaFace[iFace] * vfv->GetFaceArea(iFace);
            if (f2c[1] != UnInitIndex) // can't be non local
                lambdaCell[f2c[1]] += lambdaFace[iFace] * vfv->GetFaceArea(iFace);

            deltaLambdaFace[iFace] = std::abs((vR - vL).dot(unitNorm)) + std::sqrt(std::abs(asqrR - asqrL)) * 0.7071;
        }
        real dtMin = veryLargeReal;
#if defined(DNDS_DIST_MT_USE_OMP)
#pragma omp parallel for schedule(runtime) reduction(min: dtMin)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            // std::cout << fv->GetCellVol(iCell) << " " << (lambdaCell[iCell]) << " " << CFL << std::endl;
            // exit(0);
            dt[iCell](0) = std::min(CFL * vfv->GetCellVol(iCell) * vfv->GetCellSmoothScaleRatio(iCell) / (lambdaCell[iCell] + 1e-100), MaxDt);
            dtMin = std::min(dtMin, dt[iCell](0));
            // if (iCell == 10756)
            // {
            //     std::cout << std::endl;
            // }
        }

        MPI::Allreduce(&dtMin, &dtMinall, 1, DNDS_MPI_REAL, MPI_MIN, u.father->getMPI().comm);

        // if (uRec.father->getMPI().rank == 0)
        //     std::cout << "dt min is " << dtMinall << std::endl;
        if (!UseLocaldt)
        {
            dt.setConstant(dtMinall);
        }
        // if (uRec.father->getMPI().rank == 0)
        // log() << "dt: " << dtMin << std::endl;
    }

    DNDS_SWITCH_INTELLISENSE(
        // the real definition
        template <EulerModel model>
        ,
        // the intellisense friendly definition
        template <>
    )
    typename EulerEvaluator<model>::TU_Batch EulerEvaluator<model>::fluxFace(
        const TU_Batch &ULxy,
        const TU_Batch &URxy,
        const TU &ULMeanXy,
        const TU &URMeanXy,
        const TDiffU_Batch &DiffUxy,
        const TDiffU_Batch &DiffUxyPrim,
        const TVec_Batch &unitNorm,
        const TVec_Batch &vgXY,
        const TVec &unitNormC,
        const TVec &vgC,
        TU_Batch &FLfix,
        TU_Batch &FRfix,
        TReal_Batch &lam0V, TReal_Batch &lam123V, TReal_Batch &lam4V,
        Geom::t_index btype,
        typename Gas::RiemannSolverType rsType,
        index iFace)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS

        int nB = ULxy.cols();

        TU_Batch UMeanXy = 0.5 * (ULxy + URxy);

        // PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterB);
        real muRef = settings.idealGasProperty.muGas;
        real TRef = settings.idealGasProperty.TRef;

        /** viscous flux **/
        TU_Batch visFluxV;
        visFluxV.resizeLike(ULxy);
        for (int iB = 0; iB < nB; iB++)
        {
            real pMean, asqrMean, Hmean;
            real gamma = settings.idealGasProperty.gamma;
            TU UMeanXYC = UMeanXy(Eigen::all, iB);
            auto seqC = Eigen::seq(iB * dim, iB * dim + dim - 1);
            TDiffU DiffUxyC = DiffUxy(seqC, Eigen::all);
            TDiffU DiffUxyPrimC = DiffUxyPrim(seqC, Eigen::all);
            TVec uNormC = unitNorm(Eigen::all, iB);
            Gas::IdealGasThermal(UMeanXYC(I4), UMeanXYC(0), (UMeanXy(Seq123, iB) / UMeanXYC(0)).squaredNorm(),
                                 gamma, pMean, asqrMean, Hmean);
            real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXYC(0));
            real CSut = settings.idealGasProperty.CSutherland;
            real mufPhy, muf;
            muf = muEff(UMeanXYC, T);
            mufPhy = muf;
            // PerformanceTimer::Instance().StopTimer(PerformanceTimer::LimiterB);
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
            real muTurb = this->getMuTur(UMeanXYC, DiffUxyC, muRef, muf, iFace);
            muf += muTurb;

            real k = settings.idealGasProperty.CpGas * muTurb / 0.9 +
                     settings.idealGasProperty.CpGas * mufPhy / settings.idealGasProperty.prGas;
            TU VisFlux;
            VisFlux.resizeLike(ULMeanXy);
            VisFlux.setZero();
            Gas::ViscousFlux_IdealGas<dim>(
                UMeanXYC, DiffUxyPrimC, uNormC, pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWall,
                settings.idealGasProperty.gamma,
                muf, muTurb / (muf + verySmallReal), settings.ransUseQCR,
                k,
                settings.idealGasProperty.CpGas,
                VisFlux);

            if (pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWallInvis ||
                pBCHandler->GetTypeFromID(btype) == EulerBCType::BCSym)
            {
                // VisFlux *= 0.0;
            }
            this->visFluxTurVariable(UMeanXYC, DiffUxyPrimC, muRef, mufPhy, muTurb, uNormC, iFace, VisFlux);
            visFluxV(Eigen::all, iB) = VisFlux;
#endif
            if (!isfinite(pMean) || !isfinite(pMean) || !isfinite(pMean))
            {
                std::cout << T << std::endl;
                std::cout << muf << std::endl;
                std::cout << pMean << std::endl;
                DNDS_assert(false);
            }
        }

        auto exitFun = [&]()
        {
            std::cout << "face at" << vfv->GetFaceQuadraturePPhys(iFace, -1) << '\n';
            std::cout << "UL" << ULxy.transpose() << '\n';
            std::cout << "UR" << URxy.transpose() << std::endl;
        };

        if (pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWallInvis ||
            pBCHandler->GetTypeFromID(btype) == EulerBCType::BCSym)
        {
            // ? normal invert here?
        }

        lam0V.resize(nB);
        lam123V.resize(nB);
        lam4V.resize(nB);

        auto RSWrapper_XY =
            [&](Gas::RiemannSolverType rsType,
                auto &&UL, auto &&UR, auto &&ULm, auto &&URm, auto &&vg, auto &&n,
                real gamma, auto &&finc, real dLambda,
                real &lam0, real &lam123, real &lam4)
        {
            Gas::InviscidFlux_IdealGas_Dispatcher<dim>(rsType, UL, UR, ULm, URm, vg, n, gamma, finc, dLambda, exitFun, lam0, lam123, lam4);
        };

        TU_Batch finc;
        finc.resizeLike(ULxy);
        // TU_Batch finc1;
        // finc1.resizeLike(ULxy);
        if (settings.rsTypeWall != Gas::UnknownRS && pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWall)
        {
            rsType = settings.rsTypeWall;
        }

        auto runRsOnNorm = [&]()
        {
            if (settings.rsMeanValueEig != 0 &&
                (rsType >= Gas::Roe_M1 && rsType <= Gas::Roe_M5))
            {
                real lam0{0}, lam123{0}, lam4{0};
                Gas::InviscidFlux_IdealGas_Batch_Dispatcher<dim>(
                    rsType,
                    ULxy, URxy, ULMeanXy, URMeanXy, vgXY, vgC, unitNorm, unitNormC,
                    settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    exitFun, lam0, lam123, lam4);
                lam0V.setConstant(lam0);
                lam123V.setConstant(lam123);
                lam4V.setConstant(lam4);
            }
            else
                for (int iB = 0; iB < nB; iB++)
                {
                    RSWrapper_XY(rsType, ULxy(Eigen::all, iB), URxy(Eigen::all, iB), ULMeanXy, URMeanXy,
                                 vgXY(Eigen::all, iB), unitNorm(Eigen::all, iB),
                                 settings.idealGasProperty.gamma, finc(Eigen::all, iB), deltaLambdaFace[iFace],
                                 lam0V(iB), lam123V(iB), lam4V(iB));
                }
        };

        if (settings.rsRotateScheme == 0)
        {
            runRsOnNorm();
        }
        else
        {
            TVec veloL = ULMeanXy(Seq123) / ULMeanXy(0);
            TVec veloR = URMeanXy(Seq123) / URMeanXy(0);
            TVec diffVelo = veloR - veloL;
            real diffVeloN = diffVelo.norm();
            real veloLN = veloL.norm();
            real veloRN = veloR.norm();
            if (diffVeloN < (smallReal * 10) * (veloLN + veloRN) || diffVeloN < std::sqrt(verySmallReal))
                runRsOnNorm();
            else
            {
                TVec N1 = diffVelo / diffVeloN;
                DNDS_assert_info(std::abs(N1.norm() - 1) < 1e-5, fmt::format("{}", diffVeloN));
                TReal_Batch N1Proj = N1.transpose() * unitNorm;

                TVec_Batch N2 = unitNorm - N1 * N1Proj;
                TReal_Batch N2Proj = N2.colwise().norm().array().max(smallReal * 10);
                N2.array().rowwise() /= N2Proj.array();

                real N1ProjC = N1.dot(unitNormC);
                TVec N2C = unitNormC - N1 * N1ProjC;
                real N2CProj = std::max(N2C.norm(), smallReal * 10);
                N2C /= N2CProj;

                TVec_Batch N1B;
                N1B.resizeLike(N2);
                N1B.colwise() = N1;

                TReal_Batch lam4V1, lam0V1, lam123V1;
                lam0V1.resizeLike(lam0V);
                lam4V1.resizeLike(lam0V);
                lam123V1.resizeLike(lam0V);

                TU_Batch F1;
                F1.resizeLike(finc);

                auto rsTypeAux = settings.rsTypeAux == Gas::UnknownRS ? rsType : settings.rsTypeAux;

                if (settings.rsMeanValueEig != 0 &&
                    (rsTypeAux >= Gas::Roe_M1 && rsTypeAux <= Gas::Roe_M5))
                {
                    real lam0{0}, lam123{0}, lam4{0};
                    Gas::InviscidFlux_IdealGas_Batch_Dispatcher<dim>(
                        rsTypeAux,
                        ULxy, URxy, ULMeanXy, URMeanXy, vgXY, vgC, N1B, N1,
                        settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                        exitFun, lam0, lam123, lam4);
                    lam0V1.setConstant(lam0);
                    lam123V1.setConstant(lam123);
                    lam4V1.setConstant(lam4);
                }
                else
                    for (int iB = 0; iB < nB; iB++)
                    {
                        RSWrapper_XY(rsTypeAux, ULxy(Eigen::all, iB), URxy(Eigen::all, iB), ULMeanXy, URMeanXy,
                                     vgXY(Eigen::all, iB), N1,
                                     settings.idealGasProperty.gamma, F1(Eigen::all, iB), deltaLambdaFace[iFace],
                                     lam0V1(iB), lam123V1(iB), lam4V1(iB));
                    }

                if (settings.rsMeanValueEig != 0 &&
                    (rsType >= Gas::Roe_M1 && rsType <= Gas::Roe_M5))
                {
                    real lam0{0}, lam123{0}, lam4{0};
                    Gas::InviscidFlux_IdealGas_Batch_Dispatcher<dim>(
                        rsType,
                        ULxy, URxy, ULMeanXy, URMeanXy, vgXY, vgC, N2, N2C,
                        settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                        exitFun, lam0, lam123, lam4);
                    lam0V.setConstant(lam0);
                    lam123V.setConstant(lam123);
                    lam4V.setConstant(lam4);
                }
                else
                    for (int iB = 0; iB < nB; iB++)
                    {
                        RSWrapper_XY(rsType, ULxy(Eigen::all, iB), URxy(Eigen::all, iB), ULMeanXy, URMeanXy,
                                     vgXY(Eigen::all, iB), N2(Eigen::all, iB),
                                     settings.idealGasProperty.gamma, finc(Eigen::all, iB), deltaLambdaFace[iFace],
                                     lam0V(iB), lam123V(iB), lam4V(iB));
                    }

                finc.array().rowwise() *= N2Proj.array();
                finc.array() += F1.array().rowwise() * N1Proj.array();

                TReal_Batch N12ProjSum = N1Proj.array() + N2Proj.array();
                lam0V.array() *= N2Proj.array() / N12ProjSum.array();
                lam4V.array() *= N2Proj.array() / N12ProjSum.array();
                lam123V.array() *= N2Proj.array() / N12ProjSum.array();
                lam0V.array() += N1Proj.array() * lam0V1.array() / N12ProjSum.array();
                lam4V.array() += N1Proj.array() * lam4V1.array() / N12ProjSum.array();
                lam123V.array() += N1Proj.array() * lam123V1.array() / N12ProjSum.array();
            }
        }

#ifndef USE_ENTROPY_FIXED_LAMBDA_IN_SA
        lam123 = (std::abs(UL(1) / UL(0) - vg(0)) + std::abs(UR(1) / UR(0) - vg(0))) * 0.5; //! high fix
                                                                                            // lam123 = std::abs(UL(1) / UL(0) + UR(1) / UR(0)) * 0.5 - vg(0); //! low fix
#endif

        if constexpr (model == NS_SA || model == NS_SA_3D)
        {
            // real lambdaFaceCC = sqrt(std::abs(asqrMean)) + std::abs((UL(1) / UL(0) - vg(0)) + (UR(1) / UR(0) - vg(0))) * 0.5;
            Eigen::RowVector<real, -1> lambdaFaceCC = lam123V; //! using velo instead of velo + a
            if (settings.ransEigScheme == 1)
                lambdaFaceCC = lambdaFaceCC.array().max(lam0V.array()).max(lam4V.array());
            auto vnR = ((URxy(Seq123, Eigen::all).array().rowwise() / URxy(0, Eigen::all).array() - vgXY.array()) * unitNorm.array()).colwise().sum();
            auto vnL = ((ULxy(Seq123, Eigen::all).array().rowwise() / ULxy(0, Eigen::all).array() - vgXY.array()) * unitNorm.array()).colwise().sum();
            finc(I4 + 1, Eigen::all) =
                ((vnL * ULxy(I4 + 1, Eigen::all).array() + vnR * URxy(I4 + 1, Eigen::all).array()) -
                 (URxy(I4 + 1, Eigen::all).array() - ULxy(I4 + 1, Eigen::all).array()) * lambdaFaceCC.array()) *
                0.5;
        }
        if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
        {
            Eigen::RowVector<real, -1> lambdaFaceCC = lam123V; //! using velo instead of velo + a
            if (settings.ransEigScheme == 1)
                lambdaFaceCC = lambdaFaceCC.array().max(lam0V.array()).max(lam4V.array());
            auto vnR = ((URxy(Seq123, Eigen::all).array().rowwise() / URxy(0, Eigen::all).array() - vgXY.array()) * unitNorm.array()).colwise().sum();
            auto vnL = ((ULxy(Seq123, Eigen::all).array().rowwise() / ULxy(0, Eigen::all).array() - vgXY.array()) * unitNorm.array()).colwise().sum();
            finc(I4 + 1, Eigen::all) =
                ((vnL * ULxy(I4 + 1, Eigen::all).array() + vnR * URxy(I4 + 1, Eigen::all).array()) -
                 (URxy(I4 + 1, Eigen::all).array() - ULxy(I4 + 1, Eigen::all).array()) * lambdaFaceCC.array()) *
                0.5;
            finc(I4 + 2, Eigen::all) =
                ((vnL * ULxy(I4 + 2, Eigen::all).array() + vnR * URxy(I4 + 2, Eigen::all).array()) -
                 (URxy(I4 + 2, Eigen::all).array() - ULxy(I4 + 2, Eigen::all).array()) * lambdaFaceCC.array()) *
                0.5;
            finc(1, Eigen::all).array() += UMeanXy(I4 + 1, Eigen::all).array() * (2. / 3.); //! k's normal stress
            finc(I4, Eigen::all).array() += UMeanXy(I4 + 1, Eigen::all).array() * (2. / 3.) * UMeanXy(1, Eigen::all).array() / UMeanXy(0, Eigen::all).array();
        }

#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
        finc -= visFluxV;
#endif

        if (finc.hasNaN() || (!finc.allFinite()))
        {
            std::cout << finc.transpose() << std::endl;
            std::cout << ULxy.transpose() << std::endl;
            std::cout << URxy.transpose() << std::endl;
            std::cout << DiffUxy << std::endl;
            std::cout << unitNorm << std::endl;
            std::cout << btype << std::endl;
            DNDS_assert(false);
        }

        return -finc;
    }

    template <EulerModel model>
    typename EulerEvaluator<model>::TU EulerEvaluator<model>::source(
        const TU &UMeanXy,
        const TDiffU &DiffUxy,
        const Geom::tPoint &pPhy,
        TJacobianU &jacobian,
        index iCell,
        index ig,
        int Mode) // mode =0: source; mode = 1, diagJacobi; mode = 2,
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        TU ret;
        ret.resizeLike(UMeanXy);
        ret.setZero();
        real dWallC;
        if (ig < 0)
            dWallC = dWall[iCell].mean();
        else
            dWallC = dWall[iCell][ig];
        if (Mode == 2)
            jacobian.setZero(UMeanXy.size(), UMeanXy.size());
#ifdef DNDS_FV_EULEREVALUATOR_SOURCE_TERM_ZERO
        return ret;
#endif
        if (Mode == 0)
        {
            ret(Seq123) += settings.constMassForce(Seq012) * UMeanXy(0);
            ret(I4) += settings.constMassForce(Seq012).dot(UMeanXy(Seq123));
        }
        if (Mode == 2)
        {
            jacobian(I4, Seq123) -= settings.constMassForce(Seq012);
        }
#ifdef USE_ABS_VELO_IN_ROTATION
        if (settings.frameConstRotation.enabled)
        {
            if (Mode == 0 || Mode == 2)
                ret(Seq123) += -settings.frameConstRotation.vOmega().cross(Geom::ToThreeDim<dim>(UMeanXy(Seq123)))(Seq012);
            if (Mode == 2)
                jacobian(Seq123, Seq123) -= Geom::CrossVecToMat(-settings.frameConstRotation.vOmega())(Seq012, Seq012);
        }
#else
        if (settings.frameConstRotation.enabled)
        {
            Geom::tPoint radi = pPhy - settings.frameConstRotation.center;
            Geom::tPoint radiR = radi - settings.frameConstRotation.axis * (settings.frameConstRotation.axis.dot(radi));
            TVec mvolForce = (radiR * sqr(settings.frameConstRotation.Omega()) * UMeanXy(0))(Seq012);
            mvolForce += -2.0 * settings.frameConstRotation.vOmega().cross(Geom::ToThreeDim<dim>(UMeanXy(Seq123)))(Seq012);
            if (Mode == 0)
            {
                ret(Seq123) += mvolForce;
                ret(I4) += mvolForce.dot(UMeanXy(Seq123)) / UMeanXy(0);
            }
            if (Mode == 2)
            {
                TMat dmvolForceDrhov = Geom::CrossVecToMat(-2 * settings.frameConstRotation.vOmega())(Seq012, Seq012);
                jacobian(Seq123, Seq123) -= dmvolForceDrhov;
                jacobian(I4, Seq123) -= mvolForce + dmvolForceDrhov.transpose() * UMeanXy(Seq123) / UMeanXy(0);
                jacobian(I4, 0) -= -mvolForce.dot(UMeanXy(Seq123)) / sqr(UMeanXy(0));
            }
        }
#endif
        if constexpr (model == NS || model == NS_2D || model == NS_3D)
        {
        }
        else if constexpr (model == NS_SA || model == NS_SA_3D)
        {

            real pMean, asqrMean, Hmean;
            real gamma = settings.idealGasProperty.gamma;
            Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                                 gamma, pMean, asqrMean, Hmean);
            // ! refvalue:
            real muRef = settings.idealGasProperty.muGas;
            real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));

            real mufPhy, muf;
            mufPhy = muf = muEff(UMeanXy, T);

            real d = std::min(dWallC, std::pow(veryLargeReal, 1. / 6.));
            TU retInc;
            retInc.setZero(UMeanXy.size());
            auto sourceCaller = [&](int mode)
            {
                RANS::GetSource_SA<dim>(UMeanXy, DiffUxy, settings.idealGasProperty.muGas, muf,
                                        gamma, std::min(d, vfv->GetCellMaxLenScale(iCell) * settings.SADESScale), retInc,
                                        settings.ransSARotCorrection, mode);
            };

            if (Mode == 0)
            {
                sourceCaller(0);
            }
            else if (Mode == 1)
            {
                sourceCaller(1);
            }
            else if (Mode == 2)
            {
                sourceCaller(1);
                jacobian += retInc.asDiagonal(); //! TODO: make really block jacobian
            }
            ret += retInc;
        }
        else if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
        {
            real pMean, asqrMean, Hmean;
            real gamma = settings.idealGasProperty.gamma;
            Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                                 gamma, pMean, asqrMean, Hmean);
            // ! refvalue:
            real muRef = settings.idealGasProperty.muGas;
            real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));

            real mufPhy, muf;
            mufPhy = muf = muEff(UMeanXy, T);

            TU retInc;
            retInc.setZero(UMeanXy.size());

            TU UMeanXyFixed = UMeanXy;

            // if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            //     for (auto f : mesh->cell2face[iCell])
            //         if (pBCHandler->GetTypeFromID(mesh->GetFaceZone(f)) == BCWall)
            //         {
            //             real d1 = dWallC;
            //             real rhoOmegaaaWall = mufPhy / sqr(d1) * 800;
            //             UMeanXyFixed(I4 + 2) = rhoOmegaaaWall;
            //         }

            auto sourceCaller = [&](int mode)
            {
                if (settings.ransModel == RANSModel::RANS_KOSST)
                    RANS::GetSource_SST<dim>(UMeanXyFixed, DiffUxy, mufPhy, dWallC, retInc, mode);
                else if (settings.ransModel == RANSModel::RANS_KOWilcox)
                    RANS::GetSource_KOWilcox<dim>(UMeanXyFixed, DiffUxy, mufPhy, dWallC, retInc, mode);
                else if (settings.ransModel == RANSModel::RANS_RKE)
                    RANS::GetSource_RealizableKe<dim>(UMeanXyFixed, DiffUxy, mufPhy, dWallC, retInc, mode);
            };

            if (Mode == 0)
            {
                sourceCaller(0);
            }
            else if (Mode == 1)
            {
                sourceCaller(1);
            }
            else if (Mode == 2)
            {
                sourceCaller(1);
                jacobian += retInc.asDiagonal(); //! TODO: make really block jacobian
            }
            ret += retInc;
        }
        else
        {
            DNDS_assert(false);
        }
        // if (Mode == 1)
        //     std::cout << ret.transpose() << std::endl;
        return ret;
    }

    template <EulerModel model>
    typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBoundaryValue(
        TU &ULxy, //! warning, possible that UL is also modified
        const TU &ULMeanXy,
        index iCell, index iFace, int iG,
        const TVec &uNorm,
        const TMat &normBase,
        const Geom::tPoint &pPhysics,
        real t,
        Geom::t_index btype,
        bool fixUL,
        int geomMode)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_assert(iG >= -2);

        TU URxy;
        URxy.resizeLike(ULxy);
        auto bTypeEuler = pBCHandler->GetTypeFromID(btype);

        if (bTypeEuler == EulerBCType::BCSpecial ||
            bTypeEuler == EulerBCType::BCFar ||
            bTypeEuler == EulerBCType::BCOutP)
        {
            DNDS_assert(ULxy(0) > 0);
            if (btype == Geom::BC_ID_DEFAULT_FAR ||
                bTypeEuler == EulerBCType::BCFar ||
                bTypeEuler == EulerBCType::BCOutP)
            {
                TU far = btype >= Geom::BC_ID_DEFAULT_MAX
                             ? pBCHandler->GetValueFromID(btype)
                             : TU(settings.farFieldStaticValue);
                if (pCLDriver)
                    far(Seq123) = pCLDriver->GetAOARotation()(Seq012, Seq012) * far(Seq123);

                if (bTypeEuler == EulerBCType::BCFar)
                {
                    if (settings.frameConstRotation.enabled && pBCHandler->GetFlagFromID(btype, "frameOpt") != 0)
                        far(Seq123) = (Geom::RotateAxis(-settings.frameConstRotation.vOmega() * t) * Geom::ToThreeDim<dim>(far(Seq123)))(Seq012);
                    // std::cout << Geom::RotateAxis(settings.frameConstRotation.vOmega() * t) * Geom::RotateAxis(settings.frameConstRotation.vOmega() * t).transpose() << std::endl;
                    // DNDS_assert(false);
                }
                // fmt::print("far id: {}\n", btype);
                // std::cout << far.transpose() << std::endl;

                TU ULxyStatic = ULxy;
                if (settings.frameConstRotation.enabled) // to static frame velocity
                    TransformURotatingFrame(ULxyStatic, pPhysics, 1);

                real un = ULxy(Seq123).dot(uNorm) / ULxy(0); // using relative velo for in/out judgement
                real gamma = settings.idealGasProperty.gamma;
                real asqr, H, p;
                Gas::IdealGasThermal(ULxyStatic(I4), ULxyStatic(0), (ULxyStatic(Seq123) / ULxyStatic(0)).squaredNorm(), gamma, p, asqr, H);

                DNDS_assert(asqr >= 0);
                real a = std::sqrt(asqr);

                auto vg = this->GetFaceVGrid(iFace, iG, pPhysics);
                real vgN = vg.dot(uNorm);

                if (un - vgN - a > 0) // full outflow
                {
                    URxy = ULxyStatic;
                }
                else if (un - vgN > 0) //  1 sonic outflow, 1 sonic inflow, other outflow (subsonic out)
                {
                    TU farPrimitive, ULxyPrimitive;
                    farPrimitive.resizeLike(ULxyStatic);
                    ULxyPrimitive.resizeLike(URxy);
                    Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                    Gas::IdealGasThermalConservative2Primitive<dim>(ULxyStatic, ULxyPrimitive, gamma);
                    if (bTypeEuler == EulerBCType::BCOutP && pBCHandler->GetFlagFromID(btype, "anchorOpt") == 1)
                    {
                        {
                            TU anchorPointRel = ULxy;
                            if (anchorRecorders.count(btype)) // if doesn't have anchor value yet, use UL as anchor
                                anchorPointRel = anchorRecorders.at(btype).val;
                            TU anchorPointRelPrimitive = anchorPointRel;
                            Gas::IdealGasThermalConservative2Primitive<dim>(anchorPointRel, anchorPointRelPrimitive, gamma);
                            // rel has correct static pressure
                            // std::cout << "init Pressure " << farPrimitive(I4) << fmt::format("  UL {}, aP {}", ULxyPrimitive(I4), anchorPointRelPrimitive(I4)) << std::endl;
                            farPrimitive(I4) += std::max(ULxyPrimitive(I4) - anchorPointRelPrimitive(I4), -0.95 * farPrimitive(I4));
                            // std::cout << "anchored Pressure " << farPrimitive(I4) << std::endl;
                        }
                        // {
                        //     real pInc = 0;
                        //     if (settings.frameConstRotation.enabled && pBCHandler->GetValueExtraFromID(btype).size() >= 3)
                        //     {
                        //         real rRefSqr = settings.frameConstRotation.rVec(pBCHandler->GetValueExtraFromID(btype)({0, 1, 2})).squaredNorm();
                        //         real rCurSqr = settings.frameConstRotation.rVec(pPhysics).squaredNorm();
                        //         pInc = (rCurSqr - rRefSqr) * 0.5 * farPrimitive(0) * sqr(settings.frameConstRotation.Omega());
                        //         pInc = std::max(pInc, -0.95 * farPrimitive(I4));
                        //     }
                        //     farPrimitive(I4) += pInc;
                        // }
                    }
                    if (bTypeEuler == EulerBCType::BCOutP && pBCHandler->GetFlagFromID(btype, "anchorOpt") == 2)
                    {
                        real pInc = 0;
                        if (profileRecorders.count(btype))
                            pInc = profileRecorders.at(btype).GetPlain(settings.frameConstRotation.rVec(pPhysics).norm())(I4);
                        farPrimitive(I4) += std::max(pInc, -0.95 * farPrimitive(I4));
                    }
                    ULxyPrimitive(I4) = farPrimitive(I4); // using far pressure
                    Gas::IdealGasThermalPrimitive2Conservative<dim>(ULxyPrimitive, URxy, gamma);
                }
                else if (un - vgN + a > 0) //  1 sonic outflow, 1 sonic inflow, other inflow (subsonic in)
                {
                    TU farPrimitive, ULxyPrimitive;
                    farPrimitive.resizeLike(ULxyStatic);
                    ULxyPrimitive.resizeLike(URxy);
                    Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                    Gas::IdealGasThermalConservative2Primitive<dim>(ULxyStatic, ULxyPrimitive, gamma);
                    // farPrimitive(0) = ULxyPrimitive(0); // using inner density
                    farPrimitive(I4) = ULxyPrimitive(I4); // using inner pressure
                    Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma);
                }
                else // full inflow
                {
                    URxy = far;
                }
                if (settings.frameConstRotation.enabled) // to rotating frame velocity
                    TransformURotatingFrame(URxy, pPhysics, -1);
            }
            else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_DMR_FAR) // (no rotating)
            {
                DNDS_assert(dim > 1);
                URxy = settings.farFieldStaticValue;
                real uShock = 10;
                if constexpr (dim == 3) //* manual static dispatch
                {
                    if (((pPhysics(0) - uShock / std::sin(pi / 3) * t - 1. / 6.) -
                         pPhysics(1) / std::tan(pi / 3)) > 0)
                        URxy({0, 1, 2, 3, 4}) = Eigen::Vector<real, 5>{1.4, 0, 0, 0, 2.5};
                    else
                        URxy({0, 1, 2, 3, 4}) = Eigen::Vector<real, 5>{8, 57.157676649772960, -33, 0, 5.635e2};
                }
                else
                {
                    if (((pPhysics(0) - uShock / std::sin(pi / 3) * t - 1. / 6.) -
                         pPhysics(1) / std::tan(pi / 3)) > 0)
                        URxy({0, 1, 2, 3}) = Eigen::Vector<real, 4>{1.4, 0, 0, 2.5};
                    else
                        URxy({0, 1, 2, 3}) = Eigen::Vector<real, 4>{8, 57.157676649772960, -33, 5.635e2};
                }
            }
            else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_RT_FAR) // (no rotating)
            {
                DNDS_assert(dim > 1);
                Eigen::VectorXd far = settings.farFieldStaticValue;
                real gamma = settings.idealGasProperty.gamma;
                real un = ULxy(Seq123).dot(uNorm) / ULxy(0);
                real vsqr = (ULxy(Seq123) / ULxy(0)).squaredNorm();
                real asqr, H, p;
                Gas::IdealGasThermal(ULxy(I4), ULxy(0), vsqr, gamma, p, asqr, H);

                DNDS_assert(asqr >= 0);
                real a = std::sqrt(asqr);
                real v = -0.025 * a * cos(pPhysics(0) * 8 * pi);

                if (pPhysics(1) < 0.5)
                {

                    real rho = 2;
                    real p = 1;
                    far(0) = rho;
                    far(1) = 0;
                    far(2) = rho * v;
                    far(I4) = 0.5 * rho * sqr(v) + p / (gamma - 1);
                }
                else
                {
                    real rho = 1;
                    real p = 2.5;
                    far(0) = rho;
                    far(1) = 0;
                    far(2) = rho * v;
                    far(I4) = 0.5 * rho * sqr(v) + p / (gamma - 1);
                }

                if (un - a > 0) // full outflow
                {
                    URxy = ULxy;
                }
                else if (un > 0) //  1 sonic outflow, 1 sonic inflow, other outflow (subsonic out)
                {
                    TU farPrimitive, ULxyPrimitive;
                    farPrimitive.resizeLike(ULxy);
                    ULxyPrimitive.resizeLike(URxy);
                    Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                    Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                    ULxyPrimitive(I4) = farPrimitive(I4); // using far pressure
                    Gas::IdealGasThermalPrimitive2Conservative<dim>(ULxyPrimitive, URxy, gamma);
                }
                else if (un + a > 0) //  1 sonic outflow, 1 sonic inflow, other inflow (subsonic in)
                {
                    TU farPrimitive, ULxyPrimitive;
                    farPrimitive.resizeLike(ULxy);
                    ULxyPrimitive.resizeLike(URxy);
                    Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                    Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                    // farPrimitive(0) = ULxyPrimitive(0); // using inner density
                    farPrimitive(I4) = ULxyPrimitive(I4); // using inner pressure
                    Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma);
                }
                else // full inflow
                {
                    URxy = far;
                }
                // URxy = far; //! override
            }
            else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_IV_FAR) // (no rotating)
            {
                real chi = 5;
                real gamma = settings.idealGasProperty.gamma;
                real xc = 5 + t;
                real yc = 5 + t;
                real r = std::sqrt(sqr(pPhysics(0) - xc) + sqr(pPhysics(1) - yc));
                real dT = -(gamma - 1) / (8 * gamma * sqr(pi)) * sqr(chi) * std::exp(1 - sqr(r));
                real dux = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * -(pPhysics(1) - xc);
                real duy = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * +(pPhysics(0) - yc);
                real T = dT + 1;
                real ux = dux + 1;
                real uy = duy + 1;
                real S = 1;
                real rho = std::pow(T / S, 1 / (gamma - 1));
                real p = T * rho;

                real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + p / (gamma - 1);

                // std::cout << T << " " << rho << std::endl;
                URxy.setZero();
                URxy(0) = rho;
                URxy(1) = rho * ux;
                URxy(2) = rho * uy;
                URxy(dim + 1) = E;
            }
            else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR) // (no rotating)
            {
                real gamma = settings.idealGasProperty.gamma;
                real bdL = 0.0; // left
                real bdR = 1.0; // right
                real bdD = 0.0; // down
                real bdU = 1.0; // up

                real phi1 = -0.663324958071080;
                real phi2 = -0.422115882408869;
                real location = 0.8;
                real p1 = location + phi1 * t;
                real p2 = location + phi2 * t;
                real rho, u, v, pre;
                TU ULxyPrimitive;
                ULxyPrimitive.resizeLike(ULxy);

                Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                real rhoL = ULxyPrimitive(0);
                real uL = ULxyPrimitive(1);
                real vL = ULxyPrimitive(2);
                real preL = ULxyPrimitive(I4);
                TU farPrimitive = ULxyPrimitive;

                static const real bTol = 1e-9;
                if (std::abs(pPhysics(0) - bdL) < bTol)
                { // left, phi2
                    if (pPhysics(1) <= p2)
                    { // region 3
                        rho = 0.137992831541219;
                        u = 1.206045378311055;
                        v = 1.206045378311055;
                        pre = 0.029032258064516;
                    }
                    else
                    { // region 2
                        rho = 0.532258064516129;
                        u = 1.206045378311055;
                        v = 0.0;
                        pre = 0.3;
                    }
                }
                else if (std::abs(pPhysics(0) - bdR) < bTol)
                { // right, phi1
                    if (pPhysics(1) <= p1)
                    { // region 4
                        // rho = 0.532258064516129;
                        // u = 0.0;
                        // v = 1.206045378311055;
                        // pre = 0.3;
                        rho = rhoL;
                        u = -uL;
                        v = vL;
                        pre = preL;
                    }
                    else
                    { // region 1
                        // rho = 1.5;
                        // u = 0.0;
                        // v = 0.0;
                        // pre = 1.5;
                        rho = rhoL;
                        u = -uL;
                        v = vL;
                        pre = preL;
                    }
                }
                else if (std::abs(pPhysics(1) - bdU) < bTol)
                { // up, phi1
                    if (pPhysics(0) <= p1)
                    { // region 2
                        // rho = 0.532258064516129;
                        // u = 1.206045378311055;
                        // v = 0.0;
                        // pre = 0.3;
                        rho = rhoL;
                        u = uL;
                        v = -vL;
                        pre = preL;
                    }
                    else
                    { // region 1
                        // rho = 1.5;
                        // u = 0.0;
                        // v = 0.0;
                        // pre = 1.5;
                        rho = rhoL;
                        u = uL;
                        v = -vL;
                        pre = preL;
                    }
                }
                else if (std::abs(pPhysics(1) - bdD) < bTol)
                { // down, phi2
                    if (pPhysics(0) <= p2)
                    { // region 3
                        rho = 0.137992831541219;
                        u = 1.206045378311055;
                        v = 1.206045378311055;
                        pre = 0.029032258064516;
                    }
                    else
                    { // region 4
                        rho = 0.532258064516129;
                        u = 0.0;
                        v = 1.206045378311055;
                        pre = 0.3;
                    }
                }
                else
                {
                    rho = u = v = pre = std::nan("1");
                    DNDS_assert(false); // not valid boundary pos
                }
                farPrimitive(0) = rho;
                farPrimitive(1) = u, farPrimitive(2) = v;
                farPrimitive(I4) = pre;
                Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma);
            }
            else if (pBCHandler->GetFlagFromID(btype, "specialOpt") == 3001) // (no rotating)
            {
                // 3001 for Noh problem
                TU farPrimitive;
                Gas::IdealGasThermalConservative2Primitive<dim>(settings.farFieldStaticValue, farPrimitive, settings.idealGasProperty.gamma);
                real pInf = farPrimitive(I4);
                real r = pPhysics.norm();
                TVec velo = -pPhysics(Seq012) / (r + smallReal);
                real rho = sqr(1. + t / (r + smallReal));
                farPrimitive(0) = rho;
                farPrimitive(Seq123) = velo;
                farPrimitive(I4) = pInf; // warning: only valid for t < 0.768
                // std::cout << pPhysics.transpose() << ", " << t << ", " << pInf << ", " << rho << ", " << velo.transpose() << std::endl;
                Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, settings.idealGasProperty.gamma);
            }
            else
                DNDS_assert_info(false, fmt::format(
                                            "btype [{}] or bTypeEuler [{}] or specialOpt [{}] is not supported",
                                            btype, to_string(bTypeEuler), pBCHandler->GetFlagFromIDSoft(btype, "specialOpt")));
        }
        else if (bTypeEuler == EulerBCType::BCWallInvis ||
                 bTypeEuler == EulerBCType::BCSym) // (no rotating)
        {
            URxy = ULxy;
            if (settings.frameConstRotation.enabled)
                this->TransformURotatingFrame_ABS_VELO(URxy, pPhysics, -1);
            URxy(Seq123) -= 2 * URxy(Seq123).dot(uNorm) * uNorm; // mirrored!
            if (settings.frameConstRotation.enabled)
                this->TransformURotatingFrame_ABS_VELO(URxy, pPhysics, 1);
        }
        else if (bTypeEuler == EulerBCType::BCWall)
        {
            URxy = ULxy;
            if (geomMode == 0 || true) // now using only the physical mode
            {
                if (settings.frameConstRotation.enabled && pBCHandler->GetFlagFromID(btype, "frameOpt") == 0)
                    this->TransformURotatingFrame_ABS_VELO(URxy, pPhysics, -1);
                if (settings.frameConstRotation.enabled && pBCHandler->GetFlagFromID(btype, "frameOpt") != 0)
                    this->TransformURotatingFrame(URxy, pPhysics, 1);
                URxy(Seq123) *= -1;
                if (settings.frameConstRotation.enabled && pBCHandler->GetFlagFromID(btype, "frameOpt") == 0)
                    this->TransformURotatingFrame_ABS_VELO(URxy, pPhysics, 1);
                if (settings.frameConstRotation.enabled && pBCHandler->GetFlagFromID(btype, "frameOpt") != 0)
                    this->TransformURotatingFrame(URxy, pPhysics, -1);
            }
            else
            {
                URxy(Seq123) *= -1;
#ifdef USE_ABS_VELO_IN_ROTATION
                if (settings.frameConstRotation.enabled && pBCHandler->GetFlagFromID(btype, "frameOpt") == 0)
                    this->TransformVelocityRotatingFrame(URxy, pPhysics, 2);
#else
                if (settings.frameConstRotation.enabled && pBCHandler->GetFlagFromID(btype, "frameOpt") != 0)
                    this->TransformVelocityRotatingFrame(URxy, pPhysics, -2);
#endif
            }
            if (model == NS_SA || model == NS_SA_3D)
            {
                URxy(I4 + 1) *= -1;
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                if (fixUL)
                    ULxy(I4 + 1) = URxy(I4 + 1) = 0; //! modifing UL
#endif
            }
            if (model == NS_2EQ || model == NS_2EQ_3D)
            {
                URxy({I4 + 1, I4 + 2}) *= -1;
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                // if (fixUL)
                //     ULxy({I4 + 1, I4 + 2}).setZero(), URxy({I4 + 1, I4 + 2}).setZero(); //! modifing UL
#endif
                if (settings.ransModel == RANSModel::RANS_RKE)
                { // BC for RealizableKe
                    TVec v = (vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1) - vfv->GetCellBary(iCell))(Seq012);
                    real d1 = dWall[iCell].mean();
                    real k1 = ULMeanXy(I4 + 1) / ULMeanXy(0);

                    real pMean, asqrMean, Hmean;
                    real gamma = settings.idealGasProperty.gamma;
                    Gas::IdealGasThermal(ULMeanXy(I4), ULMeanXy(0), (ULMeanXy(Seq123) / ULMeanXy(0)).squaredNorm(),
                                         gamma, pMean, asqrMean, Hmean);
                    // ! refvalue:
                    real muRef = settings.idealGasProperty.muGas;
                    real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * ULMeanXy(0));
                    real mufPhy1;
                    mufPhy1 = muEff(ULMeanXy, T);
                    real epsWall = 2 * mufPhy1 / ULMeanXy(0) * k1 / sqr(d1);
                    URxy(I4 + 2) = 2 * epsWall * ULxy(0) - ULxy(I4 + 2);
                    // if (fixUL)
                    //     ULxy(I4 + 2) = URxy(I4 + 2) = epsWall * ULxy(0);
                }
                if (settings.ransModel == RANSModel::RANS_KOSST ||
                    settings.ransModel == RANSModel::RANS_KOWilcox)
                { // BC for SST or KOWilcox
                    real d1 = dWall[iCell].mean();
                    // real d1 = dWall[iCell].minCoeff();
                    real pMean, asqrMean, Hmean;
                    real gamma = settings.idealGasProperty.gamma;
                    Gas::IdealGasThermal(ULMeanXy(I4), ULMeanXy(0), (ULMeanXy(Seq123) / ULMeanXy(0)).squaredNorm(),
                                         gamma, pMean, asqrMean, Hmean);
                    // ! refvalue:
                    real muRef = settings.idealGasProperty.muGas;
                    real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * ULMeanXy(0));
                    real mufPhy1 = muEff(ULMeanXy, T);

                    real rhoOmegaaaWall = mufPhy1 / sqr(d1) * 800;
                    URxy(I4 + 2) = 2 * rhoOmegaaaWall - ULxy(I4 + 2);
                    // if (fixUL)
                    //     ULxy(I4 + 2) = URxy(I4 + 2) = rhoOmegaaaWall;
                }
            }
        }
        else if (bTypeEuler == EulerBCType::BCOut)
        {
            URxy = ULxy;
        }
        else if (bTypeEuler == EulerBCType::BCIn)
        {
            URxy = pBCHandler->GetValueFromID(btype);
            if (pCLDriver)
                URxy(Seq123) = pCLDriver->GetAOARotation()(Seq012, Seq012) * URxy(Seq123);
            if (bTypeEuler == EulerBCType::BCFar)
            {
                if (settings.frameConstRotation.enabled && pBCHandler->GetFlagFromID(btype, "frameOpt") != 0)
                    URxy(Seq123) = (Geom::RotateAxis(-settings.frameConstRotation.vOmega() * t) * Geom::ToThreeDim<dim>(URxy(Seq123)))(Seq012);
            }
            if (settings.frameConstRotation.enabled)
                TransformURotatingFrame(URxy, pPhysics, -1);
        }
        else if (bTypeEuler == EulerBCType::BCInPsTs)
        {
            real rvNorm = ULxy(Seq123).dot(uNorm(Seq012));
            TU ULxyStatic = ULxy;
            if (settings.frameConstRotation.enabled)
                TransformURotatingFrame(ULxyStatic, pPhysics, 1);
            TU ULxyPrimitive;
            ULxyPrimitive.resizeLike(ULxy);
            real gamma = settings.idealGasProperty.gamma;
            Gas::IdealGasThermalConservative2Primitive<dim>(ULxyStatic, ULxyPrimitive, gamma);
            TVec v = ULxyStatic(Seq123).array() / ULxyStatic(0);
            real vSqr = v.squaredNorm();
            {
                TU farPrimitive = pBCHandler->GetValueFromID(btype); // primitive passive scalar components like Nu

                real pStag = pBCHandler->GetValueFromID(btype)(0);
                real tStag = pBCHandler->GetValueFromID(btype)(1);
                vSqr = std::min(vSqr, tStag * 2 * settings.idealGasProperty.CpGas * 0.95); // incase kinetic energy exceeds internal
                real tStatic = tStag - 0.5 * vSqr / (settings.idealGasProperty.CpGas);
                real gamma = settings.idealGasProperty.gamma;
                real pStatic = pStag * std::pow(tStatic / tStag, gamma / (gamma - 1));
                real rStatic = pStatic / (settings.idealGasProperty.Rgas * tStatic);
                farPrimitive(0) = rStatic;
                // farPrimitive(Seq123) = -uNorm * std::sqrt(vSqr);
                farPrimitive(Seq123) = pBCHandler->GetValueFromID(btype)(Seq234).normalized() * std::sqrt(vSqr);
                farPrimitive(I4) = pStatic;
                Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma);
            }
            if (pCLDriver)
                URxy(Seq123) = pCLDriver->GetAOARotation()(Seq012, Seq012) * URxy(Seq123);
            if (settings.frameConstRotation.enabled)
                TransformURotatingFrame(URxy, pPhysics, -1);
        }
        else
        {
            DNDS_assert(false);
        }
        return URxy;
    }

    template <EulerModel model>
    void EulerEvaluator<model>::InitializeOutputPicker(OutputPicker &op, OutputOverlapDataRefs dataRefs)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS

        auto &eval = *this;
        auto &u = dataRefs.u;
        auto &uRec = dataRefs.uRec;
        auto &betaPP = dataRefs.betaPP;
        auto &alphaPP = dataRefs.alphaPP;

        OutputPicker::tMap outMap;
        // outMap["R"] = [&](index iCell)
        // { return u[iCell](0); };
        outMap["RU"] = [&](index iCell)
        { return u[iCell](1); };
        outMap["RV"] = [&](index iCell)
        { return u[iCell](2); };
        outMap["RV"] = [&](index iCell)
        { return u[iCell](I4 - 1); };
        outMap["RE"] = [&](index iCell)
        { return u[iCell](I4); };
        outMap["R_REC_1"] = [&](index iCell)
        { return uRec[iCell](0, 0); };
        outMap["RU_REC_1"] = [&](index iCell)
        { return uRec[iCell](1, 0); }; // TODO: to be continued to...

        // pps:
        outMap["betaPP"] = [&](index iCell)
        { return betaPP[iCell](0); };
        outMap["alphaPP"] = [&](index iCell)
        { return alphaPP[iCell](0); };
        outMap["ACond"] = [&](index iCell)
        {
            auto AI = vfv->GetCellRecMatAInv(iCell);
            Eigen::MatrixXd AIInv = AI;
            real aCond = HardEigen::EigenLeastSquareInverse(AI, AIInv);
            return aCond;
        };
        outMap["dWall"] = [&](index iCell)
        {
            return eval.dWall.at(iCell).mean();
        };
        outMap["minJacobiDetRel"] = [&](index iCell)
        {
            auto eCell = mesh->GetCellElement(iCell);
            auto qCell = vfv->GetCellQuad(iCell);
            real minDetJac = veryLargeReal;
            for (int iG = 0; iG < qCell.GetNumPoints(); iG++)
                minDetJac = std::min(vfv->GetCellJacobiDet(iCell, iG), minDetJac);
            return minDetJac * Geom::Elem::ParamSpaceVol(eCell.GetParamSpace()) / vfv->GetCellVol(iCell);
        };
        outMap["cellVolume"] = [&](index iCell)
        {
            return vfv->GetCellVol(iCell);
        };
        outMap["mut"] = [&](index iCell)
        {
            real mut = 0;
            if (model == NS_2EQ || model == NS_2EQ_3D)
            {
                TU Uxy = u[iCell];
                TDiffU GradU;
                GradU.resize(Eigen::NoChange, nVars);
                GradU.setZero();
                if constexpr (gDim == 2)
                    GradU({0, 1}, Eigen::all) =
                        vfv->GetIntPointDiffBaseValue(iCell, -1, -1, -1, std::array<int, 2>{1, 2}, 3) *
                        uRec[iCell]; // 2d specific
                else
                    GradU({0, 1, 2}, Eigen::all) =
                        vfv->GetIntPointDiffBaseValue(iCell, -1, -1, -1, std::array<int, 3>{1, 2, 3}, 4) *
                        uRec[iCell]; // 3d specific
                real pMean, asqrMean, Hmean;
                real gamma = settings.idealGasProperty.gamma;
                auto ULMeanXy = Uxy;
                Gas::IdealGasThermal(ULMeanXy(I4), ULMeanXy(0), (ULMeanXy(Seq123) / ULMeanXy(0)).squaredNorm(),
                                     gamma, pMean, asqrMean, Hmean);
                // ! refvalue:
                real muRef = settings.idealGasProperty.muGas;
                real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * ULMeanXy(0));
                real mufPhy = muEff(ULMeanXy, T);
                if (settings.ransModel == RANSModel::RANS_KOSST)
                    mut = RANS::GetMut_SST<dim>(Uxy, GradU, mufPhy, dWall[iCell].mean());
                else if (settings.ransModel == RANSModel::RANS_KOWilcox)
                    mut = RANS::GetMut_KOWilcox<dim>(Uxy, GradU, mufPhy, dWall[iCell].mean());
                else if (settings.ransModel == RANSModel::RANS_RKE)
                    mut = RANS::GetMut_RealizableKe<dim>(Uxy, GradU, mufPhy, dWall[iCell].mean());
            }

            return mut;
        };

        op.setMap(outMap);
    }
}