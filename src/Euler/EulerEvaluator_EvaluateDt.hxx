/** @file EulerEvaluator_EvaluateDt.hxx
 *  @brief Template implementations of EulerEvaluator methods for local time-step estimation,
 *         wall distance computation (CGAL AABB and Poisson-based), positivity-preserving
 *         limiters, face eigenvalue estimation, numerical flux evaluation, source terms,
 *         boundary value generation, and output variable registration.
 */
#pragma once
#include "EulerEvaluator.hpp"
#include "RANS_ke.hpp"
#include "Solver/Linear.hpp"

#define CGAL_DISABLE_ROUNDING_MATH_CHECK // for valgrind
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#undef CGAL_DISABLE_ROUNDING_MATH_CHECK

namespace DNDS::Euler
{

    static const auto model = NS_SA;
    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Collect wall boundary face triangulations for AABB-based wall distance computation.
     *
     *  Iterates over wall and isothermal-wall boundary faces, decomposes them into triangles
     *  (handling Line, Tri, and Quad element types), and appends to the output vector.
     *  When useQuadPatches is true, uses quadrature-point-based sub-patches instead of
     *  vertex-based triangulation.
     *
     *  @param useQuadPatches  If true, create triangle patches from quadrature points.
     *  @param trianglesOut    Collected triangles as 3x3 matrices (columns = vertices) (output).
     */
    void EulerEvaluator<model>::GetWallDist_CollectTriangles(
        bool useQuadPatches,
        std::vector<Eigen::Matrix<real, 3, 3>> &trianglesOut)
    {
        for (index iBnd = 0; iBnd < mesh->NumBnd(); iBnd++)
        {
            if (pBCHandler->GetTypeFromID(mesh->GetBndZone(iBnd)) == EulerBCType::BCWall ||
                pBCHandler->GetTypeFromID(mesh->GetBndZone(iBnd)) == EulerBCType::BCWallIsothermal)
            {
                index iFace = mesh->bnd2faceV[iBnd];
                auto elem = mesh->GetFaceElement(iFace);
                auto quad = vfv->GetFaceQuad(iFace);
                if (!useQuadPatches)
                {
                    if (elem.type == Geom::Elem::ElemType::Line2 || elem.type == Geom::Elem::ElemType::Line3)
                    {
                        Geom::tSmallCoords coords;
                        mesh->GetCoordsOnFace(iFace, coords);
                        Eigen::Matrix<real, 3, 3> tri;
                        tri(EigenAll, 0) = coords(EigenAll, 0);
                        tri(EigenAll, 1) = coords(EigenAll, 1);
                        tri(EigenAll, 2) = coords(EigenAll, 1) + Geom::tPoint{0., 0., vfv->GetFaceArea(iFace)};
                        trianglesOut.push_back(tri);
                    }
                    else if (elem.type == Geom::Elem::ElemType::Tri3 || elem.type == Geom::Elem::ElemType::Tri6)
                    {
                        Geom::tSmallCoords coords;
                        mesh->GetCoordsOnFace(iFace, coords);
                        Eigen::Matrix<real, 3, 3> tri;
                        tri(EigenAll, 0) = coords(EigenAll, 0);
                        tri(EigenAll, 1) = coords(EigenAll, 1);
                        tri(EigenAll, 2) = coords(EigenAll, 2);
                        trianglesOut.push_back(tri);
                    }
                    else if (elem.type == Geom::Elem::ElemType::Quad4 || elem.type == Geom::Elem::ElemType::Quad9)
                    {
                        Geom::tSmallCoords coords;
                        mesh->GetCoordsOnFace(iFace, coords);
                        Eigen::Matrix<real, 3, 3> tri;
                        tri(EigenAll, 0) = coords(EigenAll, 0);
                        tri(EigenAll, 1) = coords(EigenAll, 1);
                        tri(EigenAll, 2) = coords(EigenAll, 2);
                        trianglesOut.push_back(tri);
                        tri(EigenAll, 0) = coords(EigenAll, 0);
                        tri(EigenAll, 1) = coords(EigenAll, 2);
                        tri(EigenAll, 2) = coords(EigenAll, 3);
                        trianglesOut.push_back(tri);
                    }
                    else
                    {
                        DNDS_assert_info(false, "This elem not implemented yet for GetWallDist()");
                    }
                }
                else
                {
                    auto qPatches = Geom::Elem::GetQuadPatches(quad);
                    for (auto &qPatch : qPatches)
                    {
                        Eigen::Matrix<real, 3, 3> tri;
                        Geom::tSmallCoords coords;
                        mesh->GetCoordsOnFace(iFace, coords);
                        for (int iV = 0; iV < 3; iV++)
                            if (qPatch[iV] > 0)
                                tri(EigenAll, iV) = coords(EigenAll, qPatch[iV] - 1);
                            else if (qPatch[iV] < 0)
                                tri(EigenAll, iV) = vfv->GetFaceQuadraturePPhys(iFace, -qPatch[iV] - 1);
                            else
                                tri(EigenAll, iV) = coords(EigenAll, 1) + Geom::tPoint{0., 0., vfv->GetFaceArea(iFace)};
                        trianglesOut.push_back(tri);
                    }
                }
            }
        }
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Compute face-averaged wall distances from cell wall distances.
     *
     *  For each face, averages the wall distance of its two adjacent cells (or uses
     *  the single owner cell for boundary faces) and stores in dWallFace.
     */
    void EulerEvaluator<model>::GetWallDist_ComputeFaceDistances()
    {
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

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Compute geometric wall distance using CGAL AABB tree (all-to-all broadcast).
     *
     *  Gathers all wall triangles globally, builds a CGAL AABB tree, and queries the
     *  nearest distance for each cell's quadrature points. Uses execution serialization
     *  (wallDistExection) to limit concurrent memory usage. Clamps result to minWallDist.
     */
    void EulerEvaluator<model>::GetWallDist_AABB()
    {
        using TriArray = ArrayEigenMatrix<3, 3>;
        ssp<TriArray> TrianglesLocal, TrianglesFull;
        DNDS_MAKE_SSP(TrianglesLocal, mesh->getMPI());
        DNDS_MAKE_SSP(TrianglesFull, mesh->getMPI());

        bool useQuadPatches = (settings.wallDistScheme == 1);
        std::vector<Eigen::Matrix<real, 3, 3>> Triangles;
        GetWallDist_CollectTriangles(useQuadPatches, Triangles);

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
            typedef K::Point_3 Point;
            typedef K::Triangle_3 Triangle;
            typedef std::vector<Triangle>::iterator Iterator;
            typedef CGAL::AABB_triangle_primitive_3<K, Iterator> Primitive;
            typedef CGAL::AABB_traits_3<K, Primitive> AABB_triangle_traits;
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
                Tree tree(triangles.begin(), triangles.end());

                for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
                {
                    auto quadCell = vfv->GetCellQuad(iCell);
                    dWall[iCell].resize(quadCell.GetNumPoints());
                    for (int ig = 0; ig < quadCell.GetNumPoints(); ig++)
                    {
                        auto p = vfv->GetCellQuadraturePPhys(iCell, ig);
                        Point pQ(p[0], p[1], p[2]);
                        FT sqd = tree.squared_distance(pQ);
                        dWall[iCell][ig] = std::sqrt(sqd);
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

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Compute geometric wall distance using batched CGAL AABB tree queries.
     *
     *  Builds a local AABB tree from wall triangles (using quadrature patches), then
     *  performs batched distributed queries where cells are processed in configurable
     *  load-sized chunks. Supports refinement mode (wallDistScheme==20) for selective
     *  re-querying of cells below wallDistRefineMax.
     */
    void EulerEvaluator<model>::GetWallDist_BatchedAABB()
    {
        typedef CGAL::Simple_cartesian<double> K;
        typedef K::FT FT;
        typedef K::Point_3 Point;
        typedef K::Triangle_3 Triangle;
        typedef std::vector<Triangle>::iterator Iterator;
        typedef CGAL::AABB_triangle_primitive_3<K, Iterator> Primitive;
        typedef CGAL::AABB_traits_3<K, Primitive> AABB_triangle_traits;
        typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

        // Build local AABB tree from wall boundary triangles using quad patches.
        std::vector<Triangle> triangles;
        triangles.reserve(mesh->NumBnd() * 8 + 8);
        {
            std::vector<Eigen::Matrix<real, 3, 3>> triEigen;
            GetWallDist_CollectTriangles(/*useQuadPatches=*/true, triEigen);
            for (auto &tri : triEigen)
            {
                Point p0(tri(0, 0), tri(1, 0), tri(2, 0));
                Point p1(tri(0, 1), tri(1, 1), tri(2, 1));
                Point p2(tri(0, 2), tri(1, 2), tri(2, 2));
                triangles.push_back(Triangle(p0, p1, p2));
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

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Compute wall distance by solving a Poisson equation with pseudo-time stepping.
     *
     *  Solves a p-Poisson equation (grad^2 phi = -1) with Dirichlet BC (phi=0) on
     *  wall boundaries and Neumann BC elsewhere, using Jacobi or GMRES iterations.
     *  The wall distance is recovered from the solution as d = |grad(phi)| + sqrt(|grad(phi)|^2 + 2*phi).
     *  Configurable via settings: wallDistNJacobiSweep, wallDistIter, wallDistPoissonP, etc.
     */
    void EulerEvaluator<model>::GetWallDist_Poisson()
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
                        if (faceBCType == BCWall || faceBCType == BCWallIsothermal)
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
                        if (faceBCType == BCWall || faceBCType == BCWallIsothermal)
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

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Dispatcher for wall distance computation, selecting method based on settings.
     *
     *  Routes to GetWallDist_AABB (schemes 0,1,20), GetWallDist_BatchedAABB (schemes 2,20),
     *  or GetWallDist_Poisson (scheme 3), then computes face-averaged distances.
     */
    void EulerEvaluator<model>::GetWallDist()
    {
        if (settings.wallDistScheme == 0 || settings.wallDistScheme == 1 || settings.wallDistScheme == 20)
            GetWallDist_AABB();
        if (settings.wallDistScheme == 2 || settings.wallDistScheme == 20)
            GetWallDist_BatchedAABB();
        if (settings.wallDistScheme == 3)
            GetWallDist_Poisson();
        GetWallDist_ComputeFaceDistances();
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
    /** @brief Estimate the local time step and face spectral radii for CFL-based time stepping.
     *
     *  For each face, evaluates L/R states using 2nd-order reconstruction, computes the
     *  convective and viscous eigenvalue estimates, and stores face spectral radii
     *  (lambdaFace, lambdaFace0..4, lambdaFaceC, lambdaFaceVis). Then for each cell,
     *  sums the face spectral radii to obtain a CFL-limited local time step, optionally
     *  clamped by MaxDt or made uniform (global minimum).
     *
     *  @param dt          Local time step per cell (output).
     *  @param u           Conservative variable DOF array.
     *  @param uRec        Reconstruction coefficients.
     *  @param CFL         CFL number for time step scaling.
     *  @param dtMinall    Global minimum time step (output, MPI-reduced).
     *  @param MaxDt       Maximum allowed time step (upper clamp).
     *  @param UseLocaldt  If true, use local time stepping; otherwise set all cells to dtMinall.
     *  @param t           Current simulation time.
     *  @param flags       Bitfield flags (e.g., DT_Dont_update_lambda01234).
     */
    void EulerEvaluator<model>::EvaluateDt(
        ArrayDOFV<1> &dt,
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        real CFL, real &dtMinall, real MaxDt,
        bool UseLocaldt,
        real t,
        uint64_t flags)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "EvaluateDt 1");

        bool dont_update_lambda01234 = flags & DT_Dont_update_lambda01234;

        { // 2nd order reconstruction
            typename TVFV::template TFBoundary<nVarsFixed>
                FBoundary = [&](const TU &UL, const TU &UMean, index iCell, index iFace, int ig,
                                const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
            {
                TVec normOutV = normOut(Seq012);
                Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(normOutV);
                bool compressed = false;
                TU ULfixed = this->CompressRecPart(
                    UMean,
                    UL - UMean,
                    compressed);
                return this->generateBoundaryValue(ULfixed, UMean, iCell, iFace, ig, normOutV, normBase, pPhy, t, bType, true, 1);
            };
            vfv->DoReconstruction2ndGrad(uGradBufNoLim, u, FBoundary, settings.direct2ndRecMethod);
            uGradBufNoLim.trans.startPersistentPull();
            uGradBufNoLim.trans.waitPersistentPull();
        }
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
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
            real T_L = phys_.template temperature<dim>(UL);
            phys_.template conservativeThermal<dim>(UL, T_L, pL, asqrL, HL);
            pR = pL, HR = HL, asqrR = asqrL;
            if (f2c[1] != UnInitIndex)
            {
                TU UR = u[f2c[1]];
                this->UFromCell2Face(UR, iFace, f2c[1], 1);
                uMean = (uMean + UR) * 0.5;
                vR = UR(Seq123) / UR(0);
                real T_R = phys_.template temperature<dim>(UR);
                phys_.template conservativeThermal<dim>(UR, T_R, pR, asqrR, HR);
            }
            TDiffU GradULxy, GradURxy;
            GradULxy.resize(Eigen::NoChange, nVars);
            GradURxy.resize(Eigen::NoChange, nVars);
            GradULxy.setZero(), GradURxy.setZero();
            GradULxy(SeqG012, EigenAll) = uGradBuf[f2c[0]];
            // if constexpr (gDim == 2)
            //     GradULxy({0, 1}, EigenAll) =
            //         vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, -1, std::array<int, 2>{1, 2}, 3) *
            //         uRec[f2c[0]]; // 2d here
            // else
            //     GradULxy({0, 1, 2}, EigenAll) =
            //         vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, -1, std::array<int, 3>{1, 2, 3}, 4) *
            //         uRec[f2c[0]]; // 3d here
            this->DiffUFromCell2Face(GradULxy, iFace, f2c[0], 0);
            GradURxy = GradULxy;
            if (f2c[1] != UnInitIndex)
            {
                GradURxy(SeqG012, EigenAll) = uGradBuf[f2c[1]];
                // if constexpr (gDim == 2)
                //     GradURxy({0, 1}, EigenAll) =
                //         vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, -1, std::array<int, 2>{1, 2}, 3) *
                //         uRec[f2c[1]]; // 2d here
                // else
                //     GradURxy({0, 1, 2}, EigenAll) =
                //         vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, -1, std::array<int, 3>{1, 2, 3}, 4) *
                //         uRec[f2c[1]]; // 3d here
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
            real T = phys_.template temperature<dim>(uMean);
            real gammaEq = 0;
            real gamma = 0;
            phys_.template conservativeThermal<dim>(uMean, T, pMean, asqrMean, HMean, &gammaEq, &gamma);

            pMean = (pL + pR) * 0.5;
            real aMean = sqrt(gamma * pMean / uMean(0)); // paper

            // DNDS_assert(asqrMean >= 0);
            // real aMean = std::sqrt(asqrMean); // original
            real lambdaConvection = std::abs(veloNMean - vgN) + aMean;
            DNDS_assert_info(
                asqrL >= 0 && asqrR >= 0,
                fmt::format(" mean value violates PP! asqr: [{} {}]", asqrL, asqrR));
            real aL = std::sqrt(asqrL);
            real aR = std::sqrt(asqrR);
            lambdaConvection = std::max(aL + std::abs(veloNL - vgN), aR + std::abs(veloNR - vgN));

            // ! refvalue:
            real muRef = phys_.muRef();

            real muf = muEff(uMean, T);
            real muPhy = muf;
            real muTurb = this->getMuTur(uMean, GradUMeanXY, muRef, muf, iFace);
            muf += muTurb;

            real lamVis = muf / uMean(0) *
                          std::max(4. / 3., gamma / phys_.Pr());

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
            if (!dont_update_lambda01234)
                lambdaFace0[iFace] = lambdaFace123[iFace] = lambdaFace4[iFace] = lambdaConvection;

            // if (f2c[0] == 10756)
            // {
            //     std::cout << "----Lambdas" << std::setprecision(16) << iFace << std::endl;
            //     std::cout << lambdaConvection << std::endl;
            //     std::cout << lambdaFaceVis[iFace] << std::endl;
            //     std::cout << veloNMean << " " << aMean << std::endl;
            //     std::cout << gamma << " " << pMean << " " << uMean(0) << std::endl;
            // }

            // put to cell part to be OMP compliant
            // lambdaCell[iCellL] += lambdaFace[iFace] * vfv->GetFaceArea(iFace);
            // if (f2c[1] != UnInitIndex) // can't be non local
            //     lambdaCell[f2c[1]] += lambdaFace[iFace] * vfv->GetFaceArea(iFace);

            deltaLambdaFace[iFace] = std::max<real>({std::abs((vR - vL).dot(unitNorm) + aR - aL), std::abs((vR - vL).dot(unitNorm) - aR + aL), std::abs(aL - aR)});
        }
        real dtMin = veryLargeReal;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime) reduction(min : dtMin)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            real lambdaCellC = 0;
            for (auto iFace : mesh->cell2face[iCell])
                lambdaCellC += lambdaFace[iFace] * vfv->GetFaceArea(iFace);
            // std::cout << fv->GetCellVol(iCell) << " " << (lambdaCellC) << " " << CFL << std::endl;
            // exit(0);
            dt[iCell](0) = std::min(CFL * vfv->GetCellVol(iCell) * vfv->GetCellSmoothScaleRatio(iCell) / (lambdaCellC + 1e-100), MaxDt);
            dtMin = std::min(dtMin, dt[iCell](0));
            deltaLambdaCell[iCell](0) = 0;
            for (auto iFace : mesh->cell2face[iCell])
                deltaLambdaCell[iCell](0) = std::max(deltaLambdaCell[iCell](0), deltaLambdaFace[iFace]);
            // if (iCell == 10756)
            // {
            //     std::cout << std::endl;
            // }
        }

        deltaLambdaCell.trans.startPersistentPull();
        MPI::Allreduce(&dtMin, &dtMinall, 1, DNDS_MPI_REAL, MPI_MIN, u.father->getMPI().comm);
        deltaLambdaCell.trans.waitPersistentPull();

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
    /** @brief Evaluate the numerical flux at a single face for a batch of quadrature points.
     *
     *  Computes the inviscid and viscous numerical flux at a face given batched L/R states,
     *  gradients, and face geometry. Dispatches to the configured Riemann solver type.
     *  Outputs left/right viscous corrections (FLfix, FRfix), the total flux increment (finc),
     *  and per-quadrature-point eigenvalue estimates (lam0V, lam123V, lam4V).
     *
     *  @param ULxy        Batched left states at quadrature points.
     *  @param URxy        Batched right states at quadrature points.
     *  @param ULMeanXy    Left cell mean state.
     *  @param URMeanXy    Right cell mean state.
     *  @param DiffUxy     Batched conservative gradient at face.
     *  @param DiffUxyPrim Batched primitive gradient at face.
     *  @param unitNorm    Batched outward unit normals at quadrature points.
     *  @param vgXY        Batched grid velocity at quadrature points.
     *  @param unitNormC   Face-center unit normal.
     *  @param vgC         Face-center grid velocity.
     *  @param FLfix       Left viscous correction (output).
     *  @param FRfix       Right viscous correction (output).
     *  @param finc        Total flux increment (output).
     *  @param lam0V       Eigenvalue estimate for wave 0 (output).
     *  @param lam123V     Eigenvalue estimate for waves 1-3 (output).
     *  @param lam4V       Eigenvalue estimate for wave 4 (output).
     *  @param btype       Boundary type index of the face.
     *  @param rsType      Riemann solver type to use.
     *  @param iFace       Face index.
     *  @param ignoreVis   If true, skip viscous flux computation.
     */
    void EulerEvaluator<model>::fluxFace(
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
        TU_Batch &finc,
        TReal_Batch &lam0V, TReal_Batch &lam123V, TReal_Batch &lam4V,
        Geom::t_index btype,
        typename Gas::RiemannSolverType rsType,
        index iFace, bool ignoreVis)
    {
        finc.resizeLike(ULxy);
        DNDS_assert(&DiffUxy != &DiffUxyPrim);
        DNDS_assert(&ULMeanXy != &URMeanXy);
        DNDS_assert(&DiffUxy != &DiffUxyPrim);
        DNDS_assert(&unitNorm != &vgXY);
        DNDS_assert(&unitNormC != &vgC);
        DNDS_assert(&FLfix != &FRfix);
        DNDS_assert(&lam0V != &lam123V);
        DNDS_assert(&lam123V != &lam4V); // aliasing check not complete here

        auto fluxFace_impl =
            [&](
                const TU_Batch *DNDS_RESTRICT p_ULxy,
                const TU_Batch *DNDS_RESTRICT p_URxy,
                const TU *DNDS_RESTRICT p_ULMeanXy,
                const TU *DNDS_RESTRICT p_URMeanXy,
                const TDiffU_Batch *DNDS_RESTRICT p_DiffUxy,
                const TDiffU_Batch *DNDS_RESTRICT p_DiffUxyPrim,
                const TVec_Batch *DNDS_RESTRICT p_unitNorm,
                const TVec_Batch *DNDS_RESTRICT p_vgXY,
                const TVec *DNDS_RESTRICT p_unitNormC,
                const TVec *DNDS_RESTRICT p_vgC,
                TU_Batch *DNDS_RESTRICT p_FLfix,
                TU_Batch *DNDS_RESTRICT p_FRfix,
                TReal_Batch *DNDS_RESTRICT p_lam0V, TReal_Batch *DNDS_RESTRICT p_lam123V, TReal_Batch *DNDS_RESTRICT p_lam4V,
                Geom::t_index btype,
                typename Gas::RiemannSolverType rsType)
        // clang-format off
        {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        const TU_Batch &ULxy = *p_ULxy;
        const TU_Batch &URxy = *p_URxy;
        const TU &ULMeanXy = *p_ULMeanXy;
        const TU &URMeanXy = *p_URMeanXy;
        const TDiffU_Batch &DiffUxy = *p_DiffUxy;
        const TDiffU_Batch &DiffUxyPrim = *p_DiffUxyPrim;
        const TVec_Batch &unitNorm = *p_unitNorm;
        const TVec_Batch &vgXY = *p_vgXY;
        const TVec &unitNormC = *p_unitNormC;
        const TVec &vgC = *p_vgC;
        TU_Batch &FLfix = *p_FLfix;
        TU_Batch &FRfix = *p_FRfix;
        TReal_Batch &lam0V = *p_lam0V;
        TReal_Batch &lam123V = *p_lam123V;
        TReal_Batch &lam4V = *p_lam4V;

        int nB = ULxy.cols();

        TU_Batch UMeanXy = 0.5 * (ULxy + URxy);

        // PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterB);
        real muRef = phys_.muRef();
        real TRef = phys_.TRef();

        auto f2c = mesh->face2cell[iFace];
        real dLambda = deltaLambdaCell[f2c[0]](0);
        if (f2c[1] != UnInitIndex)
            dLambda = std::max(dLambda, deltaLambdaCell[f2c[1]](0));
        real fixScale = settings.rsFixScale;
        real incFScale = settings.rsIncFScale;
        /** viscous flux **/
        TU_Batch visFluxV;
        visFluxV.resizeLike(ULxy);
        for (int iB = 0; iB < nB; iB++)
        {
            real pMean, asqrMean, Hmean;
            TU UMeanXYC = UMeanXy(EigenAll, iB);
            auto seqC = Eigen::seq(iB * dim, iB * dim + dim - 1);
            TDiffU DiffUxyC = DiffUxy(seqC, EigenAll);
            TDiffU DiffUxyPrimC = DiffUxyPrim(seqC, EigenAll);
            TVec uNormC = unitNorm(EigenAll, iB);
            real T = phys_.template temperature<dim>(UMeanXYC);
            real gammaEq = 0;
            real gamma = 0;
            phys_.template conservativeThermal<dim>(UMeanXYC, T, pMean, asqrMean, Hmean, &gammaEq, &gamma);
            DNDS_assert_info(pMean > 0, fmt::format("{}, {}, {}", UMeanXYC(I4), UMeanXYC(0), (UMeanXYC(Seq123) / UMeanXYC(0)).squaredNorm()));
            real mufPhy, muf;
            muf = muEff(UMeanXYC, T);
            mufPhy = muf;
            // PerformanceTimer::Instance().StopTimer(PerformanceTimer::LimiterB);
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
            if (!ignoreVis)
            {
                real muTurb = this->getMuTur(UMeanXYC, DiffUxyC, muRef, muf, iFace); // TODO: make this accept primitive gradients instead
                muf += muTurb;

                real k = phys_.Cp(T, UMeanXYC) * muTurb / 0.9 +
                            phys_.Cp(T, UMeanXYC) * mufPhy / phys_.Pr();
                TU VisFlux;
                VisFlux.resizeLike(ULMeanXy);
                VisFlux.setZero();
                Gas::ViscousFlux_IdealGas<dim>(
                    UMeanXYC, DiffUxyPrimC, uNormC, pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWall,
                    gammaEq, gamma,
                    muf, muTurb / (muf + verySmallReal), settings.ransUseQCR,
                    k,
                    phys_.Cp(T, UMeanXYC),
                    VisFlux,
                    phys_.mixtureFormationRhoE(UMeanXYC));

                // Fickian species diffusion + enthalpy transport (reactive flow)
                if constexpr (Traits::isExtended)
                {
                    if (phys_.hasChemicalSource())
                    {
                        int Ns = phys_.nSpecies();
                        int Ns1 = Ns - 1;
                        int nV = static_cast<int>(UMeanXYC.size());
                        int Isp = nV - Ns1;

                        // Per-species total enthalpies h_k/U0² (sensible + formation + p/ρ, no KE)
                        std::vector<double> hBuf(Ns);
                        Chemistry::SpeciesBufferView hv{hBuf.data(), Ns};
                        phys_.speciesEnthalpies(T, pMean, UMeanXYC, hv);
                        real hLast_code = hBuf[Ns - 1];

                        real rhoFace = UMeanXYC(0);
                        for (int kk = 0; kk < Ns1; ++kk)
                        {
                            real Dk = phys_.speciesDiffusivityK(T, pMean, UMeanXYC, kk);
                            // DiffUxyPrimC columns beyond I4 are ∇Y_k (from GradientCons2Prim)
                            real gradYk_dot_n = DiffUxyPrimC(Seq012, Isp + kk).dot(uNormC);
                            // Outward species diffusion flux: J_k·n = -ρ D_k (∇Y_k·n)
                            real Jk_n = -rhoFace * Dk * gradYk_dot_n;
                            VisFlux(Isp + kk) += Jk_n;
                            // Energy enthalpy transport: Σ h_k·J_k = Σ_{k<Ns1} (h_k − h_N)·J_k
                            VisFlux(I4) += (hBuf[kk] - hLast_code) * Jk_n;
                        }
                    }
                }

                this->visFluxTurVariable(UMeanXYC, DiffUxyPrimC, muRef, mufPhy, muTurb, uNormC, iFace, VisFlux);
                if (pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWallInvis ||
                    pBCHandler->GetTypeFromID(btype) == EulerBCType::BCSym)
                {
                    VisFlux *= 0.0;
                }
                visFluxV(EigenAll, iB) = VisFlux;
            }
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
            std::cout << "ULM" << ULMeanXy.transpose() << '\n';
            std::cout << "URM" << URMeanXy.transpose() << std::endl;
        };

        if (pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWallInvis ||
            pBCHandler->GetTypeFromID(btype) == EulerBCType::BCSym)
        {
            // ? normal invert here?
        }

        lam0V.resize(nB);
        lam123V.resize(nB);
        lam4V.resize(nB);

        real T_Lm_rs = phys_.template temperature<dim>(ULMeanXy);
        real gammaEqLm_rs = phys_.template gammaEq<dim>(T_Lm_rs, ULMeanXy);
        real gammaLm_rs = phys_.gamma(T_Lm_rs, ULMeanXy);
        real T_Rm_rs = phys_.template temperature<dim>(URMeanXy);
        real gammaEqRm_rs = phys_.template gammaEq<dim>(T_Rm_rs, URMeanXy);
        real gammaRm_rs = phys_.gamma(T_Rm_rs, URMeanXy);
        real gammaEq_rs = gammaEqLm_rs;
        real gamma_rs = gammaLm_rs;

        TU gammaEqLV, gammaEqRV, gammaLV, gammaRV, rhoHLV, rhoHRV;
        if (phys_.hasChemicalSource())
        {
            gammaEqLV.resize(nB);
            gammaEqRV.resize(nB);
            gammaLV.resize(nB);
            gammaRV.resize(nB);
            rhoHLV.resize(nB);
            rhoHRV.resize(nB);
            for (int iB = 0; iB < nB; ++iB)
            {
                TU uL = ULxy(EigenAll, iB);
                TU uR = URxy(EigenAll, iB);
                real TL = phys_.template temperature<dim>(uL);
                real TR = phys_.template temperature<dim>(uR);
                gammaEqLV(iB) = phys_.template gammaEq<dim>(TL, uL);
                gammaEqRV(iB) = phys_.template gammaEq<dim>(TR, uR);
                gammaLV(iB) = phys_.gamma(TL, uL);
                gammaRV(iB) = phys_.gamma(TR, uR);
                rhoHLV(iB) = phys_.mixtureFormationRhoE(uL);
                rhoHRV(iB) = phys_.mixtureFormationRhoE(uR);
            }
        }

        auto RSWrapper_XY =
            [&](Gas::RiemannSolverType rsType,
                auto &&UL, auto &&UR, auto &&ULm, auto &&URm, auto &&vg, auto &&n,
                real gammaEq, real gamma, auto &&finc, real dLambda, real fixScale, real incFScale,
                real &lam0, real &lam123, real &lam4)
        {
            if (phys_.hasChemicalSource())
            {
                real TL = phys_.template temperature<dim>(UL);
                real TR = phys_.template temperature<dim>(UR);
                real gammaEqL = phys_.template gammaEq<dim>(TL, UL);
                real gammaEqR = phys_.template gammaEq<dim>(TR, UR);
                real gammaL = phys_.gamma(TL, UL);
                real gammaR = phys_.gamma(TR, UR);
                Gas::InviscidFlux_IdealGas_Dispatcher<dim, true>(rsType, UL, UR, ULm, URm, vg, n, gammaEq, gamma, finc, dLambda, fixScale, incFScale, exitFun, lam0, lam123, lam4,
                    phys_.mixtureFormationRhoE(UL), phys_.mixtureFormationRhoE(UR),
                    phys_.mixtureFormationRhoE(ULm), phys_.mixtureFormationRhoE(URm),
                    gammaEqL, gammaEqR, gammaEqLm_rs, gammaEqRm_rs,
                    gammaL, gammaR, gammaLm_rs, gammaRm_rs);
            }
            else
                Gas::InviscidFlux_IdealGas_Dispatcher<dim>(rsType, UL, UR, ULm, URm, vg, n, gammaEq, gamma, finc, dLambda, fixScale, incFScale, exitFun, lam0, lam123, lam4,
                    phys_.mixtureFormationRhoE(UL), phys_.mixtureFormationRhoE(UR),
                    phys_.mixtureFormationRhoE(ULm), phys_.mixtureFormationRhoE(URm));
        };

        // TU_Batch finc1;
        // finc1.resizeLike(ULxy);
        if (settings.rsTypeWall != Gas::UnknownRS &&
            (pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWall ||
             pBCHandler->GetTypeFromID(btype) == BCWallIsothermal))
        {
            rsType = settings.rsTypeWall;
        }

        auto runRsOnNorm = [&]()
        {
            if (settings.rsMeanValueEig != 0 &&
                (rsType >= Gas::Roe_M1 && rsType <= Gas::Roe_M5))
            {
                real lam0{0}, lam123{0}, lam4{0};
                if (phys_.hasChemicalSource())
                    Gas::InviscidFlux_IdealGas_Batch_Dispatcher<dim, true, true>(
                        rsType,
                        ULxy, URxy, ULMeanXy, URMeanXy, vgXY, vgC, unitNorm, unitNormC,
                        gammaEq_rs, gamma_rs, finc, dLambda, fixScale, incFScale,
                        exitFun, lam0, lam123, lam4,
                        phys_.mixtureFormationRhoE(ULMeanXy), phys_.mixtureFormationRhoE(URMeanXy),
                        rhoHLV, rhoHRV, gammaEqLV, gammaEqRV, gammaEqLm_rs, gammaEqRm_rs,
                        gammaLV, gammaRV, gammaLm_rs, gammaRm_rs);
                else
                    Gas::InviscidFlux_IdealGas_Batch_Dispatcher<dim>(
                        rsType,
                        ULxy, URxy, ULMeanXy, URMeanXy, vgXY, vgC, unitNorm, unitNormC,
                        gammaEq_rs, gamma_rs, finc, dLambda, fixScale, incFScale,
                        exitFun, lam0, lam123, lam4,
                        phys_.mixtureFormationRhoE(ULMeanXy), phys_.mixtureFormationRhoE(URMeanXy),
                        nullptr, nullptr, nullptr, nullptr);
                lam0V.setConstant(lam0);
                lam123V.setConstant(lam123);
                lam4V.setConstant(lam4);
            }
            else
                for (int iB = 0; iB < nB; iB++)
                {
                    RSWrapper_XY(rsType, ULxy(EigenAll, iB), URxy(EigenAll, iB), ULMeanXy, URMeanXy,
                                 vgXY(EigenAll, iB), unitNorm(EigenAll, iB),
                                 gammaEq_rs, gamma_rs, finc(EigenAll, iB), dLambda, fixScale, incFScale,
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
                TVec N1;
                if (settings.rsRotateScheme == 1)
                    N1 = diffVelo / diffVeloN;
                else if (settings.rsRotateScheme == 2)
                    N1 = unitNormC;
                else
                    DNDS_assert(false);
                DNDS_assert_info(std::abs(N1.norm() - 1) < 1e-5, fmt::format("{}", diffVeloN));
                N1 *= sign(N1.dot(unitNormC));
                TReal_Batch N1Proj = N1.transpose() * unitNorm;

                TVec_Batch N2 = unitNorm - N1 * N1Proj;
                TReal_Batch N2Proj = N2.colwise().norm().array().max(verySmallReal * 10);
                N2.array().rowwise() /= N2Proj.array();

                real N1ProjC = N1.dot(unitNormC);
                TVec N2C = unitNormC - N1 * N1ProjC;
                real N2CProj = std::max(N2C.norm(), verySmallReal * 10);
                N2C /= N2CProj;

                TVec_Batch N1B;
                N1B.resizeLike(N2);
                N1B.colwise() = N1;

                if (false)
                {
                    std::cout << unitNorm << "\n";
                    std::cout << "N1" << "\n";
                    std::cout << N1.transpose() << "\n";
                    std::cout << N1B.transpose() << "\n";
                    std::cout << N1Proj << "\n";
                    std::cout << "N2" << "\n";
                    std::cout << N2.transpose() << "\n";
                    std::cout << N2Proj << "\n";
                    std::cout << N2C.transpose() << "\n";
                    std::cout << N2CProj << "\n";
                    std::cout << std::endl;
                }

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
                    if (phys_.hasChemicalSource())
                        Gas::InviscidFlux_IdealGas_Batch_Dispatcher<dim, true, true>(
                            rsTypeAux,
                            ULxy, URxy, ULMeanXy, URMeanXy, vgXY, vgC, N1B, N1,
                            gammaEq_rs, gamma_rs, F1, dLambda, fixScale, incFScale,
                            exitFun, lam0, lam123, lam4,
                            phys_.mixtureFormationRhoE(ULMeanXy), phys_.mixtureFormationRhoE(URMeanXy),
                            rhoHLV, rhoHRV, gammaEqLV, gammaEqRV, gammaEqLm_rs, gammaEqRm_rs,
                            gammaLV, gammaRV, gammaLm_rs, gammaRm_rs);
                    else
                        Gas::InviscidFlux_IdealGas_Batch_Dispatcher<dim>(
                            rsTypeAux,
                            ULxy, URxy, ULMeanXy, URMeanXy, vgXY, vgC, N1B, N1,
                            gammaEq_rs, gamma_rs, F1, dLambda, fixScale, incFScale,
                            exitFun, lam0, lam123, lam4,
                            phys_.mixtureFormationRhoE(ULMeanXy), phys_.mixtureFormationRhoE(URMeanXy),
                            nullptr, nullptr, nullptr, nullptr);
                    lam0V1.setConstant(lam0);
                    lam123V1.setConstant(lam123);
                    lam4V1.setConstant(lam4);
                }
                else
                    for (int iB = 0; iB < nB; iB++)
                    {
                        RSWrapper_XY(rsTypeAux, ULxy(EigenAll, iB), URxy(EigenAll, iB), ULMeanXy, URMeanXy,
                                     vgXY(EigenAll, iB), N1,
                                     gammaEq_rs, gamma_rs, F1(EigenAll, iB), dLambda, fixScale, incFScale,
                                     lam0V1(iB), lam123V1(iB), lam4V1(iB));
                    }

                if (settings.rsMeanValueEig != 0 &&
                    (rsType >= Gas::Roe_M1 && rsType <= Gas::Roe_M5))
                {
                    real lam0{0}, lam123{0}, lam4{0};
                    if (phys_.hasChemicalSource())
                        Gas::InviscidFlux_IdealGas_Batch_Dispatcher<dim, true, true>(
                            rsType,
                            ULxy, URxy, ULMeanXy, URMeanXy, vgXY, vgC, N2, N2C,
                            gammaEq_rs, gamma_rs, finc, dLambda, fixScale, incFScale,
                            exitFun, lam0, lam123, lam4,
                            phys_.mixtureFormationRhoE(ULMeanXy), phys_.mixtureFormationRhoE(URMeanXy),
                            rhoHLV, rhoHRV, gammaEqLV, gammaEqRV, gammaEqLm_rs, gammaEqRm_rs,
                            gammaLV, gammaRV, gammaLm_rs, gammaRm_rs);
                    else
                        Gas::InviscidFlux_IdealGas_Batch_Dispatcher<dim>(
                            rsType,
                            ULxy, URxy, ULMeanXy, URMeanXy, vgXY, vgC, N2, N2C,
                            gammaEq_rs, gamma_rs, finc, dLambda, fixScale, incFScale,
                            exitFun, lam0, lam123, lam4,
                            phys_.mixtureFormationRhoE(ULMeanXy), phys_.mixtureFormationRhoE(URMeanXy),
                            nullptr, nullptr, nullptr, nullptr);
                    lam0V.setConstant(lam0);
                    lam123V.setConstant(lam123);
                    lam4V.setConstant(lam4);
                }
                else
                    for (int iB = 0; iB < nB; iB++)
                    {
                        RSWrapper_XY(rsType, ULxy(EigenAll, iB), URxy(EigenAll, iB), ULMeanXy, URMeanXy,
                                     vgXY(EigenAll, iB), N2(EigenAll, iB),
                                     gammaEq_rs, gamma_rs, finc(EigenAll, iB), dLambda, fixScale, incFScale,
                                     lam0V(iB), lam123V(iB), lam4V(iB));
                    }

                finc.array().rowwise() *= N2Proj.array();
                finc.array() += F1.array().rowwise() * N1Proj.array();

                // Recover acoustic speed a from either direction (isotropic).
                // a = max(|lam0 - lam123|, |lam4 - lam123|) — exact for all sign combos:
                //   subsonic: lam0 = a∓un, lam4 = a±un → |lam4−lam123|=a, |lam0−lam123|=a
                //   supersonic: lam0 = |un|−a, lam4 = |un|+a → both diffs = a
                TReal_Batch aV;
                aV.resizeLike(lam0V);
                aV = (lam0V1 - lam123V1).array().abs().max((lam4V1 - lam123V1).array().abs());
                // Guard: if N1 direction gives zero (static fluid), recover a from N2.
                aV = aV.array().max((lam0V - lam123V).array().abs().max((lam4V - lam123V).array().abs()));

                // N1 and N2 are orthogonal → combined normal-velo magnitude:
                //   un = sqrt(un_n1² + un_n2²) = sqrt(lam123V1² + lam123V²)
                TReal_Batch unV;
                unV.resizeLike(lam0V);
                unV = (lam123V1.array().pow(2) + lam123V.array().pow(2)).sqrt();

                // Reconstruct face-combined eigenvalues.
                lam0V = (unV - aV).array().abs();   // |un − a| (vanishes at sonic point)
                lam123V = unV;                       // |un|
                lam4V = unV + aV;                    // un + a ≥ 0 always
            }
        }

#ifndef USE_ENTROPY_FIXED_LAMBDA_IN_SA
        lam123 = (std::abs(UL(1) / UL(0) - vg(0)) + std::abs(UR(1) / UR(0) - vg(0))) * 0.5; //! high fix
                                                                                            // lam123 = std::abs(UL(1) / UL(0) + UR(1) / UR(0)) * 0.5 - vg(0); //! low fix
#endif

        if constexpr (Traits::hasSA)
        {
            // real lambdaFaceCC = sqrt(std::abs(asqrMean)) + std::abs((UL(1) / UL(0) - vg(0)) + (UR(1) / UR(0) - vg(0))) * 0.5;
            Eigen::RowVector<real, -1> lambdaFaceCC = lam123V; //! using velo instead of velo + a
            if (settings.ransEigScheme == 1)
                lambdaFaceCC = lambdaFaceCC.array().max(lam0V.array()).max(lam4V.array());
            auto vnR = ((URxy(Seq123, EigenAll).array().rowwise() / URxy(0, EigenAll).array() - vgXY.array()) * unitNorm.array()).colwise().sum();
            auto vnL = ((ULxy(Seq123, EigenAll).array().rowwise() / ULxy(0, EigenAll).array() - vgXY.array()) * unitNorm.array()).colwise().sum();
            finc(I4 + 1, EigenAll) =
                ((vnL * ULxy(I4 + 1, EigenAll).array() + vnR * URxy(I4 + 1, EigenAll).array()) -
                 (URxy(I4 + 1, EigenAll).array() - ULxy(I4 + 1, EigenAll).array()) * lambdaFaceCC.array()) *
                0.5;
        }
        if constexpr (Traits::has2EQ)
        {
            Eigen::RowVector<real, -1> lambdaFaceCC = lam123V; //! using velo instead of velo + a
            if (settings.ransEigScheme == 1)
                lambdaFaceCC = lambdaFaceCC.array().max(lam0V.array()).max(lam4V.array());
            auto vnR = ((URxy(Seq123, EigenAll).array().rowwise() / URxy(0, EigenAll).array() - vgXY.array()) * unitNorm.array()).colwise().sum();
            auto vnL = ((ULxy(Seq123, EigenAll).array().rowwise() / ULxy(0, EigenAll).array() - vgXY.array()) * unitNorm.array()).colwise().sum();
            finc(I4 + 1, EigenAll) =
                ((vnL * ULxy(I4 + 1, EigenAll).array() + vnR * URxy(I4 + 1, EigenAll).array()) -
                 (URxy(I4 + 1, EigenAll).array() - ULxy(I4 + 1, EigenAll).array()) * lambdaFaceCC.array()) *
                0.5;
            finc(I4 + 2, EigenAll) =
                ((vnL * ULxy(I4 + 2, EigenAll).array() + vnR * URxy(I4 + 2, EigenAll).array()) -
                 (URxy(I4 + 2, EigenAll).array() - ULxy(I4 + 2, EigenAll).array()) * lambdaFaceCC.array()) *
                0.5;
            finc(1, EigenAll).array() += UMeanXy(I4 + 1, EigenAll).array() * (2. / 3.); //! k's normal stress
            finc(I4, EigenAll).array() += UMeanXy(I4 + 1, EigenAll).array() * (2. / 3.) * UMeanXy(1, EigenAll).array() / UMeanXy(0, EigenAll).array();
        }
        if constexpr (Traits::isExtended)
        {
            // real lambdaFaceCC = sqrt(std::abs(asqrMean)) + std::abs((UL(1) / UL(0) - vg(0)) + (UR(1) / UR(0) - vg(0))) * 0.5;
            Eigen::RowVector<real, -1> lambdaFaceCC = lam123V; //! using velo instead of velo + a
            if (settings.ransEigScheme == 1)
                lambdaFaceCC = lambdaFaceCC.array().max(lam0V.array()).max(lam4V.array());
            auto vnR = ((URxy(Seq123, EigenAll).array().rowwise() / URxy(0, EigenAll).array() - vgXY.array()) * unitNorm.array()).colwise().sum();
            auto vnL = ((ULxy(Seq123, EigenAll).array().rowwise() / ULxy(0, EigenAll).array() - vgXY.array()) * unitNorm.array()).colwise().sum();
            finc(SeqI52Last, EigenAll) =
                ((ULxy(SeqI52Last, EigenAll).array().rowwise() * vnL.array() +
                  URxy(SeqI52Last, EigenAll).array().rowwise() * vnR.array()) -
                 (URxy(SeqI52Last, EigenAll).array() - ULxy(SeqI52Last, EigenAll).array()).rowwise() * lambdaFaceCC.array()) *
                0.5;
        }

#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
        if (!ignoreVis)
            finc -= visFluxV;
#endif

        if (finc.hasNaN() || (!finc.allFinite()))
        {
            std::cout << "finc\n"
                      << finc.transpose() << "\n";
            std::cout << "visFluxV\n"
                      << visFluxV << "\n";
            std::cout << "lam0V\n"
                      << lam0V << "\n";
            std::cout << "lam123V\n"
                      << lam0V << "\n";
            std::cout << "lam4V\n"
                      << lam0V << "\n";
            std::cout << "ULxy\n"
                      << ULxy.transpose() << "\n";
            std::cout << "URxy\n"
                      << URxy.transpose() << "\n";
            std::cout << "UMeanXy\n"
                      << UMeanXy.transpose() << "\n";
            std::cout << "DiffUxy\n"
                      << DiffUxy << "\n";
            std::cout << "DiffUxyPrim\n"
                      << DiffUxyPrim << "\n";
            std::cout << "unitNorm\n"
                      << unitNorm << "\n";
            std::cout << "btype\n"
                      << btype << std::endl;
            DNDS_assert(false);
        }

        finc *= -1.0;
        };
        // clang-format on
        return fluxFace_impl(
            &ULxy,
            &URxy,
            &ULMeanXy,
            &URMeanXy,
            &DiffUxy,
            &DiffUxyPrim,
            &unitNorm,
            &vgXY,
            &unitNormC,
            &vgC,
            &FLfix,
            &FRfix,
            &lam0V, &lam123V, &lam4V,
            btype,
            rsType);
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Compute source terms for a cell at a quadrature point.
     *
     *  Evaluates volume source terms including: constant mass force, rotating-frame
     *  fictitious forces (Coriolis and centrifugal), axisymmetric source terms,
     *  SA turbulence model source, and k-omega/k-epsilon RANS model sources.
     *
     *  @param UMeanXy   Cell mean conservative state.
     *  @param DiffUxy   Gradient of conservative variables at the quadrature point.
     *  @param pPhy      Physical coordinates of the quadrature point.
     *  @param jacobian  Source-term Jacobian matrix (output when Mode==2).
     *  @param iCell     Cell index.
     *  @param ig        Quadrature point index within the cell (-1 for cell center).
     *  @param Mode      0: return source vector; 1: return diagonal Jacobian; 2: return full Jacobian.
     *  @return Source term vector.
     */
    typename EulerEvaluator<model>::TU EulerEvaluator<model>::source(
        const TU &UMeanXy,
        const TDiffU &DiffUxy,
        const Geom::tPoint &pPhy,
        TJacobianU &jacobian,
        index iCell,
        index ig,
        int Mode,
        SourceFilter filter) // mode =0: source; mode = 1, diagJacobi; mode = 2,
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

        // === Extended models (NS_EX / NS_EX_3D): variant-based contributor dispatch ===
        if constexpr (Traits::isExtended)
        {
            SourceCellAux aux;
            aux.dWallC = dWallC;
            aux.hMax = vfv->GetCellMaxLenScale(iCell);

            real pMean, asqrMean, Hmean;
            real T = phys_.template temperature<dim>(UMeanXy);
            real gammaEq = 0;
            real gamma = 0;
            phys_.template conservativeThermal<dim>(UMeanXy, T, pMean, asqrMean, Hmean, &gammaEq, &gamma);

            aux.T = T;
            aux.gammaEq = gammaEq;
            aux.p = pMean;
            aux.pPhys = phys_.toPhysP(pMean);
            aux.muf = muEff(UMeanXy, aux.T);
            aux.rhoH_form = phys_.mixtureFormationRhoE(UMeanXy);

            SourceTermVisitor<model> visitor{ret, jacobian, UMeanXy, DiffUxy, pPhy, aux,
                                             iCell, ig, Mode, filter};
            for (auto &c : sourceContributors)
                std::visit(visitor, c);
            return ret;
        }

        // === Fixed-size models: existing if-constexpr path (unchanged) ===
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
        if (axisSymmetric)
        {
            TU uPrim;
            uPrim.resizeLike(UMeanXy);
            real T_axi = phys_.template temperature<dim>(UMeanXy);
            real gamma_axi = phys_.template gammaEq<dim>(T_axi, UMeanXy);
            Gas::IdealGasThermalConservative2Primitive(UMeanXy, uPrim, gamma_axi,
                                                       phys_.mixtureFormationRhoE(UMeanXy));
            if (Mode == 0)
                ret(2) += uPrim(I4) / std::max(verySmallReal, pPhy(1)); // y direction force
            if (Mode == 1)
                ; // not implementing axisSymmetric jacobian addition
            if (Mode == 2)
                ; // not implementing axisSymmetric jacobian addition
        }
        if constexpr (Traits::isPlainNS)
        {
        }
        else if constexpr (Traits::hasSA)
        {

            real pMean, asqrMean, Hmean;
            real T = phys_.template temperature<dim>(UMeanXy);
            real gammaEq = 0;
            real gamma = 0;
            phys_.template conservativeThermal<dim>(UMeanXy, T, pMean, asqrMean, Hmean, &gammaEq, &gamma);
            // ! refvalue:
            real muRef = phys_.muRef();

            real mufPhy, muf;
            mufPhy = muf = muEff(UMeanXy, T);

            real d = std::min(dWallC, std::pow(veryLargeReal, 1. / 6.));
            TU retInc;
            retInc.setZero(UMeanXy.size());

            //! DES mesh lengths should be cached!
            real hMax = vfv->GetCellMaxLenScale(iCell);
            real cWall = settings.SADESScale > 100.0 ? 1.0 : 0.15;
            real lLES = hMax * settings.SADESScale;
            //! missing hWallNormal!
            lLES = std::min(lLES, std::max({d * cWall, hMax * cWall}));
            auto sourceCaller = [&](int mode)
            {
                RANS::GetSource_SA<dim>(UMeanXy, DiffUxy, phys_.muRef(), muf,
                                        gamma,
                                        d, lLES, hMax, settings.SADESMode,
                                        retInc,
                                        settings.ransSARotCorrection, mode,
                                        settings.SAVersion);
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
        else if constexpr (Traits::has2EQ)
        {
            real pMean, asqrMean, Hmean;
            real T = phys_.template temperature<dim>(UMeanXy);
            real gammaEq = 0;
            real gamma = 0;
            phys_.template conservativeThermal<dim>(UMeanXy, T, pMean, asqrMean, Hmean, &gammaEq, &gamma);
            // ! refvalue:
            real muRef = phys_.muRef();

            real mufPhy, muf;
            mufPhy = muf = muEff(UMeanXy, T);

            TU retInc;
            retInc.setZero(UMeanXy.size());

            TU UMeanXyFixed = UMeanXy;

            // if constexpr (Traits::has2EQ)
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
                    RANS::GetSource_SST<dim>(UMeanXyFixed, DiffUxy, mufPhy, dWallC, vfv->GetCellMaxLenScale(iCell) * settings.SADESScale, retInc, mode);
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

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, )
    void EulerEvaluator<model>::EvaluateCellSource(
        TU &cellRHS,
        TJacobianU &cellJac,
        const TU &uCell,
        const TDiffU &cellGradU,
        index iCell,
        int jacMode,
        SourceFilter filter,
        real cellAlpha,
        bool useRecArrays,
        ArrayDOFV<nVarsFixed> *pU,
        ArrayRECV<nVarsFixed> *pURecUnlim,
        ArrayRECV<nVarsFixed> *pURec,
        bool direct2ndRec,
        real t)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        int cnvars = nVars;

        auto gCell = (useRecArrays && !direct2ndRec)
                         ? vfv->GetCellQuad(iCell)
                         : vfv->GetCellQuadO1(iCell);

        // Construct 2nd-order cell gradient if needed (source2nd / ransSource2nd)
        TDiffU cellGrad2nd;
        if (useRecArrays && (settings.ransSource2nd || settings.source2nd))
        {
            DNDS_assert(pU);
            cellGrad2nd.setZero(Eigen::NoChange, uCell.size());
            TU uC = uCell;
            for (index iFace : mesh->cell2face[iCell])
            {
                index iCellOther = mesh->CellFaceOther(iCell, iFace);
                TVec uNorm = vfv->GetFaceNormFromCell(iFace, iCell, -1, -1)(Seq012) *
                             (mesh->CellIsFaceBack(iCell, iFace) ? 1 : -1);
                TU uR;
                if (iCellOther != UnInitIndex)
                    uR = (*pU)[iCellOther], this->UFromOtherCell(uR, iFace, iCell, iCellOther, -1);
                else
                    uR = generateBoundaryValue(
                        uC, uC, iCell, iFace, -1,
                        uNorm, Geom::NormBuildLocalBaseV<dim>(uNorm),
                        vfv->GetFaceQuadraturePPhys(iFace, -1),
                        t, mesh->GetFaceZone(iFace), true, 0);
                cellGrad2nd += vfv->GetFaceArea(iFace) * uNorm * (uR - uC).transpose() * 0.5;
            }
            cellGrad2nd /= vfv->GetCellVol(iCell);
        }

        TDiffU cellGradFix;
        if (useRecArrays && settings.useSourceGradFixGG)
            cellGradFix = gradUFix[iCell] / vfv->GetCellVol(iCell);

        Eigen::Matrix<real, nVarsFixed, Eigen::Dynamic> sourceV;
        sourceV.setZero(cnvars, (jacMode == 2) ? cnvars + 1 : 2);

        Geom::Elem::SummationNoOp noOp;

        gCell.IntegrationSimple(
            sourceV,
            [&](decltype(sourceV) &finc, int iG)
            {
                int iGQ = (useRecArrays && !direct2ndRec) ? iG : -1;

                TDiffU GradU;
                GradU.resize(Eigen::NoChange, cnvars);
                GradU.setZero();

                if (!useRecArrays)
                {
                    // Simple mode: use caller-provided gradient
                    GradU = cellGradU;
                }
                else if (direct2ndRec)
                {
                    GradU(SeqG012, EigenAll) = uGradBufNoLim[iCell];
                }
                else if (settings.source2nd)
                {
                    GradU = cellGrad2nd;
                }
                else
                {
                    DNDS_assert(pURecUnlim);
                    if constexpr (gDim == 2)
                        GradU({0, 1}, EigenAll) =
                            vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iGQ, std::array<int, 2>{1, 2}, 3) *
                            (*pURecUnlim)[iCell]; // IF_NOT_NOREC = 1
                    else
                        GradU({0, 1, 2}, EigenAll) =
                            vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iGQ, std::array<int, 3>{1, 2, 3}, 4) *
                            (*pURecUnlim)[iCell]; // IF_NOT_NOREC = 1
                    if (settings.useSourceGradFixGG)
                        GradU += cellGradFix;
                    if (settings.ransSource2nd)
                    {
                        if constexpr (Traits::hasSA)
                            GradU(EigenAll, I4 + 1) = cellGrad2nd(EigenAll, I4 + 1);
                        if constexpr (Traits::has2EQ)
                            GradU(EigenAll, {I4 + 1, I4 + 2}) = cellGrad2nd(EigenAll, {I4 + 1, I4 + 2});
                    }
                }

                TU ULxy = uCell;
                if (useRecArrays && !direct2ndRec)
                {
                    DNDS_assert(pURec);
                    ULxy +=
                        (vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iGQ, std::array<int, 1>{0}, 1) *
                         (*pURec)[iCell])
                            .transpose(); // IF_NOT_NOREC = 1
                }

                finc.resizeLike(sourceV);
                TJacobianU jac;
                finc(EigenAll, 0) =
                    source(ULxy, GradU,
                           vfv->GetCellQuadraturePPhys(iCell, iGQ), jac,
                           iCell, iGQ, 0, filter);
                if (jacMode >= 1)
                {
                    TU sourceJDiag =
                        source(ULxy, GradU,
                               vfv->GetCellQuadraturePPhys(iCell, iGQ), jac,
                               iCell, iGQ, (jacMode == 2) ? 2 : 1, filter);
                    if (jacMode == 2)
                        finc(EigenAll, Eigen::seq(Eigen::fix<1>, EigenLast)) = jac;
                    else
                        finc(EigenAll, 1) = sourceJDiag;
                }

                finc *= (useRecArrays && !direct2ndRec)
                            ? vfv->GetCellJacobiDet(iCell, iG)
                            : vfv->GetCellVol(iCell) / vfv->GetCellParamVol(iCell);
                DNDS_assert(finc.allFinite());
            });
        sourceV *= cellAlpha / vfv->GetCellVol(iCell); // becomes mean value

        cellRHS += sourceV(EigenAll, 0);
        if (jacMode == 2)
            cellJac = sourceV(EigenAll, Eigen::seq(Eigen::fix<1>, EigenLast));
        else if (jacMode == 1)
            cellJac.diagonal() = sourceV(EigenAll, 1);
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Dispatcher for boundary ghost-value generation, routing to per-BC-type handlers.
     *
     *  Selects the appropriate boundary value generator based on the BC type (farfield,
     *  wall, inviscid wall, outflow, inflow, total-condition inflow, symmetry, etc.)
     *  and returns the ghost (right) state for the boundary face.
     *
     *  @param ULxy       Left (interior) state; may be modified by some handlers (e.g., fixUL).
     *  @param ULMeanXy   Left cell mean state.
     *  @param iCell      Owner cell index.
     *  @param iFace      Face index.
     *  @param iG         Quadrature point index (-1 for face center).
     *  @param uNorm      Outward unit normal vector.
     *  @param normBase   Local coordinate frame built from the normal.
     *  @param pPhysics   Physical coordinates of the quadrature point.
     *  @param t          Current simulation time.
     *  @param btype      Boundary zone identifier.
     *  @param fixUL      If true, handler may modify ULxy (e.g., enforce zero SA at wall).
     *  @param geomMode   Geometry mode for normal evaluation.
     *  @param linMode    If 1, return linearized boundary perturbation (for Jacobian).
     *  @return Ghost (right) state for the boundary face.
     */
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
        int geomMode, int linMode)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_assert(iG >= -2);

        TU URxy;
        URxy.resizeLike(ULxy);
        auto bTypeEuler = pBCHandler->GetTypeFromID(btype);

        // TODO: for all linMode == 1: implement more precise linearized BC

        if (bTypeEuler == EulerBCType::BCSpecial ||
            bTypeEuler == EulerBCType::BCFar ||
            bTypeEuler == EulerBCType::BCOutP)
        {
            if (linMode == 1)
            {
                URxy.setZero();
                return URxy;
            }
            DNDS_assert(ULxy(0) > 0);
            if (btype == Geom::BC_ID_DEFAULT_FAR ||
                bTypeEuler == EulerBCType::BCFar ||
                bTypeEuler == EulerBCType::BCOutP)
            {
                URxy = generateBV_FarField(ULxy, ULMeanXy, iCell, iFace, iG,
                                           uNorm, normBase, pPhysics, t, btype, fixUL, geomMode);
            }
            else
            {
                URxy = generateBV_SpecialFar(ULxy, ULMeanXy, iCell, iFace, iG,
                                             uNorm, normBase, pPhysics, t, btype);
            }
        }
        else if (bTypeEuler == EulerBCType::BCWallInvis ||
                 bTypeEuler == EulerBCType::BCSym)
        {
            URxy = generateBV_InviscidWall(ULxy, ULMeanXy, iCell, iFace, iG,
                                           uNorm, normBase, pPhysics, t, btype);
            if (linMode == 1)
                return URxy;
        }
        else if (bTypeEuler == EulerBCType::BCWall ||
                 bTypeEuler == EulerBCType::BCWallIsothermal)
        {
            URxy = generateBV_ViscousWall(ULxy, ULMeanXy, iCell, iFace, iG,
                                          uNorm, normBase, pPhysics, t, btype, fixUL, linMode);
            if (linMode == 1)
                return URxy;
        }
        else if (bTypeEuler == EulerBCType::BCOut)
        {
            URxy = generateBV_Outflow(ULxy, ULMeanXy, iCell, iFace, iG,
                                      uNorm, normBase, pPhysics, t, btype);
            if (linMode == 1)
            {
                URxy.setZero();
                return URxy;
            }
        }
        else if (bTypeEuler == EulerBCType::BCIn)
        {
            if (linMode == 1)
            {
                URxy.setZero();
                return URxy;
            }
            URxy = generateBV_Inflow(ULxy, ULMeanXy, iCell, iFace, iG,
                                     uNorm, normBase, pPhysics, t, btype);
        }
        else if (bTypeEuler == EulerBCType::BCInPsTs)
        {
            if (linMode == 1)
            {
                URxy.setZero();
                return URxy;
            }
            URxy = generateBV_TotalConditionInflow(ULxy, ULMeanXy, iCell, iFace, iG,
                                                   uNorm, normBase, pPhysics, t, btype);
        }
        else
        {
            DNDS_assert(false);
        }
        return URxy;
    }

    /**************************************************************************
     * Per-BC-type handlers
     **************************************************************************/

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Generate ghost state for farfield and back-pressure outlet boundary conditions.
     *
     *  Uses Riemann-invariant-based switching: supersonic outflow extrapolates the
     *  interior state; subsonic outflow imposes farfield pressure; subsonic inflow
     *  imposes farfield values but retains interior pressure; supersonic inflow uses
     *  the full farfield state. Supports anchor-point and radial-profile pressure corrections,
     *  CL-driver AoA rotation, and rotating-frame transformations.
     *
     *  @param ULxy       Left (interior) state.
     *  @param ULMeanXy   Left cell mean state.
     *  @param iCell      Owner cell index.
     *  @param iFace      Face index.
     *  @param iG         Quadrature point index.
     *  @param uNorm      Outward unit normal.
     *  @param normBase   Local coordinate frame.
     *  @param pPhysics   Physical coordinates.
     *  @param t          Current simulation time.
     *  @param btype      Boundary zone identifier.
     *  @param fixUL      If true, may modify ULxy.
     *  @param geomMode   Geometry mode.
     *  @return Ghost (right) state for the farfield face.
     */
    typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_FarField(
        TU &ULxy, const TU &ULMeanXy,
        index iCell, index iFace, int iG,
        const TVec &uNorm, const TMat &normBase,
        const Geom::tPoint &pPhysics, real t,
        Geom::t_index btype, bool fixUL, int geomMode)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        auto bTypeEuler = pBCHandler->GetTypeFromID(btype);
        TU URxy;
        URxy.resizeLike(ULxy);

        TU far = btype >= Geom::BC_ID_DEFAULT_MAX
                     ? pBCHandler->GetValueFromID(btype)
                     : TU(settings.farFieldStaticValue);
        if (pCLDriver)
            far(Seq123) = pCLDriver->GetAOARotation()(Seq012, Seq012) * far(Seq123);

        if (bTypeEuler == EulerBCType::BCFar)
        {
            if (settings.frameConstRotation.enabled && pBCHandler->GetFlagFromID(btype, "frameOpt") != 0)
                far(Seq123) = (Geom::RotateAxis(-settings.frameConstRotation.vOmega() * t) * Geom::ToThreeDim<dim>(far(Seq123)))(Seq012);
        }

        TU ULxyStatic = ULxy;
        if (settings.frameConstRotation.enabled) // to static frame velocity
            TransformURotatingFrame(ULxyStatic, pPhysics, 1);

        real un = ULxy(Seq123).dot(uNorm) / ULxy(0); // using relative velo for in/out judgement
        real T = phys_.template temperature<dim>(ULxyStatic);
        real gammaEq = 0;
        real gamma = 0;
        real asqr, H, p;
        phys_.template conservativeThermal<dim>(ULxyStatic, T, p, asqr, H, &gammaEq, &gamma);
        auto gammaEqFromTotal = [&](const TU &U)
        {
            real TState = phys_.template temperature<dim>(U);
            return phys_.template gammaEq<dim>(TState, U);
        };
        auto gammaEqFromSensibleConfig = [&](TU U)
        {
            if (settings.reactiveFlow.enabled)
                U(I4) += phys_.mixtureFormationRhoE(U);
            return gammaEqFromTotal(U);
        };

        DNDS_assert(asqr >= 0);
        real a = std::sqrt(asqr);

        auto vg = this->GetFaceVGrid(iFace, iG, pPhysics);
        real vgN = vg.dot(uNorm);

        if (un - vgN - a > 0) // full outflow
        {
            URxy = ULxyStatic;
        }
        else if (un - vgN > 0) //  subsonic out
        {
            TU farPrimitive, ULxyPrimitive;
            farPrimitive.resizeLike(ULxyStatic);
            ULxyPrimitive.resizeLike(URxy);
            real gammaEqFar = gammaEqFromSensibleConfig(far);
            Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gammaEqFar, 0 /* config, sensible ρE */);
            Gas::IdealGasThermalConservative2Primitive<dim>(ULxyStatic, ULxyPrimitive, gammaEq,
                                                            phys_.mixtureFormationRhoE(ULxyStatic));
            if (bTypeEuler == EulerBCType::BCOutP && pBCHandler->GetFlagFromID(btype, "anchorOpt") == 1)
            {
                {
                    TU anchorPointRel = ULxy;
                    if (anchorRecorders.count(btype)) // if doesn't have anchor value yet, use UL as anchor
                        anchorPointRel = anchorRecorders.at(btype).val;
                    TU anchorPointRelPrimitive = anchorPointRel;
                    Gas::IdealGasThermalConservative2Primitive<dim>(anchorPointRel, anchorPointRelPrimitive, gammaEqFromTotal(anchorPointRel),
                                                                    phys_.mixtureFormationRhoE(anchorPointRel));
                    farPrimitive(I4) += std::max(ULxyPrimitive(I4) - anchorPointRelPrimitive(I4), -0.95 * farPrimitive(I4));
                }
            }
            if (bTypeEuler == EulerBCType::BCOutP && pBCHandler->GetFlagFromID(btype, "anchorOpt") == 2)
            {
                real pInc = 0;
                if (profileRecorders.count(btype))
                    pInc = profileRecorders.at(btype).GetPlain(settings.frameConstRotation.rVec(pPhysics).norm())(I4);
                farPrimitive(I4) += std::max(pInc, -0.95 * farPrimitive(I4));
            }
            ULxyPrimitive(I4) = farPrimitive(I4); // using far pressure
            if (settings.reactiveFlow.enabled)
                phys_.template primToConservative<dim>(ULxyPrimitive, URxy);
            else
                Gas::IdealGasThermalPrimitive2Conservative<dim>(ULxyPrimitive, URxy, gammaEq, 0);
        }
        else if (un - vgN + a > 0) //  subsonic in
        {
            TU farPrimitive, ULxyPrimitive;
            farPrimitive.resizeLike(ULxyStatic);
            ULxyPrimitive.resizeLike(URxy);
            real gammaEqFar = gammaEqFromSensibleConfig(far);
            Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gammaEqFar, 0 /* config, sensible ρE */);
            Gas::IdealGasThermalConservative2Primitive<dim>(ULxyStatic, ULxyPrimitive, gammaEq,
                                                            phys_.mixtureFormationRhoE(ULxyStatic));
            farPrimitive(I4) = ULxyPrimitive(I4); // using inner pressure
            if (settings.reactiveFlow.enabled)
                phys_.template primToConservative<dim>(farPrimitive, URxy);
            else
                Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gammaEqFar, 0);
        }
        else // full inflow
        {
            URxy = far;
            if (settings.reactiveFlow.enabled)
                URxy(I4) += phys_.mixtureFormationRhoE(URxy);
        }
        if (settings.frameConstRotation.enabled) // to rotating frame velocity
            TransformURotatingFrame(URxy, pPhysics, -1);
        return URxy;
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Generate ghost state for special farfield BCs (DMR, Rayleigh-Taylor, etc.).
     *
     *  Handles built-in special boundary conditions identified by reserved BC IDs:
     *  double Mach reflection (DMR), Rayleigh-Taylor instability, and user-specified
     *  special options. Uses Riemann-invariant switching like the standard farfield.
     *
     *  @param ULxy       Left (interior) state.
     *  @param ULMeanXy   Left cell mean state.
     *  @param iCell      Owner cell index.
     *  @param iFace      Face index.
     *  @param iG         Quadrature point index.
     *  @param uNorm      Outward unit normal.
     *  @param normBase   Local coordinate frame.
     *  @param pPhysics   Physical coordinates.
     *  @param t          Current simulation time.
     *  @param btype      Boundary zone identifier (special reserved ID).
     *  @return Ghost (right) state for the special farfield face.
     */
    typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_SpecialFar(
        TU &ULxy, const TU &ULMeanXy,
        index iCell, index iFace, int iG,
        const TVec &uNorm, const TMat &normBase,
        const Geom::tPoint &pPhysics, real t,
        Geom::t_index btype)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_check_throw_info(!settings.reactiveFlow.enabled,
                              "Reactive flow does not support built-in special far-field BCs; "
                              "use explicit reactive farfield/inflow/outflow BCs instead.");
        TU URxy;
        URxy.resizeLike(ULxy);

        if (btype == Geom::BC_ID_DEFAULT_SPECIAL_DMR_FAR) // DMR
        {
            // Shock-tube BC: overwrites fluid components (dim+1) only.
            // Hardcoded Vector<real,5/4> sizes assume NS models; extended-model
            // components (SA, 2EQ, species) preserve farFieldStaticValue defaults.
            DNDS_assert(dim > 1);
            URxy = settings.farFieldStaticValue;
            real uShock = 10;
            if constexpr (dim == 3)
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
            if (settings.reactiveFlow.enabled)
                URxy(I4) += phys_.mixtureFormationRhoE(URxy);
        }
        else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_RT_FAR) // Rayleigh-Taylor
        {
            DNDS_assert(dim > 1);
            Eigen::VectorXd far = settings.farFieldStaticValue;
            real T = phys_.template temperature<dim>(ULxy);
            real gammaEq = 0;
            real gamma = 0;
            real un = ULxy(Seq123).dot(uNorm) / ULxy(0);
            real asqr, H, p;
            phys_.template conservativeThermal<dim>(ULxy, T, p, asqr, H, &gammaEq, &gamma);

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
                far(I4) = 0.5 * rho * sqr(v) + p / (gammaEq - 1);
            }
            else
            {
                real rho = 1;
                real p = 2.5;
                far(0) = rho;
                far(1) = 0;
                far(2) = rho * v;
                far(I4) = 0.5 * rho * sqr(v) + p / (gammaEq - 1);
            }

            if (un - a > 0)
                URxy = ULxy;
            else if (un > 0)
            {
                TU farPrimitive, ULxyPrimitive;
                farPrimitive.resizeLike(ULxy);
                ULxyPrimitive.resizeLike(URxy);
                Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gammaEq, 0 /* config, sensible ρE */);
                Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gammaEq,
                                                                phys_.mixtureFormationRhoE(ULxy));
                ULxyPrimitive(I4) = farPrimitive(I4);
                Gas::IdealGasThermalPrimitive2Conservative<dim>(ULxyPrimitive, URxy, gammaEq,
                                                                phys_.mixtureFormationRhoE(ULxy));
            }
            else if (un + a > 0)
            {
                TU farPrimitive, ULxyPrimitive;
                farPrimitive.resizeLike(ULxy);
                ULxyPrimitive.resizeLike(URxy);
                Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gammaEq, 0 /* config, sensible ρE */);
                Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gammaEq,
                                                                phys_.mixtureFormationRhoE(ULxy));
                farPrimitive(I4) = ULxyPrimitive(I4);
                Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gammaEq,
                                                                phys_.mixtureFormationRhoE(far));
            }
            else
            {
                URxy = far;
                if (settings.reactiveFlow.enabled)
                    URxy(I4) += phys_.mixtureFormationRhoE(URxy);
            }
        }
        else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_IV_FAR) // Isentropic Vortex
        {
            DNDS_assert_info(!settings.reactiveFlow.enabled,
                             "BCSpecial IV: Isentropic Vortex BC does not support reactive flow. "
                             "Species are zeroed and formation energy is not included.");
            real chi = 5;
            real T_iv = phys_.template temperature<dim>(ULxy);
            real gammaEq = phys_.template gammaEq<dim>(T_iv, ULxy);
            real gamma = phys_.gamma(T_iv, ULxy);
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
            real pVal = T * rho;
            real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + pVal / (gammaEq - 1);

            URxy.setZero();
            URxy(0) = rho;
            URxy(1) = rho * ux;
            URxy(2) = rho * uy;
            URxy(dim + 1) = E;
        }
        else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR) // 2D Riemann
        {
            real T = phys_.template temperature<dim>(ULxy);
            real gammaEq = phys_.template gammaEq<dim>(T, ULxy);
            real bdL = 0.0, bdR = 1.0, bdD = 0.0, bdU = 1.0;

            real phi1 = -0.663324958071080;
            real phi2 = -0.422115882408869;
            real location = 0.8;
            real p1 = location + phi1 * t;
            real p2 = location + phi2 * t;
            real rho, u, v, pre;
            TU ULxyPrimitive;
            ULxyPrimitive.resizeLike(ULxy);

            Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gammaEq,
                                                            phys_.mixtureFormationRhoE(ULxy));
            real rhoL = ULxyPrimitive(0);
            real uL = ULxyPrimitive(1);
            real vL = ULxyPrimitive(2);
            real preL = ULxyPrimitive(I4);
            TU farPrimitive = ULxyPrimitive;

            static const real bTol = 1e-9;
            if (std::abs(pPhysics(0) - bdL) < bTol)
            { // left, phi2
                if (pPhysics(1) <= p2)
                    rho = 0.137992831541219, u = 1.206045378311055, v = 1.206045378311055, pre = 0.029032258064516;
                else
                    rho = 0.532258064516129, u = 1.206045378311055, v = 0.0, pre = 0.3;
            }
            else if (std::abs(pPhysics(0) - bdR) < bTol)
            { // right, phi1
                if (pPhysics(1) <= p1)
                    rho = rhoL, u = -uL, v = vL, pre = preL;
                else
                    rho = rhoL, u = -uL, v = vL, pre = preL;
            }
            else if (std::abs(pPhysics(1) - bdU) < bTol)
            { // up, phi1
                if (pPhysics(0) <= p1)
                    rho = rhoL, u = uL, v = -vL, pre = preL;
                else
                    rho = rhoL, u = uL, v = -vL, pre = preL;
            }
            else if (std::abs(pPhysics(1) - bdD) < bTol)
            { // down, phi2
                if (pPhysics(0) <= p2)
                    rho = 0.137992831541219, u = 1.206045378311055, v = 1.206045378311055, pre = 0.029032258064516;
                else
                    rho = 0.532258064516129, u = 0.0, v = 1.206045378311055, pre = 0.3;
            }
            else
            {
                rho = u = v = pre = std::nan("1");
                DNDS_assert(false);
            }
            farPrimitive(0) = rho;
            farPrimitive(1) = u, farPrimitive(2) = v;
            farPrimitive(I4) = pre;
            Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gammaEq,
                                                            phys_.mixtureFormationRhoE(ULxy));
        }
        else if (pBCHandler->GetFlagFromID(btype, "specialOpt") == 3001) // Noh
        {
            TU farPrimitive;
            real T_noh = phys_.template temperature<dim>(settings.farFieldStaticValue);
            real gamma_noh = phys_.template gammaEq<dim>(T_noh, settings.farFieldStaticValue);
            Gas::IdealGasThermalConservative2Primitive<dim>(settings.farFieldStaticValue, farPrimitive, gamma_noh, 0 /* config, sensible ρE */);
            real pInf = farPrimitive(I4);
            real r = pPhysics.norm();
            TVec velo = -pPhysics(Seq012) / (r + smallReal);
            real rho = sqr(1. + t / (r + smallReal));
            farPrimitive(0) = rho;
            farPrimitive(Seq123) = velo;
            farPrimitive(I4) = pInf;
            Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma_noh,
                                                            phys_.mixtureFormationRhoE(settings.farFieldStaticValue));
        }
        else
            DNDS_assert_info(false, fmt::format(
                                        "btype [{}] or bTypeEuler [{}] or specialOpt [{}] is not supported",
                                        btype, to_string(pBCHandler->GetTypeFromID(btype)),
                                        pBCHandler->GetFlagFromIDSoft(btype, "specialOpt")));
        return URxy;
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Generate ghost state for inviscid wall (slip wall) and symmetry boundary conditions.
     *
     *  Mirrors the velocity component normal to the wall while preserving the
     *  tangential component, density, and energy. Handles rotating-frame transformations.
     *
     *  @param ULxy       Left (interior) state.
     *  @param ULMeanXy   Left cell mean state.
     *  @param iCell      Owner cell index.
     *  @param iFace      Face index.
     *  @param iG         Quadrature point index.
     *  @param uNorm      Outward unit normal.
     *  @param normBase   Local coordinate frame.
     *  @param pPhysics   Physical coordinates.
     *  @param t          Current simulation time.
     *  @param btype      Boundary zone identifier.
     *  @return Ghost (right) state with reflected normal velocity.
     */
    typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_InviscidWall(
        TU &ULxy, const TU &ULMeanXy,
        index iCell, index iFace, int iG,
        const TVec &uNorm, const TMat &normBase,
        const Geom::tPoint &pPhysics, real t,
        Geom::t_index btype)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        TU URxy = ULxy;
        if (settings.frameConstRotation.enabled)
            this->TransformURotatingFrame_ABS_VELO(URxy, pPhysics, -1);
        URxy(Seq123) -= 2 * URxy(Seq123).dot(uNorm) * uNorm; // mirrored!
        if (settings.frameConstRotation.enabled)
            this->TransformURotatingFrame_ABS_VELO(URxy, pPhysics, 1);
        return URxy;
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Generate ghost state for viscous (no-slip) wall boundary conditions.
     *
     *  Reverses the velocity to enforce zero velocity at the wall (ghost mirroring).
     *  For isothermal walls, adjusts density to match the prescribed wall temperature.
     *  Handles SA turbulence variable (negated or zeroed at wall) and k-omega/k-epsilon
     *  wall boundary conditions including the realizable k-epsilon wall epsilon formula.
     *
     *  @param ULxy       Left (interior) state (may be modified when fixUL is true for SA).
     *  @param ULMeanXy   Left cell mean state.
     *  @param iCell      Owner cell index.
     *  @param iFace      Face index.
     *  @param iG         Quadrature point index.
     *  @param uNorm      Outward unit normal.
     *  @param normBase   Local coordinate frame.
     *  @param pPhysics   Physical coordinates.
     *  @param t          Current simulation time.
     *  @param btype      Boundary zone identifier.
     *  @param fixUL      If true, may zero SA variables in ULxy at the wall.
     *  @param linMode    If 1, return linearized ghost perturbation for Jacobian.
     *  @return Ghost (right) state with no-slip wall condition.
     */
    typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_ViscousWall(
        TU &ULxy, const TU &ULMeanXy,
        index iCell, index iFace, int iG,
        const TVec &uNorm, const TMat &normBase,
        const Geom::tPoint &pPhysics, real t,
        Geom::t_index btype, bool fixUL, int linMode)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        auto bTypeEuler = pBCHandler->GetTypeFromID(btype);
        TU URxy = ULxy;
        if (true) // physical mode
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

        if (linMode == 1)
            return URxy;

        if (bTypeEuler == EulerBCType::BCWallIsothermal)
        {
            real temp = pBCHandler->GetValueFromID(btype)(0);
            TU URxyPrim;
            URxyPrim.resizeLike(ULxy);
            real T = phys_.template temperature<dim>(URxy);
            real gammaEq = phys_.template gammaEq<dim>(T, URxy);
            Gas::IdealGasThermalConservative2Primitive<dim>(URxy, URxyPrim, gammaEq,
                                                            phys_.mixtureFormationRhoE(URxy));
            DNDS_assert_info(URxyPrim(0) > 0 && URxyPrim(I4) > 0 && temp > 0, fmt::format("{}, {}, {}", URxyPrim(0), URxyPrim(I4), temp));
            real newDensity = URxyPrim(I4) / temp / phys_.Rgas(URxy);
            URxyPrim(0) = newDensity;
            if (phys_.hasChemicalSource())
                phys_.template primToConservative<dim>(URxyPrim, URxy);
            else
                Gas::IdealGasThermalPrimitive2Conservative<dim>(URxyPrim, URxy, gammaEq, 0);
        }
        if constexpr (Traits::hasSA)
        {
            URxy(I4 + 1) *= -1;
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
            if (fixUL)
                ULxy(I4 + 1) = URxy(I4 + 1) = 0; //! modifing UL
#endif
        }
        if constexpr (Traits::has2EQ)
        {
            URxy({I4 + 1, I4 + 2}) *= -1;
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
            // if (fixUL)
            //     ULxy({I4 + 1, I4 + 2}).setZero(), URxy({I4 + 1, I4 + 2}).setZero();
#endif
            if (settings.ransModel == RANSModel::RANS_RKE)
            { // BC for RealizableKe
                real d1 = dWall[iCell].mean();
                real k1 = ULMeanXy(I4 + 1) / ULMeanXy(0);

                real pMean, asqrMean, Hmean;
                real T = phys_.template temperature<dim>(ULMeanXy);
                phys_.template conservativeThermal<dim>(ULMeanXy, T, pMean, asqrMean, Hmean);
                real mufPhy1 = muEff(ULMeanXy, T);
                real epsWall = 2 * mufPhy1 / ULMeanXy(0) * k1 / sqr(d1);
                URxy(I4 + 2) = 2 * epsWall * ULxy(0) - ULxy(I4 + 2);
            }
            if (settings.ransModel == RANSModel::RANS_KOSST ||
                settings.ransModel == RANSModel::RANS_KOWilcox)
            { // BC for SST or KOWilcox
                real d1 = dWall[iCell].mean();
                real pMean, asqrMean, Hmean;
                real T = phys_.template temperature<dim>(ULMeanXy);
                phys_.template conservativeThermal<dim>(ULMeanXy, T, pMean, asqrMean, Hmean);
                real mufPhy1 = muEff(ULMeanXy, T);

                real rhoOmegaaaWall = mufPhy1 / sqr(d1) * RANS::kWallOmegaCoeff;
                URxy(I4 + 2) = 2 * rhoOmegaaaWall - ULxy(I4 + 2);
            }
        }
        return URxy;
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Generate ghost state for supersonic/extrapolation outflow BC.
     *
     *  Simply returns the interior state as the ghost value, allowing all waves to
     *  exit the domain without reflection.
     *
     *  @param ULxy       Left (interior) state.
     *  @param ULMeanXy   Left cell mean state.
     *  @param iCell      Owner cell index.
     *  @param iFace      Face index.
     *  @param iG         Quadrature point index.
     *  @param uNorm      Outward unit normal.
     *  @param normBase   Local coordinate frame.
     *  @param pPhysics   Physical coordinates.
     *  @param t          Current simulation time.
     *  @param btype      Boundary zone identifier.
     *  @return Ghost state equal to the interior state (full extrapolation).
     */
    typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_Outflow(
        TU &ULxy, const TU &ULMeanXy,
        index iCell, index iFace, int iG,
        const TVec &uNorm, const TMat &normBase,
        const Geom::tPoint &pPhysics, real t,
        Geom::t_index btype)
    {
        return ULxy;
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Generate ghost state for prescribed inflow boundary condition.
     *
     *  Sets the ghost state to the BC-prescribed conservative values. Applies
     *  CL-driver AoA rotation and rotating-frame transformation if applicable.
     *
     *  @param ULxy       Left (interior) state (unused for pure inflow).
     *  @param ULMeanXy   Left cell mean state.
     *  @param iCell      Owner cell index.
     *  @param iFace      Face index.
     *  @param iG         Quadrature point index.
     *  @param uNorm      Outward unit normal.
     *  @param normBase   Local coordinate frame.
     *  @param pPhysics   Physical coordinates.
     *  @param t          Current simulation time.
     *  @param btype      Boundary zone identifier.
     *  @return Ghost state set to prescribed inflow values.
     */
    typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_Inflow(
        TU &ULxy, const TU &ULMeanXy,
        index iCell, index iFace, int iG,
        const TVec &uNorm, const TMat &normBase,
        const Geom::tPoint &pPhysics, real t,
        Geom::t_index btype)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        TU URxy = pBCHandler->GetValueFromID(btype);
        if (pCLDriver)
            URxy(Seq123) = pCLDriver->GetAOARotation()(Seq012, Seq012) * URxy(Seq123);
        // Note: removed dead code that checked bTypeEuler == BCFar (unreachable in BCIn branch)
        if (settings.reactiveFlow.enabled)
            URxy(I4) += phys_.mixtureFormationRhoE(URxy);
        if (settings.frameConstRotation.enabled)
            TransformURotatingFrame(URxy, pPhysics, -1);
        return URxy;
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>
        ,
        template <>
    )
    /** @brief Generate ghost state for total-condition (stagnation) inflow BC.
     *
     *  Given prescribed stagnation pressure and temperature, computes the static
     *  conditions from the interior velocity magnitude using isentropic relations.
     *  The flow direction is taken from the BC-prescribed direction vector.
     *  Applies CL-driver AoA rotation and rotating-frame transformation.
     *
     *  @param ULxy       Left (interior) state (used for velocity magnitude).
     *  @param ULMeanXy   Left cell mean state.
     *  @param iCell      Owner cell index.
     *  @param iFace      Face index.
     *  @param iG         Quadrature point index.
     *  @param uNorm      Outward unit normal.
     *  @param normBase   Local coordinate frame.
     *  @param pPhysics   Physical coordinates.
     *  @param t          Current simulation time.
     *  @param btype      Boundary zone identifier.
     *  @return Ghost state computed from total conditions and interior velocity.
     */
    typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_TotalConditionInflow(
        TU &ULxy, const TU &ULMeanXy,
        index iCell, index iFace, int iG,
        const TVec &uNorm, const TMat &normBase,
        const Geom::tPoint &pPhysics, real t,
        Geom::t_index btype)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        TU URxy;
        URxy.resizeLike(ULxy);

        TU ULxyStatic = ULxy;
        if (settings.frameConstRotation.enabled)
            TransformURotatingFrame(ULxyStatic, pPhysics, 1);
        TVec v = ULxyStatic(Seq123).array() / ULxyStatic(0);
        real vSqr = v.squaredNorm();
        {
            TU bcValue = pBCHandler->GetValueFromID(btype);
            TU farPrimitive = bcValue; // primitive passive scalar components like species/RANS variables
            real dirNorm = bcValue(Seq234).norm();
            DNDS_check_throw_info(std::isfinite(dirNorm) && dirNorm > 1e-14,
                                  fmt::format("BCInPsTs requires a finite nonzero direction vector for boundary id {}", btype));
            real dirDotNormal = bcValue(Seq234).dot(uNorm);
            DNDS_check_throw_info(std::isfinite(dirDotNormal) && dirDotNormal < -1e-14 * dirNorm,
                                  fmt::format("BCInPsTs direction must point into the domain for boundary id {}; direction dot outward normal = {:.3e}",
                                              btype, dirDotNormal));
            farPrimitive(Seq123) = bcValue(Seq234) * (std::sqrt(vSqr) / dirNorm);
            phys_.template totalToStaticPrimitive<dim>(bcValue(0), bcValue(1), farPrimitive);
            phys_.template primToConservative<dim>(farPrimitive, URxy);
        }
        if (pCLDriver)
            URxy(Seq123) = pCLDriver->GetAOARotation()(Seq012, Seq012) * URxy(Seq123);
        if (settings.frameConstRotation.enabled)
            TransformURotatingFrame(URxy, pPhysics, -1);
        return URxy;
    }

    template <EulerModel model>
    /** @brief Register available output scalar fields in the OutputPicker.
     *
     *  Populates the output picker with named lambda functions that extract per-cell
     *  scalar quantities (conservative components, reconstruction coefficients,
     *  limiter values, wall distance, cell volume, turbulent viscosity, etc.)
     *  for use in post-processing output.
     *
     *  @param op        OutputPicker to populate with named field extractors (output).
     *  @param dataRefs  References to the solution arrays (u, uRec, betaPP, alphaPP).
     */
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
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                TU Uxy = u[iCell];
                TDiffU GradU;
                GradU.resize(Eigen::NoChange, nVars);
                GradU.setZero();
                if constexpr (gDim == 2)
                    GradU({0, 1}, EigenAll) =
                        vfv->GetIntPointDiffBaseValue(iCell, -1, -1, -1, std::array<int, 2>{1, 2}, 3) *
                        uRec[iCell]; // 2d specific
                else
                    GradU({0, 1, 2}, EigenAll) =
                        vfv->GetIntPointDiffBaseValue(iCell, -1, -1, -1, std::array<int, 3>{1, 2, 3}, 4) *
                        uRec[iCell]; // 3d specific
                real pMean, asqrMean, Hmean;
                auto ULMeanXy = Uxy;
                real T = phys_.template temperature<dim>(ULMeanXy);
                phys_.template conservativeThermal<dim>(ULMeanXy, T, pMean, asqrMean, Hmean);
                // ! refvalue:
                real muRef = phys_.muRef();
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

        // Species output fields for reactive flow
        if constexpr (Traits::isExtended)
        {
            if (settings.reactiveFlow.enabled)
            {
                for (int k = I4 + 1; k < nVars; ++k)
                    outMap["rhoY_" + std::to_string(k - (I4 + 1))] =
                        [&, k](index iCell)
                    { return u[iCell](k); };
            }
        }

        op.setMap(outMap);
    }
}
