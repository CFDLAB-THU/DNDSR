#include "Mesh_Elevation_SmoothHelpers.hpp"
#include "Geom/Quadrature.hpp"

#include <fmt/core.h>
#include <Solver/Linear.hpp>

#include "DNDS/EigenPCH.hpp"

namespace DNDS::Geom
{
    // =================================================================
    // Block sparse matrix element for the elasticity system.
    // =================================================================
    namespace
    {
        struct MatElem
        {
            index j{UnInitIndex};
            Eigen::Matrix<real, 3, 3> m{};
        };

        // Build node-to-node adjacency from cell connectivity.
        std::vector<std::unordered_set<index>> BuildNode2NodeFromCells(
            const tAdjPair &cell2node, index nLocalNodes)
        {
            std::vector<std::unordered_set<index>> node2nodeV(nLocalNodes);
            for (index iCell = 0; iCell < cell2node.Size(); iCell++)
                for (auto iN : cell2node[iCell])
                    for (auto iNOther : cell2node[iCell])
                    {
                        if (iN == iNOther)
                            continue;
                        if (iN < nLocalNodes)
                            node2nodeV[iN].insert(iNOther);
                    }
            return node2nodeV;
        }

        // Allocate block sparse matrix from node-to-node adjacency.
        std::vector<std::vector<MatElem>> AllocateBlockSparseMatrix(
            const std::vector<std::unordered_set<index>> &node2nodeV,
            index nLocalNodes)
        {
            std::vector<std::vector<MatElem>> A(nLocalNodes);
            for (index iN = 0; iN < nLocalNodes; iN++)
            {
                A[iN].resize(node2nodeV[iN].size() + 1);
                A[iN][0].j = iN;
                int curRowEnd = 1;
                for (auto iNOther : node2nodeV[iN])
                    A[iN][curRowEnd++].j = iNOther;
                for (auto &ME : A[iN])
                    ME.m.setConstant(0);
            }
            return A;
        }

        // Block-sparse matrix-vector product with MPI ghost pull.
        void BlockMatVec(
            const std::vector<std::vector<MatElem>> &A,
            tCoordPair &x, tCoordPair &Ax)
        {
            for (index iN = 0; iN < x.father->Size(); iN++)
            {
                Ax[iN].setZero();
                for (auto &ME : A[iN])
                    Ax[iN] += ME.m * x[ME.j];
            }
            Ax.trans.startPersistentPull();
            Ax.trans.waitPersistentPull();
        }

        // Block-Jacobi preconditioner with MPI ghost pull.
        void BlockJacobiPrecondition(
            const std::vector<std::vector<MatElem>> &A,
            tCoordPair &x, tCoordPair &ADIx)
        {
            for (index iN = 0; iN < x.father->Size(); iN++)
                ADIx[iN] = A[iN][0].m.fullPivLu().solve(x[iN]);
            ADIx.trans.startPersistentPull();
            ADIx.trans.waitPersistentPull();
        }
    } // anonymous namespace

    // =================================================================
    // ElevatedNodesSolveInternalSmoothV2
    // =================================================================
    void UnstructuredMesh::ElevatedNodesSolveInternalSmoothV2()
    {
        auto setupOpt = PrepareSmoothSolverSetup(*this);
        if (!setupOpt)
            return;
        auto &setup = *setupOpt;
        auto &nodesBoundInterpolated = setup.nodesBoundInterpolated;
        auto &boundInterpCoo = setup.boundInterpCoo;

        // ----- Build node-to-node adjacency -----
        auto node2nodeV = BuildNode2NodeFromCells(cell2node, coords.father->Size());

        if (mpi.rank == mRank)
            log() << "RBF set: " << boundInterpCoo.son->Size() << std::endl;

        // ----- KD-tree for nearest boundary point lookup -----
        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTreeCoordPair>,
            PointCloudKDTreeCoordPair, 3, index>;
        auto coordsI = PointCloudKDTreeCoordPair(boundInterpCoo.son);
        kdtree_t bndInterpTree(3, coordsI);

        DNDS_MPI_InsertCheck(mpi, "Got node2node");

        // ----- Initialize DOF vectors -----
        CoordPairDOF dispO2, bO2, dispO2New, dispO2PrecRHS, dispO2Inc, dispO2IncLimited;
        auto initDOF = [&](CoordPairDOF &v)
        {
            v.InitPair("V2::dof", mpi);
            v.father->Resize(coords.father->Size());
            v.TransAttach();
            v.trans.BorrowGGIndexing(coords.trans);
            v.trans.createMPITypes();
            v.trans.initPersistentPull();
        };
        initDOF(dispO2);
        initDOF(dispO2Inc);
        initDOF(dispO2New);
        initDOF(dispO2IncLimited);
        initDOF(bO2);
        initDOF(dispO2PrecRHS);

        // ----- Allocate block sparse matrix -----
        auto A = AllocateBlockSparseMatrix(node2nodeV, coords.father->Size());

        // Initialize DOF values from boundary displacement data.
        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            bO2[iN].setZero();
            dispO2[iN].setZero();
            if (coordsElevDisp[iN](0) != largeReal)
                dispO2[iN] = coordsElevDisp[iN];
            if (coordsElevDisp[iN](2) == 2 * largeReal ||
                coordsElevDisp[iN](2) == 3 * largeReal)
                dispO2[iN].setZero();
        }
        dispO2.trans.startPersistentPull();
        dispO2.trans.waitPersistentPull();

        // =============================================================
        // Assemble elasticity stiffness matrix and RHS.
        // Uses bisect-FEM: each O2 cell is bisected into O1 sub-elements,
        // linear elasticity stiffness is assembled on each sub-element.
        // =============================================================
        auto AssembleRHSMat = [&](tCoordPair &uCur)
        {
            index nFix{0}, nFixB{0};

            for (index iCell = 0; iCell < cell2node.Size(); iCell++)
            {
                auto c2n = cell2node[iCell];
                rowsize nnLoc = c2n.size();
                SmallCoordsAsVector coordsC, coordsCu;
                GetCoordsOnCell(iCell, coordsC);
                GetCoordsOnCell(iCell, coordsCu, uCur);
                tPoint cellCent = coordsC.rowwise().mean();

                // Find distance to nearest boundary point.
                real wDist{0};
                {
                    std::vector<real> sqrDists(1);
                    std::vector<index> idxFound(1);
                    auto nFound = bndInterpTree.knnSearch(
                        cellCent.data(), 1, idxFound.data(), sqrDists.data());
                    DNDS_assert(nFound == 1);
                    wDist = std::sqrt(sqrDists[0]);
                }

                MatrixXR ALoc;
                ALoc.setZero(static_cast<index>(nnLoc) * 3, static_cast<index>(nnLoc) * 3 + 1);
                auto localNodeIdx = Eigen::ArrayXi::LinSpaced(c2n.size(), 0, c2n.size() - 1);

                // Bisect the O2 element into O1 sub-elements.
                auto eCell = GetCellElement(iCell);
                int nBi = eCell.GetO2NumBisect();
                int iBiVariant = GetO2ElemBisectVariant(eCell, coordsC);
                for (int iBi = 0; iBi < nBi; iBi++)
                {
                    auto eCellSub = eCell.ObtainO2BisectElem(iBi);
                    Eigen::ArrayXi c2nSubLocal;
                    c2nSubLocal.resize(eCellSub.GetNumNodes());
                    eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, localNodeIdx, c2nSubLocal);
                    SmallCoordsAsVector coordsCSub, coordsCuSub;
                    coordsCSub.resize(3, eCellSub.GetNumNodes());
                    coordsCuSub.resize(3, eCellSub.GetNumNodes());
                    eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, coordsC, coordsCSub);
                    eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, coordsCu, coordsCuSub);
                    auto qCellSub = Elem::Quadrature{eCellSub, 6};
                    auto nnLocSub = rowsize(c2nSubLocal.size());
                    MatrixXR ALocSub, mLoc;
                    ALocSub.setZero(static_cast<index>(nnLocSub) * 3, static_cast<index>(nnLocSub) * 3 + 1);
                    mLoc.resize(6, static_cast<index>(nnLocSub) * 3);
                    Eigen::ArrayXi c2nSubLocal3;
                    c2nSubLocal3.resize(c2nSubLocal.size() * 3);
                    c2nSubLocal3(Eigen::seq(c2nSubLocal.size() * 0, c2nSubLocal.size() * 1 - 1)) = c2nSubLocal + nnLoc * 0;
                    c2nSubLocal3(Eigen::seq(c2nSubLocal.size() * 1, c2nSubLocal.size() * 2 - 1)) = c2nSubLocal + nnLoc * 1;
                    c2nSubLocal3(Eigen::seq(c2nSubLocal.size() * 2, c2nSubLocal.size() * 3 - 1)) = c2nSubLocal + nnLoc * 2;

                    // Integrate elasticity stiffness on the sub-element.
                    qCellSub.Integration(
                        ALocSub,
                        [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                        {
                            auto getJ = [&](auto coos)
                            {
                                tJacobi J = Elem::ShapeJacobianCoordD01Nj(coos, DiNj);
                                tJacobi JInv = tJacobi::Identity();
                                real JDet = UnInitReal;
                                if (dim == 2)
                                    JDet = J(EigenAll, 0).cross(J(EigenAll, 1)).stableNorm();
                                else
                                    JDet = J.determinant();
                                if (dim == 2)
                                    JInv({0, 1}, {0, 1}) = J({0, 1}, {0, 1}).inverse().eval();
                                else
                                    JInv = J.inverse().eval();
                                return std::make_tuple(J, JInv, JDet);
                            };
                            auto [J, JInv, JDet] = getJ(coordsCSub);
                            auto [Jn, JInvn, JDetn] = getJ(coordsCSub + coordsCuSub);
                            DNDS_assert(JDet > 0);

                            // Strain-displacement matrix (Voigt notation).
                            tSmallCoords m3 = JInv.transpose() * DiNj({1, 2, 3}, EigenAll);
                            mLoc.setZero();
                            mLoc(0, Eigen::seq(nnLocSub * 0, nnLocSub * 0 + nnLocSub - 1)) = m3(0, EigenAll);
                            mLoc(1, Eigen::seq(nnLocSub * 1, nnLocSub * 1 + nnLocSub - 1)) = m3(1, EigenAll);
                            mLoc(2, Eigen::seq(nnLocSub * 2, nnLocSub * 2 + nnLocSub - 1)) = m3(2, EigenAll);
                            mLoc(3, Eigen::seq(nnLocSub * 1, nnLocSub * 1 + nnLocSub - 1)) = 0.5 * m3(2, EigenAll);
                            mLoc(3, Eigen::seq(nnLocSub * 2, nnLocSub * 2 + nnLocSub - 1)) = 0.5 * m3(1, EigenAll);
                            mLoc(4, Eigen::seq(nnLocSub * 2, nnLocSub * 2 + nnLocSub - 1)) = 0.5 * m3(0, EigenAll);
                            mLoc(4, Eigen::seq(nnLocSub * 0, nnLocSub * 0 + nnLocSub - 1)) = 0.5 * m3(2, EigenAll);
                            mLoc(5, Eigen::seq(nnLocSub * 0, nnLocSub * 0 + nnLocSub - 1)) = 0.5 * m3(1, EigenAll);
                            mLoc(5, Eigen::seq(nnLocSub * 1, nnLocSub * 1 + nnLocSub - 1)) = 0.5 * m3(0, EigenAll);

                            // Constitutive matrix (isotropic linear elasticity).
                            Eigen::Matrix<real, 6, 6> DStruct;
                            DStruct.setIdentity();
                            real lam = 1;
                            real muu = 100;
                            DStruct *= muu;
                            DStruct({0, 1, 2}, {0, 1, 2}) += muu * tJacobi::Identity();
                            DStruct({0, 1, 2}, {0, 1, 2}).array() += lam;
                            DStruct *= std::pow(JDetn / JDet, -0.0) + 1;

                            vInc.resizeLike(ALocSub);
                            vInc(EigenAll, Eigen::seq(0, 3 * nnLocSub - 1)) = mLoc.transpose() * DStruct * mLoc;
                            vInc(EigenAll, EigenLast).setZero();
                            vInc *= JDet;
                        });
                    ALoc(c2nSubLocal3, c2nSubLocal3) += ALocSub(EigenAll, Eigen::seq(0, EigenLast - 1));
                    ALoc(c2nSubLocal3, EigenLast) += ALocSub(EigenAll, EigenLast);
                }

                // Zero out z-component rows/cols for 2D problems.
                if (dim != 3)
                {
                    ALoc(Eigen::seq(2 * nnLoc, 3 * nnLoc - 1), Eigen::seq(2 * nnLoc, 3 * nnLoc - 1)).setZero();
                    ALoc(EigenAll, Eigen::seq(2 * nnLoc, 3 * nnLoc - 1)).setZero();
                    ALoc(Eigen::seq(2 * nnLoc, 3 * nnLoc - 1), EigenAll).setZero();
                }

                // Scatter local matrix into global block-sparse matrix.
                for (rowsize ic2n = 0; ic2n < nnLoc; ic2n++)
                {
                    index iN = c2n[ic2n];
                    if (iN >= coords.father->Size())
                        continue;
                    for (rowsize jc2n = 0; jc2n < nnLoc; jc2n++)
                    {
                        index jN = c2n[jc2n];
                        int nMatrixFound = 0;
                        for (auto &ME : A[iN])
                            if (ME.j == jN)
                            {
                                nMatrixFound++;
                                ME.m(0, 0) += ALoc(0 * nnLoc + ic2n, 0 * nnLoc + jc2n);
                                ME.m(0, 1) += ALoc(0 * nnLoc + ic2n, 1 * nnLoc + jc2n);
                                ME.m(0, 2) += ALoc(0 * nnLoc + ic2n, 2 * nnLoc + jc2n);
                                ME.m(1, 0) += ALoc(1 * nnLoc + ic2n, 0 * nnLoc + jc2n);
                                ME.m(1, 1) += ALoc(1 * nnLoc + ic2n, 1 * nnLoc + jc2n);
                                ME.m(1, 2) += ALoc(1 * nnLoc + ic2n, 2 * nnLoc + jc2n);
                                ME.m(2, 0) += ALoc(2 * nnLoc + ic2n, 0 * nnLoc + jc2n);
                                ME.m(2, 1) += ALoc(2 * nnLoc + ic2n, 1 * nnLoc + jc2n);
                                ME.m(2, 2) += ALoc(2 * nnLoc + ic2n, 2 * nnLoc + jc2n);
                                if (isPeriodic)
                                {
                                    auto ipbi = cell2nodePbi(iCell, ic2n);
                                    auto jpbi = cell2nodePbi(iCell, jc2n);
                                    tJacobi iTrans = periodicInfo.GetVectorByBits<3, 3>(tJacobi::Identity(), ipbi);
                                    tJacobi jTrans = periodicInfo.GetVectorByBits<3, 3>(tJacobi::Identity(), jpbi);
                                    ME.m = jTrans.transpose() * ME.m * iTrans;
                                }
                            }
                        DNDS_assert(nMatrixFound == 1);
                    }

                    // RHS contribution.
                    bO2[iN](0) += ALoc(0 * nnLoc + ic2n, EigenLast);
                    bO2[iN](1) += ALoc(1 * nnLoc + ic2n, EigenLast);
                    bO2[iN](2) += ALoc(2 * nnLoc + ic2n, EigenLast);

                    // Apply Dirichlet BC for boundary-displaced nodes.
                    if (coordsElevDisp[iN](0) != largeReal)
                    {
                        nFix++;
                        real nDiag = 0;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                nDiag = ME.m.array().abs().maxCoeff();
                        bO2[iN] = coordsElevDisp[iN] * nDiag;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                ME.m = tJacobi::Identity() * nDiag;
                            else
                                ME.m.setZero();
                    }
                    // Apply Dirichlet BC for O1 nodes and other boundary markers.
                    if (coordsElevDisp[iN](2) == 2 * largeReal ||
                        coordsElevDisp[iN](2) == 3 * largeReal)
                    {
                        nFixB++;
                        real nDiag = 0;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                nDiag = ME.m.array().abs().maxCoeff();
                        bO2[iN].setZero();
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                ME.m = tJacobi::Identity() * nDiag;
                            else
                                ME.m.setZero();
                    }
                    // Constrain z-direction in 2D.
                    if (dim != 3)
                    {
                        real nDiag = 0;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                nDiag = ME.m.array().abs().maxCoeff();
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                ME.m(2, 2) += nDiag;
                    }
                }
            }

            bO2.trans.startPersistentPull();
            bO2.trans.waitPersistentPull();
            MPI::AllreduceOneIndex(nFix, MPI_SUM, mpi);
            MPI::AllreduceOneIndex(nFixB, MPI_SUM, mpi);
            if (mpi.rank == mRank)
            {
                log() << fmt::format(
                             "UnstructuredMesh === ElevatedNodesSolveInternalSmooth(): "
                             "Matrix Assembled, nFix {}, nFixB {} ",
                             nFix, nFixB)
                      << std::endl;
            }
        };

        // ----- Assemble the linear system -----
        dispO2.setConstant(0.0);
        AssembleRHSMat(dispO2);

        // ----- Solve via preconditioned GMRES -----
        Linear::GMRES_LeftPreconditioned<CoordPairDOF> gmres(5, initDOF);
        gmres.solve(
            [&](CoordPairDOF &x, CoordPairDOF &Ax)
            { BlockMatVec(A, x, Ax); },
            [&](CoordPairDOF &x, CoordPairDOF &MLx)
            { BlockJacobiPrecondition(A, x, MLx); },
            [&](CoordPairDOF &a, CoordPairDOF &b) -> real
            { return a.dot(b); },
            bO2,
            dispO2,
            elevationInfo.nIter,
            [&](int iRestart, real res, real resB)
            {
                if (mpi.rank == mRank)
                    log() << fmt::format("iRestart [{}] res [{:3e}] -> [{:3e}]",
                                         iRestart, resB, res)
                          << std::endl;
                return res < resB * 1e-10;
            });

        // ----- Apply solved displacement to coordinates -----
        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (dim == 2)
                dispO2[iN](2) = 0;
            coords[iN] += dispO2[iN];
        }
        coords.trans.pullOnce();
    }

} // namespace DNDS::Geom
