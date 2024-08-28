#include "Mesh.hpp"
#include "Quadrature.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "RadialBasisFunction.hpp"

#include <omp.h>
#include <fmt/core.h>
#include <functional>
#include <unordered_set>
#include <Solver/Linear.hpp>

#include "PointCloud.hpp"
#include <nanoflann.hpp>

#include <Eigen/Sparse>
#ifdef DNDS_USE_SUPERLU
#include <superlu_ddefs.h>
#endif

namespace DNDS::Geom
{
    void UnstructuredMesh::BuildO2FromO1Elevation(UnstructuredMesh &meshO1)
    {
        real epsSqrDist = 1e-20;

        bool O1MeshIsO1 = meshO1.IsO1();
        DNDS_assert(O1MeshIsO1);
        // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX Phase0" << std::endl;
        DNDS_assert(meshO1.adjPrimaryState == Adj_PointToLocal);
        mRank = meshO1.mRank;
        mpi = meshO1.mpi;
        dim = meshO1.dim;
        isPeriodic = meshO1.isPeriodic;
        periodicInfo = meshO1.periodicInfo;
        nNodeO1 = meshO1.coords.father->Size();
        elevState = Elevation_O1O2;

        tAdjPair cellNewNodes; // records iNode - iNodeO1
        index localNewNodeNum{0};
        DNDS_MAKE_SSP(cellNewNodes.father, mpi);
        DNDS_MAKE_SSP(cellNewNodes.son, mpi);
        cellNewNodes.father->Resize(meshO1.cell2node.father->Size());
        std::vector<Eigen::Vector<real, 3>> newCoords;

        // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX Phase1" << std::endl;

        /**********************************/ // TODO
        // add new nodes to cellNewNodes, edge and face O2 nodes belong to smallest neighboring cell, cellNewNodes record local node indices
        for (index iCell = 0; iCell < meshO1.cell2node.father->Size(); iCell++)
        {
            Elem::Element eCell = meshO1.GetCellElement(iCell);
            auto c2n = meshO1.cell2node[iCell];
            auto c2c = meshO1.cell2cell[iCell];
            index iCellGlob = meshO1.cell2node.trans.pLGhostMapping->operator()(-1, iCell);

            std::vector<index> cellNewNodeRowV;
            cellNewNodeRowV.resize(eCell.GetNumElev_O1O2());
            SmallCoordsAsVector coordsC;
            meshO1.GetCoordsOnCell(iCell, coordsC);

            for (index iNNode = 0; iNNode < eCell.GetNumElev_O1O2(); iNNode++)
            {
                auto eSpan = eCell.ObtainElevNodeSpan(iNNode);
                std::array<index, Elem::CellNumNodeMax> spanNodes;
                eCell.ExtractElevNodeSpanNodes(iNNode, c2n, spanNodes);
                auto spanNodesSrt = spanNodes;
                std::sort(spanNodesSrt.begin(), spanNodesSrt.begin() + eSpan.GetNumNodes());
                tPoint newNodeCoord;

                { // the new node method, should cope with shape func
                    SmallCoordsAsVector coordsSpan;
                    coordsSpan.resize(3, eSpan.GetNumNodes());
                    eCell.ExtractElevNodeSpanNodes(iNNode, coordsC, coordsSpan);
                    newNodeCoord = coordsSpan.rowwise().mean();

                    if (isPeriodic)
                    {
                        std::vector<NodePeriodicBits> pbi;
                        pbi.resize(eSpan.GetNumNodes());
                        eCell.ExtractElevNodeSpanNodes(iNNode, meshO1.cell2nodePbi[iCell], pbi);
                        NodePeriodicBits pbiR;
                        pbiR.setP1True(), pbiR.setP2True(), pbiR.setP3True();
                        for (auto pbie : pbi)
                            pbiR = pbiR & pbie;
                        newNodeCoord = periodicInfo.GetCoordBackByBits(newNodeCoord, pbiR);
                    }
                }

                index curMinGlobiCell = iCellGlob;
                index curMinic2c = -1;
                for (int ic2c = 0; ic2c < c2c.size(); ic2c++)
                {
                    index iCellOther = c2c[ic2c];
                    if (iCellOther == iCell)
                        continue;
                    std::vector<index> c2nOther = meshO1.cell2node[iCellOther];
                    index iCellGlobOther = meshO1.cell2node.trans.pLGhostMapping->operator()(-1, iCellOther);
                    std::sort(c2nOther.begin(), c2nOther.end());
                    if (iCellGlobOther < curMinGlobiCell &&
                        std::includes(c2nOther.begin(), c2nOther.end(), spanNodesSrt.begin(), spanNodesSrt.begin() + eSpan.GetNumNodes()))
                    {
                        curMinGlobiCell = iCellGlobOther;
                        curMinic2c = ic2c;
                    }
                }
                if (curMinGlobiCell == iCellGlob)
                {
                    newCoords.push_back(newNodeCoord);
                    cellNewNodeRowV.at(iNNode) = (localNewNodeNum++); //*
                }
                else
                {
                    cellNewNodeRowV.at(iNNode) = (-1 - curMinic2c);
                }
            }
            cellNewNodes.ResizeRow(iCell, cellNewNodeRowV.size());
            cellNewNodes[iCell] = cellNewNodeRowV;
            // std::cout << "iCell " << iCell << ":  ";
            // for (auto i : cellNewNodeRowV)
            //     std::cout << i << ", ";
            // std::cout << std::endl;
        }
        // std::cout << fmt::format("Num NewNode {} ", localNewNodeNum) << std::endl;
        // for (auto v : newCoords)
        //     std::cout << v.transpose() << std::endl;
        index numNewNode;
        MPI::Allreduce(&localNewNodeNum, &numNewNode, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        if (mpi.rank == mRank)
            log() << fmt::format("=== Mesh Elevation: Num NewNode {} ", numNewNode) << std::endl;

        /**********************************/
        // build each proc coordO2 from cellNewNodes
        DNDS_MAKE_SSP(coords.father, mpi);
        DNDS_MAKE_SSP(coords.son, mpi);
        coords.father->Resize(meshO1.coords.father->Size() + localNewNodeNum);
        for (index iC = 0; iC < meshO1.coords.father->Size(); iC++)
            coords[iC] = meshO1.coords[iC];
        for (index iC = meshO1.coords.father->Size(); iC < meshO1.coords.father->Size() + localNewNodeNum; iC++)
            coords[iC] = newCoords.at(iC - meshO1.coords.father->Size());
        coords.father->createGlobalMapping();

        // tAdj1Pair nodeLocalIdxOld;
        // DNDS_MAKE_SSP(nodeLocalIdxOld.father, mpi);
        // DNDS_MAKE_SSP(nodeLocalIdxOld.son, mpi);
        // nodeLocalIdxOld.father->Resize(meshO1.coords.father->Size());
        // for (index iC = 0; iC < meshO1.coords.father->Size(); iC++)
        //     nodeLocalIdxOld[iC][0] = iC;
        // nodeLocalIdxOld.TransAttach();
        // nodeLocalIdxOld.trans.BorrowGGIndexing(mesh->cell2node.trans);
        // nodeLocalIdxOld.trans.createMPITypes();
        // nodeLocalIdxOld.trans.pullOnce();

        /**********************************/
        // from coordO2 get cellNewNodesGlobal, and get cellNewNodesGlobalPair
        for (index iCell = 0; iCell < meshO1.cell2node.father->Size(); iCell++)
        {
            auto cellNewNodesRow = cellNewNodes[iCell];
            for (auto &iNodeNew : cellNewNodesRow)
            {
                // nodes here must be at local proc; now use global idx
                iNodeNew =
                    iNodeNew < 0
                        ? iNodeNew // negative field was filled with indicating which cell to seek the node
                        : coords.father->pLGlobalMapping->operator()(mpi.rank, iNodeNew + meshO1.coords.father->Size());
            }
        }
        cellNewNodes.TransAttach();
        cellNewNodes.trans.BorrowGGIndexing(meshO1.cell2node.trans);
        cellNewNodes.trans.createMPITypes();
        cellNewNodes.trans.pullOnce();

        coords.TransAttach();
        std::vector<DNDS::index> ghostNodesTmp;
        for (index iCell = 0; iCell < meshO1.cell2node.Size(); iCell++)
        {
            auto cellNewNodesRow = cellNewNodes[iCell];
            for (auto iNodeNew : cellNewNodesRow)
            {
                if (iNodeNew < 0)
                    continue;
                MPI_int rank;
                index val;
                if (!coords.trans.pLGlobalMapping->search(iNodeNew, rank, val))
                    DNDS_assert_info(false, "search failed");
                if (rank != mpi.rank)
                    ghostNodesTmp.push_back(iNodeNew);
            }
        }
        coords.trans.createGhostMapping(ghostNodesTmp);
        coords.trans.createMPITypes();
        coords.trans.pullOnce();

        /**********************************/
        // each cell obtain new global cell2node global state with cellNewNodesGlobalPair
        DNDS_MAKE_SSP(cell2node.father, mpi);
        DNDS_MAKE_SSP(cell2node.son, mpi);
        DNDS_MAKE_SSP(cellElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        DNDS_MAKE_SSP(cellElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        if (isPeriodic)
        {
            DNDS_MAKE_SSP(cell2nodePbi.father, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), getMPI());
            DNDS_MAKE_SSP(cell2nodePbi.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), getMPI());
        }

        cell2node.father->Resize(meshO1.cell2node.father->Size());
        cellElemInfo.father->Resize(meshO1.cell2node.father->Size());
        if (isPeriodic)
            cell2nodePbi.father->Resize(meshO1.cell2node.father->Size());
        for (index iCell = 0; iCell < meshO1.cell2node.father->Size(); iCell++)
        {
            Elem::Element eCell = meshO1.GetCellElement(iCell);
            Elem::Element eCellO2 = eCell.ObtainElevatedElem();
            cell2node.father->ResizeRow(iCell, eCellO2.GetNumNodes());
            if (isPeriodic)
                cell2nodePbi.father->ResizeRow(iCell, eCellO2.GetNumNodes());
            auto c2n = meshO1.cell2node[iCell];
            for (int ic2n = 0; ic2n < c2n.size(); ic2n++)
            {
                // fill in the O1 nodes, note that global indices for O1 nodes have changed
                index iNodeOldGlobal = meshO1.coords.trans.pLGhostMapping->operator()(-1, c2n[ic2n]);
                index nodeOldOrigLocalIdx{-1};
                int nodeOldOrigRank{-1};
                if (!meshO1.coords.trans.pLGlobalMapping->search(iNodeOldGlobal, nodeOldOrigRank, nodeOldOrigLocalIdx))
                    DNDS_assert_info(false, "search failed");
                // nodeOldOrigRank and nodeOldOrigLocalIdx is same in new
                cell2node(iCell, ic2n) =
                    coords.father->pLGlobalMapping->operator()(nodeOldOrigRank, nodeOldOrigLocalIdx); // now point to global

                // fill in node pbi
                if (isPeriodic)
                    cell2nodePbi(iCell, ic2n) = meshO1.cell2nodePbi(iCell, ic2n);
            }
            for (int ic2n = c2n.size(); ic2n < cell2node[iCell].size(); ic2n++)
                cell2node(iCell, ic2n) = -1;

            cellElemInfo(iCell, 0) = meshO1.cellElemInfo(iCell, 0);
            cellElemInfo(iCell, 0).setElemType(eCellO2.type); // update cell elem info
        }
        // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
        for (index iCell = 0; iCell < meshO1.cell2node.father->Size(); iCell++)
        {
            Elem::Element eCell = meshO1.GetCellElement(iCell);
            auto c2n = meshO1.cell2node[iCell];
            auto c2c = meshO1.cell2cell[iCell];
            index iCellGlob = meshO1.cell2node.trans.pLGhostMapping->operator()(-1, iCell);

            // std::vector<index> cellNewNodeRowV;
            // cellNewNodeRowV.resize(eCell.GetNumElev_O1O2());
            SmallCoordsAsVector coordsC;
            meshO1.GetCoordsOnCell(iCell, coordsC);

            for (int iNNode = 0; iNNode < eCell.GetNumElev_O1O2(); iNNode++)
            {
                auto eSpan = eCell.ObtainElevNodeSpan(iNNode);
                std::array<index, Elem::CellNumNodeMax> spanNodes;
                eCell.ExtractElevNodeSpanNodes(iNNode, c2n, spanNodes);
                auto spanNodesSrt = spanNodes;
                std::sort(spanNodesSrt.begin(), spanNodesSrt.begin() + eSpan.GetNumNodes());
                tPoint newNodeCoord;
                { // the new node method, should cope with shape func
                    SmallCoordsAsVector coordsSpan;
                    coordsSpan.resize(3, eSpan.GetNumNodes());
                    eCell.ExtractElevNodeSpanNodes(iNNode, coordsC, coordsSpan);
                    newNodeCoord = coordsSpan.rowwise().mean();

                    // std::cout << fmt::format("############\niCell {}, iNNode {}", iCell, iNNode) << std::endl;
                    // std::cout << newNodeCoord.transpose() << std::endl;

                    if (isPeriodic)
                    {
                        std::vector<NodePeriodicBits> pbi;
                        pbi.resize(eSpan.GetNumNodes());
                        eCell.ExtractElevNodeSpanNodes(iNNode, meshO1.cell2nodePbi[iCell], pbi);
                        NodePeriodicBits pbiR;
                        pbiR.setP1True(), pbiR.setP2True(), pbiR.setP3True();
                        for (auto pbie : pbi)
                            pbiR = pbiR & pbie;
                        // std::cout << uint(uint8_t(pbiR)) << std::endl;
                        newNodeCoord = periodicInfo.GetCoordBackByBits(newNodeCoord, pbiR);
                        cell2nodePbi[iCell][c2n.size() + iNNode] = pbiR; //* fill In the O2 cell2nodePbi
                        // ** cell2nodePbiO2: edge using edge common, face using face common
                        // ** note that this technique requires at least 2 layer each periodic direction
                    }
                }
                if (cellNewNodes[iCell][iNNode] >= 0)
                {
                    cell2node[iCell][c2n.size() + iNNode] = cellNewNodes[iCell][iNNode];
                    continue;
                }

                int nFound = 0;
                real minSqrDist = veryLargeReal;
                for (auto iNNodeC : cellNewNodes[c2c[-1 - cellNewNodes[iCell][iNNode]]])
                {
                    if (iNNodeC < 0)
                        continue;
                    // iNNodeC is global iNode
                    MPI_int rank{-1};
                    index val{-1};
                    if (!coords.trans.pLGhostMapping->search_indexAppend(iNNodeC, rank, val))
                        DNDS_assert_info(false, "search failed");
                    // std::cout << fmt::format("val {}, iNNodeC {}", val, iNNodeC) << std::endl;
                    // std::cout << (coords[val] - newNodeCoord).squaredNorm() << std::endl;
                    // std::cout << coords[val].transpose() << ", " << newNodeCoord.transpose() << std::endl;
                    real sqrDist = (coords[val] - newNodeCoord).squaredNorm();
                    minSqrDist = std::min(sqrDist, minSqrDist);
                    if (sqrDist < epsSqrDist)
                    {
                        nFound++;
                        cell2node[iCell][c2n.size() + iNNode] = iNNodeC;
                    }
                }
                DNDS_assert_info(nFound == 1, fmt::format("geometric search for elevated point failed, nFound = {}, minSqrDist = {}, pos ({},{},{})", nFound, minSqrDist,
                                                          newNodeCoord(0), newNodeCoord(1), newNodeCoord(2)));
                //* comment: way of ridding of the geometric search:
                //* use interpolated topo: face/edge
                //* or record the full vertex string for each O2 nodes' span (O2 nodes do not have siblings in the same span)
            }
        }
        /**********************************/
        // cell2cell, bnd2cell can be copied; bnd2nodeO2 created with bnd2cell and bnd2node;
        // cell2cell = meshO1.cell2cell;
        // bnd2cell = meshO1.bnd2cell;
        DNDS_MAKE_SSP(cell2cell.father, mpi);
        DNDS_MAKE_SSP(cell2cell.son, mpi);
        cell2cell.father->Resize(meshO1.cell2cell.father->Size());
        for (index iCell = 0; iCell < meshO1.cell2node.father->Size(); iCell++)
        {
            cell2cell.father->ResizeRow(iCell, meshO1.cell2cell.RowSize(iCell));
            cell2cell[iCell] = std::vector<index>{meshO1.cell2cell[iCell]};
            for (auto &iCellOther : cell2cell[iCell])
            {
                iCellOther = meshO1.cell2cell.trans.pLGhostMapping->operator()(-1, iCellOther);
            }
        }
        DNDS_MAKE_SSP(bnd2cell.father, mpi);
        DNDS_MAKE_SSP(bnd2cell.son, mpi);
        bnd2cell.father->Resize(meshO1.bnd2cell.father->Size());
        for (index iBnd = 0; iBnd < meshO1.bnd2node.father->Size(); iBnd++)
        {
            bnd2cell(iBnd, 0) = meshO1.bnd2cell(iBnd, 0);
            bnd2cell(iBnd, 0) = meshO1.cell2cell.trans.pLGhostMapping->operator()(-1, bnd2cell(iBnd, 0));
        }

        DNDS_MAKE_SSP(bnd2node.father, mpi);
        DNDS_MAKE_SSP(bnd2node.son, mpi);
        DNDS_MAKE_SSP(bndElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        DNDS_MAKE_SSP(bndElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        bnd2node.father->Resize(meshO1.bnd2node.father->Size());
        bndElemInfo.father->Resize(meshO1.bnd2node.father->Size());
        for (index iBnd = 0; iBnd < meshO1.bnd2node.father->Size(); iBnd++)
        {
            index iCell = meshO1.bnd2cell(iBnd, 0); // my own bnd2cell is to global!
            auto eCellO2 = this->GetCellElement(iCell);
            auto b2n = meshO1.bnd2node[iBnd];
            auto c2nO2 = this->cell2node[iCell];
            std::vector<index> b2nv = b2n;
            for (auto &i : b2nv)
            {
                //* note that b2nv holds O1 nodes' old and local indices
                index iNodeOldGlobal = meshO1.coords.trans.pLGhostMapping->operator()(-1, i);
                index nodeOldOrigLocalIdx{-1};
                int nodeOldOrigRank{-1};
                if (!meshO1.coords.trans.pLGlobalMapping->search(iNodeOldGlobal, nodeOldOrigRank, nodeOldOrigLocalIdx))
                    DNDS_assert_info(false, "search failed");
                // nodeOldOrigRank and nodeOldOrigLocalIdx is same in new
                i = coords.father->pLGlobalMapping->operator()(nodeOldOrigRank, nodeOldOrigLocalIdx); // now point to global
            }
            std::sort(b2nv.begin(), b2nv.end());

            int nFound{0};
            int c2fFound{-1};
            std::vector<index> b2nO2Found;
            Elem::Element eBndFound;

            for (int ic2f = 0; ic2f < eCellO2.GetNumFaces(); ic2f++)
            {
                auto eFaceO2 = eCellO2.ObtainFace(ic2f);
                std::vector<index> f2nO2;
                f2nO2.resize(eFaceO2.GetNumNodes());
                eCellO2.ExtractFaceNodes(ic2f, c2nO2, f2nO2);
                std::sort(f2nO2.begin(), f2nO2.end());
                if (std::includes(f2nO2.begin(), f2nO2.end(), b2nv.begin(), b2nv.end()))
                {
                    nFound++;
                    c2fFound = ic2f;
                    b2nO2Found = f2nO2;
                    eBndFound = eFaceO2;
                }
            }
            DNDS_assert(nFound == 1);
            bnd2node.father->ResizeRow(iBnd, b2nO2Found.size());
            bnd2node[iBnd] = b2nO2Found;
            bndElemInfo(iBnd, 0) = meshO1.bndElemInfo(iBnd, 0);
            bndElemInfo(iBnd, 0).setElemType(eBndFound.type);
        }
        adjPrimaryState = Adj_PointToGlobal;

        DNDS_MAKE_SSP(coords.son, mpi);

        // this->BuildGhostPrimary();

        // this->AdjGlobal2LocalPrimary();
        // if (mpi.rank == 0)
        // {
        //     for (index iCell = 0; iCell < meshO1.cell2node.Size(); iCell++)
        //     {
        //         std::vector<index> v1 = meshO1.cell2node[iCell];
        //         std::vector<index> v2 = cell2node[iCell];
        //         std::cout << "iCell " << iCell << std::endl;
        //         for (auto i : v1)
        //             std::cout << i << "(" << meshO1.coords[i].transpose() << ")"
        //                       << ", ";
        //         std::cout << std::endl;
        //         for (auto i : v2)
        //             std::cout << i << "(" << coords[i].transpose() << ")"
        //                       << ", ";
        //         std::cout << std::endl;

        //         std::vector<index> v3 = meshO1.cell2cell[iCell];
        //         std::vector<index> v4 = cell2cell[iCell];
        //         for (auto i : v3)
        //             std::cout << i
        //                       << ", ";
        //         std::cout << std::endl;
        //         for (auto i : v4)
        //             std::cout << i
        //                       << ", ";
        //         std::cout << std::endl;
        //     }
        // }
        // // this->AdjLocal2GlobalPrimary();
    }

    static tPoint HermiteInterpolateMidPointOnLine2WithNorm(tPoint c0, tPoint c1, tPoint n0, tPoint n1)
    {
        tPoint c01 = c1 - c0;
        tPoint c01U = c01.stableNormalized();
        real c01Len = c01.stableNorm();
        tPoint t0 = -c01U.cross(n0).cross(n0);
        tPoint t1 = -c01U.cross(n1).cross(n1);
        t0 = t0.stableNormalized() * c01Len;
        t1 = t1.stableNormalized() * c01Len;
        if (n0.norm() == 0 && n1.norm() != 0)
            return 0.25 * c0 + 0.75 * c1 - 0.25 * t1;
        if (n0.norm() != 0 && n1.norm() == 0)
            return 0.75 * c0 + 0.25 * c1 + 0.25 * t0;
        if (n0.norm() == 0 && n1.norm() == 0)
            return 0.5 * (c0 + c1);
        return 0.5 * (c0 + c1) + 0.125 * (t0 - t1);
    }

    static tPoint HermiteInterpolateMidPointOnQuad4WithNorm(
        tPoint c0, tPoint c1, tPoint c2, tPoint c3,
        tPoint n0, tPoint n1, tPoint n2, tPoint n3)
    {
        tPoint c01 = HermiteInterpolateMidPointOnLine2WithNorm(c0, c1, n0, n1);
        tPoint c12 = HermiteInterpolateMidPointOnLine2WithNorm(c1, c2, n1, n2);
        tPoint c23 = HermiteInterpolateMidPointOnLine2WithNorm(c2, c3, n2, n3);
        tPoint c30 = HermiteInterpolateMidPointOnLine2WithNorm(c3, c0, n3, n0);
        return 0.5 * (c01 + c12 + c23 + c30) - 0.25 * (c0 + c1 + c2 + c3);
    }

    void UnstructuredMesh::ElevatedNodesGetBoundarySmooth(const std::function<bool(t_index)> &FiFBndIdNeedSmooth)
    {
        // if (mpi.rank == 1)
        //     for (index iCell = 0; iCell < cell2face.Size(); iCell++)
        //     {
        //         std::cout << iCell << ": ";
        //         for (auto i : cell2face[iCell])
        //             std::cout << i << ", ";
        //         std::cout << std::endl;
        //     }
        DNDS_assert(elevState == Elevation_O1O2);
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        DNDS_assert(face2node.father);

        DNDS_MAKE_SSP(coordsElevDisp.father, mpi);
        DNDS_MAKE_SSP(coordsElevDisp.son, mpi);
        coordsElevDisp.father->Resize(coords.father->Size());
        coordsElevDisp.TransAttach();
        coordsElevDisp.trans.BorrowGGIndexing(coords.trans);
        coordsElevDisp.trans.createMPITypes();

        for (index iN = 0; iN < coords.Size(); iN++)
            coordsElevDisp[iN].setConstant(largeReal);

        // tAdj1Pair nCoordBndNum;
        // DNDS_MAKE_SSP(nCoordBndNum.father, mpi);
        // DNDS_MAKE_SSP(nCoordBndNum.son, mpi);
        // nCoordBndNum.father->Resize(coords.father->Size());
        // nCoordBndNum.TransAttach();
        // nCoordBndNum.trans.BorrowGGIndexing(coords.trans);
        // nCoordBndNum.trans.createMPITypes();

        // for (index iN = 0; iN < coords.size(); iN++)
        //     nCoordBndNum[iN][0] = 0;

        /***********************************/
        // build faceExteded info
        tAdjPair face2nodeExtended;
        tElemInfoArrayPair faceElemInfoExtended;
        tPbiPair face2nodePbiExtended;
        face2nodeExtended.father = face2node.father;
        faceElemInfoExtended.father = faceElemInfo.father;
        if (isPeriodic)
            face2nodePbiExtended.father = face2nodePbi.father;
        DNDS_MAKE_SSP(face2nodeExtended.son, mpi);
        DNDS_MAKE_SSP(faceElemInfoExtended.son, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        if (isPeriodic)
            DNDS_MAKE_SSP(face2nodePbiExtended.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);

        this->AdjLocal2GlobalFacial();
        this->AdjLocal2GlobalC2F();

        std::vector<index> faceGhostExt;
        for (index iCell = 0; iCell < cell2face.Size(); iCell++)
            for (auto iFace : cell2face[iCell])
            {
                DNDS_assert_info(iFace >= 0 || iFace == UnInitIndex, fmt::format("iFace {}", iFace));
                if (iFace == UnInitIndex) // old cell2face could contain void pointing
                    continue;
                DNDS::MPI_int rank;
                DNDS::index val;
                if (!face2node.trans.pLGlobalMapping->search(iFace, rank, val))
                    DNDS_assert_info(false, "search failed");
                if (rank != mpi.rank)
                    faceGhostExt.push_back(iFace);
                // if (mpi.rank == 1)
                // std::cout << "added Face " << iFace << std::endl;
            }
        face2nodeExtended.TransAttach();
        faceElemInfoExtended.TransAttach();
        if (isPeriodic)
            face2nodePbiExtended.TransAttach();
        face2nodeExtended.trans.createGhostMapping(faceGhostExt);
        faceElemInfoExtended.trans.BorrowGGIndexing(face2nodeExtended.trans);
        if (isPeriodic)
            face2nodePbiExtended.trans.BorrowGGIndexing(face2nodeExtended.trans);
        face2nodeExtended.trans.createMPITypes();
        faceElemInfoExtended.trans.createMPITypes();
        if (isPeriodic)
            face2nodePbiExtended.trans.createMPITypes();
        face2nodeExtended.trans.pullOnce();
        faceElemInfoExtended.trans.pullOnce();
        if (isPeriodic)
            face2nodePbiExtended.trans.pullOnce();

        // std::cout << fmt::format("rank {} faceElemInfo {} {}, face2node {} {}",
        //                          mpi.rank,
        //                          faceElemInfo.father->Size(), faceElemInfo.son->Size(),
        //                          face2node.father->Size(), face2node.son->Size())
        //           << std::endl;
        // std::cout << fmt::format("rank {} faceElemInfoExt {} {}, face2nodeExt {} {}",
        //                          mpi.rank,
        //                          faceElemInfoExtended.father->Size(), faceElemInfoExtended.son->Size(),
        //                          face2nodeExtended.father->Size(), face2nodeExtended.son->Size())
        //           << std::endl;

        this->AdjGlobal2LocalFacial();
        this->AdjGlobal2LocalC2F();
        //** a direct copy from AdjGlobal2LocalPrimary()
        auto NodeIndexGlobal2Local = [&](DNDS::index &iNodeOther)
        {
            if (iNodeOther == UnInitIndex)
                return;
            DNDS::MPI_int rank;
            DNDS::index val;
            // if (!cell2cell.trans.pLGlobalMapping->search(iCellOther, rank, val))
            //     DNDS_assert_info(false, "search failed");
            // if (rank != mpi.rank)
            //     iCellOther = -1 - iCellOther;
            auto result = coords.trans.pLGhostMapping->search_indexAppend(iNodeOther, rank, val);
            if (result)
                iNodeOther = val;
            else
                iNodeOther = -1 - iNodeOther; // mapping to un-found in father-son
        };

        // std::cout << fmt::format("rank {} Sizes ext : extFaceProc: {}->{}", mpi.rank, faceElemInfo.Size(), faceElemInfoExtended.Size()) << std::endl;
        // if (mpi.rank == 1)
        //     for (index iFace = 0; iFace < face2nodeExtended.Size(); iFace++)
        //     {
        //         std::cout << fmt::format("rank {}, face {}", mpi.rank, face2nodeExtended.trans.pLGhostMapping->operator()(-1, iFace)) << std::endl;
        //     }

        // this->AdjGlobal2LocalPrimary();

        for (index iFace = face2nodeExtended.father->Size(); iFace < face2nodeExtended.Size(); iFace++)
        {
            for (auto &iN : face2nodeExtended[iFace])
            {
                DNDS_assert_info(iN >= 0, fmt::format("rank {}, iN {}", mpi.rank, iN)); // can't be unfound node
                NodeIndexGlobal2Local(iN);
                DNDS_assert_info(iN >= 0, fmt::format("rank {}, iN {}", mpi.rank, iN)); // can't be unfound node
            }
        }
        // std::cout << mpi.rank << " rank Here XXXX 1" << std::endl;
        /***********************************/
        // build nodeNormClusters
        using t3VecsPair = ArrayPair<ArrayEigenUniMatrixBatch<3, 1>>;
        t3VecsPair nodeNormClusters;
        DNDS_MAKE_SSP(nodeNormClusters.father, mpi);
        nodeNormClusters.father->Resize(coords.father->Size(), 3, 1);
        DNDS_MAKE_SSP(nodeNormClusters.son, mpi);

        std::vector<int> nodeBndNum(coords.father->Size(), 0); //? need row-appending methods for NonUniform arrays?
        for (index iFace = 0; iFace < face2nodeExtended.Size(); iFace++)
        {
            auto faceBndID = faceElemInfoExtended(iFace, 0).zone;
            if (!FiFBndIdNeedSmooth(faceBndID))
                continue;
            for (auto iNode : face2nodeExtended[iFace])
                if (iNode < coords.father->Size() && iNode < nNodeO1) //* being local node and O1 node
                    nodeBndNum.at(iNode)++;
        }
        // if (mpi.rank == 0)
        // {
        //     // std::cout << "faceBndID: \n";
        //     // for (index iFace = 0; iFace < face2nodeExtended.Size(); iFace++)
        //     // {
        //     //     auto faceBndID = faceElemInfoExtended(iFace, 0).zone;
        //     //     std::cout << faceBndID << ", " << FiFBndIdNeedSmooth(faceBndID) << "; "
        //     //               << coords[face2node(iFace, 0)].transpose()
        //     //               << " ||| " << coords[face2node(iFace, 1)].transpose() << std::endl;
        //     // }
        //     // std::cout << "XXXXXXXXXXXX" << std::endl;
        //     std::cout << "nodeBndNum: ";
        //     for (auto i : nodeBndNum)
        //         std::cout
        //             << i << ";";
        //     std::cout << "XXXXXXXXXXXX" << std::endl;
        // }

        // std::cout << mpi.rank << " rank Here XXXX 2" << std::endl;
        for (index iNode = 0; iNode < coords.father->Size(); iNode++)
            nodeNormClusters.father->ResizeBatch(iNode, std::max(nodeBndNum.at(iNode), 0));
        for (auto &v : nodeBndNum)
            v = 0; // set to zero
        // std::cout << mpi.rank << " rank Here XXXX 3" << std::endl;
        auto GetCoordsOnFaceExtended = [&](index iFace, tSmallCoords &cs)
        {
            if (!isPeriodic)
                __GetCoords(face2nodeExtended[iFace], cs);
            else
                __GetCoordsOnElem(face2nodeExtended[iFace], face2nodePbiExtended[iFace], cs);
        };

        for (index iFace = 0; iFace < face2nodeExtended.Size(); iFace++)
        {
            auto faceBndID = faceElemInfoExtended(iFace, 0).zone;
            if (!FiFBndIdNeedSmooth(faceBndID))
                continue;
            auto eFace = Elem::Element{faceElemInfoExtended(iFace, 0).getElemType()}; // O1 faces could use only one norm (not strict for Quad4)
            auto qFace = Elem::Quadrature{eFace, 1};
            real faceArea{0};
            tPoint uNorm;
            tSmallCoords coordsF;
            GetCoordsOnFaceExtended(iFace, coordsF);
            qFace.Integration(
                faceArea,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coordsF, DiNj);
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsF, DiNj);
                    tPoint np;
                    if (dim == 2)
                        np = FacialJacobianToNormVec<2>(J);
                    else
                        np = FacialJacobianToNormVec<3>(J);
                    // np.stableNormalize(); // do not normalize to preserve face area
                    uNorm = np;
                    vInc = np.norm();
                });

            for (int if2n = 0; if2n < face2nodeExtended[iFace].size(); if2n++)
            {
                index iNode = face2nodeExtended[iFace][if2n];
                if (iNode < coords.father->Size() && iNode < nNodeO1) //* being local node and O1 node
                {
                    tPoint uNormC = uNorm;
                    if (isPeriodic)
                        uNormC = periodicInfo.GetVectorBackByBits<3, 1>(uNorm, face2nodePbiExtended[iFace][if2n]);
                    nodeNormClusters(iNode, nodeBndNum.at(iNode)++) = uNormC;
                }
            }
        }
        nodeNormClusters.TransAttach();
        nodeNormClusters.trans.BorrowGGIndexing(coords.trans);
        nodeNormClusters.trans.createMPITypes();
        MPI_Barrier(mpi.comm);
        nodeNormClusters.trans.pullOnce();

        /***********************************/
        // do interpolation
        tPoint bboxMove;
        bboxMove.setZero();
        index nMoved{0};
        std::unordered_set<index> moveds;
        for (index iFace = 0; iFace < face2nodeExtended.Size(); iFace++)
        {
            auto faceBndID = faceElemInfoExtended(iFace, 0).zone;
            // TODO: add marking for non-moving bounaries
            if (!FiFBndIdNeedSmooth(faceBndID))
            {
                if (!FaceIDIsInternal(faceBndID)) // * note don't operate on internal faces
                    for (auto iN : face2nodeExtended[iFace])
                        if (coordsElevDisp[iN](0) == largeReal && coordsElevDisp[iN](2) != 2 * largeReal)
                            coordsElevDisp[iN](2) = 3 * largeReal; // marking other boundary nodes
                continue;
            }
            // for (auto iN : face2nodeExtended[iFace])
            //     coordsElevDisp[iN](2) = 2 * largeReal;                                // marking O1 nodes
            auto eFace = Elem::Element{faceElemInfoExtended(iFace, 0).getElemType()}; // O1 faces could use only one norm (not strict for Quad4)
            auto qFace = Elem::Quadrature{eFace, 1};
            Elem::SummationNoOp noOp;
            tPoint uNorm;
            SmallCoordsAsVector coordsF;
            GetCoordsOnFaceExtended(iFace, coordsF);
            qFace.Integration(
                noOp,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coordsF, DiNj);
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsF, DiNj);
                    tPoint np;
                    if (dim == 2)
                        np = FacialJacobianToNormVec<2>(J);
                    else
                        np = FacialJacobianToNormVec<3>(J);
                    np.stableNormalize();
                    uNorm = np;
                });

            auto eFaceO1 = eFace.ObtainO1Elem();

            auto f2n = face2nodeExtended[iFace];

            std::vector<int> f2nVecSeq(f2n.size());
            for (int i = 0; i < f2n.size(); i++)
                f2nVecSeq[i] = i;
            for (int if2n = 0; if2n < eFaceO1.GetNumNodes(); if2n++)
            {
                index iNode = f2n[if2n];
                coordsElevDisp[iNode](2) = 2 * largeReal;
            }

            for (int if2n = eFaceO1.GetNumNodes(); if2n < f2n.size(); if2n++)
            {
                index iNode = f2n[if2n];
                if (iNode >= coords.father->Size())
                    continue;
                DNDS_assert(iNode >= nNodeO1);

                std::vector<index> spanO2Node;
                std::vector<int> spanO2if2n;
                SmallCoordsAsVector coordsSpan;
                int iElev = if2n - eFaceO1.GetNumNodes();
                int spanNnode = eFaceO1.ObtainElevNodeSpan(iElev).GetNumNodes();
                spanO2Node.resize(spanNnode);
                coordsSpan.resize(3, spanNnode);
                spanO2if2n.resize(spanNnode);

                eFaceO1.ExtractElevNodeSpanNodes(iElev, f2n, spanO2Node); // should use O1f2n, but O2f2n is equivalent
                eFaceO1.ExtractElevNodeSpanNodes(iElev, coordsF, coordsSpan);
                eFaceO1.ExtractElevNodeSpanNodes(iElev, f2nVecSeq, spanO2if2n);
                tPoint cooUnsmooth = coordsF(Eigen::all, if2n);

                std::vector<tPoint> edges;
                if (spanO2Node.size() == 2)
                {
                    edges.push_back((coordsSpan[1] - coordsSpan[0]).stableNormalized());
                }
                else if (spanO2Node.size() == 4)
                {
                    edges.push_back((coordsSpan[1] - coordsSpan[0]).stableNormalized());
                    edges.push_back((coordsSpan[2] - coordsSpan[1]).stableNormalized());
                    edges.push_back((coordsSpan[3] - coordsSpan[2]).stableNormalized());
                    edges.push_back((coordsSpan[0] - coordsSpan[3]).stableNormalized());
                }
                else
                    DNDS_assert(false);

                Eigen::Matrix<real, 3, 4> norms;
                Eigen::Vector<real, 4> nAdd;
                nAdd.setZero();
                norms.setZero();
                // if (mpi.rank == 1)
                //     std::cout << nodeNormClusters.RowSize(spanO2Node[0]) << " nodeNormClustersRowSize" << std::endl;

                for (int iN = 0; iN < spanO2Node.size(); iN++)
                {
                    real sinValFFMax{0};
                    for (int iNorm = 0; iNorm < nodeNormClusters.RowSize(spanO2Node[iN]); iNorm++)
                        for (int jNorm = 0; jNorm < nodeNormClusters.RowSize(spanO2Node[iN]); jNorm++)
                            sinValFFMax = std::max(
                                sinValFFMax,
                                std::sqrt(1 - sqr(std::abs(nodeNormClusters(spanO2Node[iN], iNorm)
                                                               .stableNormalized()
                                                               .dot(nodeNormClusters(spanO2Node[iN], jNorm).stableNormalized())))));
                    for (int iNorm = 0; iNorm < nodeNormClusters.RowSize(spanO2Node[iN]); iNorm++)
                    {
                        tPoint normN = nodeNormClusters(spanO2Node[iN], iNorm);
                        if (isPeriodic)
                            normN = periodicInfo.GetVectorByBits<3, 1>(normN, face2nodePbiExtended[iFace][if2n]);
                        real sinValMax = 0;
                        for (auto e : edges)
                            sinValMax = std::max(sinValMax, std::abs(e.dot(normN.stableNormalized())));
                        //! angle is here
                        //* now using FFMaxAngle also
                        if (std::max(sinValMax, sinValFFMax * 0) > std::sin(pi / 180. * elevationInfo.MaxIncludedAngle))
                            continue;
                        nAdd[iN] += 1.;
                        // norms(Eigen::all, iN) += normN * 1.;
                        norms(Eigen::all, iN) += normN.stableNormalized();

                        // std::cout << fmt::format("iFace {}, if2n {}, iN {}, iNorm{}, sinValMax {}; ====",
                        // iFace, if2n, iN, iNorm, sinValMax) << normN.transpose() << std::endl;
                    }
                }
                // norms.array().rowwise() /= nAdd.transpose().array();
                norms.colwise().normalize();
                for (int iN = 0; iN < spanO2Node.size(); iN++)
                {
                    if (nAdd[iN] < 0.1) // no found, then no mov
                        norms(Eigen::all, iN).setZero();
                    // DNDS_assert(nAdd[iN] > 0.1);
                }

                tPoint cooSmooth;
                tPoint cooInc;
                if (spanO2Node.size() == 2)
                {
                    cooSmooth = HermiteInterpolateMidPointOnLine2WithNorm(
                        coordsSpan[0], coordsSpan[1], norms(Eigen::all, 0), norms(Eigen::all, 1));
                    cooInc = cooSmooth - cooUnsmooth;
                }
                else // definitely is spanO2Node.size() == 4
                {
                    f2n[if2n - 1];
                    DNDS_assert(f2n.size() == 9); // has to be a Quad9
                    cooInc = 0.5 * (coordsElevDisp[f2n[if2n - 4]] +
                                    coordsElevDisp[f2n[if2n - 3]] +
                                    coordsElevDisp[f2n[if2n - 2]] +
                                    coordsElevDisp[f2n[if2n - 1]]);
                    if (isPeriodic)
                    {
                        cooInc =
                            0.5 *
                            (periodicInfo.GetVectorByBits<3, 1>(coordsElevDisp[f2n[if2n - 4]], face2nodePbiExtended[iFace][if2n - 4]) +
                             periodicInfo.GetVectorByBits<3, 1>(coordsElevDisp[f2n[if2n - 3]], face2nodePbiExtended[iFace][if2n - 3]) +
                             periodicInfo.GetVectorByBits<3, 1>(coordsElevDisp[f2n[if2n - 2]], face2nodePbiExtended[iFace][if2n - 2]) +
                             periodicInfo.GetVectorByBits<3, 1>(coordsElevDisp[f2n[if2n - 1]], face2nodePbiExtended[iFace][if2n - 1]));
                    }

                    cooSmooth = HermiteInterpolateMidPointOnQuad4WithNorm(
                        coordsSpan[0], coordsSpan[1], coordsSpan[2], coordsSpan[3],
                        norms(Eigen::all, 0), norms(Eigen::all, 1), norms(Eigen::all, 2), norms(Eigen::all, 3));

                    cooInc = cooSmooth - cooUnsmooth;
                }

                if (isPeriodic)
                    cooInc = periodicInfo.GetVectorBackByBits<3, 1>(cooInc, face2nodePbiExtended[iFace][if2n]);
                // if (cooInc.stableNorm() > 0) //! could use a threshold
                {

                    coordsElevDisp[iNode] = cooInc * 1; // maybe output the increment directly?
                    nMoved++;
                    moveds.insert(iNode);
                    bboxMove = bboxMove.array().max(cooInc.array().abs());
                }

                // std::cout << mpi.rank << " rank; iNode " << iNode << " alterwith " << coordsElevDisp[iNode].transpose() << std::endl;
            }
        }
        tPoint bboxMoveT;
        MPI::Allreduce(bboxMove.data(), bboxMoveT.data(), 3, DNDS_MPI_REAL, MPI_MAX, mpi.comm);

        coordsElevDisp.trans.pullOnce();
        nMoved = moveds.size();
        MPI::Allreduce(&nMoved, &nTotalMoved, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        if (mpi.rank == mRank)
            log() << fmt::format(
                         "UnstructuredMesh === ElevatedNodesGetBoundarySmooth: Smoothing Complete, total Moved [{}], moving Vec Bnd {:.2g},{:.2g},{:.2g}",
                         nTotalMoved, bboxMoveT(0), bboxMoveT(1), bboxMoveT(2))
                  << std::endl;

        // std::cout << mpi.rank << " rank Here XXXX -1" << std::endl;
    }

    struct CoordPairDOF : public tCoordPair
    {
        real dot(CoordPairDOF &R)
        {
            real ret = 0;
            for (index i = 0; i < this->father->Size(); i++)
                ret += (*this)[i].dot(R[i]);
            real retSum;
            MPI::Allreduce(&ret, &retSum, 1, DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            return retSum;
        }

        real norm2()
        {
            return std::sqrt(this->dot(*this));
        }

        void addTo(CoordPairDOF &R, real alpha)
        {
            for (index i = 0; i < this->Size(); i++)
                (*this)[i] += R[i] * alpha;
        }

        void setConstant(real v)
        {
            for (index i = 0; i < this->Size(); i++)
                (*this)[i].setConstant(v);
        }

        void operator=(CoordPairDOF &R)
        {
            for (index i = 0; i < this->Size(); i++)
                (*this)[i] = R[i];
        }

        void operator*=(real r)
        {
            for (index i = 0; i < this->Size(); i++)
                (*this)[i] *= r;
        }
    };

    struct PointCloudKDTreeCoordPair
    {
        tCoord ref;
        using coord_t = real; //!< The type of each coordinate
        PointCloudKDTreeCoordPair(tCoord &v)
        {
            ref = v;
        }

        // Must return the number of data points
        inline size_t
        kdtree_get_point_count() const
        {
            DNDS_assert(ref);
            return ref->Size();
        }

        inline real kdtree_get_pt(const size_t idx, const size_t dim) const
        {
            DNDS_assert(ref);
            return ref->operator[](idx)(dim);
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX & /* bb */) const
        {
            return false;
        }
    };

    void UnstructuredMesh::ElevatedNodesSolveInternalSmooth()
    {
        DNDS_assert(elevState == Elevation_O1O2);
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        DNDS_assert(face2node.father);
        DNDS_assert(nTotalMoved >= 0);
        if (!nTotalMoved)
        {
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth() early exit for no nodes were moved";
            return;
        }

        DNDS_MPI_InsertCheck(mpi, "Got node2node");

        coordsElevDisp.trans.initPersistentPull(); // only holds local nodes

        std::unordered_set<index> nodesBoundInterpolated;
        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (coordsElevDisp[iN](0) != largeReal || coordsElevDisp[iN](2) == 2 * largeReal)
            {
                nodesBoundInterpolated.insert(iN);
            }
        }

        tCoordPair boundInterpCoo;
        tCoordPair boundInterpVal;
        DNDS_MAKE_SSP(boundInterpCoo.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoo.son, mpi);
        DNDS_MAKE_SSP(boundInterpVal.father, mpi);
        DNDS_MAKE_SSP(boundInterpVal.son, mpi);

        boundInterpCoo.father->Resize(nodesBoundInterpolated.size());
        boundInterpVal.father->Resize(nodesBoundInterpolated.size());

        index top{0};
        for (auto iN : nodesBoundInterpolated)
        {
            boundInterpCoo[top] = coords[iN];
            boundInterpVal[top] = (coordsElevDisp[iN](0) != largeReal) ? tPoint{coordsElevDisp[iN]} : tPoint::Zero();
            top++;
        }
        std::vector<index> boundInterpPullIdx;

        index boundInterpGlobSize = boundInterpCoo.father->globalSize();
        boundInterpPullIdx.resize(boundInterpGlobSize);
        for (index i = 0; i < boundInterpPullIdx.size(); i++)
            boundInterpPullIdx[i] = i;
        boundInterpCoo.father->createGlobalMapping();
        boundInterpCoo.TransAttach();
        boundInterpCoo.trans.createGhostMapping(boundInterpPullIdx);
        boundInterpCoo.trans.createMPITypes();
        boundInterpCoo.trans.pullOnce();

        boundInterpVal.TransAttach();
        boundInterpVal.trans.BorrowGGIndexing(boundInterpCoo.trans);
        boundInterpVal.trans.createMPITypes();
        boundInterpVal.trans.pullOnce();

        if (mpi.rank == mRank)
            log() << "RBF set: " << boundInterpCoo.son->Size() << std::endl;

        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTreeCoordPair>,
            PointCloudKDTreeCoordPair,
            3,
            index>;
        auto coordsI = PointCloudKDTreeCoordPair(boundInterpCoo.son);
        // for (index iF = 0; iF < boundInterpCoo.son->Size(); iF++)
        // {
        //     std::cout << boundInterpVal.sfaon->operator[](iF).transpose() << std::endl;
        // }
        kdtree_t bndInterpTree(3, coordsI);

        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (nodesBoundInterpolated.count(iN))
                continue;

            index nFind = elevationInfo.nSearch;
            tPoint cooC = coords[iN];

            std::vector<index> idxFound;
            Eigen::Vector<real, Eigen::Dynamic> outDistancesSqr;
            idxFound.resize(nFind);
            outDistancesSqr.resize(nFind);

            index nFound = bndInterpTree.knnSearch(cooC.data(), nFind, idxFound.data(), outDistancesSqr.data());
            DNDS_assert(nFound >= 1);
            idxFound.resize(nFound);
            outDistancesSqr.resize(nFound);
            tSmallCoords coordBnd;
            tSmallCoords dispBnd;
            coordBnd.resize(3, nFound);
            dispBnd.resize(3, nFound);
            for (index iF = 0; iF < nFound; iF++)
            {
                coordBnd(Eigen::all, iF) = (*boundInterpCoo.son)[idxFound[iF]];
                dispBnd(Eigen::all, iF) = (*boundInterpVal.son)[idxFound[iF]];
            }
            real RMin = std::sqrt(outDistancesSqr.minCoeff());
            tPoint dispC;
            dispC.setZero();

            if (RMin < sqr(elevationInfo.RBFRadius * (KernelIsCompact(elevationInfo.kernel) ? 1 : 5)))
            {
                tPoint coordBndC = coordBnd.rowwise().mean();
                tSmallCoords coordBndRel = coordBnd.colwise() - coordBndC;
                Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>
                    coefs = RBF::RBFInterpolateSolveCoefsNoPoly(coordBndRel, dispBnd.transpose(), elevationInfo.RBFRadius, elevationInfo.kernel);
                tSmallCoords qs;
                qs.resize(3, 1);
                qs = cooC - coordBndC;
                tPoint dCur =
                    (RBF::RBFCPC2(coordBndRel, qs, elevationInfo.RBFRadius, elevationInfo.kernel).transpose() * coefs.topRows(coordBndRel.cols()) +
                     coefs(Eigen::last - 3, Eigen::all) +
                     qs.transpose() * coefs.bottomRows(3))
                        .transpose();
                dispC = dCur;
            }

            coordsElevDisp[iN] = dispC;
        }

        coordsElevDisp.trans.startPersistentPull();
        coordsElevDisp.trans.waitPersistentPull();

        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            // if(dim == 2)
            //     dispO2[iN](2) = 0;
            // if(dispO2[iN].norm() != 0)
            //     std::cout << dispO2[iN].transpose() << std::endl;
            if (coordsElevDisp[iN](0) != largeReal)
                coords[iN] += coordsElevDisp[iN];
        }
        coords.trans.pullOnce();
    }

    void UnstructuredMesh::ElevatedNodesSolveInternalSmoothV1Old()
    {
        DNDS_assert(elevState == Elevation_O1O2);
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        DNDS_assert(face2node.father);
        DNDS_assert(nTotalMoved >= 0);
        if (!nTotalMoved)
        {
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth() early exit for no nodes were moved";
            return;
        }

        DNDS_MPI_InsertCheck(mpi, "Got node2node");

        coordsElevDisp.trans.initPersistentPull(); // only holds local nodes

        std::unordered_set<index> nodesBoundInterpolated;
        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (coordsElevDisp[iN](0) != largeReal || coordsElevDisp[iN](2) == 2 * largeReal)
            {
                nodesBoundInterpolated.insert(iN);
            }
        }

        tCoordPair boundInterpCoo;
        tCoordPair boundInterpVal;
        DNDS_MAKE_SSP(boundInterpCoo.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoo.son, mpi);
        DNDS_MAKE_SSP(boundInterpVal.father, mpi);
        DNDS_MAKE_SSP(boundInterpVal.son, mpi);

        boundInterpCoo.father->Resize(nodesBoundInterpolated.size());
        boundInterpVal.father->Resize(nodesBoundInterpolated.size());

        index top{0};
        for (auto iN : nodesBoundInterpolated)
        {
            boundInterpCoo[top] = coords[iN];
            boundInterpVal[top] = (coordsElevDisp[iN](0) != largeReal) ? tPoint{coordsElevDisp[iN]} : tPoint::Zero();
            top++;
        }
        std::vector<index> boundInterpPullIdx;

        index boundInterpGlobSize = boundInterpCoo.father->globalSize();
        boundInterpPullIdx.resize(boundInterpGlobSize);
        for (index i = 0; i < boundInterpPullIdx.size(); i++)
            boundInterpPullIdx[i] = i;
        boundInterpCoo.father->createGlobalMapping();
        boundInterpCoo.TransAttach();
        boundInterpCoo.trans.createGhostMapping(boundInterpPullIdx);
        boundInterpCoo.trans.createMPITypes();
        boundInterpCoo.trans.pullOnce();

        boundInterpVal.TransAttach();
        boundInterpVal.trans.BorrowGGIndexing(boundInterpCoo.trans);
        boundInterpVal.trans.createMPITypes();
        boundInterpVal.trans.pullOnce();

        if (mpi.rank == mRank)
            log() << "RBF set: " << boundInterpCoo.son->Size() << std::endl;

        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTreeCoordPair>,
            PointCloudKDTreeCoordPair,
            3,
            index>;
        using kdtree_tcoo = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTree>,
            PointCloudKDTree,
            3,
            index>;
        auto coordsI = PointCloudKDTreeCoordPair(boundInterpCoo.son);
        kdtree_t bndInterpTree(3, coordsI);

        tCoordPair boundInterpCoef;
        using tScalarPair = DNDS::ArrayPair<DNDS::ParArray<real, 1>>;
        tScalarPair boundInterpR;

        DNDS_MAKE_SSP(boundInterpCoef.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoef.son, mpi);
        DNDS_MAKE_SSP(boundInterpR.father, mpi);
        DNDS_MAKE_SSP(boundInterpR.son, mpi);
        boundInterpCoef.father->Resize(mpi.rank == mRank ? boundInterpGlobSize : 0);
        boundInterpR.father->Resize(mpi.rank == mRank ? boundInterpGlobSize : 0);

        if (mpi.rank == mRank) // that dirty work
        {

            Eigen::SparseMatrix<real> M(boundInterpGlobSize, boundInterpGlobSize);
            M.reserve(Eigen::Vector<index, Eigen::Dynamic>::Constant(boundInterpGlobSize, elevationInfo.nSearch));
            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> f;
            f.resize(boundInterpGlobSize, 3);
            for (index iN = 0; iN < boundInterpGlobSize; iN++)
            {
                std::cout << iN << std::endl;
                tPoint cooC = (*boundInterpCoo.son)[iN];

                index nFind = elevationInfo.nSearch;
                std::vector<index> idxFound;
                Eigen::Vector<real, Eigen::Dynamic> outDistancesSqr;
                idxFound.resize(nFind);
                outDistancesSqr.resize(nFind);
                index nFound = bndInterpTree.knnSearch(cooC.data(), nFind, idxFound.data(), outDistancesSqr.data());
                DNDS_assert(nFound >= 1);
                idxFound.resize(nFound);
                outDistancesSqr.resize(nFound);

                real RMin = outDistancesSqr.minCoeff();
                real RMax = outDistancesSqr.maxCoeff();
                real RRBF = elevationInfo.RBFRadius * std::sqrt(RMax);
                boundInterpR(iN, 0) = RRBF;
                DNDS_assert(RRBF > 0);
                Eigen::Vector<real, Eigen::Dynamic> outDists = outDistancesSqr.array().sqrt() * (1. / RRBF);
                auto fBasis = RBF::FRBFBasis(outDists, elevationInfo.kernel);
                for (index in2n = 0; in2n < nFound; in2n++)
                    M.insert(iN, idxFound[in2n]) = fBasis(in2n);

                f(iN, Eigen::all) = (*boundInterpVal.son)[iN].transpose();
            }

            log() << "RBF assembled: " << boundInterpCoo.son->Size() << std::endl;
            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> coefs;
            M.makeCompressed();
            Eigen::SparseLU<Eigen::SparseMatrix<real>, Eigen::COLAMDOrdering<int>> LUSolver;
            LUSolver.analyzePattern(M);
            LUSolver.factorize(M);
            coefs = LUSolver.solve(f);
            for (index iN = 0; iN < boundInterpGlobSize; iN++)
                boundInterpCoef[iN] = coefs(iN, Eigen::all).transpose();
        }

        boundInterpCoef.father->createGlobalMapping();
        boundInterpCoef.TransAttach();
        boundInterpCoef.trans.createGhostMapping(boundInterpPullIdx);
        boundInterpCoef.trans.createMPITypes();
        boundInterpCoef.trans.pullOnce();

        boundInterpR.TransAttach();
        boundInterpR.trans.BorrowGGIndexing(boundInterpCoef.trans);
        boundInterpR.trans.createMPITypes();
        boundInterpR.trans.pullOnce();

        if (mpi.rank == mRank)
        {
            for (index iN = 0; iN < boundInterpGlobSize; iN++)
            {
                std::cout << "coef: " << iN << " " << boundInterpCoef.son->operator[](iN).transpose() << std::endl;
            }
        }

        PointCloudKDTree insidePts;
        insidePts.pts.reserve(coords.father->Size() - nodesBoundInterpolated.size());
        std::vector<index> insideNodes;
        insideNodes.reserve(insidePts.pts.size());
        for (index iN = 0; iN < coords.father->Size(); iN++)
            if (!nodesBoundInterpolated.count(iN))
                insidePts.pts.push_back(coords[iN]), insideNodes.push_back(iN), coordsElevDisp[iN].setZero();
        kdtree_tcoo nodesDstTree(3, insidePts);
        for (index iN = 0; iN < boundInterpGlobSize; iN++)
        {

            tPoint cooC = (*boundInterpCoo.son)[iN];
            std::vector<std::pair<DNDS::index, DNDS::real>> IndicesDists;
            IndicesDists.reserve(elevationInfo.nSearch * 5);
            real RRBF = boundInterpR.son->operator()(iN, 0);
            nanoflann::SearchParams params{}; // default params
            index nFound = nodesDstTree.radiusSearch(cooC.data(), RRBF, IndicesDists, params);
            Eigen::Vector<real, Eigen::Dynamic> outDists;
            outDists.resize(IndicesDists.size());
            DNDS_assert(RRBF > 0);
            for (index i = 0; i < IndicesDists.size(); i++)
                outDists[i] = std::sqrt(IndicesDists[i].second) / RRBF;

            auto fBasis = RBF::FRBFBasis(outDists, elevationInfo.kernel);
            for (index i = 0; i < IndicesDists.size(); i++)
            {
                coordsElevDisp[insideNodes[IndicesDists[i].first]] +=
                    fBasis(i, 0) *
                    boundInterpCoef.son->operator[](iN);
            }
        }

        // for (index iN = 0; iN < coords.father->Size(); iN++)
        // {
        //     if (nodesBoundInterpolated.count(iN))
        //         continue;

        //     index nFind = elevationInfo.nSearch * 5; // for safety
        //     tPoint cooC = coords[iN];

        //     std::vector<index> idxFound;
        //     Eigen::Vector<real, Eigen::Dynamic> outDistancesSqr;
        //     idxFound.resize(nFind);
        //     outDistancesSqr.resize(nFind);

        //     index nFound = bndInterpTree.knnSearch(cooC.data(), nFind, idxFound.data(), outDistancesSqr.data());
        //     DNDS_assert(nFound >= 1);
        //     idxFound.resize(nFound);
        //     outDistancesSqr.resize(nFound);

        //     Eigen::Vector<real, Eigen::Dynamic> foundRRBFs;
        //     foundRRBFs.resize(nFound);
        //     tSmallCoords coefsC;
        //     coefsC.resize(3, nFound);
        //     for (index iF = 0; iF < nFound; iF++)
        //         foundRRBFs[iF] = (*boundInterpR.son)(idxFound[iF], 0),
        //         coefsC(Eigen::all, iF) = (*boundInterpCoef.son)[idxFound[iF]].transpose();
        //     Eigen::Vector<real, Eigen::Dynamic> outDists = outDistancesSqr.array().sqrt() / foundRRBFs.array();
        //     auto fBasis = RBF::FRBFBasis(outDists, elevationInfo.kernel);
        //     tPoint dispC = coefsC * fBasis;
        //     coordsElevDisp[iN] = dispC;
        // }

        coordsElevDisp.trans.startPersistentPull();
        coordsElevDisp.trans.waitPersistentPull();

        for (index iN = nNodeO1; iN < coords.father->Size(); iN++)
        {
            // if(dim == 2)
            //     dispO2[iN](2) = 0;
            // if(dispO2[iN].norm() != 0)
            //     std::cout << dispO2[iN].transpose() << std::endl;
            if (coordsElevDisp[iN](0) != largeReal)
                coords[iN] += coordsElevDisp[iN];
        }
        coords.trans.pullOnce();
    }

    void UnstructuredMesh::ElevatedNodesSolveInternalSmoothV1()
    {
        DNDS_assert(elevState == Elevation_O1O2);
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        DNDS_assert(face2node.father);
        DNDS_assert(nTotalMoved >= 0);
        if (!nTotalMoved)
        {
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth() early exit for no nodes were moved";
            return;
        }

        DNDS_MPI_InsertCheck(mpi, "Got node2node");

        coordsElevDisp.trans.initPersistentPull(); // only holds local nodes

        std::unordered_set<index> nodesBoundInterpolated;
        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (coordsElevDisp[iN](0) != largeReal || coordsElevDisp[iN](2) == 2 * largeReal)
            {
                nodesBoundInterpolated.insert(iN);
            }
        }

        tCoordPair boundInterpCoo;
        tCoordPair boundInterpVal;
        DNDS_MAKE_SSP(boundInterpCoo.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoo.son, mpi);
        DNDS_MAKE_SSP(boundInterpVal.father, mpi);
        DNDS_MAKE_SSP(boundInterpVal.son, mpi);

        boundInterpCoo.father->Resize(nodesBoundInterpolated.size());
        boundInterpVal.father->Resize(nodesBoundInterpolated.size());

        index top{0};
        for (auto iN : nodesBoundInterpolated)
        {
            boundInterpCoo[top] = coords[iN];
            boundInterpVal[top] = (coordsElevDisp[iN](0) != largeReal) ? tPoint{coordsElevDisp[iN]} : tPoint::Zero();
            top++;
        }
        std::vector<index> boundInterpPullIdx;

        index boundInterpGlobSize = boundInterpCoo.father->globalSize();
        index boundInterpLocSize = nodesBoundInterpolated.size();
        boundInterpPullIdx.resize(boundInterpGlobSize);
        for (index i = 0; i < boundInterpPullIdx.size(); i++)
            boundInterpPullIdx[i] = i;
        boundInterpCoo.father->createGlobalMapping();
        boundInterpCoo.TransAttach();
        boundInterpCoo.trans.createGhostMapping(boundInterpPullIdx);
        boundInterpCoo.trans.createMPITypes();
        boundInterpCoo.trans.pullOnce();

        boundInterpVal.TransAttach();
        boundInterpVal.trans.BorrowGGIndexing(boundInterpCoo.trans);
        boundInterpVal.trans.createMPITypes();
        boundInterpVal.trans.pullOnce();

        if (mpi.rank == mRank)
            log() << "RBF set: " << boundInterpCoo.son->Size() << std::endl;

        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTreeCoordPair>,
            PointCloudKDTreeCoordPair,
            3,
            index>;
        using kdtree_tcoo = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTree>,
            PointCloudKDTree,
            3,
            index>;
        auto coordsI = PointCloudKDTreeCoordPair(boundInterpCoo.son);
        kdtree_t bndInterpTree(3, coordsI);

        CoordPairDOF boundInterpCoef, boundInterpCoefRHS;
        using tScalarPair = DNDS::ArrayPair<DNDS::ParArray<real, 1>>;
        tScalarPair boundInterpR;
        std::cout << "HEre-2" << std::endl;
        DNDS_MAKE_SSP(boundInterpCoef.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoef.son, mpi);
        DNDS_MAKE_SSP(boundInterpCoefRHS.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoefRHS.son, mpi);
        DNDS_MAKE_SSP(boundInterpR.father, mpi);
        DNDS_MAKE_SSP(boundInterpR.son, mpi);
        boundInterpCoef.father->Resize(boundInterpCoo.father->Size());
        boundInterpCoefRHS.father->Resize(boundInterpCoo.father->Size());
        boundInterpR.father->Resize(boundInterpCoo.father->Size());
        std::vector<std::vector<std::pair<index, real>>> MatC;
        std::vector<index> boundInterpPullingIndexSolving;
        MatC.resize(boundInterpLocSize);
        std::cout << "HEre-1" << std::endl;
        for (index iN = 0; iN < boundInterpLocSize; iN++)
        {
            tPoint cooC = (*boundInterpCoo.father)[iN];

            index nFind = elevationInfo.nSearch;
            std::vector<index> idxFound;
            Eigen::Vector<real, Eigen::Dynamic> outDistancesSqr;

            /**********************************************/
            idxFound.resize(nFind);
            outDistancesSqr.resize(nFind);
            index nFound = bndInterpTree.knnSearch(cooC.data(), nFind, idxFound.data(), outDistancesSqr.data());
            DNDS_assert(nFound >= 1);
            idxFound.resize(nFound);
            outDistancesSqr.resize(nFound);

            real RMin = std::sqrt(outDistancesSqr.minCoeff());
            real RMax = std::sqrt(outDistancesSqr.maxCoeff());
            real RRBF = elevationInfo.RBFRadius * RMax;
            /***********************************************/
            // std::vector<std::pair<DNDS::index, DNDS::real>> indicesDists;
            // indicesDists.reserve(nFind);

            // nanoflann::SearchParams searchParams;
            // index nFound = bndInterpTree.radiusSearch(cooC.data(), elevationInfo.RBFRadius, indicesDists, searchParams);
            // idxFound.reserve(nFound);
            // outDistancesSqr.resize(nFound);
            // for (auto v : indicesDists)
            //     outDistancesSqr(idxFound.size()) = v.second, idxFound.push_back(v.first);
            // real RRBF = elevationInfo.RBFRadius;
            /***********************************************/

            boundInterpR(iN, 0) = RRBF;
            DNDS_assert(RRBF > 0);
            Eigen::Vector<real, Eigen::Dynamic> outDists = outDistancesSqr.array().pow(0.5 / elevationInfo.RBFPower) * (1. / std::pow(RRBF, 1. / elevationInfo.RBFPower));
            auto fBasis = RBF::FRBFBasis(outDists, elevationInfo.kernel);
            MatC[iN].resize(nFound);
            for (index in2n = 0; in2n < nFound; in2n++)
            {
                MatC[iN][in2n] = std::make_pair(index(idxFound[in2n]), real(fBasis(in2n)));
                // idxFound[in2n] is a global indexing!
                auto [search_good, rank, val] = boundInterpCoo.trans.pLGlobalMapping->search(idxFound[in2n]);
                DNDS_assert(search_good);
                if (rank != mpi.rank)
                    boundInterpPullingIndexSolving.push_back(idxFound[in2n]);
            }

            boundInterpCoefRHS[iN] = (*boundInterpVal.father)[iN];
            boundInterpCoef[iN].setZero(); // init val
            // if(mpi.rank == 0)
            // {
            //     std::cout << "mat row at " << cooC.transpose() << "R = " << RRBF << "\n";
            //     for (auto idx : idxFound)
            //         std::cout << idx << " ";
            //     std::cout << "\n"
            //               << fBasis.transpose() << "\n"
            //               << outDists.transpose() << "\n"
            //               << boundInterpCoefRHS[iN].transpose() << std::endl;
            // }
        }

        if (false)
        { // use superlu_dist to solve

            // gridinfo_t grid;
            // superlu_gridinit(mpi.comm, 1, mpi.size, &grid);
            // DNDS_assert(grid.iam < mpi.size && grid.iam >= 0);

            // std::vector<double> nzval, b;
            // std::vector<int_t> colind;
            // std::vector<int_t> rowptr;
            // rowptr.resize(boundInterpLocSize + 1);
            // rowptr[0] = 0;
            // for (index i = 0; i < boundInterpLocSize; i++)
            //     rowptr[i + 1] = rowptr[i] + MatC[i].size();
            // colind.resize(rowptr.back());
            // nzval.resize(rowptr.back());
            // for (index i = 0; i < boundInterpLocSize; i++)
            //     for (index i2j = 0; i2j < MatC[i].size(); i2j++)
            //     {
            //         colind[rowptr[i] + i2j] = MatC[i][i2j].first;
            //         nzval[rowptr[i] + i2j] = MatC[i][i2j].second;
            //     }
            // b.resize(boundInterpLocSize * 3);
            // for (index i = 0; i < boundInterpLocSize; i++)
            // {
            //     b[boundInterpLocSize * 0 + i] = boundInterpCoefRHS[i](0);
            //     b[boundInterpLocSize * 1 + i] = boundInterpCoefRHS[i](1);
            //     b[boundInterpLocSize * 2 + i] = boundInterpCoefRHS[i](2);
            // }

            // SuperMatrix Aloc;
            // dCreate_CompRowLoc_Matrix_dist(
            //     &Aloc, boundInterpGlobSize, boundInterpGlobSize, nzval.size(), boundInterpLocSize,
            //     boundInterpCoo.trans.pLGlobalMapping->operator()(mpi.rank, 0), nzval.data(), colind.data(), rowptr.data(),
            //     SLU_NR_loc, SLU_D, SLU_GE);

            // superlu_dist_options_t options;
            // set_default_options_dist(&options);
            // SuperLUStat_t stat;
            // PStatInit(&stat);
            // dScalePermstruct_t ScalePermstruct;
            // dScalePermstructInit(Aloc.nrow, Aloc.ncol, &ScalePermstruct);
            // dLUstruct_t LUstruct;
            // dLUstructInit(Aloc.ncol, &LUstruct);
            // dSOLVEstruct_t SOLVEstruct;

            // tPoint berr;
            // berr.setZero();
            // int info;

            // pdgssvx(&options, &Aloc, &ScalePermstruct,
            //         b.data(), boundInterpLocSize, 3,
            //         &grid, &LUstruct, &SOLVEstruct, berr.data(), &stat, &info);
            // for (index i = 0; i < boundInterpLocSize; i++)
            // {
            //     boundInterpCoef[i](0) = b[boundInterpLocSize * 0 + i];
            //     boundInterpCoef[i](1) = b[boundInterpLocSize * 1 + i];
            //     boundInterpCoef[i](2) = b[boundInterpLocSize * 2 + i];
            // }
            // PStatPrint(&options, &stat, &grid);
            // DNDS_assert(info == 0);

            // PStatFree(&stat);
            // // TODO: sanitize with valgrind!
            // // Destroy_CompRowLoc_Matrix_dist(&Aloc); // use vector for strorage, this causes double free
            // SUPERLU_FREE(Aloc.Store);
            // dScalePermstructFree(&ScalePermstruct);
            // dDestroy_LU(Aloc.ncol, &grid, &LUstruct);
            // dLUstructFree(&LUstruct);
            // if (options.SolveInitialized)
            // {
            //     dSolveFinalize(&options, &SOLVEstruct);
            // }

            // superlu_gridexit(&grid);
        }
        if (true)
        { // use GMRES to solve
            std::cout << "HEre0" << std::endl;
            boundInterpCoef.father->createGlobalMapping();
            boundInterpCoef.TransAttach();
            boundInterpCoef.trans.createGhostMapping(boundInterpPullingIndexSolving);
            boundInterpCoef.trans.createMPITypes();
            boundInterpCoef.trans.initPersistentPull();

            boundInterpCoefRHS.TransAttach();
            boundInterpCoefRHS.trans.BorrowGGIndexing(boundInterpCoef.trans);
            boundInterpCoefRHS.trans.createMPITypes();
            boundInterpCoefRHS.trans.initPersistentPull();

            std::cout << "HEre1" << std::endl;
            for (index iN = 0; iN < boundInterpLocSize; iN++)
            {
                for (index in2n = 0; in2n < MatC[iN].size(); in2n++)
                {
                    // MatC[iN][in2n].first;
                    auto [search_good, rank, val] = boundInterpCoef.trans.pLGhostMapping->search_indexAppend(MatC[iN][in2n].first);
                    DNDS_assert(search_good);
                    DNDS_assert(val < boundInterpCoef.Size());
                    MatC[iN][in2n].first = val; // to local
                }
            }
            Eigen::SparseMatrix<real> M(boundInterpLocSize, boundInterpLocSize);
            M.reserve(Eigen::Vector<index, Eigen::Dynamic>::Constant(boundInterpLocSize, elevationInfo.nSearch));
            for (index iN = 0; iN < boundInterpLocSize; iN++)
            {
                for (index in2n = 0; in2n < MatC[iN].size(); in2n++)
                {
                    if (MatC[iN][in2n].first < boundInterpLocSize)
                        M.insert(iN, MatC[iN][in2n].first) = MatC[iN][in2n].second;
                    // ,
                    //              std::cout << fmt::format("rank {}, inserting {},{}={}", mpi.rank, iN, MatC[iN][in2n].first, MatC[iN][in2n].second) << std::endl;
                }
            }
            Eigen::SparseLU<Eigen::SparseMatrix<real>, Eigen::COLAMDOrdering<int>> LUSolver;
            if (boundInterpLocSize)
            {
                LUSolver.analyzePattern(M);
                LUSolver.factorize(M); // full LU perconditioner
            }
            std::cout << "HEre2" << std::endl;

            Linear::GMRES_LeftPreconditioned<CoordPairDOF> gmres(
                5,
                [&](CoordPairDOF &v)
                {
                    DNDS_MAKE_SSP(v.father, mpi);
                    DNDS_MAKE_SSP(v.son, mpi);
                    v.father->Resize(boundInterpCoef.father->Size());
                    v.TransAttach();
                    v.trans.BorrowGGIndexing(boundInterpCoef.trans);
                    v.trans.createMPITypes();
                    v.trans.initPersistentPull();
                });
            std::cout << "HEre3" << std::endl;
            boundInterpCoef.trans.startPersistentPull();
            boundInterpCoef.trans.waitPersistentPull();
            boundInterpCoefRHS.trans.startPersistentPull();
            boundInterpCoefRHS.trans.waitPersistentPull();
            gmres.solve(
                [&](CoordPairDOF &x, CoordPairDOF &Ax)
                {
                    x.trans.startPersistentPull();
                    x.trans.waitPersistentPull();
                    for (index iN = 0; iN < boundInterpLocSize; iN++)
                    {
                        Ax[iN].setZero();
                        for (index in2n = 0; in2n < MatC[iN].size(); in2n++)
                        {
                            Ax[iN] += x[MatC[iN][in2n].first] * MatC[iN][in2n].second;
                        }
                    }
                    Ax.trans.startPersistentPull();
                    Ax.trans.waitPersistentPull();
                },
                [&](CoordPairDOF &x, CoordPairDOF &MLX)
                {
                    x.trans.startPersistentPull();
                    x.trans.waitPersistentPull();
                    Eigen::Matrix<real, Eigen::Dynamic, 3> xVal;
                    xVal.resize(boundInterpLocSize, 3);
                    for (index iN = 0; iN < boundInterpLocSize; iN++)
                        xVal(iN, Eigen::all) = x[iN].transpose();
                    Eigen::Matrix<real, Eigen::Dynamic, 3> xValPrec;
                    if (boundInterpLocSize)
                        xValPrec = LUSolver.solve(xVal);
                    for (index iN = 0; iN < boundInterpLocSize; iN++)
                        MLX[iN] = xValPrec(iN, Eigen::all).transpose();
                    MLX = x;
                    MLX.trans.startPersistentPull();
                    MLX.trans.waitPersistentPull();
                },
                [&](CoordPairDOF &a, CoordPairDOF &b) -> real
                {
                    return a.dot(b);
                },
                boundInterpCoefRHS,
                boundInterpCoef,
                elevationInfo.nIter,
                [&](int iRestart, real res, real resB)
                {
                    if (mpi.rank == mRank)
                        log() << fmt::format("iRestart {}, res {}, resB {}", iRestart, res, resB) << std::endl;
                    return res < resB * 1e-6;
                }

            );
            boundInterpCoef.trans.startPersistentPull();
            boundInterpCoef.trans.waitPersistentPull();
        }

        boundInterpCoef.father->createGlobalMapping();
        boundInterpCoef.TransAttach();
        boundInterpCoef.trans.createGhostMapping(boundInterpPullIdx);
        boundInterpCoef.trans.createMPITypes();
        boundInterpCoef.trans.pullOnce();

        boundInterpR.TransAttach();
        boundInterpR.trans.BorrowGGIndexing(boundInterpCoo.trans);
        boundInterpR.trans.createMPITypes();
        boundInterpR.trans.pullOnce();

        // if (mpi.rank == mRank)
        // {
        //     for (index iN = 0; iN < boundInterpGlobSize; iN++)
        //     {
        //         std::cout << "pt - coef: R " << boundInterpR.son->operator()(iN, 0)
        //                   << " coo " << boundInterpCoo.son->operator[](iN).transpose()
        //                   << " -> " << boundInterpCoef.son->operator[](iN).transpose() << std::endl;
        //     }
        // }

        PointCloudKDTree insidePts;
        insidePts.pts.reserve(coords.father->Size() - boundInterpLocSize);
        std::vector<index> insideNodes;
        insideNodes.reserve(insidePts.pts.size());
        for (index iN = 0; iN < coords.father->Size(); iN++)
            if (!nodesBoundInterpolated.count(iN))
                insidePts.pts.push_back(coords[iN]), insideNodes.push_back(iN), coordsElevDisp[iN].setZero();
        kdtree_tcoo nodesDstTree(3, insidePts);
        for (index iN = 0; iN < boundInterpGlobSize; iN++)
        {

            tPoint cooC = (*boundInterpCoo.son)[iN];
            std::vector<std::pair<DNDS::index, DNDS::real>> IndicesDists;
            IndicesDists.reserve(elevationInfo.nSearch * 5);
            real RRBF = boundInterpR.son->operator()(iN, 0);
            nanoflann::SearchParams params{}; // default params
            index nFound = nodesDstTree.radiusSearch(cooC.data(), RRBF, IndicesDists, params);
            Eigen::Vector<real, Eigen::Dynamic> outDists;
            outDists.resize(IndicesDists.size());
            DNDS_assert(RRBF > 0);
            for (index i = 0; i < IndicesDists.size(); i++)
                outDists[i] = std::sqrt(IndicesDists[i].second) / std::pow(RRBF, 1. / elevationInfo.RBFPower);

            auto fBasis = RBF::FRBFBasis(outDists, elevationInfo.kernel);
            for (index i = 0; i < IndicesDists.size(); i++)
            {
                coordsElevDisp[insideNodes[IndicesDists[i].first]] +=
                    fBasis(i, 0) *
                    boundInterpCoef.son->operator[](iN);
            }
        }

        // for (index iN = 0; iN < coords.father->Size(); iN++)
        // {
        //     if (nodesBoundInterpolated.count(iN))
        //         continue;

        //     index nFind = elevationInfo.nSearch * 5; // for safety
        //     tPoint cooC = coords[iN];

        //     std::vector<index> idxFound;
        //     Eigen::Vector<real, Eigen::Dynamic> outDistancesSqr;
        //     idxFound.resize(nFind);
        //     outDistancesSqr.resize(nFind);

        //     index nFound = bndInterpTree.knnSearch(cooC.data(), nFind, idxFound.data(), outDistancesSqr.data());
        //     DNDS_assert(nFound >= 1);
        //     idxFound.resize(nFound);
        //     outDistancesSqr.resize(nFound);

        //     Eigen::Vector<real, Eigen::Dynamic> foundRRBFs;
        //     foundRRBFs.resize(nFound);
        //     tSmallCoords coefsC;
        //     coefsC.resize(3, nFound);
        //     for (index iF = 0; iF < nFound; iF++)
        //         foundRRBFs[iF] = (*boundInterpR.son)(idxFound[iF], 0),
        //         coefsC(Eigen::all, iF) = (*boundInterpCoef.son)[idxFound[iF]].transpose();
        //     Eigen::Vector<real, Eigen::Dynamic> outDists = outDistancesSqr.array().sqrt() / foundRRBFs.array();
        //     auto fBasis = RBF::FRBFBasis(outDists, elevationInfo.kernel);
        //     tPoint dispC = coefsC * fBasis;
        //     coordsElevDisp[iN] = dispC;
        // }

        coordsElevDisp.trans.startPersistentPull();
        coordsElevDisp.trans.waitPersistentPull();

        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            // if(dim == 2)
            //     dispO2[iN](2) = 0;
            // if(dispO2[iN].norm() != 0)
            //     std::cout << dispO2[iN].transpose() << std::endl;
            if (coordsElevDisp[iN](0) != largeReal)
                coords[iN] += coordsElevDisp[iN];
        }
        coords.trans.pullOnce();
    }

    void UnstructuredMesh::ElevatedNodesSolveInternalSmoothV2()
    {
        DNDS_assert(elevState == Elevation_O1O2);
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        DNDS_assert(face2node.father);
        DNDS_assert(nTotalMoved >= 0);
        if (!nTotalMoved)
        {
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth() early exit for no nodes were moved";
            return;
        }

        std::unordered_set<index> nodesBoundInterpolated;
        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (coordsElevDisp[iN](0) != largeReal || coordsElevDisp[iN](2) == 2 * largeReal)
            {
                nodesBoundInterpolated.insert(iN);
            }
        }

        tCoordPair boundInterpCoo;
        tCoordPair boundInterpVal;
        DNDS_MAKE_SSP(boundInterpCoo.father, mpi);
        DNDS_MAKE_SSP(boundInterpCoo.son, mpi);
        DNDS_MAKE_SSP(boundInterpVal.father, mpi);
        DNDS_MAKE_SSP(boundInterpVal.son, mpi);

        boundInterpCoo.father->Resize(nodesBoundInterpolated.size());
        boundInterpVal.father->Resize(nodesBoundInterpolated.size());

        index top{0};
        for (auto iN : nodesBoundInterpolated)
        {
            boundInterpCoo[top] = coords[iN];
            boundInterpVal[top] = (coordsElevDisp[iN](0) != largeReal) ? tPoint{coordsElevDisp[iN]} : tPoint::Zero();
            top++;
        }
        std::vector<index> boundInterpPullIdx;

        index boundInterpGlobSize = boundInterpCoo.father->globalSize();
        index boundInterpLocSize = nodesBoundInterpolated.size();
        boundInterpPullIdx.resize(boundInterpGlobSize);
        for (index i = 0; i < boundInterpPullIdx.size(); i++)
            boundInterpPullIdx[i] = i;
        boundInterpCoo.father->createGlobalMapping();
        boundInterpCoo.TransAttach();
        boundInterpCoo.trans.createGhostMapping(boundInterpPullIdx);
        boundInterpCoo.trans.createMPITypes();
        boundInterpCoo.trans.pullOnce();

        boundInterpVal.TransAttach();
        boundInterpVal.trans.BorrowGGIndexing(boundInterpCoo.trans);
        boundInterpVal.trans.createMPITypes();
        boundInterpVal.trans.pullOnce();

        std::vector<std::unordered_set<index>> node2nodeV;
        node2nodeV.resize(coords.father->Size());
        for (index iCell = 0; iCell < cell2node.Size(); iCell++)
        {
            for (auto iN : cell2node[iCell])
                for (auto iNOther : cell2node[iCell])
                {
                    if (iN == iNOther)
                        continue;
                    if (iN < coords.father->Size())
                    {
                        node2nodeV[iN].insert(iNOther);
                    }
                }
        }

        if (mpi.rank == mRank)
            log() << "RBF set: " << boundInterpCoo.son->Size() << std::endl;

        using kdtree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<real, PointCloudKDTreeCoordPair>,
            PointCloudKDTreeCoordPair,
            3,
            index>;
        auto coordsI = PointCloudKDTreeCoordPair(boundInterpCoo.son);
        kdtree_t bndInterpTree(3, coordsI);

        DNDS_MPI_InsertCheck(mpi, "Got node2node");
        // std::cout << "hereXXXXXXXXXXXXXXX 0" << std::endl;

        CoordPairDOF dispO2, bO2, dispO2New, dispO2PrecRHS, dispO2Inc, dispO2IncLimited;
        auto initDOF = [&](CoordPairDOF &dispO2)
        {
            DNDS_MAKE_SSP(dispO2.father, mpi);
            DNDS_MAKE_SSP(dispO2.son, mpi);
            dispO2.father->Resize(coords.father->Size()); // O1 part is unused
            dispO2.TransAttach();
            dispO2.trans.BorrowGGIndexing(coords.trans); // should be subset of coord?
            dispO2.trans.createMPITypes();
            dispO2.trans.initPersistentPull();
        };
        initDOF(dispO2);
        initDOF(dispO2Inc);
        initDOF(dispO2New);
        initDOF(dispO2IncLimited);
        initDOF(bO2);
        initDOF(dispO2PrecRHS);

        struct MatElem
        {
            index j;
            Eigen::Matrix<real, 3, 3> m;
        };
        std::vector<std::vector<MatElem>> A; // stiff matrixTo O2 coords
        A.resize(coords.father->Size());
        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            A[iN].resize(node2nodeV[iN].size() + 1);
            A[iN][0].j = iN;
            int curRowEnd = 1;
            for (auto iNOther : node2nodeV[iN])
            {
                A[iN][curRowEnd++].j = iNOther;
            }
            for (auto &ME : A[iN])
                ME.m.setConstant(0);
            bO2[iN].setZero();
            dispO2[iN].setZero();
            if (coordsElevDisp[iN](0) != largeReal)
                dispO2[iN] = coordsElevDisp[iN];
            if (coordsElevDisp[iN](2) == 2 * largeReal || coordsElevDisp[iN](2) == 3 * largeReal)
                dispO2[iN].setZero();
        }
        dispO2.trans.startPersistentPull();
        dispO2.trans.waitPersistentPull();

        auto AssembleRHSMat = [&](tCoordPair &uCur)
        {
            index nFix{0}, nFixB{0};
            if (false) // fem method
                for (index iCell = 0; iCell < cell2node.Size(); iCell++)
                {
                    auto c2n = cell2node[iCell];
                    rowsize nnLoc = c2n.size();
                    tSmallCoords coordsC;
                    GetCoordsOnCell(iCell, coordsC);
                    tPoint cellCent = coordsC.rowwise().mean();
                    real wDist{0};
                    {
                        std::vector<real> sqrDists(1);
                        std::vector<index> idxFound(1);
                        auto nFound = bndInterpTree.knnSearch(cellCent.data(), 1, idxFound.data(), sqrDists.data());
                        DNDS_assert(nFound == 1);
                        wDist = std::sqrt(sqrDists[0]);
                    }

                    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> ALoc, mLoc;
                    ALoc.setZero(3 * nnLoc, 3 * nnLoc + 1);
                    mLoc.resize(6, 3 * nnLoc);
                    auto eCell = GetCellElement(iCell);
                    auto qCell = Elem::Quadrature{eCell, 2}; // for O2 FEM
                    qCell.Integration(
                        ALoc,
                        [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                        {
                            // J = dx_i/dxii_j, Jinv = dxii_i/dx_j
                            tJacobi J = Elem::ShapeJacobianCoordD01Nj(coordsC, DiNj);
                            tJacobi JInv = tJacobi::Identity();
                            real JDet;
                            if (dim == 2)
                                JDet = J(Eigen::all, 0).cross(J(Eigen::all, 1)).stableNorm();
                            else
                                JDet = J.determinant();
                            DNDS_assert(JDet > 0);
                            if (dim == 2)
                                JInv({0, 1}, {0, 1}) = J({0, 1}, {0, 1}).inverse().eval();
                            else
                                JInv = J.inverse().eval();
                            // JInv.transpose() * DiNj({1,2,3}, Eigen::all) * u (size = nnode,3) => du_j/dxi
                            tSmallCoords m3 = JInv.transpose() * DiNj({1, 2, 3}, Eigen::all);
                            mLoc(0, Eigen::seq(nnLoc * 0, nnLoc * 0 + nnLoc - 1)) = m3(0, Eigen::all);
                            mLoc(1, Eigen::seq(nnLoc * 1, nnLoc * 1 + nnLoc - 1)) = m3(1, Eigen::all);
                            mLoc(2, Eigen::seq(nnLoc * 2, nnLoc * 2 + nnLoc - 1)) = m3(2, Eigen::all);

                            mLoc(3, Eigen::seq(nnLoc * 1, nnLoc * 1 + nnLoc - 1)) = 0.5 * m3(2, Eigen::all);
                            mLoc(3, Eigen::seq(nnLoc * 2, nnLoc * 2 + nnLoc - 1)) = 0.5 * m3(1, Eigen::all);
                            mLoc(4, Eigen::seq(nnLoc * 2, nnLoc * 2 + nnLoc - 1)) = 0.5 * m3(0, Eigen::all);
                            mLoc(4, Eigen::seq(nnLoc * 0, nnLoc * 0 + nnLoc - 1)) = 0.5 * m3(2, Eigen::all);
                            mLoc(5, Eigen::seq(nnLoc * 0, nnLoc * 0 + nnLoc - 1)) = 0.5 * m3(1, Eigen::all);
                            mLoc(5, Eigen::seq(nnLoc * 1, nnLoc * 1 + nnLoc - 1)) = 0.5 * m3(0, Eigen::all);

                            Eigen::Matrix<real, 6, 6> DStruct;
                            DStruct.setIdentity();
                            // real lam = std::pow((wDist / 1e-6), -1);
                            real lam = 1;
                            real muu = 1;
                            DStruct *= muu;
                            DStruct({0, 1, 2}, {0, 1, 2}) += muu * tJacobi::Identity();
                            DStruct({0, 1, 2}, {0, 1, 2}).array() += lam;
                            // DStruct *= (1. / JDet);
                            // DStruct *= std::pow((wDist / 1e-6), -10);

                            vInc.resizeLike(ALoc);
                            vInc(Eigen::all, Eigen::seq(0, 3 * nnLoc - 1)) = mLoc.transpose() * DStruct * mLoc;
                            vInc(Eigen::all, Eigen::last).setZero(); // no node force, thus zero
                            vInc *= JDet;
                        });
                    // for (int ic2n = 0; ic2n < nnLoc; ic2n++)
                    //     for (int jc2n = 0; jc2n < nnLoc; jc2n++)
                    //     {

                    //         if (ic2n == jc2n)
                    //             continue;
                    //         real lBA = (coordsC(Eigen::all, jc2n) - coordsC(Eigen::all, ic2n)).norm();
                    //         tPoint nBA = (coordsC(Eigen::all, jc2n) - coordsC(Eigen::all, ic2n)).normalized();
                    //         // real K = std::pow(lBA, 0);
                    //         real K = std::pow((wDist / elevationInfo.refDWall), -1);
                    //         tJacobi mnBA = nBA * nBA.transpose() * K;
                    //         for (int iii = 0; iii < 3; iii++)
                    //             for (int jjj = 0; jjj < 3; jjj++)
                    //             {
                    //                 ALoc(iii * nnLoc + ic2n, jjj * nnLoc + ic2n) += mnBA(iii, jjj);
                    //                 ALoc(iii * nnLoc + ic2n, jjj * nnLoc + jc2n) -= mnBA(iii, jjj);
                    //             }
                    //}
                    if (dim != 3)
                    {
                        ALoc(Eigen::seq(2 * nnLoc, 3 * nnLoc - 1), Eigen::seq(2 * nnLoc, 3 * nnLoc - 1)).setZero();
                        ALoc(Eigen::all, Eigen::seq(2 * nnLoc, 3 * nnLoc - 1)).setZero();
                        ALoc(Eigen::seq(2 * nnLoc, 3 * nnLoc - 1), Eigen::all).setZero();
                    }

                    for (rowsize ic2n = 0; ic2n < nnLoc; ic2n++)
                    {
                        index iN = c2n[ic2n];
                        if (iN >= coords.father->Size())
                            continue; // not local row
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
                        bO2[iN](0) += ALoc(0 * nnLoc + ic2n, Eigen::last);
                        bO2[iN](1) += ALoc(1 * nnLoc + ic2n, Eigen::last);
                        bO2[iN](2) += ALoc(2 * nnLoc + ic2n, Eigen::last); // last col is b
                        if (coordsElevDisp[iN](0) != largeReal)
                        {
                            nFix++;
                            real nDiag = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    nDiag = ME.m.array().abs().maxCoeff();
                            bO2[iN] = coordsElevDisp[iN] * nDiag;
                            // bO2[iN].setZero();
                            // if (coords[iN].norm() < 2)
                            //     bO2[iN](1) = sqr(coords[iN](0)) * 0.1 * nDiag;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    ME.m = tJacobi::Identity() * nDiag;
                                else
                                    ME.m.setZero();
                        }
                        if (coordsElevDisp[iN](2) == 2 * largeReal || coordsElevDisp[iN](2) == 3 * largeReal)
                        {
                            nFixB++;
                            real nDiag = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    nDiag = ME.m.array().abs().maxCoeff();
                            bO2[iN].setZero();
                            // bO2[iN].setZero();
                            // if (coords[iN].norm() < 2)
                            //     bO2[iN](1) = sqr(coords[iN](0)) * 0.1 * nDiag;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    ME.m = tJacobi::Identity() * nDiag;
                                else
                                    ME.m.setZero();
                        }
                        if (dim != 3)
                        {
                            real nDiag = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    nDiag = ME.m.array().abs().maxCoeff();
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    ME.m(2, 2) += nDiag; // the z-direction need a constraint
                        }
                    }
                }

            if (true) // bisect fem method
                for (index iCell = 0; iCell < cell2node.Size(); iCell++)
                {
                    auto c2n = cell2node[iCell];
                    rowsize nnLoc = c2n.size();
                    SmallCoordsAsVector coordsC, coordsCu;
                    GetCoordsOnCell(iCell, coordsC);
                    GetCoordsOnCell(iCell, coordsCu, uCur);
                    tPoint cellCent = coordsC.rowwise().mean();
                    real wDist{0};
                    {
                        std::vector<real> sqrDists(1);
                        std::vector<index> idxFound(1);
                        auto nFound = bndInterpTree.knnSearch(cellCent.data(), 1, idxFound.data(), sqrDists.data());
                        DNDS_assert(nFound == 1);
                        wDist = std::sqrt(sqrDists[0]);
                    }

                    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> ALoc;
                    ALoc.setZero(3 * nnLoc, 3 * nnLoc + 1);
                    auto localNodeIdx = Eigen::ArrayXi::LinSpaced(c2n.size(), 0, c2n.size() - 1);

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
                        auto qCellSub = Elem::Quadrature{eCellSub, 6}; // for O1 FEM
                        rowsize nnLocSub = rowsize(c2nSubLocal.size());
                        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> ALocSub, mLoc;
                        ALocSub.setZero(3 * nnLocSub, 3 * nnLocSub + 1);
                        mLoc.resize(6, 3 * nnLocSub);
                        Eigen::ArrayXi c2nSubLocal3;
                        c2nSubLocal3.resize(c2nSubLocal.size() * 3);
                        c2nSubLocal3(Eigen::seq(c2nSubLocal.size() * 0, c2nSubLocal.size() * 1 - 1)) = c2nSubLocal + nnLoc * 0;
                        c2nSubLocal3(Eigen::seq(c2nSubLocal.size() * 1, c2nSubLocal.size() * 2 - 1)) = c2nSubLocal + nnLoc * 1;
                        c2nSubLocal3(Eigen::seq(c2nSubLocal.size() * 2, c2nSubLocal.size() * 3 - 1)) = c2nSubLocal + nnLoc * 2;

                        qCellSub.Integration(
                            ALocSub,
                            [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                            {
                                // J = dx_i/dxii_j, Jinv = dxii_i/dx_j
                                auto getJ = [&](auto coos)
                                {
                                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coos, DiNj);
                                    tJacobi JInv = tJacobi::Identity();
                                    real JDet;
                                    if (dim == 2)
                                        JDet = J(Eigen::all, 0).cross(J(Eigen::all, 1)).stableNorm();
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
                                // JInv.transpose() * DiNj({1,2,3}, Eigen::all) * u (size = nnode,3) => du_j/dxi
                                tSmallCoords m3 = JInv.transpose() * DiNj({1, 2, 3}, Eigen::all);
                                mLoc.setZero();
                                mLoc(0, Eigen::seq(nnLocSub * 0, nnLocSub * 0 + nnLocSub - 1)) = m3(0, Eigen::all);
                                mLoc(1, Eigen::seq(nnLocSub * 1, nnLocSub * 1 + nnLocSub - 1)) = m3(1, Eigen::all);
                                mLoc(2, Eigen::seq(nnLocSub * 2, nnLocSub * 2 + nnLocSub - 1)) = m3(2, Eigen::all);

                                mLoc(3, Eigen::seq(nnLocSub * 1, nnLocSub * 1 + nnLocSub - 1)) = 0.5 * m3(2, Eigen::all);
                                mLoc(3, Eigen::seq(nnLocSub * 2, nnLocSub * 2 + nnLocSub - 1)) = 0.5 * m3(1, Eigen::all);
                                mLoc(4, Eigen::seq(nnLocSub * 2, nnLocSub * 2 + nnLocSub - 1)) = 0.5 * m3(0, Eigen::all);
                                mLoc(4, Eigen::seq(nnLocSub * 0, nnLocSub * 0 + nnLocSub - 1)) = 0.5 * m3(2, Eigen::all);
                                mLoc(5, Eigen::seq(nnLocSub * 0, nnLocSub * 0 + nnLocSub - 1)) = 0.5 * m3(1, Eigen::all);
                                mLoc(5, Eigen::seq(nnLocSub * 1, nnLocSub * 1 + nnLocSub - 1)) = 0.5 * m3(0, Eigen::all);

                                Eigen::Matrix<real, 6, 6> DStruct;
                                DStruct.setIdentity();
                                // real lam = std::pow((wDist / 1e-6), -1);
                                real lam = 1;
                                real muu = 100;
                                // lam *= std::pow(JDetn / JDet, -1) + 1;
                                DStruct *= muu;
                                DStruct({0, 1, 2}, {0, 1, 2}) += muu * tJacobi::Identity();
                                DStruct({0, 1, 2}, {0, 1, 2}).array() += lam;
                                DStruct *= std::pow(JDetn / JDet, -0.0) + 1;
                                // DStruct *= (1. / JDet);
                                // DStruct *= std::pow((wDist / 1e-6), -10);
                                // DStruct *= std::pow((wDist / elevationInfo.refDWall), -1);

                                vInc.resizeLike(ALocSub);
                                vInc(Eigen::all, Eigen::seq(0, 3 * nnLocSub - 1)) = mLoc.transpose() * DStruct * mLoc;
                                vInc(Eigen::all, Eigen::last).setZero(); // no node force, thus zero
                                vInc *= JDet;
                            });
                        ALoc(c2nSubLocal3, c2nSubLocal3) += ALocSub(Eigen::all, Eigen::seq(0, Eigen::last - 1));
                        ALoc(c2nSubLocal3, Eigen::last) += ALocSub(Eigen::all, Eigen::last);
                    }

                    if (dim != 3)
                    {
                        ALoc(Eigen::seq(2 * nnLoc, 3 * nnLoc - 1), Eigen::seq(2 * nnLoc, 3 * nnLoc - 1)).setZero();
                        ALoc(Eigen::all, Eigen::seq(2 * nnLoc, 3 * nnLoc - 1)).setZero();
                        ALoc(Eigen::seq(2 * nnLoc, 3 * nnLoc - 1), Eigen::all).setZero();
                    }

                    for (rowsize ic2n = 0; ic2n < nnLoc; ic2n++)
                    {
                        index iN = c2n[ic2n];
                        if (iN >= coords.father->Size())
                            continue; // not local row
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
                        bO2[iN](0) += ALoc(0 * nnLoc + ic2n, Eigen::last);
                        bO2[iN](1) += ALoc(1 * nnLoc + ic2n, Eigen::last);
                        bO2[iN](2) += ALoc(2 * nnLoc + ic2n, Eigen::last); // last col is b
                        if (coordsElevDisp[iN](0) != largeReal)
                        {
                            nFix++;
                            real nDiag = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    nDiag = ME.m.array().abs().maxCoeff();
                            bO2[iN] = coordsElevDisp[iN] * nDiag;
                            // bO2[iN].setZero();
                            // if (coords[iN].norm() < 2)
                            //     bO2[iN](1) = sqr(coords[iN](0)) * 0.1 * nDiag;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    ME.m = tJacobi::Identity() * nDiag;
                                else
                                    ME.m.setZero();
                        }
                        if (coordsElevDisp[iN](2) == 2 * largeReal || coordsElevDisp[iN](2) == 3 * largeReal || (iN < nNodeO1 && false))
                        {
                            nFixB++;
                            real nDiag = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    nDiag = ME.m.array().abs().maxCoeff();
                            bO2[iN].setZero();
                            // bO2[iN].setZero();
                            // if (coords[iN].norm() < 2)
                            //     bO2[iN](1) = sqr(coords[iN](0)) * 0.1 * nDiag;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    ME.m = tJacobi::Identity() * nDiag;
                                else
                                    ME.m.setZero();
                        }
                        if (dim != 3)
                        {
                            real nDiag = 0;
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    nDiag = ME.m.array().abs().maxCoeff();
                            for (auto &ME : A[iN])
                                if (ME.j == iN)
                                    ME.m(2, 2) += nDiag; // the z-direction need a constraint
                        }
                    }
                }

            if (false)
                for (index iN = 0; iN < A.size(); iN++)
                {
                    for (auto jN : node2nodeV[iN])
                    {
                        real lBA = (coords[jN] - coords[iN]).norm();
                        tPoint nBA = (coords[jN] - coords[iN]).normalized();
                        real K = std::pow(lBA, -0.2);
                        // real K = std::pow((wDist / elevationInfo.refDWall), -1);
                        tJacobi mnBA = nBA * nBA.transpose() * K;
                        for (auto &ME : A[iN])
                            if (ME.j == jN)
                            {
                                A[iN][0].m += mnBA;
                                ME.m -= mnBA;
                            }
                    }
                    if (coordsElevDisp[iN](0) != largeReal)
                    {
                        nFix++;
                        real nDiag = 0;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                nDiag = ME.m.array().abs().maxCoeff();
                        bO2[iN] = coordsElevDisp[iN] * nDiag;
                        // bO2[iN].setZero();
                        // if (coords[iN].norm() < 2)
                        //     bO2[iN](1) = sqr(coords[iN](0)) * 0.1 * nDiag;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                ME.m = tJacobi::Identity() * nDiag;
                            else
                                ME.m.setZero();
                    }
                    if (coordsElevDisp[iN](2) == 2 * largeReal || coordsElevDisp[iN](2) == 3 * largeReal)
                    {
                        nFixB++;
                        real nDiag = 0;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                nDiag = ME.m.array().abs().maxCoeff();
                        bO2[iN].setZero();
                        // bO2[iN].setZero();
                        // if (coords[iN].norm() < 2)
                        //     bO2[iN](1) = sqr(coords[iN](0)) * 0.1 * nDiag;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                ME.m = tJacobi::Identity() * nDiag;
                            else
                                ME.m.setZero();
                    }
                    if (dim != 3)
                    {
                        real nDiag = 0;
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                nDiag = ME.m.array().abs().maxCoeff();
                        for (auto &ME : A[iN])
                            if (ME.j == iN)
                                ME.m(2, 2) += nDiag; // the z-direction need a constraint
                    }
                }

            bO2.trans.startPersistentPull();
            bO2.trans.waitPersistentPull();
            // for (index iN = 0; iN < A.size(); iN++)
            //     for (auto &ME : A[iN])
            //         std::cout << "ij: " << iN << " " << ME.j << "\n"
            //                   << ME.m << std::endl;
            MPI::AllreduceOneIndex(nFix, MPI_SUM, mpi);
            MPI::AllreduceOneIndex(nFixB, MPI_SUM, mpi);
            if (mpi.rank == mRank)
            {
                log() << fmt::format("UnstructuredMesh === ElevatedNodesSolveInternalSmooth(): Matrix Assembled, nFix {}, nFixB {} ",
                                     nFix, nFixB)
                      << std::endl;
            }
        };
        dispO2.setConstant(0.0);
        AssembleRHSMat(dispO2);

        auto LimitDisp = [&](tCoordPair &uCur, tCoordPair &duCur, tCoordPair &duCurLim)
        {
            index nFix{0};
            real minLim{1};
            if (true) // bisect fem method
                for (index iCell = 0; iCell < cell2node.Size(); iCell++)
                {
                    auto c2n = cell2node[iCell];
                    rowsize nnLoc = c2n.size();
                    SmallCoordsAsVector coordsC, coordsCu, coordsCuinc;
                    GetCoordsOnCell(iCell, coordsC);
                    GetCoordsOnCell(iCell, coordsCu, uCur);
                    GetCoordsOnCell(iCell, coordsCuinc, duCur);
                    auto localNodeIdx = Eigen::ArrayXi::LinSpaced(c2n.size(), 0, c2n.size() - 1);

                    auto eCell = GetCellElement(iCell);
                    int nBi = eCell.GetO2NumBisect();
                    int iBiVariant = GetO2ElemBisectVariant(eCell, coordsC);

                    auto FJDetRatio = [&](real alpha) -> real
                    {
                        real JDetRatio = 1;
                        for (int iBi = 0; iBi < nBi; iBi++)
                        {
                            auto eCellSub = eCell.ObtainO2BisectElem(iBi);
                            Eigen::ArrayXi c2nSubLocal;
                            c2nSubLocal.resize(eCellSub.GetNumNodes());
                            eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, localNodeIdx, c2nSubLocal);
                            SmallCoordsAsVector coordsCSub, coordsCuSub, coordsCuincSub;
                            coordsCSub.resize(3, eCellSub.GetNumNodes());
                            coordsCuSub.resize(3, eCellSub.GetNumNodes());
                            coordsCuincSub.resize(3, eCellSub.GetNumNodes());
                            eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, coordsC, coordsCSub);
                            eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, coordsCu, coordsCuSub);
                            eCell.ExtractO2BisectElemNodes(iBi, iBiVariant, coordsCuinc, coordsCuincSub);
                            auto qCellSub = Elem::Quadrature{eCellSub, 2}; // for O1 FEM
                            Elem::SummationNoOp NoOp;
                            qCellSub.Integration(
                                NoOp,
                                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                                {
                                    // J = dx_i/dxii_j, Jinv = dxii_i/dx_j
                                    auto getJ = [&](auto coos)
                                    {
                                        tJacobi J = Elem::ShapeJacobianCoordD01Nj(coos, DiNj);
                                        tJacobi JInv = tJacobi::Identity();
                                        real JDet;
                                        if (dim == 2)
                                            JDet = J(Eigen::all, 0).cross(J(Eigen::all, 1))(2); // ! note: for 2D, incompatible with curve surface situation
                                        else
                                            JDet = J.determinant();
                                        if (dim == 2)
                                            JInv({0, 1}, {0, 1}) = J({0, 1}, {0, 1}).inverse().eval();
                                        else
                                            JInv = J.inverse().eval();
                                        return std::make_tuple(J, JInv, JDet);
                                    };
                                    auto [J, JInv, JDet] = getJ(coordsCSub + coordsCuSub);
                                    DNDS_assert(JDet > 0);
                                    auto [Jn, JInvn, JDetn] = getJ(coordsCSub + coordsCuSub + coordsCuincSub * alpha);
                                    JDetRatio = std::min(JDetRatio, JDetn / JDet);
                                });
                        }
                        return JDetRatio;
                    };
                    real aLim = 1;
                    real decay = 0.5;
                    real minJDetRatio = 0.1;
                    for (int i = 0; i < 20; i++)
                        if (FJDetRatio(aLim) < minJDetRatio)
                            aLim *= decay;
                    for (auto iN : c2n)
                        if (iN < duCurLim.father->Size())
                            duCurLim[iN] = duCur[iN].array().sign() * duCurLim[iN].array().abs().min(duCur[iN].array().abs() * aLim);
                    minLim = std::min(minLim, aLim);
                    if (aLim < 1)
                        nFix++;
                }

            duCurLim.trans.startPersistentPull();
            duCurLim.trans.waitPersistentPull();
            // for (index iN = 0; iN < A.size(); iN++)
            //     for (auto &ME : A[iN])
            //         std::cout << "ij: " << iN << " " << ME.j << "\n"
            //                   << ME.m << std::endl;
            MPI::AllreduceOneIndex(nFix, MPI_SUM, mpi);
            MPI::AllreduceOneReal(minLim, MPI_MIN, mpi);
            if (mpi.rank == mRank)
            {
                log() << fmt::format("UnstructuredMesh === ElevatedNodesSolveInternalSmooth(): Disp Limited, nLim {}, minLim {:.3e} ",
                                     nFix, minLim)
                      << std::endl;
            }
            return minLim;
        };

        auto MatVec = [&](tCoordPair &x, tCoordPair &Ax)
        {
            for (index iN = 0; iN < x.father->Size(); iN++)
            {
                Ax[iN].setZero();
                for (auto &ME : A[iN])
                    Ax[iN] += ME.m * x[ME.j];
            }
            Ax.trans.startPersistentPull();
            Ax.trans.waitPersistentPull();
        };

        struct InitForEigenSparse
        {
            using value_type = index;
            std::vector<std::vector<MatElem>> &A;
            index operator[](index i) const
            {
                return A[i / 3].size() * 3;
            }
        } initForEigenSparse{A};

        Eigen::SparseMatrix<real> ADiag(dispO2.father->Size() * 3, dispO2.father->Size() * 3);
        ADiag.reserve(initForEigenSparse);
        // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
        for (index iN = 0; iN < dispO2.father->Size(); iN++)
        {
            for (auto &ME : A[iN])
                if (ME.j < dispO2.father->Size())
                    for (int iii = 0; iii < 3; iii++)
                        for (int jjj = 0; jjj < 3; jjj++)
                            ADiag.insert(iN * 3 + iii, ME.j * 3 + jjj) =
                                ME.m(iii, jjj);
        }
        if (mpi.rank == mRank)
        {
            log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth(): Eigen Matrix Filled" << std::endl;
        }
        // std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
        Eigen::SparseLU<Eigen::SparseMatrix<real>, Eigen::COLAMDOrdering<int>> LUSolver;
        if (dispO2.father->Size())
        {
            // LUSolver.analyzePattern(ADiag);
            // LUSolver.factorize(ADiag); // full LU perconditioner
            // std::cout << "det " << LUSolver.absDeterminant() << std::endl;
        }
        if (mpi.rank == mRank)
        {
            log() << "UnstructuredMesh === ElevatedNodesSolveInternalSmooth(): LU Factorized" << std::endl;
        }

        auto DiagSolveVec = [&](tCoordPair &x, tCoordPair &ADIx)
        {
            Eigen::Vector<real, Eigen::Dynamic> v;
            v.resize(x.father->Size() * 3);
            for (index iN = 0; iN < x.father->Size(); iN++)
            {
                v[iN * 3 + 0] = x[iN](0);
                v[iN * 3 + 1] = x[iN](1);
                v[iN * 3 + 2] = x[iN](2);
            }
            Eigen::Vector<real, Eigen::Dynamic> sol;
            if (dispO2.father->Size())
            {
                sol = LUSolver.solve(v);
            }
            for (index iN = 0; iN < x.father->Size(); iN++)
            {
                ADIx[iN](0) = sol[iN * 3 + 0];
                ADIx[iN](1) = sol[iN * 3 + 1];
                ADIx[iN](2) = sol[iN * 3 + 2];
            }
            ADIx.trans.startPersistentPull();
            ADIx.trans.waitPersistentPull();
        };
        auto JacobiPrec = [&](tCoordPair &x, tCoordPair &ADIx)
        {
            for (index iN = 0; iN < x.father->Size(); iN++)
            {
                ADIx[iN] = A[iN][0].m.fullPivLu().solve(x[iN]);
            }
            ADIx.trans.startPersistentPull();
            ADIx.trans.waitPersistentPull();
        };
        // for(int iIter = 1; iIter <= 200; iIter ++)
        // {
        //     real incC = UpdateDispO2Jacobi(dispO2, dispO2, false); // JOR iteration
        //     dispO2.trans.startPersistentPull();
        //     dispO2.trans.waitPersistentPull();
        //     if(mpi.rank == mRank)
        //         log() << fmt::format("iIter [{}] Jacobi Increment [{:3e}]", iIter,incC) << std::endl;
        // }

        /**********************************************/
        // Linear GMRES

        Linear::GMRES_LeftPreconditioned<CoordPairDOF> gmres(5, initDOF);
        gmres.solve(
            [&](CoordPairDOF &x, CoordPairDOF &Ax)
            {
                MatVec(x, Ax);
            },
            [&](CoordPairDOF &x, CoordPairDOF &MLx)
            {
                JacobiPrec(x, MLx);
            },
            [&](CoordPairDOF &a, CoordPairDOF &b) -> real
            {
                return a.dot(b);
            },
            bO2,
            dispO2,
            elevationInfo.nIter,
            [&](int iRestart, real res, real resB)
            {
                // std::cout << mpi.rank << "gmres here" << std::endl;
                if (mpi.rank == mRank)
                    log() << fmt::format("iRestart [{}] res [{:3e}] -> [{:3e}]", iRestart, resB, res) << std::endl;
                return res < resB * 1e-10;
            });

        /**********************************************/
        // Nonlin

        // Linear::GMRES_LeftPreconditioned<CoordPairDOF> gmres(5, initDOF);
        // real incNorm0{0}, rhsNorm0{0};
        // dispO2.setConstant(0.0);
        // for (int iIter = 1; iIter <= elevationInfo.nIter; iIter++)
        // {
        //     bO2.setConstant(0.0); // remember this
        //     AssembleRHSMat(dispO2);
        //     MatVec(dispO2, dispO2PrecRHS);  // "dispO2PrecRHS" = Ax
        //     bO2.addTo(dispO2PrecRHS, -1.0); // RHS
        //     real rhsNorm = bO2.norm2();
        //     dispO2Inc.setConstant(0.0);

        //     gmres.solve(
        //         [&](CoordPairDOF &x, CoordPairDOF &Ax)
        //         {
        //             MatVec(x, Ax);
        //         },
        //         [&](CoordPairDOF &x, CoordPairDOF &MLx)
        //         {
        //             JacobiPrec(x, MLx);
        //         },
        //         bO2,
        //         dispO2Inc,
        //         2,
        //         [&](int iRestart, real res, real resB)
        //         {
        //             // std::cout << mpi.rank << "gmres here" << std::endl;
        //             if (mpi.rank == mRank)
        //                 log() << fmt::format("step [{}]          iRestart [{}] res [{:3e}] -> [{:3e}]", iIter, iRestart, resB, res) << std::endl;
        //             return res < resB * 1e-10;
        //         });
        //     dispO2IncLimited = dispO2Inc;
        //     real minLim = LimitDisp(dispO2, dispO2Inc, dispO2IncLimited);
        //     dispO2IncLimited = dispO2Inc;
        //     dispO2IncLimited *= minLim;
        //     dispO2.addTo(dispO2IncLimited, 1.0);
        //     real incNorm = dispO2IncLimited.norm2();
        //     if (iIter == 1)
        //         incNorm0 = incNorm, rhsNorm0 = rhsNorm;
        //     if (mpi.rank == mRank)
        //         log() << fmt::format("step [{}] increment [{:3e}] -> [{:3e}]  rhs [{:3e}] -> [{:3e}]", iIter, incNorm0, incNorm, rhsNorm0, rhsNorm) << std::endl;
        // }
        /**********************************************/
        // dispO2 = bO2;

        for (index iN = 0; iN < coords.father->Size(); iN++)
        {
            if (dim == 2)
                dispO2[iN](2) = 0;
            // if(dispO2[iN].norm() != 0)
            //     std::cout << dispO2[iN].transpose() << std::endl;
            coords[iN] += dispO2[iN];
        }
        coords.trans.pullOnce();
    }
}
