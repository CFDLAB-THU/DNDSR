#pragma once

#include "Mesh.hpp"

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <fmt/core.h>
#include "Geom/Mesh/Mesh_DeviceView.hpp"

namespace DNDS::Geom
{
    // =================================================================
    // File-local helpers for InterpolateFace
    // =================================================================
    namespace
    {
        /**
         * \brief Result of face enumeration from cell connectivity.
         */
        struct FaceEnumerationResult
        {
            std::vector<std::vector<DNDS::index>> face2nodeV;
            std::vector<std::vector<NodePeriodicBits>> face2nodePbiV;
            std::vector<std::pair<DNDS::index, DNDS::index>> face2cellV;
            std::vector<ElemInfo> faceElemInfoV;
            DNDS::index nFaces = 0;
        };

        /**
         * \brief Enumerate unique faces from cell-to-node connectivity.
         *
         * Iterates over all cells (local + ghost), extracts topological faces,
         * deduplicates by sorted vertex comparison (with periodic-aware matching
         * when \p isPeriodic is true), and populates \p cell2face entries.
         *
         * \param[in,out] cell2face     Cell-to-face pair (rows resized and entries set).
         * \param[in]     cellElemInfo  Cell element info (for element type).
         * \param[in]     cell2node     Cell-to-node connectivity.
         * \param[in]     cell2nodePbi  Cell-to-node periodic bits (used when isPeriodic).
         * \param[in]     cell2cellSize Total number of cells (father + son).
         * \param[in]     coordsSize    Total number of nodes (father + son), for node2face sizing.
         * \param[in]     isPeriodic    Whether periodic mesh handling is enabled.
         */
        [[maybe_unused]] FaceEnumerationResult EnumerateFacesFromCells(
            tAdjPair &cell2face,
            const tElemInfoArrayPair &cellElemInfo,
            const tAdjPair &cell2node,
            const tPbiPair &cell2nodePbi,
            DNDS::index cell2cellSize,
            DNDS::index coordsSize,
            bool isPeriodic)
        {
            FaceEnumerationResult result;
            auto &face2nodeV = result.face2nodeV;
            auto &face2nodePbiV = result.face2nodePbiV;
            auto &face2cellV = result.face2cellV;
            auto &faceElemInfoV = result.faceElemInfoV;
            auto &nFaces = result.nFaces;

            std::vector<std::vector<DNDS::index>> node2face(coordsSize);

            using idx_pbi_pair = std::pair<index, NodePeriodicBits>;
            auto idx_pbi_pair_less = [](idx_pbi_pair &L, idx_pbi_pair &R)
            { return L.first == R.first ? uint8_t(L.second) < uint8_t(R.second) : L.first < R.first; };
            auto idx_pbi_pair_eq = [](idx_pbi_pair &L, idx_pbi_pair &R)
            { return L.first == R.first && uint8_t(L.second) == uint8_t(R.second); };

            for (DNDS::index iCell = 0; iCell < cell2cellSize; iCell++)
            {
                auto eCell = Elem::Element{cellElemInfo[iCell]->getElemType()};
                cell2face.ResizeRow(iCell, eCell.GetNumFaces());
                for (int ic2f = 0; ic2f < eCell.GetNumFaces(); ic2f++)
                {
                    auto eFace = eCell.ObtainFace(ic2f);
                    std::vector<DNDS::index> faceNodes(eFace.GetNumNodes());
                    eCell.ExtractFaceNodes(ic2f, cell2node[iCell], faceNodes);
                    DNDS::index iFound = -1;
                    std::vector<DNDS::index> faceVerts(faceNodes.begin(), faceNodes.begin() + eFace.GetNumVertices());
                    std::sort(faceVerts.begin(), faceVerts.end());

                    std::vector<NodePeriodicBits> faceNodePeriodicBits;
                    std::vector<idx_pbi_pair> faceNodeNodePeriodicBits;
                    if (isPeriodic)
                    {
                        faceNodePeriodicBits.resize(eFace.GetNumNodes());
                        faceNodeNodePeriodicBits.resize(eFace.GetNumNodes());
                        eCell.ExtractFaceNodes(ic2f, cell2nodePbi[iCell], faceNodePeriodicBits);
                        for (size_t i = 0; i < faceNodeNodePeriodicBits.size(); i++)
                            faceNodeNodePeriodicBits[i].first = faceNodes[i], faceNodeNodePeriodicBits[i].second = faceNodePeriodicBits[i];
                        std::sort(faceNodeNodePeriodicBits.begin(), faceNodeNodePeriodicBits.end(),
                                  idx_pbi_pair_less);
                        for (size_t i = 1; i < faceNodeNodePeriodicBits.size(); i++)
                        {
                            DNDS_assert_info(!idx_pbi_pair_eq(faceNodeNodePeriodicBits[i - 1], faceNodeNodePeriodicBits[i]),
                                             "the face has identical (periodic) nodes");
                        }
                    }

                    for (auto iV : faceVerts)
                        if (iFound < 0)
                            for (auto iFOther : node2face[iV])
                            {
                                auto eFaceOther = Elem::Element{faceElemInfoV[iFOther].getElemType()};
                                if (eFaceOther.type != eFace.type)
                                    continue;
                                std::vector<DNDS::index> faceVertsOther(
                                    face2nodeV[iFOther].begin(),
                                    face2nodeV[iFOther].begin() + eFace.GetNumVertices());
                                std::sort(faceVertsOther.begin(), faceVertsOther.end());
                                if (std::equal(faceVerts.begin(), faceVerts.end(), faceVertsOther.begin(), faceVertsOther.end()))
                                {
                                    if (isPeriodic)
                                    {
                                        std::vector<std::pair<index, NodePeriodicBits>> faceNodeNodePeriodicBitsOther(eFaceOther.GetNumNodes());
                                        for (size_t i = 0; i < faceNodeNodePeriodicBitsOther.size(); i++)
                                            faceNodeNodePeriodicBitsOther[i].first = face2nodeV[iFOther][i],
                                            faceNodeNodePeriodicBitsOther[i].second = face2nodePbiV[iFOther][i];
                                        std::sort(faceNodeNodePeriodicBitsOther.begin(), faceNodeNodePeriodicBitsOther.end(),
                                                  idx_pbi_pair_less);
                                        auto v0 = faceNodeNodePeriodicBits.at(0).second ^ faceNodeNodePeriodicBitsOther.at(0).second;
                                        DNDS_assert(faceNodeNodePeriodicBitsOther.size() == faceNodeNodePeriodicBits.size());
                                        bool collaborating = true;
                                        for (size_t i = 1; i < faceNodeNodePeriodicBitsOther.size(); i++)
                                            if ((faceNodeNodePeriodicBits[i].second ^ faceNodeNodePeriodicBitsOther[i].second) != v0)
                                                collaborating = false;
                                        if (collaborating)
                                            iFound = iFOther;
                                    }
                                    else
                                        iFound = iFOther;
                                }
                            }
                    if (iFound < 0)
                    {
                        // * face not existent yet
                        face2nodeV.emplace_back(faceNodes); // note: faceVerts invalid here!
                        if (isPeriodic)
                            face2nodePbiV.emplace_back(faceNodePeriodicBits);
                        face2cellV.emplace_back(iCell, DNDS::UnInitIndex);
                        // important note: f2nPbi node pbi is always same as cell f2c[0]'s corresponding nodes
                        faceElemInfoV.emplace_back(ElemInfo{eFace.type, 0});
                        for (auto iV : faceVerts)
                            node2face[iV].push_back(nFaces);
                        cell2face(iCell, ic2f) = nFaces;
                        nFaces++;
                    }
                    else
                    {
                        DNDS_assert(face2cellV[iFound].second == DNDS::UnInitIndex);
                        face2cellV[iFound].second = iCell;
                        cell2face(iCell, ic2f) = iFound;
                    }
                }
            }
            return result;
        }

        /**
         * \brief Result of face collection/filtering pass.
         */
        struct FaceCollectionResult
        {
            std::vector<index> iFaceAllToCollected;         ///< mapping: old face idx -> collected idx (or UnInitIndex/-1)
            std::vector<std::vector<index>> faceSendLocals; ///< per-rank lists of local face indices to send as ghost
            index nFacesNew = 0;
        };

        /**
         * \brief Filter faces: discard ghost-only and duplicate cross-rank faces,
         *        assign ownership to the rank with the lower rank ID.
         *
         * \param[in] faceElemInfoV     Per-face element info (zone for internal/bnd distinction).
         * \param[in] face2cellV        Per-face cell pair (first, second).
         * \param[in] nFaces            Total enumerated faces.
         * \param[in] nLocalCells       Number of local (father) cells.
         * \param[in] pLGhostMapping    Ghost mapping from cell2node.trans (for global index lookup).
         * \param[in] pLGlobalMapping   Global mapping from cell2node.father (for rank search).
         * \param[in] nMPIRanks         Number of MPI ranks.
         * \param[in] myRank            This process's MPI rank.
         */
        [[maybe_unused]] FaceCollectionResult CollectFaces(
            const std::vector<ElemInfo> &faceElemInfoV,
            const std::vector<std::pair<DNDS::index, DNDS::index>> &face2cellV,
            DNDS::index nFaces,
            DNDS::index nLocalCells,
            const DNDS::OffsetAscendIndexMapping &ghostMapping,
            const DNDS::GlobalOffsetsMapping &globalMapping,
            DNDS::MPI_int nMPIRanks,
            DNDS::MPI_int myRank)
        {
            FaceCollectionResult result;
            result.iFaceAllToCollected.resize(nFaces);
            result.faceSendLocals.resize(nMPIRanks);
            result.nFacesNew = 0;

            for (index iFace = 0; iFace < nFaces; iFace++)
            {
                if (faceElemInfoV[iFace].zone <= 0) // if internal
                {
                    // NOLINTNEXTLINE(bugprone-branch-clone): distinct conditions with same outcome
                    if (face2cellV[iFace].second == UnInitIndex && face2cellV[iFace].first >= nLocalCells) // has not other cell with ghost parent
                        result.iFaceAllToCollected[iFace] = UnInitIndex;                                   // * discard
                    // NOLINTNEXTLINE(bugprone-branch-clone): distinct conditions with same outcome
                    else if (face2cellV[iFace].first >= nLocalCells &&
                             face2cellV[iFace].second >= nLocalCells)    // both sides ghost
                        result.iFaceAllToCollected[iFace] = UnInitIndex; // * discard
                    else if (face2cellV[iFace].first >= nLocalCells ||
                             face2cellV[iFace].second >= nLocalCells)
                    {
                        DNDS_assert(face2cellV[iFace].second >= nLocalCells); // should only be the internal as first
                        // * check both sided's info //TODO: optimize so that pLGhostMapping returns rank directly ?
                        index cellGlobL = ghostMapping(-1, face2cellV[iFace].first);
                        index cellGlobR = ghostMapping(-1, face2cellV[iFace].second);
                        MPI_int rankL = UnInitMPIInt, rankR = UnInitMPIInt;
                        index valL = UnInitIndex, valR = UnInitIndex;
                        auto retL = globalMapping.search(cellGlobL, rankL, valL);
                        auto retR = globalMapping.search(cellGlobR, rankR, valR);
                        DNDS_assert(retL && retR && (rankL != rankR));
                        if (rankL > rankR)
                        {
                            result.iFaceAllToCollected[iFace] = -1; // * discard but with ghost
                        }
                        else
                        {
                            DNDS_assert(rankL == myRank);
                            result.faceSendLocals[rankR].push_back(result.nFacesNew);
                            result.iFaceAllToCollected[iFace] = result.nFacesNew++; //*use
                        }
                    }
                    else
                    {
                        result.iFaceAllToCollected[iFace] = result.nFacesNew++; //*use
                    }
                }
                else // all bnds would be non duplicate
                {
                    result.iFaceAllToCollected[iFace] = result.nFacesNew++; //*use
                }
            }
            return result;
        }

        /**
         * \brief Copy collected (non-discarded) face data from temporary vectors
         *        into the resized face member arrays, then remap cell2face entries.
         *
         * \param[in]  faceEnum           Face enumeration result (temp vectors).
         * \param[in]  faceCollect        Face collection result (mapping + nFacesNew).
         * \param[out] face2cell          Face-to-cell pair (father resized and filled).
         * \param[out] face2node          Face-to-node pair (father resized and filled).
         * \param[out] face2nodePbi       Face-to-node periodic bits pair (if periodic).
         * \param[out] faceElemInfo        Face element info pair (father resized and filled).
         * \param[in,out] cell2face       Cell-to-face pair (entries remapped via iFaceAllToCollected).
         * \param[in]  isPeriodic         Whether periodic mesh handling is enabled.
         * \param[in]  mpiComm            MPI communicator for barrier.
         */
        [[maybe_unused]] void CompactFacesAndRemapCell2Face(
            const FaceEnumerationResult &faceEnum,
            const FaceCollectionResult &faceCollect,
            tAdj2Pair &face2cell,
            tAdjPair &face2node,
            tPbiPair &face2nodePbi,
            tElemInfoArrayPair &faceElemInfo,
            tAdjPair &cell2face,
            bool isPeriodic,
            MPI_Comm mpiComm)
        {
            const auto &face2nodeV = faceEnum.face2nodeV;
            const auto &face2nodePbiV = faceEnum.face2nodePbiV;
            const auto &face2cellV = faceEnum.face2cellV;
            const auto &faceElemInfoV = faceEnum.faceElemInfoV;
            const auto &iFaceAllToCollected = faceCollect.iFaceAllToCollected;
            index nFacesNew = faceCollect.nFacesNew;
            index nFaces = faceEnum.nFaces;

            face2cell.father->Resize(nFacesNew);
            face2node.father->Resize(nFacesNew);
            if (isPeriodic)
                face2nodePbi.father->Resize(nFacesNew);
            faceElemInfo.father->Resize(nFacesNew); //! considering globally duplicate faces
            nFacesNew = 0;
            for (DNDS::index iFace = 0; iFace < nFaces; iFace++)
            {
                if (iFaceAllToCollected[iFace] >= 0) // ! -1 is also ignored!
                {
                    face2node.ResizeRow(nFacesNew, face2nodeV[iFace].size());
                    for (DNDS::rowsize if2n = 0; if2n < face2node.RowSize(nFacesNew); if2n++)
                        face2node(nFacesNew, if2n) = face2nodeV[iFace][if2n];
                    if (isPeriodic)
                    {
                        DNDS_assert(face2nodeV[iFace].size() == face2nodePbiV[iFace].size());
                        face2nodePbi.ResizeRow(nFacesNew, face2nodePbiV[iFace].size());
                        for (DNDS::rowsize if2n = 0; if2n < face2nodePbi.RowSize(nFacesNew); if2n++)
                            face2nodePbi(nFacesNew, if2n) = face2nodePbiV[iFace][if2n];
                    }
                    face2cell(nFacesNew, 0) = face2cellV[iFace].first;
                    face2cell(nFacesNew, 1) = face2cellV[iFace].second;
                    faceElemInfo(nFacesNew, 0) = faceElemInfoV[iFace];
                    nFacesNew++;
                }
            }

            MPI::Barrier(mpiComm);
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
            for (DNDS::index iCell = 0; iCell < cell2face.Size(); iCell++) // convert face indices pointers
            {
                for (rowsize ic2f = 0; ic2f < cell2face.RowSize(iCell); ic2f++)
                {
                    cell2face(iCell, ic2f) = iFaceAllToCollected[cell2face(iCell, ic2f)]; // Uninit if to discard
                }
            }
        }

        /**
         * \brief Match boundary elements to faces, populating faceElemInfo zone IDs
         *        and building bnd2face / face2bnd mappings.
         *
         * For each boundary, finds the face that shares the same node set via
         * the parent cell's cell2face connectivity.
         */
        void MatchBoundariesToFaces(
            const tElemInfoArrayPair &bndElemInfo,
            const tAdj2Pair &bnd2cell,
            const tAdjPair &bnd2node,
            const tAdjPair &cell2face,
            const tAdjPair &face2node,
            const tAdjPair &cell2node,
            tElemInfoArrayPair &faceElemInfo,
            std::vector<index> &bnd2faceV,
            std::unordered_map<index, index> &face2bndM,
            tAdj1Pair &face2bnd,
            tAdj1Pair &bnd2face)
        {
            bnd2faceV.resize(bndElemInfo.father->Size(), -1); // this mapping only uses main (father) part
            face2bndM.reserve(bndElemInfo.father->Size());
            std::unordered_map<index, index> iFace2iBnd;
            for (DNDS::index iBnd = 0; iBnd < bndElemInfo.father->Size(); iBnd++)
            {
                DNDS::index pCell = bnd2cell(iBnd, 0);
                std::vector<DNDS::index> b2nRow = bnd2node[iBnd];
                std::sort(b2nRow.begin(), b2nRow.end());
                int nFound = 0;
                auto faceID = bndElemInfo[iBnd]->zone;
                for (int ic2f = 0; ic2f < cell2face.RowSize(pCell); ic2f++)
                {
                    auto iFace = cell2face(pCell, ic2f);
                    if (iFace < 0) //==-1, pointing to ghost face
                        continue;
                    std::vector<DNDS::index> f2nRow = face2node[iFace];
                    std::sort(f2nRow.begin(), f2nRow.end());
                    if (std::equal(b2nRow.begin(), b2nRow.end(), f2nRow.begin(), f2nRow.end()))
                    {
                        if (iFace2iBnd.count(iFace))
                        {
                            DNDS_assert(FaceIDIsPeriodic(faceID)); // only periodic gets to be duplicated
                            index iBndOther = iFace2iBnd[iFace];
                            index iCellA = bnd2cell[iBnd][0];
                            index iCellB = bnd2cell[iBndOther][0];
                            DNDS_assert(iCellA < cell2node.father->Size()); // both points to local non-ghost cells
                            DNDS_assert(iCellB < cell2node.father->Size()); // both points to local non-ghost cells
                            // std::cout << iCellA << " vs " << iCellB << std::endl;
                            if (iCellA > iCellB) // make face corresponds with f2c[0]'s bnd if possible
                                continue;
                        }
                        iFace2iBnd[iFace] = iBnd;
                        nFound++; // two things:
                        // if is periodic, then only gets the bnd info of the main cell's bnd;
                        // if is external bc, then must be non-ghost face
                        faceElemInfo(iFace, 0) = bndElemInfo(iBnd, 0);
                        bnd2faceV[iBnd] = iFace;
                        face2bndM[iFace] = iBnd;
                        DNDS_assert_info(FaceIDIsExternalBC(faceID) ||
                                             FaceIDIsPeriodic(faceID),
                                         "bnd elem should have a BC id not interior");
                    }
                }
                DNDS_assert(nFound > 0 || (FaceIDIsPeriodic(faceID) && nFound == 0)); // periodic could miss the face
            }

            face2bnd.father->Resize(face2node.father->Size());
            for (index iFace = 0; iFace < face2bnd.father->Size(); iFace++)
                face2bnd.father->operator()(iFace) = face2bndM.count(iFace) ? face2bndM[iFace] : UnInitIndex;
            bnd2face.father->Resize(bnd2node.father->Size());
            for (index iBnd = 0; iBnd < bnd2node.father->Size(); iBnd++)
                bnd2face.father->operator()(iBnd) = bnd2faceV.at(iBnd);
        }

        /**
         * \brief Match received ghost faces to cells and assign cell2face entries
         *        that were previously marked -1 (pointing to ghost).
         *
         * For each ghost face, iterates over both its cells, finds the matching
         * topological face slot in cell2face, and assigns the ghost face index.
         */
        [[maybe_unused]] void AssignGhostFacesToCells(
            const tAdj2 &face2cellSon,
            const tAdj &face2nodeSon,
            const tElemInfoArray &faceElemInfoSon,
            tAdjPair &cell2face,
            const tAdjPair &cell2node,
            const tElemInfoArrayPair &cellElemInfo,
            DNDS::index nFatherFaces)
        {
#ifdef DNDS_USE_OMP
#    pragma omp parallel for
#endif
            for (DNDS::index iFace = 0; iFace < face2cellSon->Size(); iFace++) // face2cell points to local now
            {
                // before: first points to inner, //!relies on the order of setting face2cell
                DNDS_assert((*face2cellSon)(iFace, 0) >= cell2node.father->Size());
                auto eFace = Elem::Element{(*faceElemInfoSon)(iFace, 0).getElemType()};
                auto faceVerts = std::vector<index>((*face2nodeSon)[iFace].begin(), (*face2nodeSon)[iFace].begin() + eFace.GetNumVertices());
                std::sort(faceVerts.begin(), faceVerts.end()); //* do not forget to do set operation sort first
                for (rowsize if2c = 0; if2c < 2; if2c++)
                {
                    index iCell = (*face2cellSon)(iFace, if2c);
                    auto cell2faceRow = cell2face[iCell];
                    auto cellNodes = cell2node[iCell];
                    auto eCell = Elem::Element{cellElemInfo(iCell, 0).getElemType()};
                    bool found = false;
                    for (rowsize ic2f = 0; ic2f < cell2face.RowSize(iCell); ic2f++)
                    {
                        auto eFace = eCell.ObtainFace(ic2f);
                        std::vector<index> faceNodesC(eFace.GetNumNodes());
                        eCell.ExtractFaceNodes(ic2f, cellNodes, faceNodesC);
                        std::sort(faceNodesC.begin(), faceNodesC.end());
                        if (std::includes(faceNodesC.begin(), faceNodesC.end(), faceVerts.begin(), faceVerts.end()))
                        {
                            DNDS_assert(cell2face(iCell, ic2f) == -1);
                            cell2face(iCell, ic2f) = iFace + nFatherFaces; // remember is ghost
                            found = true;
                        }
                    }
                    DNDS_assert(found);
                }
            }
        }
    } // anonymous namespace
}