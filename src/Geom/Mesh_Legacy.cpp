/// @file Mesh_Legacy.cpp
/// @brief Legacy implementations of mesh pipeline methods, preserved for
///        comparison testing against DSL-based replacements.
///
/// Contains:
///   - GeneralCell2NodeToNode2Cell (static helper)
///   - RecoverNode2CellAndNode2BndLegacy
///   - RecoverCell2CellAndBnd2CellLegacy
///
/// These methods are functionally identical to the pre-DSL code. The DSL
/// replacements (RecoverNode2CellAndNode2Bnd, RecoverCell2CellAndBnd2Cell)
/// use MeshConnectivity::Inverse and ComposeFiltered respectively.

#include "Mesh.hpp"

#include <unordered_map>
#include <unordered_set>
#include <set>
#include <fmt/core.h>

namespace DNDS::Geom
{
    using tIndexMapFunc = std::function<index(index)>;

    static void GeneralCell2NodeToNode2Cell(
        tCoordPair &coords, tAdjPair &cell2node, tAdjPair &node2cell,
        const tIndexMapFunc &CellIndexLocal2Global_NoSon,
        const tIndexMapFunc &NodeIndexLocal2Global_NoSon)
    {
        const auto &mpi = coords.father->getMPI();
        std::unordered_set<index> ghostNodesCompactSet;
        std::vector<index> ghostNodesCompact;
        std::unordered_map<index, std::unordered_set<index>> node2CellLocalRecord;

        for (index iCell = 0; iCell < cell2node.father->Size(); iCell++)
            for (auto iNode : cell2node[iCell])
            {
                auto [ret, rank, val] = coords.father->pLGlobalMapping->search(iNode);
                DNDS_assert_info(ret, "search failed");
                if (rank != mpi.rank)
                    ghostNodesCompact.push_back(iNode), ghostNodesCompactSet.insert(iNode);
                node2CellLocalRecord[iNode].insert(CellIndexLocal2Global_NoSon(iCell));
            }

        // MPI_Barrier(mpi.comm);
        // std::cout << "here2 " << std::endl;

        tAdj node2cellPast; // + node2cell * a triplet to deal with reverse inserting
        node2cell.InitPair("node2cell", mpi);
        node2cellPast = make_ssp<tAdj::element_type>(ObjName{"node2cellPast"}, mpi);
        //* fill into father
        node2cell.father->Resize(coords.father->Size());
        for (index iNode = 0; iNode < coords.father->Size(); iNode++)
        {
            index iNodeG = NodeIndexLocal2Global_NoSon(iNode);
            if (node2CellLocalRecord.count(iNodeG))
            {
                node2cell.ResizeRow(iNode, node2CellLocalRecord[iNodeG].size());
                rowsize in2c = 0;
                for (auto v : node2CellLocalRecord[iNodeG])
                    node2cell(iNode, in2c++) = v;
            }
        }
        node2cell.TransAttach();
        node2cell.trans.createFatherGlobalMapping();
        node2cell.trans.createGhostMapping(ghostNodesCompact);
        //* fill into son
        node2cell.son->Resize(node2cell.trans.pLGhostMapping->ghostIndex.size());
        // std::unordered_set<index> touched; // only used for checking
        for (auto &[k, s] : node2CellLocalRecord)
        {
            MPI_int rank{-1};
            index val{-1};
            if (!node2cell.trans.pLGhostMapping->search(k, rank, val))
                DNDS_assert_info(false, "search failed");
            if (rank >= 0)
            {
                node2cell.son->ResizeRow(val, s.size());
                rowsize in2c = 0;
                for (auto v : s)
                    node2cell.son->operator()(val, in2c++) = v;
                // touched.insert(val);
            }
        }
        // DNDS_assert(touched.size() == node2cell.son->Size());

        // node2cell.trans.pLGhostMapping->pushingIndexGlobal; // where to receive in a push
        DNDS::ArrayTransformerType<tAdj::element_type>::Type node2cellPastTrans;
        node2cellPastTrans.setFatherSon(node2cell.son, node2cellPast);
        node2cellPastTrans.createFatherGlobalMapping();
        std::vector<index> pushSonSeries(node2cell.son->Size());
        for (index i = 0; i < node2cell.son->Size(); i++)
            pushSonSeries[i] = i;
        node2cellPastTrans.createGhostMapping(pushSonSeries, node2cell.trans.pLGhostMapping->ghostStart);
        node2cellPastTrans.createMPITypes();

        node2cellPastTrans.pullOnce();
        DNDS_assert(DNDS::size_to_index(node2cell.trans.pLGhostMapping->ghostIndex.size()) == node2cell.son->Size());
        DNDS_assert(DNDS::size_to_index(node2cell.trans.pLGhostMapping->pushingIndexGlobal.size()) == node2cellPast->Size());
        // * this state of triplet: node2cell.father - node2cell.son - node2cellPast forms a "unique pushing" for the pair node2cell
        // * should be made into some standard
        for (index i = 0; i < node2cellPast->Size(); i++)
        {
            index iNodeG = node2cell.trans.pLGhostMapping->pushingIndexGlobal[i]; //?should be right
            for (auto iCell : (*node2cellPast)[i])
                node2CellLocalRecord[iNodeG].insert(iCell);
        }
        // MPISerialDo(
        //     mpi,
        //     [&]()
        //     {
        //         for (auto &[k, s] : node2CellLocalRecord)
        //         {
        //             if (NodeIndexGlobal2Local_NoSon(k) >= 0 && s.size() != 4)
        //                 std::cout << k << ", " << s.size() << "; " << std::flush;
        //         }
        //         std::cout << std::endl;
        //     });

        // reset pair
        node2cell.InitPair("node2cell", mpi);
        //* fill into father
        node2cell.father->Resize(coords.father->Size());
        for (index iNode = 0; iNode < coords.father->Size(); iNode++)
        {
            index iNodeG = NodeIndexLocal2Global_NoSon(iNode);
            if (node2CellLocalRecord.count(iNodeG))
            {
                node2cell.ResizeRow(iNode, node2CellLocalRecord[iNodeG].size());
                rowsize in2c = 0;
                for (auto v : node2CellLocalRecord[iNodeG])
                    node2cell(iNode, in2c++) = v;
            }
        }
    }

    void UnstructuredMesh::
        RecoverNode2CellAndNode2BndLegacy()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
        DNDS_assert(coords.father);
        DNDS_assert(cell2node.father);
        DNDS_assert(bnd2node.father);

        /*****************************************************/
        // * first recover node2cell

        if (!coords.father->pLGlobalMapping)
            coords.father->createGlobalMapping(); // for NodeIndexLocal2Global_NoSon
        if (!cell2node.father->pLGlobalMapping)
            cell2node.father->createGlobalMapping(); // for CellIndexLocal2Global_NoSon

        GeneralCell2NodeToNode2Cell(
            coords, cell2node, node2cell,
            [this](index v)
            { return this->CellIndexLocal2Global_NoSon(v); },
            [this](index v)
            { return this->NodeIndexLocal2Global_NoSon(v); });

        if (!bnd2node.father->pLGlobalMapping)
            bnd2node.father->createGlobalMapping(); // for BndIndexLocal2Global_NoSon
        GeneralCell2NodeToNode2Cell(
            coords, bnd2node, node2bnd,
            [this](index v)
            { return this->BndIndexLocal2Global_NoSon(v); },
            [this](index v)
            { return this->NodeIndexLocal2Global_NoSon(v); });

        this->adjN2CBState = Adj_PointToGlobal;

        // if (mpi.rank == 0)
        // {
        //     for (index i = 0; i < node2cell.father->Size(); i++)
        //         std::cout << node2cell.RowSize(i) - 4 << std::endl;
        //     for (index i = 0; i < node2bnd.father->Size(); i++)
        //         std::cout << node2bnd.RowSize(i) + 10 << std::endl;
        // }

        // node2cell.TransAttach();
        // node2cell.trans.createFatherGlobalMapping();
        // node2cell.trans.createGhostMapping(ghostNodesCompact);
        // node2cell.trans.createMPITypes();
        // node2cell.trans.pullOnce();

        // mesh->node2cell.TransAttach();
        // mesh->node2cell.trans.BorrowGGIndexing(mesh->coords);
        // mesh->node2cell.trans.createMPITypes();
        // mesh->node2cell.trans.pullOnce();
    }

    void UnstructuredMesh::RecoverCell2CellAndBnd2CellLegacy()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
        DNDS_assert(adjN2CBState == Adj_PointToGlobal);
        DNDS_assert(coords.father);
        DNDS_assert(cell2node.father);
        DNDS_assert(bnd2node.father);
        DNDS_assert(node2cell.father);

        coords.TransAttach();
        coords.trans.createFatherGlobalMapping(); // for NodeIndexLocal2Global_NoSon
        cell2node.TransAttach();
        cell2node.trans.createFatherGlobalMapping(); // for CellIndexLocal2Global_NoSon
        bnd2node.TransAttach();
        bnd2node.trans.createFatherGlobalMapping(); // for BndIndexLocal2Global_NoSon

        // std::cout << "RecoverCell2CellAndBnd2Cell here1" << std::endl;

        std::unordered_set<index> ghostNodesCompactSet;
        for (index i = 0; i < cell2node.father->Size(); i++)
            for (auto in : cell2node.father->operator[](i))
                if (NodeIndexGlobal2Local_NoSon(in) < 0)
                    ghostNodesCompactSet.insert(in);
        for (index i = 0; i < bnd2node.father->Size(); i++)
            for (auto in : bnd2node.father->operator[](i))
                if (NodeIndexGlobal2Local_NoSon(in) < 0)
                    ghostNodesCompactSet.insert(in);
        std::vector<index> ghostNodes;
        ghostNodes.reserve(ghostNodesCompactSet.size());
        for (auto v : ghostNodesCompactSet)
            ghostNodes.push_back(v);
        // std::cout << "RecoverCell2CellAndBnd2Cell here2" << std::endl;
        node2cell.son = make_ssp<decltype(node2cell.son)::element_type>(ObjName{"node2cell.son"}, mpi);
        node2cell.TransAttach();
        node2cell.trans.createFatherGlobalMapping();
        node2cell.trans.createGhostMapping(ghostNodes);
        node2cell.trans.createMPITypes(); //! warning, this is not actual final official trans, just needed temporarily
        node2cell.trans.pullOnce();

        std::unordered_map<index, index> iNodeGlobal2LocalAppendInNode2Cell;

        for (index i = 0; i < node2cell.Size(); i++)
            iNodeGlobal2LocalAppendInNode2Cell[node2cell.trans.pLGhostMapping->operator()(-1, i)] = i;

        for (index i = 0; i < cell2node.father->Size(); i++)
            for (auto in : cell2node.father->operator[](i))
                DNDS_assert(iNodeGlobal2LocalAppendInNode2Cell.count(in));

        cell2cell.InitPair("cell2cell", mpi); // actual outputs need empty but constructed son
        cell2cell.father->Resize(cell2node.father->Size());
        for (index i = 0; i < cell2node.father->Size(); i++)
        {
            std::set<index> cellRec;
            for (auto in : cell2node.father->operator[](i))
            {
                DNDS_assert(iNodeGlobal2LocalAppendInNode2Cell.count(in));
                for (auto ico : node2cell[iNodeGlobal2LocalAppendInNode2Cell.at(in)])
                    cellRec.insert(ico);
            }
            auto ret = cellRec.erase(CellIndexLocal2Global_NoSon(i));
            DNDS_assert(ret == 1);
            cell2cell.father->ResizeRow(i, cellRec.size());
            rowsize ic2c = 0;
            for (auto v : cellRec)
                cell2cell.father->operator()(i, ic2c++) = v;
        }
        // std::cout << "RecoverCell2CellAndBnd2Cell here2.5" << std::endl;
        bnd2cell.InitPair("bnd2cell", mpi); // actual outputs need empty but constructed son
        bnd2cell.father->Resize(bnd2node.father->Size());

        // For periodic meshes, store per-bnd candidate cell sets from the node
        // intersection pass, then resolve via pbi check in a second pass after
        // ghost-pulling cell2node/cell2nodePbi for the candidate cells.
        std::vector<std::set<index>> bndCellCandidates;
        if (isPeriodic)
            bndCellCandidates.resize(bnd2node.father->Size());

        for (index i = 0; i < bnd2node.father->Size(); i++)
        {
            std::set<index> cellRecCur;
            bool initDone{false};
            // std::cout << "RecoverCell2CellAndBnd2Cell here L1 1" << std::endl;
            for (auto in : bnd2node.father->operator[](i))
            {
                DNDS_assert(iNodeGlobal2LocalAppendInNode2Cell.count(in));
                std::set<index> cellRecCurNew;
                for (auto ico : node2cell[iNodeGlobal2LocalAppendInNode2Cell.at(in)])
                    if (!initDone || cellRecCur.count(ico))
                        cellRecCurNew.insert(ico);
                std::swap(cellRecCur, cellRecCurNew);
                initDone = true;
            }

            // Periodic pbi check and bnd2cell assignment are deferred to the
            // second pass below (after ghost-pulling cell2node/cell2nodePbi for
            // the candidate cells). Store candidates per bnd for now.
            if (isPeriodic)
                bndCellCandidates[i] = std::move(cellRecCur);
            else
            {
                DNDS_assert_info(cellRecCur.size() == 1, fmt::format("cellRecCur.size() is [{}]", cellRecCur.size()));
                bnd2cell.father->operator()(i, 0) = *cellRecCur.begin();
                bnd2cell.father->operator()(i, 1) = UnInitIndex;
            }
        }

        /************************************************************/
        // Periodic: ghost-pull cell2node and cell2nodePbi for all candidate
        // cells, then do the pbi filter and donor/receiver assignment.
        /************************************************************/
        if (isPeriodic)
        {
            // Collect all unique candidate cells across all periodic bnds
            std::vector<index> neededCells;
            for (index i = 0; i < bnd2cell.father->Size(); i++)
            {
                if (!Geom::FaceIDIsPeriodic(bndElemInfo.father->operator()(i, 0).zone))
                    continue;
                for (auto ic : bndCellCandidates[i])
                    neededCells.push_back(ic);
            }

            cell2nodePbi.son = make_ssp<decltype(cell2nodePbi.son)::element_type>(ObjName{"cell2nodePbi.son"}, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
            cell2nodePbi.TransAttach();
            cell2nodePbi.trans.createFatherGlobalMapping();
            cell2nodePbi.trans.createGhostMapping(neededCells); //! warning, this is not actual final official trans, just needed temporarily
            cell2nodePbi.trans.createMPITypes();
            cell2nodePbi.trans.pullOnce();
            cell2node.son = make_ssp<decltype(cell2node.son)::element_type>(ObjName{"cell2node.son"}, mpi);
            cell2node.BorrowAndPull(cell2nodePbi); //! warning, this is not actual final official trans, just needed temporarily

            // Now do the periodic pbi filter for each bnd
            for (index i = 0; i < bnd2cell.father->Size(); i++)
            {
                if (!Geom::FaceIDIsPeriodic(bndElemInfo.father->operator()(i, 0).zone))
                    continue;

                auto &cellRecCur = bndCellCandidates[i]; // the pre-pbi candidates (from node intersection)
                std::set<index> cellRecFiltered;
                for (auto ic : cellRecCur)
                {
                    auto [ret, rank, icAppend] = cell2nodePbi.trans.pLGhostMapping->search_indexAppend(ic);
                    DNDS_assert_info(ret, fmt::format("periodic bnd {} candidate cell {} not found in ghost mapping", i, ic));

                    bool cellContainsBnd = true;
                    for (int ib2n = 0; ib2n < bnd2node.father->operator[](i).size(); ib2n++)
                    {
                        index iNode = bnd2node.father->operator[](i)[ib2n];
                        auto iNodePbi = bnd2nodePbi.father->operator[](i)[ib2n];
                        int nIndexMatchNode{0}, nIndexPBIMatchNode{0};
                        for (int ic2n = 0; ic2n < cell2node[icAppend].size(); ic2n++)
                            if (iNode == cell2node(icAppend, ic2n))
                            {
                                nIndexMatchNode++;
                                if (iNodePbi == cell2nodePbi(icAppend, ic2n))
                                    nIndexPBIMatchNode++;
                            }
                        DNDS_assert(nIndexMatchNode >= 1);
                        if (nIndexPBIMatchNode == 0)
                            cellContainsBnd = false;
                    }
                    if (cellContainsBnd)
                        cellRecFiltered.insert(ic);
                }

                // Periodic bnd: 2 cells or 1 (self-periodic)
                DNDS_assert_info(cellRecFiltered.size() == 2 || cellRecFiltered.size() == 1,
                                 fmt::format("periodic bnd {} has {} cells after pbi filter", i, cellRecFiltered.size()));
                auto it = cellRecFiltered.begin();
                bnd2cell.father->operator()(i, 0) = *it;
                if (cellRecFiltered.size() == 2)
                    ++it;
                bnd2cell.father->operator()(i, 1) = *it;
            }

            // Swap check: ensure bnd2cell(i, 0) is the donor-side cell
            for (index i = 0; i < bnd2cell.father->Size(); i++)
                if (bnd2cell(i, 1) != UnInitIndex && bnd2cell(i, 0) != bnd2cell(i, 1) /* no need to check if both sides are same*/)
                {
                    index ic0 = bnd2cell(i, 0);
                    index ic1 = bnd2cell(i, 1);
                    auto [ret0, rank0, ic0L] = cell2nodePbi.trans.pLGhostMapping->search_indexAppend(ic0);
                    auto [ret1, rank1, ic1L] = cell2nodePbi.trans.pLGhostMapping->search_indexAppend(ic1);
                    // std::cout << fmt::format("ic0L ic1L, {},{}", ic0L, ic1L) << std::endl;
                    DNDS_assert_info(ret0 && ret1, "search failed");
                    std::vector<Geom::NodePeriodicBits> pbi0, pbi1;

                    for (auto in : bnd2node[i])
                    {
                        bool found0{false}, found1{false};
                        for (rowsize ic2n = 0; ic2n < cell2node[ic0L].size(); ic2n++)
                            if (cell2node[ic0L][ic2n] == in)
                                found0 = true, pbi0.push_back(cell2nodePbi(ic0L, ic2n));
                        for (rowsize ic2n = 0; ic2n < cell2node[ic1L].size(); ic2n++)
                            if (cell2node[ic1L][ic2n] == in)
                                found1 = true, pbi1.push_back(cell2nodePbi(ic1L, ic2n));
                        DNDS_assert(found0 && found1);
                    }
                    auto cleanOtherBits = [&](Geom::NodePeriodicBits &b)
                    {
                        if (Geom::FaceIDIsPeriodic1(bndElemInfo(i, 0).zone))
                            b = b & nodePB1;
                        else if (Geom::FaceIDIsPeriodic2(bndElemInfo(i, 0).zone))
                            b = b & nodePB2;
                        else if (Geom::FaceIDIsPeriodic3(bndElemInfo(i, 0).zone))
                            b = b & nodePB3;
                    };
                    for (auto &b : pbi0)
                        cleanOtherBits(b);
                    for (auto &b : pbi1)
                        cleanOtherBits(b);

                    Geom::NodePeriodicBits bndBit;
                    if (bndElemInfo(i, 0).zone == Geom::BC_ID_PERIODIC_1_DONOR)
                        bndBit.setP1True();
                    else if (bndElemInfo(i, 0).zone == Geom::BC_ID_PERIODIC_2_DONOR)
                        bndBit.setP2True();
                    else if (bndElemInfo(i, 0).zone == Geom::BC_ID_PERIODIC_3_DONOR)
                        bndBit.setP3True();
                    else if (!Geom::FaceIDIsPeriodic(bndElemInfo(i, 0).zone))
                        DNDS_assert_info(false, "this bnd with both sides has to be periodic");

                    bool match0{true}, match1{true};
                    for (auto b : pbi0)
                        if (b ^ bndBit)
                            match0 = false;
                    for (auto b : pbi1)
                        if (b ^ bndBit)
                            match1 = false;
                    if (match0 && !match1)
                        ; // keep
                    else if (match1 && !match0)
                        std::swap(bnd2cell(i, 0), bnd2cell(i, 1));
                    else
                    {
                        if (mpi.rank >= 0)
                        {
                            for (auto b : pbi0)
                                DNDS::log() << b << ", ";
                            DNDS::log() << " ---- ";
                            for (auto b : pbi1)
                                DNDS::log() << b << ", ";
                            DNDS::log() << " ---- " << bndBit;
                            DNDS::log() << std::endl;
                        }
                        DNDS_assert_info(false,
                                         match0
                                             ? "this periodic bnd matches both sides"
                                             : "this periodic bnd matches no sides");
                    }
                    // if (mpi.rank == 0)
                    // {
                    //     if (match1 && !match0)
                    //         std::cout << "swap " << i << std::endl;
                    // }
                }
        }
    }

} // namespace DNDS::Geom
