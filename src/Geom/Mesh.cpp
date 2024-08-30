#include "Mesh.hpp"

#include <cstdlib>
#include <string>
#include <map>
#include <omp.h>
#include <filesystem>
#include <unordered_map>
#include <fmt/core.h>
#include <Eigen/Sparse>

#include <unordered_set>

namespace DNDS::Geom
{

    // void UnstructuredMeshSerialRW::InterpolateTopology() //!could be useful for parallel?
    // {
    //     // count node 2 face
    //     DNDS_MAKE_SSP(cell2faceSerial, mesh->getMPI());
    //     DNDS_MAKE_SSP(face2cellSerial, mesh->getMPI());
    //     DNDS_MAKE_SSP(face2nodeSerial, mesh->getMPI());
    //     DNDS_MAKE_SSP(faceElemInfoSerial, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());

    //     if (mRank != mesh->getMPI().rank)
    //         return;

    //     for (DNDS::index iCell = 0; iCell <cell2nodeSerial->Size(); iCell++)
    //     {

    //         // TODO
    //     }
    // }

    /*******************************************************************************************************************/
    /*******************************************************************************************************************/
    /*******************************************************************************************************************/

    /**
     * Used for generating pushing data structure
     *  @todo: //TODO test on parallel re-distributing
     */
    // for one-shot usage, partition data corresponds to mpi

    template <class TPartitionIdx>
    void Partition2LocalIdx(
        const std::vector<TPartitionIdx> &partition,
        std::vector<DNDS::index> &localPush,
        std::vector<DNDS::index> &localPushStart, const DNDS::MPIInfo &mpi)
    {
        // localPushStart.resize(mpi.size);
        std::vector<DNDS::index> localPushSizes(mpi.size, 0);
        for (auto r : partition)
        {
            DNDS_assert(r < mpi.size);
            localPushSizes[r]++;
        }
        DNDS::AccumulateRowSize(localPushSizes, localPushStart);
        localPush.resize(localPushStart[mpi.size]);
        localPushSizes.assign(mpi.size, 0);
        DNDS_assert(partition.size() == localPush.size());
        for (DNDS::index i = 0; i < partition.size(); i++)
            localPush[localPushStart[partition[i]] + (localPushSizes[partition[i]]++)] = i;
    }

    /**
     * Serial2Global is used for converting adj data to point to reordered global
     * @todo: //TODO: get fully serial version
     *  @todo: //TODO test on parallel re-distributing
     */
    template <class TPartitionIdx>
    void Partition2Serial2Global(
        const std::vector<TPartitionIdx> &partition,
        std::vector<DNDS::index> &serial2Global, const DNDS::MPIInfo &mpi, DNDS::MPI_int nPart)
    {
        serial2Global.resize(partition.size());
        /****************************************/
        std::vector<DNDS::index> numberAtLocal(nPart, 0);
        for (auto r : partition)
            numberAtLocal[r]++;
        std::vector<DNDS::index> numberTotal(nPart), numberPrev(nPart);
        MPI::Allreduce(numberAtLocal.data(), numberTotal.data(), nPart, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        MPI_Scan(numberAtLocal.data(), numberPrev.data(), nPart, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        std::vector<DNDS::index> numberTotalPlc(nPart + 1);
        numberTotalPlc[0] = 0;
        for (DNDS::MPI_int r = 0; r < nPart; r++)
            numberTotalPlc[r + 1] = numberTotalPlc[r] + numberTotal[r], numberPrev[r] -= numberAtLocal[r];
        // 2 things here: accumulate total and subtract local from prev
        /****************************************/
        numberAtLocal.assign(numberAtLocal.size(), 0);
        DNDS::index iFill = 0;
        for (auto r : partition)
            serial2Global[iFill++] = (numberAtLocal[r]++) + numberTotalPlc[r] + numberPrev[r];
    }

    /**
     * In this section, Serial means un-reordered index, i-e
     * pointing to the *serial data, cell2node then pointing to the
     * cell2node global indices:
     * 0 1; 2 3
     *
     * Global means the re-ordered data's global indices
     * if partition ==
     * 0 1; 0 1
     * then Serial 2 Global is:
     * 0 2; 1 3
     *
     * if in proc 0, a cell refers to node 2, then must be seen in JSG ghost, gets global output == 1
     *
     * comm complexity: same as data comm
     * @todo: //TODO test on parallel re-distributing
     * @todo: //TODO: fully serial emulator
     */

    template <class TAdj = tAdj1>
    void ConvertAdjSerial2Global(TAdj &arraySerialAdj,
                                 const std::vector<DNDS::index> &partitionJSerial2Global,
                                 const DNDS::MPIInfo &mpi)
    {
        // IndexArray JSG(IndexArray::tContext(partitionJSerial2Global.size()), mpi);
        // forEachInArray(
        //     JSG, [&](IndexArray::tComponent &e, index i)
        //     { e[0] = partitionJSerial2Global[i]; });
        // JSGGhost.createGlobalMapping();
        tAdj1 JSG, JSGGhost;
        DNDS_MAKE_SSP(JSG, mpi);
        DNDS_MAKE_SSP(JSGGhost, mpi);
        JSG->Resize(partitionJSerial2Global.size());
        for (DNDS::index i = 0; i < JSG->Size(); i++)
            (*JSG)(i, 0) = partitionJSerial2Global[i];
        JSG->createGlobalMapping();
        std::vector<DNDS::index> ghostJSerialQuery;

        // get ghost
        DNDS::index nGhost = 0;
        for (DNDS::index i = 0; i < arraySerialAdj->Size(); i++)
        {
            for (DNDS::rowsize j = 0; j < arraySerialAdj->RowSize(i); j++)
            {
                DNDS::index v = (*arraySerialAdj)(i, j);
                if (v == DNDS::UnInitIndex)
                    break;
                DNDS::MPI_int rank = -1;
                DNDS::index val = -1;
                if (!JSG->pLGlobalMapping->search(v, rank, val))
                    DNDS_assert_info(false, "search failed");
                if (rank != mpi.rank) //! excluding self
                    nGhost++;
            }
        }
        ghostJSerialQuery.reserve(nGhost);
        for (DNDS::index i = 0; i < arraySerialAdj->Size(); i++)
        {
            for (DNDS::rowsize j = 0; j < arraySerialAdj->RowSize(i); j++)
            {
                DNDS::index v = (*arraySerialAdj)(i, j);
                if (v == DNDS::UnInitIndex)
                    break;
                DNDS::MPI_int rank = -1;
                DNDS::index val = -1;
                JSG->pLGlobalMapping->search(v, rank, val);
                if (rank != mpi.rank) //! excluding self
                    ghostJSerialQuery.push_back(v);
            }
        }
        // PrintVec(ghostJSerialQuery, std::cout);
        typename DNDS::ArrayTransformerType<tAdj1::element_type>::Type JSGTrans;
        JSGTrans.setFatherSon(JSG, JSGGhost);
        JSGTrans.createGhostMapping(ghostJSerialQuery);
        JSGTrans.createMPITypes();
        JSGTrans.pullOnce();

        for (DNDS::index i = 0; i < arraySerialAdj->Size(); i++)
        {
            for (DNDS::rowsize j = 0; j < arraySerialAdj->RowSize(i); j++)
            {
                DNDS::index &v = (*arraySerialAdj)(i, j);
                if (v == DNDS::UnInitIndex)
                    break;
                DNDS::MPI_int rank = -1;
                DNDS::index val = -1;
                if (!JSGTrans.pLGhostMapping->search(v, rank, val))
                    DNDS_assert_info(false, "search failed");
                if (rank == -1)
                    v = (*JSG)(val, 0);
                else
                    v = (*JSGGhost)(val, 0);
            }
        }
    }

    /**
     * @todo: //TODO: fully serial emulator
     */
    template <class TArr = tAdj1>
    void TransferDataSerial2Global(TArr &arraySerial,
                                   TArr &arrayDist,
                                   const std::vector<DNDS::index> &pushIndex,
                                   const std::vector<DNDS::index> &pushIndexStart,
                                   const DNDS::MPIInfo &mpi)
    {
        typename DNDS::ArrayTransformerType<typename TArr::element_type>::Type trans;
        trans.setFatherSon(arraySerial, arrayDist);
        trans.createFatherGlobalMapping();
        trans.createGhostMapping(pushIndex, pushIndexStart);
        trans.createMPITypes();
        trans.pullOnce();
    }

    //! inefficient, use Partition2Serial2Global ! only used for convenient comparison
    void PushInfo2Serial2Global(std::vector<DNDS::index> &serial2Global,
                                DNDS::index localSize,
                                const std::vector<DNDS::index> &pushIndex,
                                const std::vector<DNDS::index> &pushIndexStart,
                                const DNDS::MPIInfo &mpi)
    {
        tIndPair Serial2Global;
        DNDS_MAKE_SSP(Serial2Global.father, mpi);
        DNDS_MAKE_SSP(Serial2Global.son, mpi);
        Serial2Global.father->Resize(localSize);
        Serial2Global.TransAttach();
        Serial2Global.trans.createFatherGlobalMapping();
        Serial2Global.trans.createGhostMapping(pushIndex, pushIndexStart);
        Serial2Global.trans.createMPITypes();
        Serial2Global.son->createGlobalMapping();
        // Set son to son's global
        for (DNDS::index iSon = 0; iSon < Serial2Global.son->Size(); iSon++)
            (*Serial2Global.son)[iSon] = Serial2Global.son->pLGlobalMapping->operator()(mpi.rank, iSon);
        Serial2Global.trans.pushOnce();
        serial2Global.resize(localSize);
        for (DNDS::index iFat = 0; iFat < Serial2Global.father->Size(); iFat++)
            serial2Global[iFat] = Serial2Global.father->operator[](iFat);
    }

    // template <class TAdj = tAdj1>
    // void ConvertAdjSerial2Global(TAdj &arraySerialAdj,
    //                              const std::vector<DNDS::index> &partitionJSerial2Global,
    //                              const DNDS::MPIInfo &mpi)
    // {
    // }
    /*******************************************************************************************************************/
    /*******************************************************************************************************************/
    /*******************************************************************************************************************/

    void UnstructuredMeshSerialRW::
        PartitionReorderToMeshCell2Cell()
    {
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing  PartitionReorderToMeshCell2Cell" << std::endl;
        DNDS_assert(cnPart == mesh->getMPI().size);
        // * 1: get the nodal partition
        nodePartition.resize(coordSerial->Size(), static_cast<DNDS::MPI_int>(INT32_MAX));
        for (DNDS::index iCell = 0; iCell < cell2nodeSerial->Size(); iCell++)
            for (DNDS::rowsize ic2n = 0; ic2n < (*cell2nodeSerial).RowSize(iCell); ic2n++)
                nodePartition[(*cell2nodeSerial)(iCell, ic2n)] = std::min(nodePartition[(*cell2nodeSerial)(iCell, ic2n)], cellPartition.at(iCell));
        // * 1: get the bnd partition
        bndPartition.resize(bnd2cellSerial->Size());
        for (DNDS::index iBnd = 0; iBnd < bnd2cellSerial->Size(); iBnd++)
            bndPartition[iBnd] = cellPartition[(*bnd2cellSerial)(iBnd, 0)];

        std::vector<DNDS::index> cell_push, cell_pushStart, node_push, node_pushStart, bnd_push, bnd_pushStart;
        Partition2LocalIdx(cellPartition, cell_push, cell_pushStart, mesh->getMPI());
        Partition2LocalIdx(nodePartition, node_push, node_pushStart, mesh->getMPI());
        Partition2LocalIdx(bndPartition, bnd_push, bnd_pushStart, mesh->getMPI());
        std::vector<DNDS::index> cell_Serial2Global, node_Serial2Global, bnd_Serial2Global;
        Partition2Serial2Global(cellPartition, cell_Serial2Global, mesh->getMPI(), mesh->getMPI().size);
        Partition2Serial2Global(nodePartition, node_Serial2Global, mesh->getMPI(), mesh->getMPI().size);
        // Partition2Serial2Global(bndPartition, bnd_Serial2Global, mesh->getMPI(), mesh->getMPI().size);//seems not needed for now
        // PushInfo2Serial2Global(cell_Serial2Global, cellPartition.size(), cell_push, cell_pushStart, mesh->getMPI());//*safe validation version
        // PushInfo2Serial2Global(node_Serial2Global, nodePartition.size(), node_push, node_pushStart, mesh->getMPI());//*safe validation version
        // PushInfo2Serial2Global(bnd_Serial2Global, bndPartition.size(), bnd_push, bnd_pushStart, mesh->getMPI());    //*safe validation version
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing PartitionReorderToMeshCell2Cell ConvertAdjSerial2Global" << std::endl;
        ConvertAdjSerial2Global(cell2nodeSerial, node_Serial2Global, mesh->getMPI());
        ConvertAdjSerial2Global(cell2cellSerial, cell_Serial2Global, mesh->getMPI());
        ConvertAdjSerial2Global(bnd2nodeSerial, node_Serial2Global, mesh->getMPI());
        ConvertAdjSerial2Global(bnd2cellSerial, cell_Serial2Global, mesh->getMPI());

        DNDS_MAKE_SSP(mesh->coords.father, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->coords.son, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->cellElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        DNDS_MAKE_SSP(mesh->cellElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        DNDS_MAKE_SSP(mesh->bndElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        DNDS_MAKE_SSP(mesh->bndElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        DNDS_MAKE_SSP(mesh->cell2node.father, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->cell2node.son, mesh->getMPI());
        if (mesh->isPeriodic)
        {
            DNDS_MAKE_SSP(mesh->cell2nodePbi.father, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->getMPI());
            DNDS_MAKE_SSP(mesh->cell2nodePbi.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->getMPI());
        }
        DNDS_MAKE_SSP(mesh->cell2cell.father, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->cell2cell.son, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->bnd2node.father, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->bnd2node.son, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->bnd2cell.father, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->bnd2cell.son, mesh->getMPI());

        // coord transferring
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing PartitionReorderToMeshCell2Cell Trasfer Data Coord" << std::endl;
        TransferDataSerial2Global(coordSerial, mesh->coords.father, node_push, node_pushStart, mesh->getMPI());

        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing PartitionReorderToMeshCell2Cell Trasfer Data Cell" << std::endl;
        // cells transferring
        TransferDataSerial2Global(cell2cellSerial, mesh->cell2cell.father, cell_push, cell_pushStart, mesh->getMPI());
        TransferDataSerial2Global(cell2nodeSerial, mesh->cell2node.father, cell_push, cell_pushStart, mesh->getMPI());
        if (mesh->isPeriodic)
            TransferDataSerial2Global(cell2nodePbiSerial, mesh->cell2nodePbi.father, cell_push, cell_pushStart, mesh->getMPI());
        TransferDataSerial2Global(cellElemInfoSerial, mesh->cellElemInfo.father, cell_push, cell_pushStart, mesh->getMPI());
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing PartitionReorderToMeshCell2Cell Doing Trasfer Data Bnd" << std::endl;
        // bnds transferring
        TransferDataSerial2Global(bnd2cellSerial, mesh->bnd2cell.father, bnd_push, bnd_pushStart, mesh->getMPI());
        TransferDataSerial2Global(bnd2nodeSerial, mesh->bnd2node.father, bnd_push, bnd_pushStart, mesh->getMPI());
        TransferDataSerial2Global(bndElemInfoSerial, mesh->bndElemInfo.father, bnd_push, bnd_pushStart, mesh->getMPI());

        {
            DNDS::MPISerialDo(mesh->getMPI(), [&]()
                              { std::cout << "[" << mesh->getMPI().rank << ": nCell " << mesh->cell2cell.father->Size() << "] " << std::flush; });
            MPI::Barrier(mesh->getMPI().comm);
            if (mesh->getMPI().rank == 0)
                std::cout << std::endl;
            DNDS::MPISerialDo(mesh->getMPI(), [&]()
                              { std::cout << "[" << mesh->getMPI().rank << ": nNode " << mesh->coords.father->Size() << "] " << std::flush; });
            MPI::Barrier(mesh->getMPI().comm);
            if (mesh->getMPI().rank == 0)
                std::cout << std::endl;
            DNDS::MPISerialDo(mesh->getMPI(), [&]()
                              { std::cout << "[" << mesh->getMPI().rank << ": nBnd " << mesh->bnd2node.father->Size() << "] " << std::flush; });
            MPI::Barrier(mesh->getMPI().comm);
            if (mesh->getMPI().rank == 0)
                std::cout << std::endl;
        }
        mesh->adjPrimaryState = Adj_PointToGlobal;
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Done  PartitionReorderToMeshCell2Cell" << std::endl;
    }

    void UnstructuredMeshSerialRW::
        BuildSerialOut()
    {
        DNDS_assert(mesh->adjPrimaryState == Adj_PointToGlobal);
        mode = SerialOutput;
        dataIsSerialIn = false;
        dataIsSerialOut = true;

        std::vector<DNDS::index> serialPullCell;
        std::vector<DNDS::index> serialPullNode;
        // std::vector<DNDS::index> serialPullBnd;

        DNDS::index numCellGlobal = mesh->cellElemInfo.father->globalSize();
        // DNDS::index numBndGlobal = mesh->bndElemInfo.father->globalSize();
        DNDS::index numNodeGlobal = mesh->coords.father->globalSize();

        if (mesh->getMPI().rank == mRank)
        {
            serialPullCell.resize(numCellGlobal);
            serialPullNode.resize(numNodeGlobal);
            // serialPullBnd.reserve(numBndGlobal);
            for (DNDS::index i = 0; i < numCellGlobal; i++)
                serialPullCell[i] = i;
            for (DNDS::index i = 0; i < numNodeGlobal; i++)
                serialPullNode[i] = i;
            // for (DNDS::index i = 0; i < numBndGlobal; i++)
            //     serialPullBnd[i] = i;
        }
        DNDS_MAKE_SSP(cell2nodeSerial, mesh->getMPI());
        if (mesh->isPeriodic)
            DNDS_MAKE_SSP(cell2nodePbiSerial, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->getMPI());
        // DNDS_MAKE_SSP(bnd2nodeSerial, mesh->getMPI());
        DNDS_MAKE_SSP(coordSerial, mesh->getMPI());
        DNDS_MAKE_SSP(cellElemInfoSerial, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        // DNDS_MAKE_SSP(bndElemInfoSerial, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        // DNDS_MAKE_SSP(bnd2cellSerial, mesh->getMPI());// not needed yet

        coordSerialOutTrans.setFatherSon(mesh->coords.father, coordSerial);
        cell2nodeSerialOutTrans.setFatherSon(mesh->cell2node.father, cell2nodeSerial);
        if (mesh->isPeriodic)
            cell2nodePbiSerialOutTrans.setFatherSon(mesh->cell2nodePbi.father, cell2nodePbiSerial);
        // bnd2nodeSerialOutTrans.setFatherSon(mesh->bnd2node.father, bnd2nodeSerial);
        cellElemInfoSerialOutTrans.setFatherSon(mesh->cellElemInfo.father, cellElemInfoSerial);
        // bndElemInfoSerialOutTrans.setFatherSon(mesh->bndElemInfo.father, bndElemInfoSerial);

        // Father could already have global mapping, result should be the same
        coordSerialOutTrans.createFatherGlobalMapping();
        cell2nodeSerialOutTrans.createFatherGlobalMapping();
        if (mesh->isPeriodic)
            cell2nodePbiSerialOutTrans.createFatherGlobalMapping();
        // bnd2nodeSerialOutTrans.createFatherGlobalMapping();
        cellElemInfoSerialOutTrans.createFatherGlobalMapping();
        // bndElemInfoSerialOutTrans.createFatherGlobalMapping();

        coordSerialOutTrans.createGhostMapping(serialPullNode);
        cell2nodeSerialOutTrans.createGhostMapping(serialPullCell);
        if (mesh->isPeriodic)
            cell2nodePbiSerialOutTrans.createGhostMapping(serialPullCell);
        // bnd2nodeSerialOutTrans.createGhostMapping(serialPullBnd);
        cellElemInfoSerialOutTrans.BorrowGGIndexing(cell2nodeSerialOutTrans); // accidentally rewrites mesh->cellElemInfo.father's global mapping but ok
        // bndElemInfoSerialOutTrans.BorrowGGIndexing(bnd2nodeSerialOutTrans);

        coordSerialOutTrans.createMPITypes();
        cell2nodeSerialOutTrans.createMPITypes();
        if (mesh->isPeriodic)
            cell2nodePbiSerialOutTrans.createMPITypes();
        // bnd2nodeSerialOutTrans.createMPITypes();
        cellElemInfoSerialOutTrans.createMPITypes();
        // bndElemInfoSerialOutTrans.createMPITypes();

        coordSerialOutTrans.pullOnce();
        cell2nodeSerialOutTrans.pullOnce();
        if (mesh->isPeriodic)
            cell2nodePbiSerialOutTrans.pullOnce();
        // bnd2nodeSerialOutTrans.pullOnce();
        cellElemInfoSerialOutTrans.pullOnce();
        // bndElemInfoSerialOutTrans.pullOnce();
        if (mesh->getMPI().rank == mRank)
        {
            std::cout << "UnstructuredMeshSerialRW === BuildSerialOut Done " << std::endl;
        }
    }

}

namespace DNDS::Geom
{

    bool UnstructuredMesh::IsO1()
    {
        using namespace Elem;
        int hasBad = 0;
        for (index iCell = 0; iCell < cellElemInfo.Size(); iCell++)
        {
            auto eType = cellElemInfo(iCell, 0).getElemType();
            if (eType == ElemType::Line2 ||
                eType == ElemType::Tri3 ||
                eType == ElemType::Quad4 ||
                eType == ElemType::Tet4 ||
                eType == ElemType::Hex8 ||
                eType == ElemType::Prism6 ||
                eType == ElemType::Pyramid5)
            {
                continue;
            }
            else
            {
                hasBad = 1;
                break;
            }
        }
        int hasBadAll;
        MPI::Allreduce(&hasBad, &hasBadAll, 1, MPI_INT, MPI_SUM, mpi.comm);
        return hasBad == 0;
    }

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
                auto [ret, rank, val] = coords.trans.pLGlobalMapping->search(iNode);
                DNDS_assert_info(ret, "search failed");
                if (rank != mpi.rank)
                    ghostNodesCompact.push_back(iNode), ghostNodesCompactSet.insert(iNode);
                node2CellLocalRecord[iNode].insert(CellIndexLocal2Global_NoSon(iCell));
            }

        // MPI_Barrier(mpi.comm);
        // std::cout << "here2 " << std::endl;

        tAdj node2cellPast; // + node2cell * a triplet to deal with reverse inserting
        DNDS_MAKE_SSP(node2cell.father, mpi);
        DNDS_MAKE_SSP(node2cell.son, mpi);
        DNDS_MAKE_SSP(node2cellPast, mpi);
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
        DNDS_assert(node2cell.trans.pLGhostMapping->ghostIndex.size() == node2cell.son->Size());
        DNDS_assert(node2cell.trans.pLGhostMapping->pushingIndexGlobal.size() == node2cellPast->Size());
        // * this state of triplet: node2cell.father - node2cell.son - node2cellPast forms a "unique pushing" for the pair node2cell
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
        DNDS_MAKE_SSP(node2cell.father, mpi);
        DNDS_MAKE_SSP(node2cell.son, mpi);
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

    void UnstructuredMesh::RecoverNode2CellAndNode2Bnd()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
        DNDS_assert(coords.father);
        DNDS_assert(cell2node.father);
        DNDS_assert(bnd2node.father);

        /*****************************************************/
        // * first recover node2cell

        coords.TransAttach();
        coords.trans.createFatherGlobalMapping(); // for NodeIndexLocal2Global_NoSon
        cell2node.TransAttach();
        cell2node.trans.createFatherGlobalMapping(); // for CellIndexLocal2Global_NoSon
        GeneralCell2NodeToNode2Cell(
            coords, cell2node, node2cell,
            [this](index v)
            { return this->CellIndexLocal2Global_NoSon(v); },
            [this](index v)
            { return this->NodeIndexLocal2Global_NoSon(v); });

        bnd2node.TransAttach();
        bnd2node.trans.createFatherGlobalMapping(); // for BndIndexLocal2Global_NoSon
        GeneralCell2NodeToNode2Cell(
            coords, bnd2node, node2bnd,
            [this](index v)
            { return this->BndIndexLocal2Global_NoSon(v); },
            [this](index v)
            { return this->NodeIndexLocal2Global_NoSon(v); });
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
    }

    void UnstructuredMesh::
        BuildGhostPrimary()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
        /********************************/
        // cells
        {
            cell2cell.TransAttach();
            cell2node.TransAttach();
            if (isPeriodic)
                cell2nodePbi.TransAttach();
            cellElemInfo.TransAttach();

            cell2cell.trans.createFatherGlobalMapping();

            std::vector<DNDS::index> ghostCells;
            for (DNDS::index iCell = 0; iCell < cell2cell.father->Size(); iCell++)
            {
                for (DNDS::rowsize ic2c = 0; ic2c < cell2cell.father->RowSize(iCell); ic2c++)
                {
                    auto iCellOther = (*cell2cell.father)(iCell, ic2c);
                    DNDS::MPI_int rank;
                    DNDS::index val;
                    if (!cell2cell.trans.pLGlobalMapping->search(iCellOther, rank, val))
                        DNDS_assert_info(false, "search failed");
                    if (rank != mpi.rank)
                        ghostCells.push_back(iCellOther);
                }
            }
            cell2cell.trans.createGhostMapping(ghostCells);

            cell2node.trans.BorrowGGIndexing(cell2cell.trans);
            if (isPeriodic)
                cell2nodePbi.trans.BorrowGGIndexing(cell2cell.trans);
            cellElemInfo.trans.BorrowGGIndexing(cell2cell.trans);

            cell2cell.trans.createMPITypes();
            cell2node.trans.createMPITypes();
            if (isPeriodic)
                cell2nodePbi.trans.createMPITypes();
            cellElemInfo.trans.createMPITypes();

            cell2cell.trans.pullOnce();
            cell2node.trans.pullOnce();
            if (isPeriodic)
                cell2nodePbi.trans.pullOnce();
            cellElemInfo.trans.pullOnce();
        }
        // if(mpi.rank == 0)
        //     std::cout <<"XXXXXXXXXXXXXXXXXXXXXXXXX" <<std::endl;
        // if(mpi.rank == 0)
        //     for(index iC = 0; iC < coords.father->Size(); iC ++)
        //         std::cout << coords.father->operator[](iC).transpose() << std::endl;

        /********************************/
        // cells done, go on to nodes
        {
            coords.TransAttach();
            coords.trans.createFatherGlobalMapping();

            std::vector<DNDS::index> ghostNodes;
            for (DNDS::index iCell = 0; iCell < cell2cell.Size(); iCell++) // note doing full (son + father) traverse
            {
                for (DNDS::rowsize ic2c = 0; ic2c < cell2node.RowSize(iCell); ic2c++)
                {
                    auto iNode = cell2node(iCell, ic2c);
                    DNDS::MPI_int rank;
                    DNDS::index val;
                    if (!coords.trans.pLGlobalMapping->search(iNode, rank, val))
                        DNDS_assert_info(false, "search failed");
                    if (rank != mpi.rank)
                        ghostNodes.push_back(iNode);
                }
            }
            coords.trans.createGhostMapping(ghostNodes);
            coords.trans.createMPITypes();
            coords.trans.pullOnce();
        }
        // if(mpi.rank == 0)
        //     std::cout <<"XXXXXXXXXXXXXXXXXXXXXXXXX" <<std::endl;
        // if(mpi.rank == 0)
        //     for(index iC = 0; iC < coords.Size(); iC ++)
        //         std::cout << coords.operator[](iC).transpose() << std::endl;

        /********************************/
        // bnds: dummy now, no actual comm
        {

            bnd2cell.TransAttach();
            bnd2node.TransAttach();
            bndElemInfo.TransAttach();

            bnd2cell.trans.createFatherGlobalMapping();

            std::vector<DNDS::index> ghostBnds; // no ghosted bnds now
            bnd2cell.trans.createGhostMapping(ghostBnds);

            bnd2node.trans.BorrowGGIndexing(bnd2cell.trans);
            bndElemInfo.trans.BorrowGGIndexing(bnd2cell.trans);

            bnd2cell.trans.createMPITypes();
            bnd2node.trans.createMPITypes();
            bndElemInfo.trans.createMPITypes();

            bnd2cell.trans.pullOnce();
            bnd2node.trans.pullOnce();
            bndElemInfo.trans.pullOnce();
        }
    }

    void UnstructuredMesh::
        AdjGlobal2LocalPrimary()
    {
        // needs results of BuildGhostPrimary()
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);

        /**********************************/
        // convert bnd2cell, bnd2node, cell2cell, cell2node ptrs global to local
        for (DNDS::index iCell = 0; iCell < cell2cell.Size(); iCell++)
            for (DNDS::rowsize j = 0; j < cell2cell.RowSize(iCell); j++)
                cell2cell(iCell, j) = CellIndexGlobal2Local(cell2cell(iCell, j));

        for (DNDS::index iBnd = 0; iBnd < bnd2cell.Size(); iBnd++)
            for (DNDS::rowsize j = 0; j < bnd2cell.RowSize(iBnd); j++)
                bnd2cell(iBnd, j) = CellIndexGlobal2Local(bnd2cell(iBnd, j)),
                               DNDS_assert(j == 0 ? bnd2cell(iBnd, j) >= 0 : true); // must be inside

        for (DNDS::index iCell = 0; iCell < cell2node.Size(); iCell++)
            for (DNDS::rowsize j = 0; j < cell2node.RowSize(iCell); j++)
                cell2node(iCell, j) = NodeIndexGlobal2Local(cell2node(iCell, j)),
                                 DNDS_assert(cell2node(iCell, j) >= 0);

        for (DNDS::index iBnd = 0; iBnd < bnd2node.Size(); iBnd++)
            for (DNDS::rowsize j = 0; j < bnd2node.RowSize(iBnd); j++)
                bnd2node(iBnd, j) = NodeIndexGlobal2Local(bnd2node(iBnd, j)),
                               DNDS_assert(bnd2node(iBnd, j) >= 0);
        /**********************************/
        adjPrimaryState = Adj_PointToLocal;
    }

    void UnstructuredMesh::
        AdjLocal2GlobalPrimary()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        /**********************************/
        // convert bnd2cell, bnd2node, cell2cell, cell2node ptrs local to global
        /**********************************/
        // convert bnd2cell, bnd2node, cell2cell, cell2node ptrs global to local
        for (DNDS::index iCell = 0; iCell < cell2cell.Size(); iCell++)
            for (DNDS::rowsize j = 0; j < cell2cell.RowSize(iCell); j++)
                cell2cell(iCell, j) = CellIndexLocal2Global(cell2cell(iCell, j));

        for (DNDS::index iBnd = 0; iBnd < bnd2cell.Size(); iBnd++)
            for (DNDS::rowsize j = 0; j < bnd2cell.RowSize(iBnd); j++)
                bnd2cell(iBnd, j) = CellIndexLocal2Global(bnd2cell(iBnd, j)),
                               DNDS_assert(j == 0 ? bnd2cell(iBnd, j) >= 0 : true); // must be inside

        for (DNDS::index iCell = 0; iCell < cell2node.Size(); iCell++)
            for (DNDS::rowsize j = 0; j < cell2node.RowSize(iCell); j++)
                cell2node(iCell, j) = NodeIndexLocal2Global(cell2node(iCell, j)),
                                 DNDS_assert(cell2node(iCell, j) >= 0);

        for (DNDS::index iBnd = 0; iBnd < bnd2node.Size(); iBnd++)
            for (DNDS::rowsize j = 0; j < bnd2node.RowSize(iBnd); j++)
                bnd2node(iBnd, j) = NodeIndexLocal2Global(bnd2node(iBnd, j)),
                               DNDS_assert(bnd2node(iBnd, j) >= 0);
        /**********************************/
        adjPrimaryState = Adj_PointToGlobal;
    }

    void UnstructuredMesh::
        AdjGlobal2LocalPrimaryForBnd() // a reduction of primary version
    {
        // needs results of BuildGhostPrimary()
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
        /**********************************/
        // convert cell2node ptrs global to local
        for (DNDS::index iCell = 0; iCell < cell2node.Size(); iCell++)
            for (DNDS::rowsize j = 0; j < cell2node.RowSize(iCell); j++)
                cell2node(iCell, j) = NodeIndexGlobal2Local(cell2node(iCell, j)),
                                 DNDS_assert(cell2node(iCell, j) >= 0);
        /**********************************/
        adjPrimaryState = Adj_PointToLocal;
    }

    void UnstructuredMesh::
        AdjLocal2GlobalPrimaryForBnd() // a reduction of primary version
    {
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        /**********************************/
        // convert cell2node ptrs local to global
        for (DNDS::index iCell = 0; iCell < cell2node.Size(); iCell++)
            for (DNDS::rowsize j = 0; j < cell2node.RowSize(iCell); j++)
                cell2node(iCell, j) = NodeIndexLocal2Global(cell2node(iCell, j)),
                                 DNDS_assert(cell2node(iCell, j) >= 0);
        /**********************************/
        adjPrimaryState = Adj_PointToGlobal;
    }

    void UnstructuredMesh::
        AdjGlobal2LocalFacial()
    {
        DNDS_assert(adjFacialState == Adj_PointToGlobal);
        /**********************************/
        // convert face2cell ptrs and face2node ptrs global to local
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iFace = 0; iFace < face2cell.Size(); iFace++)
        {
            for (rowsize if2n = 0; if2n < face2node.RowSize(iFace); if2n++)
            {
                index &iNode = face2node(iFace, if2n);
                index val;
                MPI_int rank;
                auto ret = coords.trans.pLGhostMapping->search_indexAppend(iNode, rank, val);
                DNDS_assert(ret);
                iNode = val;
            }
            for (rowsize if2c = 0; if2c < 2; if2c++)
            {
                index &iCell = face2cell(iFace, if2c);
                if (iCell != UnInitIndex) // is not a bnd
                {
                    index val;
                    MPI_int rank;
                    auto ret = cell2node.trans.pLGhostMapping->search_indexAppend(iCell, rank, val);
                    DNDS_assert(ret);
                    iCell = val;
                }
            }
        }
        /**********************************/
        adjFacialState = Adj_PointToLocal;
    }

    void UnstructuredMesh::
        AdjLocal2GlobalFacial()
    {
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        /**********************************/
        // convert face2cell ptrs and face2node ptrs to global
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iFace = 0; iFace < face2cell.Size(); iFace++)
        {
            for (rowsize if2n = 0; if2n < face2node.RowSize(iFace); if2n++)
            {
                index &iNode = face2node(iFace, if2n);
                iNode = coords.trans.pLGhostMapping->operator()(-1, iNode);
            }
            for (rowsize if2c = 0; if2c < 2; if2c++)
            {
                index &iCell = face2cell(iFace, if2c);
                if (iCell != UnInitIndex) // is not a bnd
                    iCell = cell2node.trans.pLGhostMapping->operator()(-1, iCell);
            }
        }
        // MPI::Barrier(mpi.comm);
        /**********************************/
        adjFacialState = Adj_PointToGlobal;
    }

    void UnstructuredMesh::
        AdjLocal2GlobalC2F()
    {
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        /**********************************/
        // convert cell2face

        auto FaceIndexLocal2Global = [&](DNDS::index &iF)
        {
            if (iF == UnInitIndex)
                return;
            if (iF < 0) // mapping to un-found in father-son
                iF = -1 - iF;
            else
                iF = face2node.trans.pLGhostMapping->operator()(-1, iF);
        };

#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iCell = 0; iCell < cell2face.Size(); iCell++)
        {
            for (rowsize ic2f = 0; ic2f < cell2face.RowSize(iCell); ic2f++)
            {
                index &iFace = cell2face(iCell, ic2f);
                FaceIndexLocal2Global(iFace);
            }
        }
        // MPI::Barrier(mpi.comm);
        /**********************************/
        adjC2FState = Adj_PointToGlobal;
    }

    void UnstructuredMesh::
        AdjGlobal2LocalC2F()
    {
        DNDS_assert(adjC2FState == Adj_PointToGlobal);
        /**********************************/
        // convert cell2face

        auto FaceIndexGlobal2Local = [&](DNDS::index &iF)
        {
            if (iF == UnInitIndex)
                return;
            DNDS::MPI_int rank;
            DNDS::index val;
            auto result = face2node.trans.pLGhostMapping->search_indexAppend(iF, rank, val);
            if (result)
                iF = val;
            else
                iF = -1 - iF; // mapping to un-found in father-son
        };

#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iCell = 0; iCell < cell2face.Size(); iCell++)
        {
            for (rowsize ic2f = 0; ic2f < cell2face.RowSize(iCell); ic2f++)
            {
                index &iFace = cell2face(iCell, ic2f);
                FaceIndexGlobal2Local(iFace);
            }
        }
        /**********************************/
        adjC2FState = Adj_PointToLocal;
    }

    /// @todo //TODO: handle periodic cases
    void UnstructuredMesh::
        InterpolateFace()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToLocal); // And also should have primary ghost comm

        DNDS_MAKE_SSP(cell2face.father, mpi);
        DNDS_MAKE_SSP(cell2face.son, mpi);
        DNDS_MAKE_SSP(face2cell.father, mpi);
        DNDS_MAKE_SSP(face2cell.son, mpi);
        DNDS_MAKE_SSP(face2node.father, mpi);
        DNDS_MAKE_SSP(face2node.son, mpi);
        if (isPeriodic)
        {
            DNDS_MAKE_SSP(face2nodePbi.father, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
            DNDS_MAKE_SSP(face2nodePbi.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
        }
        DNDS_MAKE_SSP(faceElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        DNDS_MAKE_SSP(faceElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);

        cell2face.father->Resize(cell2cell.father->Size()); //!
        cell2face.son->Resize(cell2cell.son->Size());
        std::vector<std::vector<DNDS::index>> node2face(coords.Size());
        std::vector<std::vector<DNDS::index>> face2nodeV;
        std::vector<std::vector<NodePeriodicBits>> face2nodePbiV;
        std::vector<std::pair<DNDS::index, DNDS::index>> face2cellV;
        std::vector<ElemInfo> faceElemInfoV;

        DNDS::index nFaces = 0;
        for (DNDS::index iCell = 0; iCell < cell2cell.Size(); iCell++)
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
                std::vector<std::pair<index, NodePeriodicBits>> faceNodeNodePeriodicBits;
                if (isPeriodic)
                {
                    faceNodePeriodicBits.resize(eFace.GetNumNodes());
                    faceNodeNodePeriodicBits.resize(eFace.GetNumNodes());
                    eCell.ExtractFaceNodes(ic2f, cell2nodePbi[iCell], faceNodePeriodicBits);
                    for (rowsize i = 0; i < faceNodeNodePeriodicBits.size(); i++)
                        faceNodeNodePeriodicBits[i].first = faceNodes[i], faceNodeNodePeriodicBits[i].second = faceNodePeriodicBits[i];
                    std::sort(faceNodeNodePeriodicBits.begin(), faceNodeNodePeriodicBits.end(),
                              [](auto &L, auto &R)
                              { return L.first < R.first; });
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
                                    for (rowsize i = 0; i < faceNodeNodePeriodicBitsOther.size(); i++)
                                        faceNodeNodePeriodicBitsOther[i].first = face2nodeV[iFOther][i],
                                        faceNodeNodePeriodicBitsOther[i].second = face2nodePbiV[iFOther][i];
                                    std::sort(faceNodeNodePeriodicBitsOther.begin(), faceNodeNodePeriodicBitsOther.end(),
                                              [](auto &L, auto &R)
                                              { return L.first < R.first; });
                                    auto v0 = faceNodeNodePeriodicBits.at(0).second ^ faceNodeNodePeriodicBitsOther.at(0).second;
                                    DNDS_assert(faceNodeNodePeriodicBitsOther.size() == faceNodeNodePeriodicBits.size());
                                    bool collaborating = true;
                                    for (rowsize i = 1; i < faceNodeNodePeriodicBitsOther.size(); i++)
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
                    face2cellV.emplace_back(std::make_pair(iCell, DNDS::UnInitIndex));
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
        node2face.clear(); // no need

        // std::cout << fmt::format("=== rank {}, nFaces {}", mpi.rank, nFaces) << std::endl;
        // if(mpi.rank == 0)
        //     for(auto row : face2nodeV)
        //     {
        //         std::cout << "Face ";
        //         for(auto i : row)
        //             std::cout << i <<" (" << coords[i].transpose() << ") " << ", ";
        //         std::cout << std::endl;
        //     }

        /*************************************/
        // ! collect!
        std::vector<index> iFaceAllToCollected(nFaces);
        std::vector<std::vector<index>> faceSendLocals(mpi.size);
        index nFacesNew = 0;
        for (index iFace = 0; iFace < nFaces; iFace++)
        {
            if (faceElemInfoV[iFace].zone <= 0) // if internal
            {
                if (face2cellV[iFace].second == UnInitIndex && face2cellV[iFace].first >= cell2face.father->Size()) // has not other cell with ghost parent
                    iFaceAllToCollected[iFace] = UnInitIndex;                                                       // * discard
                else if (face2cellV[iFace].first >= cell2face.father->Size() &&
                         face2cellV[iFace].second >= cell2face.father->Size()) // both sides ghost
                    iFaceAllToCollected[iFace] = UnInitIndex;                  // * discard
                else if (face2cellV[iFace].first >= cell2face.father->Size() ||
                         face2cellV[iFace].second >= cell2face.father->Size())
                {
                    DNDS_assert(face2cellV[iFace].second >= cell2face.father->Size()); // should only be the internal as first
                    // * check both sided's info //TODO: optimize so that pLGhostMapping returns rank directly ?
                    index cellGlobL = cell2node.trans.pLGhostMapping->operator()(-1, face2cellV[iFace].first);
                    index cellGlobR = cell2node.trans.pLGhostMapping->operator()(-1, face2cellV[iFace].second);
                    MPI_int rankL, rankR;
                    index valL, valR;
                    auto retL = cell2node.father->pLGlobalMapping->search(cellGlobL, rankL, valL);
                    auto retR = cell2node.father->pLGlobalMapping->search(cellGlobR, rankR, valR);
                    DNDS_assert(retL && retR && (rankL != rankR));
                    if (rankL > rankR)
                    {
                        iFaceAllToCollected[iFace] = -1; // * discard but with ghost
                    }
                    else
                    {
                        DNDS_assert(rankL == mpi.rank);
                        faceSendLocals[rankR].push_back(nFacesNew);
                        iFaceAllToCollected[iFace] = nFacesNew++; //*use
                    }
                }
                else
                {
                    iFaceAllToCollected[iFace] = nFacesNew++; //*use
                }
            }
            else // all bnds would be non duplicate
            {
                iFaceAllToCollected[iFace] = nFacesNew++; //*use
            }
        }
        // std::cout << fmt::format("=== rank {}, nFacesNew {}", mpi.rank, nFacesNew) << std::endl;

        /**********************************/
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

        MPI::Barrier(mpi.comm);
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iCell = 0; iCell < cell2face.Size(); iCell++) // convert face indices pointers
        {
            for (rowsize ic2f = 0; ic2f < cell2face.RowSize(iCell); ic2f++)
            {
                cell2face(iCell, ic2f) = iFaceAllToCollected[cell2face(iCell, ic2f)]; // Uninit if to discard
            }
        }
        adjFacialState = Adj_PointToLocal;

        // now face built except for the ghost part
        /*****************************************/

        /**********************************/
        // put bnd elem info into faces
        bnd2face.resize(bndElemInfo.Size(), -1);
        face2bnd.reserve(bndElemInfo.Size());
        std::unordered_map<index, index> iFace2iBnd;
        for (DNDS::index iBnd = 0; iBnd < bndElemInfo.Size(); iBnd++)
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
                        if (iCellA > iCellB)
                            continue;
                    }
                    iFace2iBnd[iFace] = iBnd;
                    nFound++; // two things:
                    // if is periodic, then only gets the bnd info of the main cell's bnd;
                    // if is external bc, then must be non-ghost face
                    faceElemInfo(iFace, 0) = bndElemInfo(iBnd, 0);
                    bnd2face[iBnd] = iFace;
                    face2bnd[iFace] = iBnd;
                    DNDS_assert_info(FaceIDIsExternalBC(faceID) ||
                                         FaceIDIsPeriodic(faceID),
                                     "bnd elem should have a BC id not interior");
                }
            }
            DNDS_assert(nFound > 0 || (FaceIDIsPeriodic(faceID) && nFound == 0)); // periodic could miss the face
        }

        /**********************************/
        // alter face2node and face2cell to point to global
        this->AdjLocal2GlobalFacial();

        // comm on the faces
        std::vector<index> faceSendLocalsIdx;
        std::vector<index> faceSendLocalsStarts(mpi.size + 1);
        faceSendLocalsStarts[0] = 0;
        for (MPI_int r = 0; r < mpi.size; r++)
            faceSendLocalsStarts[r + 1] = faceSendLocalsStarts[r] + faceSendLocals[r].size();
        faceSendLocalsIdx.resize(faceSendLocalsStarts.back());
        for (MPI_int r = 0; r < mpi.size; r++)
            std::copy(faceSendLocals[r].begin(), faceSendLocals[r].end(), faceSendLocalsIdx.begin() + faceSendLocalsStarts[r]);

        face2node.father->Compress(); // before comm
        if (isPeriodic)
            face2nodePbi.father->Compress();
        face2cell.TransAttach();
        face2node.TransAttach();
        if (isPeriodic)
            face2nodePbi.TransAttach();
        faceElemInfo.TransAttach();

        face2cell.trans.createFatherGlobalMapping();
        face2cell.trans.createGhostMapping(faceSendLocalsIdx, faceSendLocalsStarts);
        face2node.trans.BorrowGGIndexing(face2cell.trans);
        if (isPeriodic)
            face2nodePbi.trans.BorrowGGIndexing(face2cell.trans);
        faceElemInfo.trans.BorrowGGIndexing(face2cell.trans);

        face2cell.trans.createMPITypes();
        face2node.trans.createMPITypes();
        if (isPeriodic)
            face2nodePbi.trans.createMPITypes();
        faceElemInfo.trans.createMPITypes();

        face2cell.trans.pullOnce();
        face2node.trans.pullOnce();
        if (isPeriodic)
            face2nodePbi.trans.pullOnce();
        faceElemInfo.trans.pullOnce();

        this->AdjGlobal2LocalFacial();
        // alter face2node and face2cell to point to local
        /**********************************/

        /**********************************/
        // tend to unattended cell2face with pointing to ghost
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (DNDS::index iFace = 0; iFace < face2cell.son->Size(); iFace++) // face2cell points to local now
        {
            // before: first points to inner, //!relies on the order of setting face2cell
            DNDS_assert((*face2cell.son)(iFace, 0) >= cell2node.father->Size());
            auto eFace = Elem::Element{(*faceElemInfo.son)(iFace, 0).getElemType()};
            auto faceVerts = std::vector<index>((*face2node.son)[iFace].begin(), (*face2node.son)[iFace].begin() + eFace.GetNumVertices());
            std::sort(faceVerts.begin(), faceVerts.end()); //* do not forget to do set operation sort first
            for (rowsize if2c = 0; if2c < 2; if2c++)
            {
                index iCell = (*face2cell.son)(iFace, if2c);
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
                        cell2face(iCell, ic2f) = iFace + face2cell.father->Size(); // remember is ghost
                        found = true;
                    }
                }
                DNDS_assert(found);
            }
        }

        cell2face.father->Compress();
        cell2face.son->Compress();
        adjC2FState = Adj_PointToLocal;

        this->AdjLocal2GlobalC2F();
        cell2face.TransAttach();
        cell2face.trans.BorrowGGIndexing(cell2node.trans);
        cell2face.trans.createMPITypes();
        cell2face.trans.pullOnce();
        this->AdjGlobal2LocalC2F();

        for (DNDS::index iFace = 0; iFace < faceElemInfo.Size(); iFace++)
        {
            if (FaceIDIsPeriodicMain(faceElemInfo(iFace, 0).zone))
            {
                // DNDS_assert(false);
            }
        }
        for (DNDS::index iFace = 0; iFace < faceElemInfo.Size(); iFace++)
        {
            if (FaceIDIsPeriodicDonor(faceElemInfo(iFace, 0).zone))
            {
                // DNDS_assert(false);
            }
        }

        // DNDS::MPISerialDo(mpi, [&]()
        //                   { std::cout << "Rank " << mpi.rank << " : nFace " << face2node.Size() << std::endl; });
        // DNDS::MPISerialDo(mpi, [&]()
        //                   { std::cout << "Rank " << mpi.rank << " : nC2C " << cell2cell.father->DataSize() << std::endl; });
        auto gSize = face2node.father->globalSize(); //! sync call!!!
        if (mpi.rank == 0)
            log() << "UnstructuredMesh === InterpolateFace: total faces " << gSize << std::endl;
    }

    void UnstructuredMesh::
        AssertOnFaces()
    {

        //* some assertions on faces
        std::vector<uint16_t> cCont(cell2cell.Size(), 0); // simulate flux
        for (DNDS::index iFace = 0; iFace < faceElemInfo.Size(); iFace++)
        {
            auto faceID = faceElemInfo(iFace, 0).zone;
            if (FaceIDIsInternal(faceID))
            {
                // if (FaceIDIsPeriodic(faceID))
                // {
                //     // TODO: tend to the case of face is PeriodicDonor with Main in same proc
                //     continue;
                // }
                // if (face2cell[iFace][0] < cell2cell.father->Size()) // other side prime cell, periodic also
                DNDS_assert_info(face2cell[iFace][1] != DNDS::UnInitIndex,
                                 fmt::format(
                                     "Face {} is internal, but f2c[1] is null, at {},{},{} - {},{},{}", iFace,
                                     coords[face2node[iFace][0]](0),
                                     coords[face2node[iFace][0]](1),
                                     coords[face2node[iFace][0]](2),
                                     face2node[iFace].size() > 1 ? coords[face2node[iFace][1]](0) : 0.,
                                     face2node[iFace].size() > 1 ? coords[face2node[iFace][1]](1) : 0.,
                                     face2node[iFace].size() > 1 ? coords[face2node[iFace][1]](2) : 0.)); // Assert has enough cell donors
                DNDS_assert(face2cell[iFace][0] >= 0 && face2cell[iFace][0] < cell2cell.Size());
                DNDS_assert(face2cell[iFace][1] >= 0 && face2cell[iFace][1] < cell2cell.Size());
                cCont[face2cell[iFace][0]]++;
                cCont[face2cell[iFace][1]]++;
            }
            else // a external BC
            {
                DNDS_assert(face2cell[iFace][1] == DNDS::UnInitIndex);
                DNDS_assert(face2cell[iFace][0] >= 0 && face2cell[iFace][0] < cell2cell.father->Size());
                cCont[face2cell[iFace][0]]++;
            }
        }
        for (DNDS::index iCell = 0; iCell < cellElemInfo.father->Size(); iCell++) // for every non-ghost
        {
            for (auto iFace : cell2face[iCell])
            {
                DNDS_assert(iFace >= 0 && iFace < face2cell.Size());
                DNDS_assert(face2cell[iFace][0] == iCell || face2cell[iFace][1] == iCell);
            }
            DNDS_assert(cCont[iCell] == cell2face.RowSize(iCell));
        }
    }

    void UnstructuredMesh::
        WriteSerialize(SerializerBase *serializer, const std::string &name)
    {
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);

        auto cwd = serializer->GetCurrentPath();
        serializer->CreatePath(name);
        serializer->GoToPath(name);

        serializer->WriteString("mesh", "UnstructuredMesh");
        serializer->WriteIndex("dim", dim);
        serializer->WriteIndex("MPIRank", mpi.rank);
        serializer->WriteIndex("MPISize", mpi.size);
        serializer->WriteInt("isPeriodic", isPeriodic);

        coords.WriteSerialize(serializer, "coords");
        cell2node.WriteSerialize(serializer, "cell2node");
        cell2cell.WriteSerialize(serializer, "cell2cell");
        cellElemInfo.WriteSerialize(serializer, "cellElemInfo");
        bnd2node.WriteSerialize(serializer, "bnd2node");
        bnd2cell.WriteSerialize(serializer, "bnd2cell");
        bndElemInfo.WriteSerialize(serializer, "bndElemInfo");
        if (isPeriodic)
        {
            cell2nodePbi.WriteSerialize(serializer, "cell2nodePbi");
            periodicInfo.WriteSerializer(serializer, "periodicInfo");
        }

        serializer->GoToPath(cwd);
    }

    void UnstructuredMesh::
        ReadSerialize(SerializerBase *serializer, const std::string &name)
    {
        auto cwd = serializer->GetCurrentPath();
        // serializer->CreatePath(name);//! remember no create!
        serializer->GoToPath(name);

        std::string meshRead;
        index dimRead, rankRead, sizeRead;
        int isPeriodicRead;
        serializer->ReadString("mesh", meshRead);
        serializer->ReadIndex("dim", dimRead);
        serializer->ReadIndex("MPIRank", rankRead);
        serializer->ReadIndex("MPISize", sizeRead);
        serializer->ReadInt("isPeriodic", isPeriodicRead);
        isPeriodic = bool(isPeriodicRead);
        DNDS_assert(meshRead == "UnstructuredMesh");
        DNDS_assert(dimRead == dim);
        DNDS_assert(rankRead == mpi.rank && sizeRead == mpi.size);

        // make the empty arrays
        auto mesh = this;
        DNDS_MAKE_SSP(mesh->coords.father, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->coords.son, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->cellElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        DNDS_MAKE_SSP(mesh->cellElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        DNDS_MAKE_SSP(mesh->bndElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        DNDS_MAKE_SSP(mesh->bndElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        DNDS_MAKE_SSP(mesh->cell2node.father, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->cell2node.son, mesh->getMPI());
        if (isPeriodic)
        {
            DNDS_MAKE_SSP(mesh->cell2nodePbi.father, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->getMPI());
            DNDS_MAKE_SSP(mesh->cell2nodePbi.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->getMPI());
        }
        DNDS_MAKE_SSP(mesh->cell2cell.father, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->cell2cell.son, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->bnd2node.father, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->bnd2node.son, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->bnd2cell.father, mesh->getMPI());
        DNDS_MAKE_SSP(mesh->bnd2cell.son, mesh->getMPI());

        coords.ReadSerialize(serializer, "coords");
        cell2node.ReadSerialize(serializer, "cell2node");
        cell2cell.ReadSerialize(serializer, "cell2cell");
        cellElemInfo.ReadSerialize(serializer, "cellElemInfo");
        bnd2node.ReadSerialize(serializer, "bnd2node");
        bnd2cell.ReadSerialize(serializer, "bnd2cell");
        bndElemInfo.ReadSerialize(serializer, "bndElemInfo");
        if (isPeriodic)
        {
            cell2nodePbi.ReadSerialize(serializer, "cell2nodePbi");
            periodicInfo.ReadSerializer(serializer, "periodicInfo");
        }

        // after matters:
        coords.trans.createMPITypes();
        cell2node.trans.createMPITypes();
        cell2cell.trans.createMPITypes();
        cellElemInfo.trans.createMPITypes();
        bnd2node.trans.createMPITypes();
        bnd2cell.trans.createMPITypes();
        bndElemInfo.trans.createMPITypes();
        if (isPeriodic)
            cell2nodePbi.trans.createMPITypes();
        adjPrimaryState = Adj_PointToLocal; // the file is pointing to local

        index nCellG = this->NumCellGlobal(); // collective call!
        index nNodeG = this->NumNodeGlobal(); // collective call!
        if (mpi.rank == mRank)
        {
            log() << "UnstructuredMesh === ReadSerialize "
                  << "Global NumCell [ " << nCellG << " ]" << std::endl;
            log() << "UnstructuredMesh === ReadSerialize "
                  << "Global NumNode [ " << nNodeG << " ]" << std::endl;
        }
        MPISerialDo(mpi, [&]()
                    { log() << "    Rank: " << mpi.rank << " nCell " << this->NumCell() << " nCellGhost " << this->NumCellGhost() << std::endl; });
        MPISerialDo(mpi, [&]()
                    { log() << "    Rank: " << mpi.rank << " nNode " << this->NumNode() << " nNodeGhost " << this->NumNodeGhost() << std::endl; });

        serializer->GoToPath(cwd);
    }

    void UnstructuredMesh::ConstructBndMesh(UnstructuredMesh &bMesh)
    {
        DNDS_assert(bMesh.dim == dim - 1 && bMesh.mpi == mpi);
        DNDS_MAKE_SSP(bMesh.cell2node.father, mpi);
        DNDS_MAKE_SSP(bMesh.cell2node.son, mpi);
        DNDS_MAKE_SSP(bMesh.coords.father, mpi);
        DNDS_MAKE_SSP(bMesh.coords.son, mpi);
        if (isPeriodic)
        {
            bMesh.isPeriodic = true;
            bMesh.periodicInfo = this->periodicInfo;
            DNDS_MAKE_SSP(bMesh.cell2nodePbi.father, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
            DNDS_MAKE_SSP(bMesh.cell2nodePbi.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
        }
        DNDS_MAKE_SSP(bMesh.cellElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        DNDS_MAKE_SSP(bMesh.cellElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);

        node2bndNode.resize(this->NumNodeProc(), -1);
        index bndNodeCount{0};
        for (index iBnd = 0; iBnd < this->NumBnd(); iBnd++) //! bnd has no ghost!
            for (auto iNode : bnd2node.father->operator[](iBnd))
                if (node2bndNode.at(iNode) == -1)
                    node2bndNode.at(iNode) = bndNodeCount++;
        bMesh.node2parentNode.resize(bndNodeCount);
        for (index iNode = 0; iNode < node2bndNode.size(); iNode++)
            if (node2bndNode[iNode] >= 0)
                bMesh.node2parentNode.at(node2bndNode[iNode]) = iNode;
        bMesh.coords.father->Resize(bndNodeCount);
        // std::cout << bndNodeCount << std::endl;
        for (index iBNode = 0; iBNode < bndNodeCount; iBNode++)
        {
            bMesh.coords[iBNode] = coords[bMesh.node2parentNode[iBNode]];
        }
        index nBndCellUse{0};
        for (index iB = 0; iB < this->NumBnd(); iB++)
            if (!FaceIDIsPeriodic(this->bndElemInfo(iB, 0).zone))
                nBndCellUse++;
        bMesh.cell2node.father->Resize(nBndCellUse);
        if (isPeriodic)
            bMesh.cell2nodePbi.father->Resize(nBndCellUse);
        bMesh.cellElemInfo.father->Resize(nBndCellUse);
        bMesh.cell2parentCell.resize(nBndCellUse, -1);
        nBndCellUse = 0;
        for (index iB = 0; iB < this->NumBnd(); iB++)
        {
            if (FaceIDIsPeriodic(this->bndElemInfo(iB, 0).zone))
                continue;
            bMesh.cell2parentCell.at(nBndCellUse) = iB;
            bMesh.cell2node.ResizeRow(nBndCellUse, bnd2node.RowSize(iB));
            if (isPeriodic)
                bMesh.cell2nodePbi.ResizeRow(nBndCellUse, bnd2node.RowSize(iB));
            bMesh.cellElemInfo(nBndCellUse, 0) = bndElemInfo(iB, 0);

            for (rowsize ib2n = 0; ib2n < bnd2node.RowSize(iB); ib2n++)
            {
                if (bnd2face.at(iB) < 0) // where bnd has not a face!
                    bMesh.cell2node[nBndCellUse][ib2n] = node2bndNode.at(bnd2node[iB][ib2n]);
                else
                    bMesh.cell2node[nBndCellUse][ib2n] = node2bndNode.at(face2node[bnd2face.at(iB)][ib2n]); //* respect the face ordering if possible
                DNDS_assert(node2bndNode.at(bnd2node[iB][ib2n]) >= 0);

                if (isPeriodic)
                {
                    if (bnd2face.at(iB) < 0)                                              // where bnd has not a face!
                        bMesh.cell2nodePbi[nBndCellUse][ib2n] = Geom::NodePeriodicBits{}; // a invalid value
                    else
                    {
                        bMesh.cell2nodePbi[nBndCellUse][ib2n] = face2nodePbi[bnd2face.at(iB)][ib2n];
                    }
                }
            }
            nBndCellUse++;
        }

        bMesh.cell2node.father->Compress();

        bMesh.coords.father->createGlobalMapping();
        bMesh.cell2node.father->createGlobalMapping();

        bMesh.coords.TransAttach();
        bMesh.cell2node.TransAttach();
        bMesh.coords.trans.createGhostMapping(std::vector<int>{});
        bMesh.cell2node.trans.createGhostMapping(std::vector<int>{});

        bMesh.adjPrimaryState = Adj_PointToLocal;
        if (mpi.rank == mRank)
            log() << "UnstructuredMesh === ConstructBndMesh Done" << std::endl;
    }

    void UnstructuredMesh::ObtainSymmetricSymbolicFactorization(Direct::SerialSymLUStructure &symLU, Direct::DirectPrecControl control)
    {
        if (control.useDirectPrec)
            symLU.ObtainSymmetricSymbolicFactorization(
                cell2cellFaceVLocal,
                control.getILUCode());
    }
}