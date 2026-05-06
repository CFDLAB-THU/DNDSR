#include "Mesh.hpp"
#include "Mesh_PartitionHelpers.hpp"
#include "Mesh_InterpolateHelpers.hpp"
#include "MeshConnectivity.hpp"
#include "MeshConnectivity_StateChecked.hpp"
#include "Solver/Direct.hpp"

#include <cstdlib>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <fmt/core.h>
#include "DNDS/EigenPCH.hpp"
#include "Geom/Mesh/Mesh_DeviceView.hpp"
#include "SerialAdjReordering.hpp"

#include "DNDS/Device/DeviceStorage.hxx"

namespace DNDS
{
    // DNDS_DEVICE_STORAGE_BASE_DELETER_INST(Geom::ElemInfo, )
    // DNDS_DEVICE_STORAGE_INST(Geom::ElemInfo, DeviceBackend::Host, )
}

namespace DNDS::Geom
{

    /**
     * \brief Reserved skeleton for parallel topology interpolation.
     *
     * This commented-out method is preserved as a placeholder for a future generic
     * topology management API. Currently, mesh topology (cell2face, face2cell, etc.)
     * is constructed serially and distributed. However, a parallel interpolation
     * capability may be needed for:
     *   - Dynamic mesh adaptation (refinement/coarsening)
     *   - Load balancing with topology migration
     *   - Multi-physics coupling with different mesh partitions
     *
     * If implementing parallel topology construction, this skeleton provides the
     * basic structure for building face-based adjacencies from cell2node without
     * requiring a serial global mesh on rank 0.
     *
     * \todo Evaluate need for parallel topology API in Phase 5 refactoring.
     */

    /*******************************************************************************************************************/
    /*******************************************************************************************************************/
    /*******************************************************************************************************************/

    // Partition2LocalIdx, Partition2Serial2Global, ConvertAdjSerial2Global,
    // TransferDataSerial2Global are now in Mesh_PartitionHelpers.hpp
    // so they can be shared with Mesh_ReadSerializeDistributed.cpp.

    //! inefficient, use Partition2Serial2Global ! only used for convenient comparison
    void PushInfo2Serial2Global(std::vector<DNDS::index> &serial2Global,
                                DNDS::index localSize,
                                const std::vector<DNDS::index> &pushIndex,
                                const std::vector<DNDS::index> &pushIndexStart,
                                const DNDS::MPIInfo &mpi)
    {
        tIndPair Serial2Global;
        Serial2Global.InitPair("Serial2Global", mpi);
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
        // !cell2cell discarded
        // ConvertAdjSerial2Global(cell2cellSerial, cell_Serial2Global, mesh->getMPI());
        ConvertAdjSerial2Global(bnd2nodeSerial, node_Serial2Global, mesh->getMPI());
        ConvertAdjSerial2Global(bnd2cellSerial, cell_Serial2Global, mesh->getMPI());

        mesh->coords.InitPair("coords", mesh->getMPI());
        mesh->cellElemInfo.InitPair("cellElemInfo", ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        mesh->bndElemInfo.InitPair("bndElemInfo", ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());
        mesh->cell2node.InitPair("cell2node", mesh->getMPI());
        mesh->cell2cellOrig.InitPair("cell2cellOrig", mesh->getMPI());
        mesh->node2nodeOrig.InitPair("node2nodeOrig", mesh->getMPI());
        mesh->bnd2bndOrig.InitPair("bnd2bndOrig", mesh->getMPI());
        if (mesh->isPeriodic)
            mesh->cell2nodePbi.InitPair("cell2nodePbi", mesh->getMPI());
        // !cell2cell discarded
        mesh->bnd2node.InitPair("bnd2node", mesh->getMPI());
        mesh->bnd2cell.InitPair("bnd2cell", mesh->getMPI());
        if (mesh->isPeriodic)
            mesh->bnd2nodePbi.InitPair("bnd2nodePbi", mesh->getMPI());

        // coord transferring
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing PartitionReorderToMeshCell2Cell Trasfer Data Coord" << std::endl;
        TransferDataSerial2Global(coordSerial, mesh->coords.father, node_push, node_pushStart, mesh->getMPI());
        TransferDataSerial2Global(node2nodeOrigSerial, mesh->node2nodeOrig.father, node_push, node_pushStart, mesh->getMPI());

        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing PartitionReorderToMeshCell2Cell Trasfer Data Cell" << std::endl;
        // cells transferring
        // !cell2cell discarded
        // TransferDataSerial2Global(cell2cellSerial, mesh->cell2cell.father, cell_push, cell_pushStart, mesh->getMPI());
        TransferDataSerial2Global(cell2nodeSerial, mesh->cell2node.father, cell_push, cell_pushStart, mesh->getMPI());
        TransferDataSerial2Global(cell2cellOrigSerial, mesh->cell2cellOrig.father, cell_push, cell_pushStart, mesh->getMPI());
        if (mesh->isPeriodic)
            TransferDataSerial2Global(cell2nodePbiSerial, mesh->cell2nodePbi.father, cell_push, cell_pushStart, mesh->getMPI());
        TransferDataSerial2Global(cellElemInfoSerial, mesh->cellElemInfo.father, cell_push, cell_pushStart, mesh->getMPI());
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Doing PartitionReorderToMeshCell2Cell Trasfer Data Bnd" << std::endl;
        // bnds transferring
        TransferDataSerial2Global(bnd2cellSerial, mesh->bnd2cell.father, bnd_push, bnd_pushStart, mesh->getMPI());
        TransferDataSerial2Global(bnd2nodeSerial, mesh->bnd2node.father, bnd_push, bnd_pushStart, mesh->getMPI());
        TransferDataSerial2Global(bndElemInfoSerial, mesh->bndElemInfo.father, bnd_push, bnd_pushStart, mesh->getMPI());
        TransferDataSerial2Global(bnd2bndOrigSerial, mesh->bnd2bndOrig.father, bnd_push, bnd_pushStart, mesh->getMPI());
        if (mesh->isPeriodic)
            TransferDataSerial2Global(bnd2nodePbiSerial, mesh->bnd2nodePbi.father, bnd_push, bnd_pushStart, mesh->getMPI());

        {
            DNDS::MPISerialDo(mesh->getMPI(), [&]()
                              { log() << "[" << mesh->getMPI().rank << ": nCell " << mesh->cell2node.father->Size() << "] " << (((mesh->getMPI().rank + 1) % 10) ? "" : "\n") << std::flush; });
            MPI::Barrier(mesh->getMPI().comm);
            if (mesh->getMPI().rank == 0)
                log() << std::endl;
            DNDS::MPISerialDo(mesh->getMPI(), [&]()
                              { log() << "[" << mesh->getMPI().rank << ": nNode " << mesh->coords.father->Size() << "] " << (((mesh->getMPI().rank + 1) % 10) ? "" : "\n") << std::flush; });
            MPI::Barrier(mesh->getMPI().comm);
            if (mesh->getMPI().rank == 0)
                log() << std::endl;
            DNDS::MPISerialDo(mesh->getMPI(), [&]()
                              { log() << "[" << mesh->getMPI().rank << ": nBnd " << mesh->bnd2node.father->Size() << "] " << (((mesh->getMPI().rank + 1) % 10) ? "" : "\n") << std::flush; });
            MPI::Barrier(mesh->getMPI().comm);
            if (mesh->getMPI().rank == 0)
                log() << std::endl;
        }
        mesh->adjPrimaryState = Adj_PointToGlobal;
        mesh->cell2node.idx.markGlobal();
        mesh->bnd2node.idx.markGlobal();
        mesh->bnd2cell.idx.markGlobal();
        if (mesh->getMPI().rank == mRank)
            DNDS::log() << "UnstructuredMeshSerialRW === Done  PartitionReorderToMeshCell2Cell" << std::endl;
    }

    void UnstructuredMeshSerialRW::
        BuildSerialOut()
    {
        DNDS_assert(mesh->adjPrimaryState == Adj_PointToGlobal);
        DNDS_assert(mesh->cell2node.isGlobal());
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
        cell2nodeSerial = make_ssp<decltype(cell2nodeSerial)::element_type>(ObjName{"cell2nodeSerial"}, mesh->getMPI());
        if (mesh->isPeriodic)
            cell2nodePbiSerial = make_ssp<decltype(cell2nodePbiSerial)::element_type>(ObjName{"cell2nodePbiSerial"}, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mesh->getMPI());
        coordSerial = make_ssp<decltype(coordSerial)::element_type>(ObjName{"coordSerial"}, mesh->getMPI());
        cellElemInfoSerial = make_ssp<decltype(cellElemInfoSerial)::element_type>(ObjName{"cellElemInfoSerial"}, ElemInfo::CommType(), ElemInfo::CommMult(), mesh->getMPI());

        coordSerialOutTrans.setFatherSon(mesh->coords.father, coordSerial);
        cell2nodeSerialOutTrans.setFatherSon(mesh->cell2node.father, cell2nodeSerial);
        if (mesh->isPeriodic)
            cell2nodePbiSerialOutTrans.setFatherSon(mesh->cell2nodePbi.father, cell2nodePbiSerial);
        // bnd2nodeSerialOutTrans.setFatherSon(mesh->bnd2node.father, bnd2nodeSerial);
        cellElemInfoSerialOutTrans.setFatherSon(mesh->cellElemInfo.father, cellElemInfoSerial);
        // bndElemInfoSerialOutTrans.setFatherSon(mesh->bndElemInfo.father, bndElemInfoSerial);

        // Reuse existing father global mappings to avoid side-effects on
        // the mesh's own pLGlobalMapping pointers.  createFatherGlobalMapping()
        // would call father->createGlobalMapping() which replaces the shared_ptr
        // on the father array, affecting the mesh's own transformer.
        auto reuseOrCreateFatherGlobalMapping = [](auto &trans)
        {
            if (trans.father->pLGlobalMapping)
                trans.pLGlobalMapping = trans.father->pLGlobalMapping;
            else
                trans.createFatherGlobalMapping();
        };
        reuseOrCreateFatherGlobalMapping(coordSerialOutTrans);
        reuseOrCreateFatherGlobalMapping(cell2nodeSerialOutTrans);
        if (mesh->isPeriodic)
            reuseOrCreateFatherGlobalMapping(cell2nodePbiSerialOutTrans);
        // reuseOrCreateFatherGlobalMapping(bnd2nodeSerialOutTrans);
        reuseOrCreateFatherGlobalMapping(cellElemInfoSerialOutTrans);
        // reuseOrCreateFatherGlobalMapping(bndElemInfoSerialOutTrans);

        coordSerialOutTrans.createGhostMapping(serialPullNode);
        cell2nodeSerialOutTrans.createGhostMapping(serialPullCell);
        if (mesh->isPeriodic)
            cell2nodePbiSerialOutTrans.createGhostMapping(serialPullCell);
        // bnd2nodeSerialOutTrans.createGhostMapping(serialPullBnd);
        // cellElemInfo shares the same cell indexing as cell2node: copy
        // ghost + global mappings without overwriting the father's pointer.
        cellElemInfoSerialOutTrans.pLGhostMapping = cell2nodeSerialOutTrans.pLGhostMapping;
        cellElemInfoSerialOutTrans.pLGlobalMapping = cell2nodeSerialOutTrans.pLGlobalMapping;
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
            DNDS::log() << "UnstructuredMeshSerialRW === BuildSerialOut Done " << std::endl;
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
        int hasBadAll = 0;
        MPI::Allreduce(&hasBad, &hasBadAll, 1, MPI_INT, MPI_SUM, mpi.comm);
        return hasBad == 0;
    }

    bool UnstructuredMesh::IsO2()
    {
        using namespace Elem;
        int hasBad = 0;
        for (index iCell = 0; iCell < cellElemInfo.Size(); iCell++)
        {
            auto eType = cellElemInfo(iCell, 0).getElemType();
            if (eType == ElemType::Line3 ||
                eType == ElemType::Tri6 ||
                eType == ElemType::Quad9 ||
                eType == ElemType::Tet10 ||
                eType == ElemType::Hex27 ||
                eType == ElemType::Prism18 ||
                eType == ElemType::Pyramid14)
            {
                continue;
            }
            else
            {
                hasBad = 1;
                break;
            }
        }
        int hasBadAll = 0;
        MPI::Allreduce(&hasBad, &hasBadAll, 1, MPI_INT, MPI_SUM, mpi.comm);
        return hasBad == 0;
    }

    void UnstructuredMesh::SetPeriodicGeometry(
        const tPoint &translation1,
        const tPoint &rotationCenter1,
        const tPoint &eulerAngles1,
        const tPoint &translation2,
        const tPoint &rotationCenter2,
        const tPoint &eulerAngles2,
        const tPoint &translation3,
        const tPoint &rotationCenter3,
        const tPoint &eulerAngles3)
    {
        periodicInfo.translation[1].map() = translation1;
        periodicInfo.translation[2].map() = translation2;
        periodicInfo.translation[3].map() = translation3;
        periodicInfo.rotationCenter[1].map() = rotationCenter1;
        periodicInfo.rotationCenter[2].map() = rotationCenter2;
        periodicInfo.rotationCenter[3].map() = rotationCenter3;
        periodicInfo.rotation[1].map() =
            Geom::RotZ(eulerAngles1[2]) *
            Geom::RotY(eulerAngles1[1]) *
            Geom::RotX(eulerAngles1[0]);
        periodicInfo.rotation[2].map() =
            Geom::RotZ(eulerAngles2[2]) *
            Geom::RotY(eulerAngles2[1]) *
            Geom::RotX(eulerAngles2[0]);
        periodicInfo.rotation[3].map() =
            Geom::RotZ(eulerAngles3[2]) *
            Geom::RotY(eulerAngles3[1]) *
            Geom::RotX(eulerAngles3[0]);
    }

    void UnstructuredMesh::
        RecoverNode2CellAndNode2Bnd()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
        DNDS_assert(cell2node.isGlobal() && bnd2node.isGlobal());
        DNDS_assert(coords.father);
        DNDS_assert(cell2node.father);
        DNDS_assert(bnd2node.father);

        if (!coords.father->pLGlobalMapping)
            coords.father->createGlobalMapping();
        if (!cell2node.father->pLGlobalMapping)
            cell2node.father->createGlobalMapping();

        // Ensure ghost mappings exist so IndexLocal2Global / IndexGlobal2Local work
        // (father-only at this stage — no ghost cells/nodes yet).
        coords.TransAttach();
        coords.trans.createFatherGlobalMapping();
        EnsureGhostMapping(coords);
        cell2node.TransAttach();
        cell2node.trans.createFatherGlobalMapping();
        EnsureGhostMapping(cell2node);

        // node2cell via DSL Inverse (cell2node must be in global state)
        // Result is already AdjPairTracked with father adopted, state = Global.
        node2cell = CheckedInverse(
            cell2node, coords,
            coords.father->Size(), mpi);

        // node2bnd via DSL Inverse (bnd2node must be in global state)
        if (!bnd2node.father->pLGlobalMapping)
            bnd2node.father->createGlobalMapping();
        bnd2node.TransAttach();
        bnd2node.trans.createFatherGlobalMapping();
        EnsureGhostMapping(bnd2node);

        node2bnd = CheckedInverse(
            bnd2node, coords,
            coords.father->Size(), mpi);

        this->adjN2CBState = Adj_PointToGlobal;
        // markGlobal already done by CheckedInverse
    }

    void UnstructuredMesh::RecoverCell2CellAndBnd2Cell()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
        DNDS_assert(cell2node.isGlobal() && bnd2node.isGlobal());
        DNDS_assert(adjN2CBState == Adj_PointToGlobal);
        DNDS_assert(node2cell.isGlobal() && node2bnd.isGlobal());
        DNDS_assert(coords.father);
        DNDS_assert(cell2node.father);
        DNDS_assert(bnd2node.father);
        DNDS_assert(node2cell.father);

        coords.TransAttach();
        coords.trans.createFatherGlobalMapping();
        EnsureGhostMapping(coords);
        cell2node.TransAttach();
        cell2node.trans.createFatherGlobalMapping();
        EnsureGhostMapping(cell2node);
        bnd2node.TransAttach();
        bnd2node.trans.createFatherGlobalMapping();
        EnsureGhostMapping(bnd2node);

        // Ghost-pull node2cell for off-rank nodes referenced by local cells and bnds.
        // Use evaluateGhostTree: Cell → Cell2Node → Node ∪ Bnd → Bnd2Node → Node.
        {
            MeshConnectivity dagN2CB;
            fillRegistry(dagN2CB);

            GhostSpec n2cbSpec{{
                {EntityKind::Cell, {Adj::Cell2Node}, EntityKind::Node},
                {EntityKind::Bnd, {Adj::Bnd2Node}, EntityKind::Node},
            }};
            auto n2cbResult = dagN2CB.evaluateGhostTree(
                CompiledGhostTree::compile(n2cbSpec), mpi);

            auto itN = n2cbResult.ghostIndices.find(EntityKind::Node);
            std::vector<index> ghostNodes;
            if (itN != n2cbResult.ghostIndices.end())
                ghostNodes = std::move(itN->second);

            node2cell.son = make_ssp<decltype(node2cell.son)::element_type>(ObjName{"node2cell.son"}, mpi);
            node2cell.TransAttach();
            node2cell.trans.createFatherGlobalMapping();
            node2cell.trans.createGhostMapping(ghostNodes);
            node2cell.trans.createMPITypes();
            node2cell.trans.pullOnce();
        }

        // Build global→local-appended map for node2cell
        std::unordered_map<index, index> nodeG2L;
        for (index i = 0; i < node2cell.Size(); i++)
            nodeG2L[node2cell.trans.pLGhostMapping->operator()(-1, i)] = i;

        // cell2cell via DSL ComposeFiltered (inputs must be in global state)
        // Result is AdjPairTracked with father adopted, state = Global.
        cell2cell = CheckedComposeFiltered(
            cell2node, node2cell,
            nodeG2L,
            SharedCountPredicate{.minShared = 1, .removeSelf = true});

        // bnd2cell via ComposeFiltered with per-bnd node-count predicate.
        // For periodic meshes, additionally uses pbi containment matchExtra.
        bnd2cell.InitPair("bnd2cell", mpi);
        bnd2cell.father->Resize(bnd2node.father->Size());

        // For periodic: ghost-pull cell2node and cell2nodePbi for all cells
        // reachable through node2cell, so the pbi matchExtra can access them.
        std::unordered_map<index, index> cellG2LAppended; // cell global → local-appended in cell2node/cell2nodePbi
        if (isPeriodic)
        {
            // Collect all unique cell globals from node2cell (father+son)
            std::unordered_set<index> allCellGlobals;
            for (index iNode = 0; iNode < node2cell.Size(); iNode++)
                for (auto ic : node2cell[iNode])
                    allCellGlobals.insert(ic);
            std::vector<index> neededCells;
            for (auto ic : allCellGlobals)
                neededCells.push_back(ic);

            cell2nodePbi.son = make_ssp<decltype(cell2nodePbi.son)::element_type>(
                ObjName{"cell2nodePbi.son"}, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
            cell2nodePbi.TransAttach();
            cell2nodePbi.trans.createFatherGlobalMapping();
            cell2nodePbi.trans.createGhostMapping(neededCells);
            cell2nodePbi.trans.createMPITypes();
            cell2nodePbi.trans.pullOnce();
            cell2node.son = make_ssp<decltype(cell2node.son)::element_type>(ObjName{"cell2node.son"}, mpi);
            cell2node.BorrowAndPull(cell2nodePbi);

            // Build global→local-appended map
            for (index i = 0; i < cell2node.Size(); i++)
                cellG2LAppended[cell2nodePbi.trans.pLGhostMapping->operator()(-1, i)] = i;
        }

        // Per-bnd predicate: keep cells sharing ALL bnd nodes (set intersection)
        auto bndAllNodesPred = [this](index aBndGlobal, index cCellGlobal, int nShared) -> bool
        {
            // Map aBndGlobal to local bnd index to get row size
            index aBndLocal = this->BndIndexGlobal2Local(aBndGlobal);
            DNDS_assert(aBndLocal >= 0);
            return nShared >= bnd2node.father->RowSize(aBndLocal);
        };

        // matchExtra for periodic: pbi containment check
        std::function<bool(index, index, const std::vector<index> &)> bndMatchExtra = nullptr;
        if (isPeriodic)
        {
            bndMatchExtra = [this, &cellG2LAppended](
                                index aBndLocal, index cCellGlobal,
                                const std::vector<index> & /*sharedNodes*/) -> bool
            {
                auto itC = cellG2LAppended.find(cCellGlobal);
                if (itC == cellG2LAppended.end())
                    return false;
                index cLocal = itC->second;

                // Every (node, pbi) in the bnd must appear in the cell
                for (rowsize ib2n = 0; ib2n < bnd2node.father->RowSize(aBndLocal); ib2n++)
                {
                    index iNode = bnd2node.father->operator()(aBndLocal, ib2n);
                    auto iNodePbi = bnd2nodePbi.father->operator()(aBndLocal, ib2n);
                    bool found = false;
                    for (rowsize ic2n = 0; ic2n < cell2node[cLocal].size(); ic2n++)
                    {
                        if (cell2node(cLocal, ic2n) == iNode &&
                            cell2nodePbi(cLocal, ic2n) == iNodePbi)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                        return false;
                }
                return true;
            };
        }

        // bnd2cell via ComposeFiltered (inputs must be in global state)
        // Result is AdjPairTracked<tAdj2Pair> with father adopted, state = Global.
        bnd2cell = CheckedComposeFiltered<NonUniformSize, NonUniformSize, 2>(
            bnd2node, node2cell,
            nodeG2L,
            bndAllNodesPred,
            bndMatchExtra);

        // Periodic fixup: when a periodic bnd has only 1 matching cell (slot 1 == UnInitIndex),
        // fill slot 1 with slot 0 (self-reference) to match legacy behavior.
        if (isPeriodic)
        {
            for (index i = 0; i < bnd2cell.father->Size(); i++)
            {
                if (Geom::FaceIDIsPeriodic(bndElemInfo.father->operator()(i, 0).zone) &&
                    bnd2cell.father->operator()(i, 1) == UnInitIndex)
                {
                    bnd2cell.father->operator()(i, 1) = bnd2cell.father->operator()(i, 0);
                }
            }
        }

        // Periodic: donor/main swap check — ensure bnd2cell(i, 0) is the donor-side cell.
        // (This domain-specific logic is kept as-is.)
        if (isPeriodic)
        {
            for (index i = 0; i < bnd2cell.father->Size(); i++)
                if (bnd2cell(i, 1) != UnInitIndex && bnd2cell(i, 0) != bnd2cell(i, 1))
                {
                    index ic0 = bnd2cell(i, 0);
                    index ic1 = bnd2cell(i, 1);
                    auto [ret0, rank0, ic0L] = cell2nodePbi.trans.pLGhostMapping->search_indexAppend(ic0);
                    auto [ret1, rank1, ic1L] = cell2nodePbi.trans.pLGhostMapping->search_indexAppend(ic1);
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
                        ;
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
                }
        }

        // markGlobal already done by CheckedComposeFiltered
    }

    void UnstructuredMesh::
        BuildGhostPrimary(int nGhostLayers)
    {
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
        DNDS_assert(cell2node.isGlobal() && bnd2node.isGlobal());
        DNDS_assert(nGhostLayers >= 1);
        DNDS_assert(cell2cell.idx.state() == Adj_PointToGlobal || cell2cell.idx.state() == Adj_Unknown);
        DNDS_assert(bnd2cell.idx.state() == Adj_PointToGlobal || bnd2cell.idx.state() == Adj_Unknown);
        DNDS_assert(cell2cell.father && cell2cell.father->Size() == this->NumCell());
        DNDS_assert(bnd2cell.father && bnd2cell.father->Size() == this->NumBnd());

        /********************************/
        // Attach transformers and create father global mappings.
        cell2cell.TransAttach();
        cell2node.TransAttach();
        cell2cellOrig.TransAttach();
        if (isPeriodic)
            cell2nodePbi.TransAttach();
        cellElemInfo.TransAttach();
        coords.TransAttach();
        node2nodeOrig.TransAttach();
        bnd2cell.TransAttach();
        bnd2node.TransAttach();
        if (isPeriodic)
            bnd2nodePbi.TransAttach();
        bndElemInfo.TransAttach();
        bnd2bndOrig.TransAttach();
        node2bnd.TransAttach();

        cell2cell.trans.createFatherGlobalMapping();
        coords.trans.createFatherGlobalMapping();
        bnd2cell.trans.createFatherGlobalMapping();
        node2bnd.trans.createFatherGlobalMapping();

        /********************************/
        // Unified ghost evaluation: single DAG, all inputs father-only.
        // The evaluator handles scratch pulls internally.
        MeshConnectivity dag;
        fillRegistry(dag);

        auto spec = GhostSpec::defaultPrimary(nGhostLayers);
        auto tree = CompiledGhostTree::compile(spec);
        auto ghostResult = dag.evaluateGhostTree(tree, mpi);

        /********************************/
        // Apply ghost sets to real arrays.

        // --- Cells ---
        {
            auto it = ghostResult.ghostIndices.find(EntityKind::Cell);
            std::vector<DNDS::index> ghostCells;
            if (it != ghostResult.ghostIndices.end())
                ghostCells = std::move(it->second);
            cell2cell.trans.createGhostMapping(ghostCells);
            cell2cell.trans.createMPITypes();
            cell2cell.trans.pullOnce();
            cell2node.BorrowAndPull(cell2cell);
            cell2cellOrig.BorrowAndPull(cell2cell);
            if (isPeriodic)
                cell2nodePbi.BorrowAndPull(cell2cell);
            cellElemInfo.BorrowAndPull(cell2cell);
        }

        // --- Nodes ---
        {
            auto itN = ghostResult.ghostIndices.find(EntityKind::Node);
            std::vector<DNDS::index> ghostNodes;
            if (itN != ghostResult.ghostIndices.end())
                ghostNodes = std::move(itN->second);
            coords.trans.createGhostMapping(ghostNodes);
            coords.trans.createMPITypes();
            coords.trans.pullOnce();
            node2nodeOrig.BorrowAndPull(coords);
        }

        // --- Bnds ---
        {
            DNDS_assert(node2bnd.father);
            DNDS_assert(this->adjN2CBState == Adj_PointToGlobal);
            DNDS_assert(node2bnd.isGlobal());

            auto itB = ghostResult.ghostIndices.find(EntityKind::Bnd);
            std::vector<DNDS::index> ghostBnds;
            if (itB != ghostResult.ghostIndices.end())
                ghostBnds = std::move(itB->second);
            bnd2cell.trans.createGhostMapping(ghostBnds);
            bnd2cell.trans.createMPITypes();
            bnd2cell.trans.pullOnce();
            bnd2node.BorrowAndPull(bnd2cell);
            if (isPeriodic)
                bnd2nodePbi.BorrowAndPull(bnd2cell);
            bndElemInfo.BorrowAndPull(bnd2cell);
            bnd2bndOrig.BorrowAndPull(bnd2cell);

            // Ghost bnds may reference nodes not yet in the coord ghost layer.
            {
                std::vector<DNDS::index> extraGhostNodes;
                for (DNDS::index iBnd = bnd2node.father->Size(); iBnd < bnd2node.Size(); iBnd++)
                    for (DNDS::rowsize j = 0; j < bnd2node.RowSize(iBnd); j++)
                    {
                        auto iNode = bnd2node(iBnd, j);
                        DNDS::MPI_int rank = UnInitMPIInt;
                        DNDS::index val = UnInitIndex;
                        if (!coords.trans.pLGhostMapping->search_indexAppend(iNode, rank, val))
                            extraGhostNodes.push_back(iNode);
                    }
                DNDS::index nExtraLocal = extraGhostNodes.size();
                DNDS::index nExtraGlobal{0};
                MPI::Allreduce(&nExtraLocal, &nExtraGlobal, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
                if (nExtraGlobal > 0)
                {
                    auto &existingGhost = coords.trans.pLGhostMapping->ghostIndex;
                    std::vector<DNDS::index> allGhostNodes(existingGhost.begin(), existingGhost.end());
                    allGhostNodes.insert(allGhostNodes.end(), extraGhostNodes.begin(), extraGhostNodes.end());
                    coords.trans.createGhostMapping(allGhostNodes);
                    node2nodeOrig.trans.BorrowGGIndexing(coords.trans);
                    coords.trans.createMPITypes();
                    node2nodeOrig.trans.createMPITypes();
                    coords.trans.pullOnce();
                    node2nodeOrig.trans.pullOnce();
                }
            }
        }

        // === Wire per-adjacency target mappings ===
        {
            auto cellGhostMap = cellElemInfo.trans.pLGhostMapping;
            auto nodeGhostMap = coords.trans.pLGhostMapping;
            auto bndGhostMap = bndElemInfo.trans.pLGhostMapping;

            cell2node.idx.wireTargetMapping(nodeGhostMap);
            bnd2node.idx.wireTargetMapping(nodeGhostMap);
            cell2cell.idx.wireTargetMapping(cellGhostMap);
            bnd2cell.idx.wireTargetMapping(cellGhostMap);
            node2cell.idx.wireTargetMapping(cellGhostMap);
            node2bnd.idx.wireTargetMapping(bndGhostMap);
        }
    }

    void UnstructuredMesh::
        AdjGlobal2LocalPrimary()
    {
        // needs results of BuildGhostPrimary()
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
        DNDS_assert(cell2node.isGlobal() && bnd2node.isGlobal() && cell2cell.isGlobal() && bnd2cell.isGlobal());
        DNDS_assert_info(cell2node.idx.isWired(), "cell2node target mapping not wired");
        DNDS_assert_info(cell2cell.idx.isWired(), "cell2cell target mapping not wired");
        DNDS_assert_info(bnd2node.idx.isWired(), "bnd2node target mapping not wired");
        DNDS_assert_info(bnd2cell.idx.isWired(), "bnd2cell target mapping not wired");

        /**********************************/
        cell2cell.toLocal();
        cell2node.toLocal();
        bnd2node.toLocal();

        // bnd2cell: use toLocal() then assert father bnds' slot-0 cell is inside
        bnd2cell.toLocal();
        for (DNDS::index iBnd = 0; iBnd < bnd2cell.father->Size(); iBnd++)
            DNDS_assert(bnd2cell(iBnd, 0) >= 0); // father bnd's owner cell must be found
        /**********************************/
        adjPrimaryState = Adj_PointToLocal;
    }

    void UnstructuredMesh::
        AdjLocal2GlobalPrimary()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(cell2node.isLocal() && bnd2node.isLocal() && cell2cell.isLocal() && bnd2cell.isLocal());
        DNDS_assert_info(cell2node.idx.isWired(), "cell2node target mapping not wired");
        DNDS_assert_info(cell2cell.idx.isWired(), "cell2cell target mapping not wired");
        DNDS_assert_info(bnd2node.idx.isWired(), "bnd2node target mapping not wired");
        DNDS_assert_info(bnd2cell.idx.isWired(), "bnd2cell target mapping not wired");

        /**********************************/
        cell2cell.toGlobal();
        bnd2cell.toGlobal();
        cell2node.toGlobal();
        bnd2node.toGlobal();
        /**********************************/
        adjPrimaryState = Adj_PointToGlobal;
    }

    void UnstructuredMesh::
        AdjGlobal2LocalPrimaryForBnd() // a reduction of primary version
    {
        // needs results of BuildGhostPrimary()
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
        DNDS_assert(cell2node.isGlobal());
        DNDS_assert_info(cell2node.idx.isWired(), "cell2node target mapping not wired");
        /**********************************/
        cell2node.toLocal();
        /**********************************/
        adjPrimaryState = Adj_PointToLocal;
    }

    void UnstructuredMesh::
        AdjLocal2GlobalPrimaryForBnd() // a reduction of primary version
    {
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(cell2node.isLocal());
        DNDS_assert_info(cell2node.idx.isWired(), "cell2node target mapping not wired");
        /**********************************/
        cell2node.toGlobal();
        /**********************************/
        adjPrimaryState = Adj_PointToGlobal;
    }

    void UnstructuredMesh::
        AdjGlobal2LocalFacial()
    {
        DNDS_assert(adjFacialState == Adj_PointToGlobal);
        DNDS_assert(face2node.isGlobal() && face2cell.isGlobal());
        DNDS_assert_info(face2node.idx.isWired(), "face2node target mapping not wired");
        DNDS_assert_info(face2cell.idx.isWired(), "face2cell target mapping not wired");
        /**********************************/
        face2node.toLocalOMP();
        face2cell.toLocalOMP();
        // face2bnd is not wired until MatchFaceBoundary runs; skip if not yet built.
        if (face2bnd.idx.isWired())
        {
            DNDS_assert(face2bnd.isGlobal());
            face2bnd.toLocal();
        }
        /**********************************/
        adjFacialState = Adj_PointToLocal;
    }

    void UnstructuredMesh::
        AdjLocal2GlobalFacial()
    {
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(face2node.isLocal() && face2cell.isLocal() && face2bnd.isLocal());
        DNDS_assert_info(face2node.idx.isWired(), "face2node target mapping not wired");
        DNDS_assert_info(face2cell.idx.isWired(), "face2cell target mapping not wired");
        DNDS_assert_info(face2bnd.idx.isWired(), "face2bnd target mapping not wired");
        /**********************************/
        face2node.toGlobalOMP();
        face2cell.toGlobalOMP();
        face2bnd.toGlobal();
        // MPI::Barrier(mpi.comm);
        /**********************************/
        adjFacialState = Adj_PointToGlobal;
    }

    void UnstructuredMesh::
        AdjLocal2GlobalC2F()
    {
        DNDS_assert(adjC2FState == Adj_PointToLocal);
        DNDS_assert(cell2face.isLocal() && bnd2face.isLocal());
        DNDS_assert_info(cell2face.idx.isWired(), "cell2face target mapping not wired");
        DNDS_assert_info(bnd2face.idx.isWired(), "bnd2face target mapping not wired");
        /**********************************/
        cell2face.toGlobalOMP();
        bnd2face.toGlobal();
        /**********************************/
        adjC2FState = Adj_PointToGlobal;
    }

    void UnstructuredMesh::
        AdjGlobal2LocalC2F()
    {
        DNDS_assert(adjC2FState == Adj_PointToGlobal);
        DNDS_assert(cell2face.isGlobal() && bnd2face.isGlobal());
        DNDS_assert_info(cell2face.idx.isWired(), "cell2face target mapping not wired");
        DNDS_assert_info(bnd2face.idx.isWired(), "bnd2face target mapping not wired");
        /**********************************/
        cell2face.toLocalOMP();
        bnd2face.toLocal();
        /**********************************/
        adjC2FState = Adj_PointToLocal;
    }

    void UnstructuredMesh::
        BuildGhostN2CB()
    {
        DNDS_assert(adjN2CBState == Adj_PointToGlobal);
        DNDS_assert(node2cell.isGlobal() && node2bnd.isGlobal());

        DNDS_assert(coords.trans.father && coords.trans.pLGhostMapping);

        node2cell.TransAttach();
        node2cell.trans.BorrowGGIndexing(coords.trans);
        node2cell.trans.createMPITypes();
        node2cell.trans.pullOnce();

        node2bnd.TransAttach();
        node2bnd.trans.BorrowGGIndexing(coords.trans);
        node2bnd.trans.createMPITypes();
        node2bnd.trans.pullOnce();

        node2cell.idx.markGlobal();
        node2bnd.idx.markGlobal();
    }

    void UnstructuredMesh::
        AdjLocal2GlobalN2CB()
    {
        DNDS_assert(adjN2CBState == Adj_PointToLocal);
        DNDS_assert(node2cell.isLocal() && node2bnd.isLocal());
        DNDS_assert_info(node2cell.idx.isWired(), "node2cell target mapping not wired");
        DNDS_assert_info(node2bnd.idx.isWired(), "node2bnd target mapping not wired");
        /**********************************/
        node2cell.toGlobalOMP();
        node2bnd.toGlobalOMP();
        /**********************************/
        adjN2CBState = Adj_PointToGlobal;
    }

    void UnstructuredMesh::
        AdjGlobal2LocalN2CB()
    {
        DNDS_assert(adjN2CBState == Adj_PointToGlobal);
        DNDS_assert(node2cell.isGlobal() && node2bnd.isGlobal());
        DNDS_assert_info(node2cell.idx.isWired(), "node2cell target mapping not wired");
        DNDS_assert_info(node2bnd.idx.isWired(), "node2bnd target mapping not wired");
        /**********************************/
        node2cell.toLocalOMP();
        node2bnd.toLocalOMP();
        /**********************************/
        adjN2CBState = Adj_PointToLocal;
    }

    void UnstructuredMesh::AssertOnN2CB()
    {
        for (index iNode = NumNode(); iNode < NumNodeProc(); iNode++)
        {
            int nCellAdjIn = 0;
            for (auto iCell : node2cell[iNode])
                if (iCell >= 0)
                {
                    DNDS_assert(iCell < NumCellProc());
                    nCellAdjIn++;
                    for (auto iNodeOther : cell2node[iCell])
                    {
                        DNDS_assert(iNodeOther < NumNodeProc() && iNodeOther >= 0);
                    }
                }
            DNDS_assert(nCellAdjIn);
        }

        std::set<index> bnd_main_nodes;
        for (index iNode = 0; iNode < NumNode(); iNode++)
        {
            for (auto iCell : node2cell[iNode])
            {
                DNDS_assert(iCell < NumCellProc() && iCell >= 0);
                for (auto iNodeOther : cell2node[iCell])
                {
                    DNDS_assert(iNodeOther < NumNodeProc() && iNodeOther >= 0);
                    if (iNodeOther >= NumNode())
                        bnd_main_nodes.insert(iNode);
                }
            }
        }
        std::map<index, int> bnd_main_nodes_adj_ghost_num;
        for (index iNode : bnd_main_nodes)
            bnd_main_nodes_adj_ghost_num[iNode] = 0;
        for (index iNode = NumNode(); iNode < NumNodeProc(); iNode++)
            for (auto iCell : node2cell[iNode])
                if (iCell >= 0)
                    for (auto iNodeOther : cell2node[iCell])
                        if (iNodeOther >= 0 && iNodeOther < NumNode())
                            bnd_main_nodes_adj_ghost_num.at(iNodeOther)++;
        for (auto [iNode, num_ghost_adj] : bnd_main_nodes_adj_ghost_num)
            DNDS_assert(num_ghost_adj);
    }

    void UnstructuredMesh::BuildCell2CellFace()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(cell2node.isLocal() && bnd2node.isLocal());
        DNDS_assert(adjFacialState == Adj_PointToLocal);
        DNDS_assert(face2cell.isLocal() && face2node.isLocal());
        DNDS_assert_info(cellElemInfo.trans.pLGhostMapping, "trans of cellElemInfo needed but not built");

        cell2cellFace.InitPair("cell2cellFace", mpi);
        cell2cellFace.father->Resize(this->NumCell());
        for (index iCell = 0; iCell < this->NumCell(); iCell++)
        {
            cell2cellFace.ResizeRow(iCell, cell2face[iCell].size());
            for (rowsize ic2f = 0; ic2f < cell2face[iCell].size(); ic2f++)
            {
                index iFace = cell2face[iCell][ic2f];
                index iCellOther = this->CellFaceOther(iCell, iFace);
                DNDS_assert(iCellOther < this->NumCellProc());
                cell2cellFace[iCell][ic2f] = this->CellIndexLocal2Global(iCellOther);
            }
        }
        cell2cellFace.father->Compress();
        cell2cellFace.TransAttach();
        cell2cellFace.trans.BorrowGGIndexing(cell2node.trans);
        cell2cellFace.trans.createMPITypes();
        cell2cellFace.trans.pullOnce(); // warning! to pull the adj, must be in global state!
        adjC2CFaceState = Adj_PointToGlobal;
        cell2cellFace.idx.markGlobal();

        // Wire cell2cellFace target mapping (points to cells)
        cell2cellFace.idx.wireTargetMapping(cellElemInfo.trans.pLGhostMapping);
    }

    void UnstructuredMesh::AdjLocal2GlobalC2CFace()
    {
        // needs results of BuildGhostPrimary()
        DNDS_assert(adjC2CFaceState == Adj_PointToLocal);
        DNDS_assert(cell2cellFace.isLocal());
        DNDS_assert_info(cell2cellFace.idx.isWired(), "cell2cellFace target mapping not wired");

        /**********************************/
        cell2cellFace.toGlobal();
        /**********************************/

        adjC2CFaceState = Adj_PointToGlobal;
    }

    void UnstructuredMesh::AdjGlobal2LocalC2CFace()
    {
        // needs results of BuildGhostPrimary()
        DNDS_assert(adjC2CFaceState == Adj_PointToGlobal);
        DNDS_assert(cell2cellFace.isGlobal());
        DNDS_assert_info(cell2cellFace.idx.isWired(), "cell2cellFace target mapping not wired");

        /**********************************/
        cell2cellFace.toLocal();
        /**********************************/

        adjC2CFaceState = Adj_PointToLocal;
    }

    void UnstructuredMesh::
        InterpolateFace()
    {
        DNDS_assert(adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(cell2node.isLocal() && cell2cell.isLocal() && bnd2node.isLocal());

        // === Section A: Allocate face-related array pairs ===
        cell2face.InitPair("cell2face", mpi);
        face2cell.InitPair("face2cell", mpi);
        face2node.InitPair("face2node", mpi);
        if (isPeriodic)
            face2nodePbi.InitPair("face2nodePbi", mpi);
        faceElemInfo.InitPair("faceElemInfo", mpi);
        face2bnd.InitPair("face2bnd", mpi);
        bnd2face.InitPair("bnd2face", mpi);

        cell2face.father->Resize(cell2cell.father->Size());
        cell2face.son->Resize(cell2cell.son->Size());

        index nCellAll = cell2cell.Size();
        index nNodeAll = coords.Size();
        index nLocalCells = cell2cell.father->Size();

        // === Section B: Build SubEntityQueryPbi ===
        SubEntityQueryPbi faceQuery;
        faceQuery.numSubEntities = [this](index iParent) -> int
        {
            return Elem::Element{cellElemInfo[iParent]->getElemType()}.GetNumFaces();
        };
        faceQuery.describe = [this](index iParent, int iSub) -> SubEntityDesc
        {
            auto eParent = Elem::Element{cellElemInfo[iParent]->getElemType()};
            auto eFace = eParent.ObtainFace(iSub);
            return SubEntityDesc{eFace.GetNumVertices(), eFace.GetNumNodes(),
                                 static_cast<t_index>(eFace.type)};
        };
        faceQuery.extractNodes = [this](index iParent, int iSub,
                                        const std::function<index(int)> &parentNodes,
                                        index *out)
        {
            auto eParent = Elem::Element{cellElemInfo[iParent]->getElemType()};
            auto eFace = eParent.ObtainFace(iSub);
            std::vector<index> pNodes(eParent.GetNumNodes());
            for (int i = 0; i < eParent.GetNumNodes(); i++)
                pNodes[i] = parentNodes(i);
            std::vector<index> fNodes(eFace.GetNumNodes());
            eParent.ExtractFaceNodes(iSub, pNodes, fNodes);
            for (int i = 0; i < eFace.GetNumNodes(); i++)
                out[i] = fNodes[i];
        };
        if (isPeriodic)
        {
            faceQuery.matchExtra = [this](index iParent, int iSub,
                                          index /*iCandEntity*/,
                                          index candidateParent, int candidateSub) -> bool
            {
                auto eParentA = Elem::Element{cellElemInfo[iParent]->getElemType()};
                auto eFaceA = eParentA.ObtainFace(iSub);
                int nFaceNodes = eFaceA.GetNumNodes();
                std::vector<index> nodesA(nFaceNodes);
                eParentA.ExtractFaceNodes(iSub, cell2node[iParent], nodesA);
                std::vector<NodePeriodicBits> pbiA(nFaceNodes);
                eParentA.ExtractFaceNodes(iSub, cell2nodePbi[iParent], pbiA);
                auto eParentB = Elem::Element{cellElemInfo[candidateParent]->getElemType()};
                std::vector<index> nodesB(nFaceNodes);
                eParentB.ExtractFaceNodes(candidateSub, cell2node[candidateParent], nodesB);
                std::vector<NodePeriodicBits> pbiB(nFaceNodes);
                eParentB.ExtractFaceNodes(candidateSub, cell2nodePbi[candidateParent], pbiB);
                using idx_pbi = std::pair<index, NodePeriodicBits>;
                auto cmp = [](const idx_pbi &L, const idx_pbi &R)
                { return L.first == R.first ? uint8_t(L.second) < uint8_t(R.second)
                                            : L.first < R.first; };
                std::vector<idx_pbi> pairsA(nFaceNodes), pairsB(nFaceNodes);
                for (int i = 0; i < nFaceNodes; i++)
                {
                    pairsA[i] = {nodesA[i], pbiA[i]};
                    pairsB[i] = {nodesB[i], pbiB[i]};
                }
                std::sort(pairsA.begin(), pairsA.end(), cmp);
                std::sort(pairsB.begin(), pairsB.end(), cmp);
                auto v0 = pairsA[0].second ^ pairsB[0].second;
                for (int i = 1; i < nFaceNodes; i++)
                    if ((pairsA[i].second ^ pairsB[i].second) != v0)
                        return false;
                return true;
            };
            faceQuery.extractPbi = [this](index iParent, int iSub,
                                          const std::function<NodePeriodicBits(int)> &parentPbi,
                                          NodePeriodicBits *out)
            {
                auto eParent = Elem::Element{cellElemInfo[iParent]->getElemType()};
                auto eFace = eParent.ObtainFace(iSub);
                std::vector<NodePeriodicBits> pPbi(eParent.GetNumNodes());
                for (int i = 0; i < eParent.GetNumNodes(); i++)
                    pPbi[i] = parentPbi(i);
                std::vector<NodePeriodicBits> fPbi(eFace.GetNumNodes());
                eParent.ExtractFaceNodes(iSub, pPbi, fPbi);
                for (int i = 0; i < eFace.GetNumNodes(); i++)
                    out[i] = fPbi[i];
            };
        }

        // === Section C: Ownership resolver ===
        OwnershipResolverMulti faceOwnership =
            [this](const std::vector<index> &parents,
                   const std::vector<MPI_int> &parentRanks,
                   index nLocal) -> OwnershipDecision
        {
            MPI_int minRank = parentRanks[0];
            for (size_t i = 1; i < parentRanks.size(); i++)
                if (parentRanks[i] < minRank)
                    minRank = parentRanks[i];
            bool anyLocal = false;
            for (auto p : parents)
                if (p < nLocal)
                    anyLocal = true;
            if (!anyLocal)
                return {false, {}};
            if (minRank != mpi.rank)
                return {false, {}};
            std::vector<MPI_int> peers;
            for (size_t i = 0; i < parents.size(); i++)
                if (parents[i] >= nLocal && parentRanks[i] != mpi.rank)
                    peers.push_back(parentRanks[i]);
            std::sort(peers.begin(), peers.end());
            peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
            return {true, std::move(peers)};
        };

        // === Section D: InterpolateGlobal (e2p_rs=2: face2cell is fixed-width) ===
        auto globalResult = CheckedInterpolateGlobal<NonUniformSize, 2>(
            cell2node, isPeriodic ? cell2nodePbi : tPbiPair{},
            *cell2node.trans.pLGhostMapping,
            *cell2node.father->pLGlobalMapping,
            *coords.trans.pLGhostMapping,
            faceQuery, nLocalCells, nCellAll, nNodeAll,
            faceOwnership, mpi);

        index nOwnedFaces = globalResult.nOwnedEntities;

        // === Section E: Adopt owned face data directly into mesh arrays ===
        face2node.father = globalResult.entity2node.father;
        face2node.TransAttach();
        face2node.trans.createFatherGlobalMapping();

        // entity2parent is already ArrayAdjacencyPair<2> — adopt directly, no copy.
        face2cell.father = globalResult.entity2parent.father;

        faceElemInfo.father = globalResult.entityElemInfo.father;
        if (isPeriodic)
            face2nodePbi.father = globalResult.entity2nodePbi.father;

        // Populate cell2face from parent2entity (global face IDs).
        for (index iCell = 0; iCell < nCellAll; iCell++)
        {
            rowsize nFaces = globalResult.parent2entity.RowSize(iCell);
            cell2face.ResizeRow(iCell, nFaces);
            for (rowsize j = 0; j < nFaces; j++)
                cell2face(iCell, j) = globalResult.parent2entity(iCell, j);
        }
        cell2face.father->Compress();
        cell2face.son->Compress();

        auto gSize = face2node.father->globalSize();
        if (mpi.rank == 0)
            log() << "UnstructuredMesh === InterpolateFace: total faces " << gSize << std::endl;

        BuildGhostFace();
        MatchFaceBoundary();
    }

    void UnstructuredMesh::
        BuildGhostFace()
    {
        // Determine ghost faces via evaluateGhostTree(Cell → Cell2Face → Face).
        // cell2face is cell-indexed but has no own pLGlobalMapping — borrow from cell2node.
        if (!cell2face.father->pLGlobalMapping)
            cell2face.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
        {
            MeshConnectivity dag;
            fillRegistry(dag);

            GhostSpec ghostSpec;
            ghostSpec.chains.push_back(GhostChain{
                EntityKind::Cell,
                {Adj::Cell2Face},
                EntityKind::Face,
            });
            auto ghostTree = CompiledGhostTree::compile(ghostSpec);
            auto ghostResult = dag.evaluateGhostTree(ghostTree, mpi);

            auto &ghostFaces = ghostResult.ghostIndices[EntityKind::Face];

            face2cell.TransAttach();
            face2cell.trans.createFatherGlobalMapping();
            face2cell.trans.createGhostMapping(ghostFaces);
            face2cell.trans.createMPITypes();
            face2cell.trans.pullOnce();
            face2node.trans.BorrowGGIndexing(face2cell.trans);
            face2node.trans.createMPITypes();
            face2node.trans.pullOnce();
            faceElemInfo.TransAttach();
            faceElemInfo.trans.BorrowGGIndexing(face2cell.trans);
            faceElemInfo.trans.createMPITypes();
            faceElemInfo.trans.pullOnce();
            if (isPeriodic)
            {
                face2nodePbi.TransAttach();
                face2nodePbi.trans.BorrowGGIndexing(face2cell.trans);
                face2nodePbi.trans.createMPITypes();
                face2nodePbi.trans.pullOnce();
            }
        }

        // Wire per-adjacency target mappings for facial adjacencies.
        // Ghost mappings are available after the ghost-tree pull above.
        {
            auto cellGhostMap = cellElemInfo.trans.pLGhostMapping;
            auto nodeGhostMap = coords.trans.pLGhostMapping;
            face2node.idx.wireTargetMapping(nodeGhostMap);
            face2cell.idx.wireTargetMapping(cellGhostMap);
        }

        // Convert face arrays to local indices.
        adjFacialState = Adj_PointToGlobal;
        face2node.idx.markGlobal();
        face2cell.idx.markGlobal();
        AdjGlobal2LocalFacial();

        // Convert cell2face to local face indices (father + son).
        // Wire target mapping first (face ghost mapping available from BuildGhostFace).
        {
            auto faceGhostMap = face2cell.trans.pLGhostMapping;
            cell2face.idx.wireTargetMapping(faceGhostMap);
            bnd2face.idx.wireTargetMapping(faceGhostMap);
        }
        adjC2FState = Adj_PointToGlobal;
        cell2face.idx.markGlobal();
        bnd2face.idx.markGlobal();
        cell2face.toLocalOMP();
        adjC2FState = Adj_PointToLocal;
        // bnd2face has no entries yet (populated in MatchFaceBoundary),
        // but its state must track cell2face for the group invariant.
        bnd2face.toLocal();
    }

    void UnstructuredMesh::
        MatchFaceBoundary()
    {
        MatchBoundariesToFaces(
            bndElemInfo, bnd2cell, bnd2node, cell2face, face2node, cell2node,
            faceElemInfo, bnd2faceV, face2bndM, face2bnd, bnd2face);

        // Re-pull faceElemInfo so ghost faces inherit the owning rank's zone.
        // MatchBoundariesToFaces assigns periodic zones (main vs donor) based
        // on local boundary elements, which can disagree across ranks for the
        // same physical face. The owning rank's assignment is authoritative.
        faceElemInfo.trans.pullOnce();

        // face2bnd.father now contains local bnd indices (from MatchBoundariesToFaces).
        // Wire + markLocal, then convert to global before the ghost pull so that
        // remote ranks receive rank-independent global bnd indices.
        face2bnd.idx.wireTargetMapping(bndElemInfo.trans.pLGhostMapping);
        face2bnd.idx.markLocal();
        face2bnd.toGlobal();
        face2bnd.BorrowAndPull(face2cell);
        face2bnd.toLocal(); // converts both father and son: global → local

        // Communicate cell2face and bnd2face ghost data.
        // cell2face/bnd2face are already wired (BuildGhostFace).
        this->AdjLocal2GlobalC2F();
        cell2face.TransAttach();
        cell2face.trans.BorrowGGIndexing(cell2node.trans);
        cell2face.trans.createMPITypes();
        cell2face.trans.pullOnce();
        bnd2face.TransAttach();
        bnd2face.trans.BorrowGGIndexing(bnd2node.trans);
        bnd2face.trans.createMPITypes();
        bnd2face.trans.pullOnce();

        this->AdjGlobal2LocalC2F();
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
        WriteSerialize(Serializer::SerializerBaseSSP serializerP, const std::string &name)
    {
        // TODO: Make these read/write of mesh independent of ghost part!
        DNDS_assert(adjPrimaryState == Adj_PointToGlobal);
        DNDS_assert(cell2node.isGlobal() && bnd2node.isGlobal() && cell2cell.isGlobal() && bnd2cell.isGlobal());

        auto cwd = serializerP->GetCurrentPath();
        serializerP->CreatePath(name);
        serializerP->GoToPath(name);

        serializerP->WriteString("mesh", "UnstructuredMesh");
        serializerP->WriteIndex("dim", dim);
        if (serializerP->IsPerRank())
            serializerP->WriteIndex("MPIRank", mpi.rank);
        serializerP->WriteIndex("MPISize", mpi.size);
        serializerP->WriteInt("isPeriodic", isPeriodic);

        coords.WriteSerialize(serializerP, "coords");
        cell2node.WriteSerialize(serializerP, "cell2node");
        // cell2cell.WriteSerialize(serializerP, "cell2cell");
        cellElemInfo.WriteSerialize(serializerP, "cellElemInfo");
        bnd2node.WriteSerialize(serializerP, "bnd2node");
        // bnd2cell.WriteSerialize(serializerP, "bnd2cell");
        bndElemInfo.WriteSerialize(serializerP, "bndElemInfo");
        if (isPeriodic)
        {
            cell2nodePbi.WriteSerialize(serializerP, "cell2nodePbi");
            bnd2nodePbi.WriteSerialize(serializerP, "bnd2nodePbi");
            periodicInfo.WriteSerializer(serializerP, "periodicInfo");
        }
        bnd2bndOrig.WriteSerialize(serializerP, "bnd2bndOrig");
        cell2cellOrig.WriteSerialize(serializerP, "cell2cellOrig");
        node2nodeOrig.WriteSerialize(serializerP, "node2nodeOrig");

        serializerP->GoToPath(cwd);
    }

    void UnstructuredMesh::
        ReadSerialize(Serializer::SerializerBaseSSP serializerP, const std::string &name)
    {
        auto cwd = serializerP->GetCurrentPath();
        // serializerP->CreatePath(name);//! remember no create!
        serializerP->GoToPath(name);

        std::string meshRead;
        index dimRead{0}, rankRead{0}, sizeRead{0};
        int isPeriodicRead = 0;
        serializerP->ReadString("mesh", meshRead);
        serializerP->ReadIndex("dim", dimRead);
        if (serializerP->IsPerRank())
            serializerP->ReadIndex("MPIRank", rankRead);
        serializerP->ReadIndex("MPISize", sizeRead);
        serializerP->ReadInt("isPeriodic", isPeriodicRead);
        isPeriodic = bool(isPeriodicRead);
        DNDS_assert(meshRead == "UnstructuredMesh");
        DNDS_assert(dimRead == dim);
        DNDS_assert((!serializerP->IsPerRank() || rankRead == mpi.rank) && sizeRead == mpi.size);

        // make the empty arrays
        coords.InitPair("coords", getMPI());
        cellElemInfo.InitPair("cellElemInfo", getMPI());
        bndElemInfo.InitPair("bndElemInfo", getMPI());
        cell2node.InitPair("cell2node", getMPI());
        if (isPeriodic)
        {
            cell2nodePbi.InitPair("cell2nodePbi", getMPI());
            bnd2nodePbi.InitPair("bnd2nodePbi", getMPI());
        }
        bnd2node.InitPair("bnd2node", getMPI());

        bnd2bndOrig.InitPair("bnd2bndOrig", getMPI());
        cell2cellOrig.InitPair("cell2cellOrig", getMPI());
        node2nodeOrig.InitPair("node2nodeOrig", getMPI());

        coords.ReadSerialize(serializerP, "coords");
        cell2node.ReadSerialize(serializerP, "cell2node");
        // cell2cell.ReadSerialize(serializerP, "cell2cell");
        cellElemInfo.ReadSerialize(serializerP, "cellElemInfo");
        bnd2node.ReadSerialize(serializerP, "bnd2node");
        // bnd2cell.ReadSerialize(serializerP, "bnd2cell");
        bndElemInfo.ReadSerialize(serializerP, "bndElemInfo");
        if (isPeriodic)
        {
            cell2nodePbi.ReadSerialize(serializerP, "cell2nodePbi");
            bnd2nodePbi.ReadSerialize(serializerP, "bnd2nodePbi");
            periodicInfo.ReadSerializer(serializerP, "periodicInfo");
        }
        bnd2bndOrig.ReadSerialize(serializerP, "bnd2bndOrig");
        cell2cellOrig.ReadSerialize(serializerP, "cell2cellOrig");
        node2nodeOrig.ReadSerialize(serializerP, "node2nodeOrig");

        // after matters:
        coords.trans.createMPITypes();
        cell2node.trans.createMPITypes();
        // cell2cell.trans.createMPITypes();
        cellElemInfo.trans.createMPITypes();
        bnd2node.trans.createMPITypes();
        // bnd2cell.trans.createMPITypes();
        bndElemInfo.trans.createMPITypes();
        if (isPeriodic)
        {
            cell2nodePbi.trans.createMPITypes();
            bnd2nodePbi.trans.createMPITypes();
        }
        bnd2bndOrig.trans.createMPITypes();
        cell2cellOrig.trans.createMPITypes();
        node2nodeOrig.trans.createMPITypes();
        adjPrimaryState = Adj_PointToGlobal; // the file is pointing to local
        cell2node.idx.markGlobal();
        bnd2node.idx.markGlobal();

        index nCellG = this->NumCellGlobal(); // collective call!
        index nNodeG = this->NumNodeGlobal(); // collective call!
        index nNodeB = this->NumBndGlobal();  // collective call!
        if (mpi.rank == mRank)
        {
            log() << "UnstructuredMesh === ReadSerialize "
                  << "Global NumCell [ " << nCellG << " ]" << std::endl;
            log() << "UnstructuredMesh === ReadSerialize "
                  << "Global NumNode [ " << nNodeG << " ]" << std::endl;
            log() << "UnstructuredMesh === ReadSerialize "
                  << "Global NumBnd  [ " << nNodeB << " ]" << std::endl;
        }
        MPISerialDo(mpi, [&]()
                    { log() << "    Rank: " << mpi.rank << " nCell " << this->NumCell() << " nCellGhost " << this->NumCellGhost() << std::endl; });
        MPISerialDo(mpi, [&]()
                    { log() << "    Rank: " << mpi.rank << " nNode " << this->NumNode() << " nNodeGhost " << this->NumNodeGhost() << std::endl; });

        serializerP->GoToPath(cwd);
    }

    void UnstructuredMesh::ConstructBndMesh(UnstructuredMesh &bMesh)
    {
        DNDS_assert(bMesh.dim == dim - 1 && bMesh.mpi == mpi);
        bMesh.cell2node.InitPair("bMesh.cell2node", mpi);
        bMesh.coords.InitPair("bMesh.coords", mpi);
        if (isPeriodic)
        {
            bMesh.isPeriodic = true;
            bMesh.periodicInfo = this->periodicInfo;
            bMesh.cell2nodePbi.InitPair("bMesh.cell2nodePbi", mpi);
        }
        bMesh.cellElemInfo.InitPair("bMesh.cellElemInfo", mpi);
        bMesh.cell2cellOrig.InitPair("bMesh.cell2cellOrig", mpi);
        bMesh.node2nodeOrig.InitPair("bMesh.node2nodeOrig", mpi);

        bMesh.bnd2cell.InitPair("bMesh.bnd2cell", mpi);       // which will remain 0 sized
        bMesh.bnd2node.InitPair("bMesh.bnd2node", mpi);       // which will remain 0 sized
        bMesh.bndElemInfo.InitPair("bMesh.bndElemInfo", mpi); // which will remain 0 sized
        bMesh.bnd2bndOrig.InitPair("bMesh.bnd2bndOrig", mpi); // which will remain 0 sized
        if (isPeriodic)
            bMesh.bnd2nodePbi.InitPair("bMesh.bnd2nodePbi", mpi); // which will remain 0 sized

        tIndPair node2bndNodeGlobal;
        node2bndNodeGlobal.InitPair("node2bndNodeGlobal", mpi);

        node2bndNodeGlobal.father->Resize(this->NumNode());
        node2bndNodeGlobal.TransAttach();
        node2bndNodeGlobal.trans.BorrowGGIndexing(coords.trans);
        node2bndNodeGlobal.trans.createMPITypes();
        index bndNodeCount{0}, bndNodeStart{0};
        {
            for (index i = 0; i < node2bndNodeGlobal.Size(); i++)
                node2bndNodeGlobal[i] = 0;
            for (index iBnd = 0; iBnd < this->NumBnd(); iBnd++)
                for (auto iNode : bnd2node.father->operator[](iBnd))
                    node2bndNodeGlobal[iNode]++; // now stores num-reference

            {
                auto node2bndNodeGlobalPast = make_ssp<tInd::element_type>(ObjName{"node2bndNodeGlobalPast"}, mpi);
                DNDS::ArrayTransformerType<tInd::element_type>::Type node2bndNodeGlobalPastTrans;
                node2bndNodeGlobalPastTrans.setFatherSon(node2bndNodeGlobal.son, node2bndNodeGlobalPast);
                node2bndNodeGlobalPastTrans.createFatherGlobalMapping();
                std::vector<index> pushSonSeries(node2bndNodeGlobal.son->Size());
                for (size_t i = 0; i < pushSonSeries.size(); i++)
                    pushSonSeries[i] = DNDS::size_to_index(i);
                node2bndNodeGlobalPastTrans.createGhostMapping(pushSonSeries, node2bndNodeGlobal.trans.pLGhostMapping->ghostStart);
                node2bndNodeGlobalPastTrans.createMPITypes();
                node2bndNodeGlobalPastTrans.pullOnce();
                DNDS_assert(DNDS::size_to_index(node2bndNodeGlobal.trans.pLGhostMapping->ghostIndex.size()) == node2bndNodeGlobal.son->Size());
                DNDS_assert(DNDS::size_to_index(node2bndNodeGlobal.trans.pLGhostMapping->pushingIndexGlobal.size()) == node2bndNodeGlobalPast->Size());
                for (index i = 0; i < node2bndNodeGlobalPast->Size(); i++)
                {
                    index iNodeG = node2bndNodeGlobal.trans.pLGhostMapping->pushingIndexGlobal[i]; //?should be right
                    auto [ret, rank, iNodeL] = node2bndNodeGlobal.trans.pLGlobalMapping->search(iNodeG);
                    DNDS_assert(ret && rank == node2bndNodeGlobal.father->getMPI().rank); // has to be local main
                    node2bndNodeGlobal[iNodeL] += (*node2bndNodeGlobalPast)[i];
                }
            }
            {
                for (index i = 0; i < node2bndNodeGlobal.father->Size(); i++)
                    if (node2bndNodeGlobal[i] == 0)
                        node2bndNodeGlobal[i] = UnInitIndex;
                    else
                        node2bndNodeGlobal[i] = bndNodeCount++;
            }
            MPI::Scan(&bndNodeCount, &bndNodeStart, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            bndNodeStart -= bndNodeCount;
            DNDS_assert(bndNodeStart >= 0 && bndNodeCount >= 0);
            for (index i = 0; i < node2bndNodeGlobal.Size(); i++)
                if (node2bndNodeGlobal[i] >= 0)
                    node2bndNodeGlobal[i] += bndNodeStart;
        }
        node2bndNodeGlobal.trans.pullOnce();
        bMesh.coords.father->Resize(bndNodeCount);
        bMesh.coords.TransAttach();
        bMesh.coords.trans.createFatherGlobalMapping();
        std::vector<index> bndNodePullingSet;
        for (index iNode = 0; iNode < this->NumNodeProc(); iNode++)
        {
            if (node2bndNodeGlobal[iNode] < 0)
                continue;
            // then node2bndNodeGlobal[iNode] >= 0, which covers all bnd2node entries
            auto [ret, rank, iNodeL_in_bnd] = bMesh.coords.trans.pLGlobalMapping->search(node2bndNodeGlobal[iNode]);
            DNDS_assert(ret);
            if (rank != node2bndNodeGlobal.father->getMPI().rank)
                bndNodePullingSet.push_back(node2bndNodeGlobal[iNode]); // bndNodePullingSet now also covers bnd2node needs
        }
        bMesh.coords.trans.createGhostMapping(bndNodePullingSet);
        bMesh.coords.trans.createMPITypes();
        bMesh.node2nodeOrig.father->Resize(bndNodeCount);
        bMesh.node2nodeOrig.TransAttach();
        bMesh.node2nodeOrig.trans.BorrowGGIndexing(bMesh.coords.trans);
        bMesh.node2nodeOrig.trans.createMPITypes();

        // std::cout << mpi.rank << ", " << bndNodePullingSet.size() << ", " << bndNodeCount << std::endl;
        node2bndNode.resize(this->NumNodeProc(), -1);
        bMesh.node2parentNode.resize(bMesh.coords.Size());
        for (index iN = 0; iN < node2bndNodeGlobal.Size(); iN++)
            node2bndNode.at(iN) = bMesh.NodeIndexGlobal2Local(node2bndNodeGlobal[iN]),
            DNDS_assert(node2bndNodeGlobal[iN] != UnInitIndex ? (node2bndNode.at(iN) >= 0) : true);
        for (size_t iNode = 0; iNode < node2bndNode.size(); iNode++)
            if (node2bndNode[iNode] >= 0)
                bMesh.node2parentNode.at(node2bndNode[iNode]) = iNode;

        // std::cout << mpi.rank << ", " << bndNodeCount << std::endl;
        for (index iBNode = 0; iBNode < bMesh.coords.Size(); iBNode++)
            bMesh.coords[iBNode] = coords[bMesh.node2parentNode[iBNode]];
        bMesh.coords.trans.pullOnce(); // excessive, should be identical
        for (index iBNode = 0; iBNode < bMesh.node2nodeOrig.Size(); iBNode++)
            bMesh.node2nodeOrig[iBNode] = node2nodeOrig[bMesh.node2parentNode[iBNode]];

        index nBndCellUse{0};
        for (index iB = 0; iB < this->NumBnd(); iB++)
            if (!FaceIDIsPeriodic(this->bndElemInfo(iB, 0).zone))
                nBndCellUse++;
        bMesh.cell2node.father->Resize(nBndCellUse);
        if (isPeriodic)
            bMesh.cell2nodePbi.father->Resize(nBndCellUse);
        bMesh.cell2cellOrig.father->Resize(nBndCellUse);
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
            bMesh.cell2cellOrig(nBndCellUse, 0) = bnd2bndOrig(iB, 0);

            for (rowsize ib2n = 0; ib2n < bnd2node.RowSize(iB); ib2n++)
            {
                if (bnd2faceV.at(iB) < 0) // where bnd has not a face!
                    bMesh.cell2node[nBndCellUse][ib2n] = node2bndNode.at(bnd2node[iB][ib2n]);
                else
                    bMesh.cell2node[nBndCellUse][ib2n] = node2bndNode.at(face2node[bnd2faceV.at(iB)][ib2n]); //* respect the face ordering if possible // this can be omitted if all bnds used are not periodic
                DNDS_assert(node2bndNode.at(bnd2node[iB][ib2n]) >= 0);
                if (isPeriodic)
                {
                    if (bnd2faceV.at(iB) < 0)                                             // where bnd has not a face!
                        bMesh.cell2nodePbi[nBndCellUse][ib2n] = Geom::NodePeriodicBits{}; // a invalid value
                    else
                    {
                        bMesh.cell2nodePbi[nBndCellUse][ib2n] = face2nodePbi[bnd2faceV.at(iB)][ib2n];
                    }
                }
            }
            nBndCellUse++;
        }

        bMesh.cell2node.father->Compress();
        bMesh.cell2node.father->createGlobalMapping();
        bMesh.cell2node.TransAttach();
        bMesh.cell2node.trans.createGhostMapping(std::vector<int>{}); // now bnd mesh has no valid cell2cell and ghost cells

        // Ensure global mappings exist on all arrays that BuildSerialOut
        // (and other consumers) may query via globalSize().  The ghost
        // mappings were already set up for coords/node2nodeOrig above;
        // cellElemInfo and cell2cellOrig only need the global mapping.
        bMesh.cellElemInfo.father->createGlobalMapping();
        bMesh.cell2cellOrig.father->createGlobalMapping();

        bMesh.adjPrimaryState = Adj_PointToLocal;
        // bMesh.cell2node was populated directly with local indices.
        // Wire the target mapping (bMesh's node ghost mapping), then mark local.
        bMesh.cell2node.idx.wireTargetMapping(bMesh.coords.trans.pLGhostMapping);
        bMesh.cell2node.idx.markLocal();
        if (mpi.rank == mRank)
            log() << "UnstructuredMesh === ConstructBndMesh Done" << std::endl;
    }

    void UnstructuredMesh::ObtainSymmetricSymbolicFactorization(Direct::SerialSymLUStructure &symLU, Direct::DirectPrecControl control) const
    {
        if (control.useDirectPrec)
            symLU.ObtainSymmetricSymbolicFactorization(
                cell2cellFaceVLocalParts,
                this->localPartitionStarts,
                control.getILUCode());
    }

    void UnstructuredMesh::fillRegistry(
        MeshConnectivity &dag) const
    {
        fillRegistry(dag, {});
    }

    void UnstructuredMesh::fillRegistry(
        MeshConnectivity &dag,
        const std::unordered_set<AdjKind, AdjKindHash> &skip) const
    {
        dag.meshDim = dim;

        // --- Register adjacency arrays (father-only shallow copy) ---
        auto tryAdj = [&](AdjKind kind, const auto &pair)
        {
            if (pair.father && skip.find(kind) == skip.end())
                dag.registerAdj(kind, pair);
        };

        // Primary
        tryAdj(Adj::Cell2Node, cell2node);
        tryAdj(Adj::Cell2Cell, cell2cell);
        tryAdj(Adj::Bnd2Node, bnd2node);
        tryAdj(Adj::Bnd2Cell, bnd2cell);
        // N2CB
        tryAdj(Adj::Node2Cell, node2cell);
        tryAdj(Adj::Node2Bnd, node2bnd);
        // Facial
        tryAdj(Adj::Cell2Face, cell2face);
        tryAdj(Adj::Face2Node, face2node);
        tryAdj(Adj::Face2Cell, face2cell);
        tryAdj(Adj::Face2Bnd, face2bnd);
        tryAdj(Adj::Bnd2Face, bnd2face);
        // C2CFace
        tryAdj(Adj::Cell2CellFace, cell2cellFace);

        // --- Register global mappings ---
        // For each entity kind, find the first adj array whose father
        // carries a valid pLGlobalMapping.  All adj arrays for the same
        // entity kind share equivalent offsets (same father Size), so
        // any one suffices.  check_throw if a mapping is needed (i.e.,
        // at least one adj was registered whose .from is this kind) but
        // none is found.

        // Candidate sources per entity kind (ordered by typical availability).
        // All adj arrays for the same entity kind have equivalent offsets
        // (same father Size), so any one suffices.
        auto firstValid = [](std::initializer_list<ssp<GlobalOffsetsMapping>> candidates)
            -> ssp<GlobalOffsetsMapping>
        {
            for (auto &gm : candidates)
                if (gm)
                    return gm;
            return nullptr;
        };

        auto getGM = [](const auto &pair) -> ssp<GlobalOffsetsMapping>
        {
            if (pair.father && pair.father->pLGlobalMapping)
                return pair.father->pLGlobalMapping;
            return nullptr;
        };

        auto cellMap = firstValid({getGM(cell2node), getGM(cell2cell), getGM(cell2face)});
        auto nodeMap = coords.father ? coords.father->pLGlobalMapping : nullptr;
        auto bndMap = firstValid({getGM(bnd2node), getGM(bnd2cell), getGM(bnd2face)});
        auto faceMap = firstValid({getGM(face2node), getGM(face2cell), getGM(face2bnd)});

        // Register if found; check_throw if any registered adj needs it
        auto regMap = [&](EntityKind kind, const ssp<GlobalOffsetsMapping> &gm)
        {
            if (gm)
            {
                dag.registerGlobalMapping(kind, gm);
            }
            else
            {
                // Verify no registered adj has this kind as its from-entity
                for (auto &[adjKind, _] : dag.adjRegistry)
                    DNDS_check_throw_info(
                        adjKind.from != kind,
                        fmt::format("fillRegistry: no pLGlobalMapping found for EntityKind {} "
                                    "but adj {} requires it",
                                    static_cast<int>(kind), adjKindName(adjKind)));
            }
        };

        regMap(EntityKind::Cell, cellMap);
        regMap(EntityKind::Node, nodeMap);
        regMap(EntityKind::Bnd, bndMap);
        regMap(EntityKind::Face, faceMap);
    }

    // =================================================================
    // File-local helpers for ReorderLocalCells
    // =================================================================
    // Cell permutation helper: extracted to shared header.
    // See Mesh_CellPermutation.hpp for CellPermutationResult + ComputeCellPermutation.
} // namespace DNDS::Geom
#include "Mesh_CellPermutation.hpp"
namespace DNDS::Geom
{

    void UnstructuredMesh::ReorderLocalCellsLegacy(int nParts, int nPartsInner)
    {
        DNDS_assert(this->adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(cell2node.isLocal() && bnd2node.isLocal() && cell2cell.isLocal() && bnd2cell.isLocal());
        DNDS_check_throw(int64_t(nParts) * int64_t(nPartsInner) < std::numeric_limits<int>::max());
        nParts = std::max(nParts, 1);
        nPartsInner = std::max(nPartsInner, 1);

        // Section A: Compute cell permutation (Metis partition + contiguous sort)
        auto cell2cellFaceV = this->GetCell2CellFaceVLocal();
        auto perm = detail::ComputeCellPermutation(
            cell2cellFaceV, cell2cell, NumCell(), nParts, nPartsInner);
        this->localPartitionStarts = std::move(perm.localPartitionStarts);

        MPI::AllreduceOneIndex(perm.bwOld, MPI_MAX, mpi);
        MPI::AllreduceOneIndex(perm.bwNew, MPI_MAX, mpi);
        if (mpi.rank == mRank)
            log()
                << fmt::format("UnstructuredMesh === ReorderLocalCells, nPart0 [{}], got reordering, bw [{}] to [{}]", nParts, perm.bwOld, perm.bwNew) << std::endl;

        // Section B: Build cellOld2NewArr for MPI communication of new cell indices
        tAdj1Pair cellOld2NewArr;
        cellOld2NewArr.InitPair("cellOld2NewArr", mpi);
        cellOld2NewArr.father->Resize(this->NumCell());
        for (index iCell = 0; iCell < this->NumCell(); iCell++)
            (*cellOld2NewArr.father)[iCell][0] = this->CellIndexLocal2Global_NoSon(perm.cellOld2New.at(iCell)); // this is right but bad syntax
        cellOld2NewArr.TransAttach();

        std::set<index> cellsNonLocalFull;

        auto record_iCellNonLocal = [&](index iCell)
        {
            if (iCell >= 0 && iCell < cell2node.father->pLGlobalMapping->globalSize())
            {
                if (std::get<1>(cell2node.father->pLGlobalMapping->search(iCell)) != mpi.rank)
                    cellsNonLocalFull.insert(iCell);
            }
            else
                DNDS_assert(iCell == UnInitIndex);
        };

        if (this->adjFacialState != Adj_Unknown && this->face2cell.isBuilt())
        {
            DNDS_assert(this->adjFacialState == Adj_PointToLocal);
            DNDS_assert(face2cell.isLocal() && face2node.isLocal());
            this->AdjLocal2GlobalFacial();
            // see face2cell
            for (index iF = 0; iF < this->NumFace(); iF++)
                for (rowsize if2c = 0; if2c < 2; if2c++)
                    record_iCellNonLocal(face2cell(iF, if2c));
        }
        if (this->adjC2FState != Adj_Unknown && this->cell2face.isBuilt())
        {
            DNDS_assert(this->adjC2FState == Adj_PointToLocal);
            DNDS_assert(cell2face.isLocal() && bnd2face.isLocal());
            this->AdjLocal2GlobalC2F();
        }
        if (this->adjN2CBState != Adj_Unknown && this->node2cell.isBuilt())
        {
            DNDS_assert(this->adjN2CBState == Adj_PointToLocal);
            DNDS_assert(node2cell.isLocal() && node2bnd.isLocal());
            this->AdjLocal2GlobalN2CB();
            // see node2cell
            for (index iN = 0; iN < this->NumNode(); iN++)
                for (rowsize in2c = 0; in2c < node2cell[iN].size(); in2c++)
                    record_iCellNonLocal(node2cell(iN, in2c));
        }
        this->AdjLocal2GlobalPrimary();
        // see cell2cell
        for (index iC = 0; iC < this->NumCell(); iC++)
            for (rowsize ic2c = 0; ic2c < cell2cell[iC].size(); ic2c++)
                record_iCellNonLocal(cell2cell(iC, ic2c));
        // see bnd2cell
        for (index iB = 0; iB < this->NumBnd(); iB++)
            for (rowsize ib2c = 0; ib2c < 2; ib2c++)
                record_iCellNonLocal(bnd2cell(iB, ib2c));

        for (auto iCellGhost : cell2node.trans.pLGhostMapping->ghostIndex)
            cellsNonLocalFull.insert(iCellGhost); // ! should have no effect normally

        cellOld2NewArr.trans.createFatherGlobalMapping();
        cellOld2NewArr.trans.createGhostMapping(std::vector<index>(cellsNonLocalFull.begin(), cellsNonLocalFull.end()));
        cellOld2NewArr.trans.createMPITypes();
        cellOld2NewArr.trans.pullOnce();

        // Section C: Replace cell indices in RHS of xxx2cell arrays
        auto replaceCellIndexToNew = [&](index iCellOld) -> index
        {
            if (iCellOld == UnInitIndex)
                return UnInitIndex;
            auto [ret, rank, val] = cellOld2NewArr.trans.pLGhostMapping->search_indexAppend(iCellOld);
            DNDS_assert(ret);
            return cellOld2NewArr(val, 0);
        };

        if (this->adjFacialState != Adj_Unknown && this->face2cell.isBuilt())
            for (index iFace = 0; iFace < this->NumFace(); iFace++)
                for (index &iCell : face2cell[iFace])
                    iCell = replaceCellIndexToNew(iCell);
        if (this->adjN2CBState != Adj_Unknown && this->node2cell.isBuilt())
            for (index iNode = 0; iNode < this->NumNode(); iNode++)
                for (index &iCell : node2cell[iNode])
                    iCell = replaceCellIndexToNew(iCell);
        for (index iCell = 0; iCell < this->NumCell(); iCell++)
            for (index &iCellOther : cell2cell[iCell])
                iCellOther = replaceCellIndexToNew(iCellOther);
        for (index iBnd = 0; iBnd < this->NumBnd(); iBnd++)
            for (index &iCell : bnd2cell[iBnd])
                iCell = replaceCellIndexToNew(iCell);

        // Section D: Pull ghost data for xxx2cell
        if (this->adjFacialState != Adj_Unknown && this->face2cell.isBuilt())
            face2cell.trans.pullOnce();
        if (this->adjN2CBState != Adj_Unknown && this->node2cell.isBuilt())
            node2cell.trans.pullOnce();
        cell2cell.trans.pullOnce(); // should be unnecessary
        bnd2cell.trans.pullOnce();

        // Section E: Permute LHS of cell2xxx arrays
        auto cellOld2NewLocal = [&](index iCell) -> index
        { return this->CellIndexGlobal2Local_NoSon(cellOld2NewArr(iCell, 0)); };

        PermuteRows(cell2node, this->NumCell(), cellOld2NewLocal);
        PermuteRows(cell2cell, this->NumCell(), cellOld2NewLocal);
        PermuteRows(cell2cellOrig, this->NumCell(), cellOld2NewLocal);
        if (this->adjC2FState != Adj_Unknown && this->cell2face.isBuilt())
            PermuteRows(cell2face, this->NumCell(), cellOld2NewLocal);
        if (this->isPeriodic)
            PermuteRows(cell2nodePbi, this->NumCell(), cellOld2NewLocal);
        PermuteRows(cellElemInfo, this->NumCell(), cellOld2NewLocal);

        // Section F: Rebuild ghost mappings with new cell indices
        { // cell2node
            std::vector<index> ghostCellGlobalsNew = cell2node.trans.pLGhostMapping->ghostIndex;
            for (index &iCell : ghostCellGlobalsNew)
                iCell = replaceCellIndexToNew(iCell);
            cell2node.trans.createGhostMapping(ghostCellGlobalsNew);
            cell2node.trans.createMPITypes();
            cell2node.trans.pullOnce();
        }
        { // cell2cell
            cell2cell.trans.BorrowGGIndexing(cell2node.trans);
            cell2cell.trans.createMPITypes();
            cell2cell.trans.pullOnce();
        }
        { // cell2cellOrig
            cell2cellOrig.trans.BorrowGGIndexing(cell2node.trans);
            cell2cellOrig.trans.createMPITypes();
            cell2cellOrig.trans.pullOnce();
        }
        if (this->adjC2FState != Adj_Unknown && this->cell2face.isBuilt())
        { // cell2face
            cell2face.trans.BorrowGGIndexing(cell2node.trans);
            cell2face.trans.createMPITypes();
            cell2face.trans.pullOnce();
        }
        if (this->isPeriodic)
        { // cell2nodePbi
            cell2nodePbi.trans.BorrowGGIndexing(cell2node.trans);
            cell2nodePbi.trans.createMPITypes();
            cell2nodePbi.trans.pullOnce();
        }
        { // cellElemInfo
            cellElemInfo.trans.BorrowGGIndexing(cell2node.trans);
            cellElemInfo.trans.createMPITypes();
            cellElemInfo.trans.pullOnce();
        }

        // Re-wire per-adjacency target mappings: ghost mappings were rebuilt
        // above with new cell indices, so the previously captured pointers are
        // stale.  All xxx2cell adjacencies use cellGhostMap; xxx2bnd use
        // bndGhostMap; xxx2node use nodeGhostMap.
        {
            auto cellGhostMap = cellElemInfo.trans.pLGhostMapping;
            auto bndGhostMap = bndElemInfo.trans.pLGhostMapping;

            cell2cell.idx.wireTargetMapping(cellGhostMap);
            bnd2cell.idx.wireTargetMapping(cellGhostMap);
            node2cell.idx.wireTargetMapping(cellGhostMap);
            face2cell.idx.wireTargetMapping(cellGhostMap);
            node2bnd.idx.wireTargetMapping(bndGhostMap);
        }

        // Section G: Restore all adjacencies to local indices
        if (this->adjFacialState != Adj_Unknown && this->face2cell.isBuilt())
        {
            this->AdjGlobal2LocalFacial();
        }
        if (this->adjC2FState != Adj_Unknown && this->cell2face.isBuilt())
        {
            this->AdjGlobal2LocalC2F();
        }
        if (this->adjN2CBState != Adj_Unknown && this->node2cell.isBuilt())
        {
            this->AdjGlobal2LocalN2CB();
        }
        this->AdjGlobal2LocalPrimary();
        if (mpi.rank == mRank)
            log() << fmt::format("UnstructuredMesh === ReorderLocalCells finished") << std::endl;
    }
}