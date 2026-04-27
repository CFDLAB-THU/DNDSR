#include "Mesh.hpp"
#include "MeshConnectivity.hpp"
#include "Mesh_PartitionHelpers.hpp"
#include "Geom/Metis.hpp"

#include <unordered_set>

namespace DNDS::Geom
{
    /**
     * Helper: read a single ArrayPair's father using EvenSplit offset.
     * Creates father+son, reads father from H5 with even-split.
     * The son is allocated but empty (size 0).
     */
    template <class TArray>
    static void EvenSplitReadFather(
        ssp<TArray> &father, ssp<TArray> &son,
        const MPIInfo &mpi,
        Serializer::SerializerBaseSSP serializerP,
        const std::string &name)
    {
        father = std::make_shared<TArray>(mpi);
        son = std::make_shared<TArray>(mpi);
        auto offset = Serializer::ArrayGlobalOffset_EvenSplit;
        father->ReadSerializer(serializerP, name, offset);
    }

    /// Overload for arrays that need CommType/CommMult (ElemInfo, NodePeriodicBits).
    template <class TArray>
    static void EvenSplitReadFather(
        ssp<TArray> &father, ssp<TArray> &son,
        const MPIInfo &mpi,
        Serializer::SerializerBaseSSP serializerP,
        const std::string &name,
        MPI_Datatype commType, int commMult)
    {
        father = std::make_shared<TArray>(commType, commMult, mpi);
        son = std::make_shared<TArray>(commType, commMult, mpi);
        auto offset = Serializer::ArrayGlobalOffset_EvenSplit;
        father->ReadSerializer(serializerP, name, offset);
    }

    // =================================================================
    // Top-level orchestrator
    // =================================================================

    void UnstructuredMesh::
        ReadSerializeAndDistribute(
            Serializer::SerializerBaseSSP serializerP,
            const std::string &name,
            const PartitionOptions &partitionOptions)
    {
        DNDS_check_throw_info(!serializerP->IsPerRank(),
                              "ReadSerializeAndDistribute requires a collective (H5) serializer");

        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: begin" << std::endl;

        auto cwd = serializerP->GetCurrentPath();
        serializerP->GoToPath(name);

        // Step 0+1: read metadata and all primary arrays with even-split
        ReadDistributed_EvenSplitRead(serializerP);
        serializerP->GoToPath(cwd);

        // Step 2+3: build adjacencies and filter to facial cell2cell
        auto cell2cellFacial = ReadDistributed_BuildFacialCell2Cell();

        // Step 4: ParMetis repartition
        auto cellPartition = ReadDistributed_PartitionParMetis(
            cell2cellFacial, partitionOptions);
        cell2cellFacial.reset(); // free before redistribution

        // Step 5: derive entity partitions and redistribute
        auto partitions = ReadDistributed_DeriveEntityPartitions(std::move(cellPartition));
        ReadDistributed_Redistribute(partitions);

        // Step 6: log final stats
        coords.father->createGlobalMapping();
        cell2node.father->createGlobalMapping();
        bnd2node.father->createGlobalMapping();

        {
            index nCellG = this->NumCellGlobal();
            index nNodeG = this->NumNodeGlobal();
            index nBndG = this->NumBndGlobal();
            if (mpi.rank == 0)
            {
                log() << "UnstructuredMesh === ReadSerializeAndDistribute done, "
                      << fmt::format("nCellGlobal [{}], nNodeGlobal [{}], nBndGlobal [{}]",
                                     nCellG, nNodeG, nBndG)
                      << std::endl;
            }
            MPISerialDo(mpi, [&]()
                        { log() << fmt::format("    Rank {}: nCell {}, nNode {}, nBnd {}",
                                               mpi.rank, this->NumCell(), this->NumNode(), this->NumBnd())
                                << std::endl; });
        }
    }

    // =================================================================
    // Step 0+1: Even-split read
    // =================================================================

    void UnstructuredMesh::
        ReadDistributed_EvenSplitRead(
            Serializer::SerializerBaseSSP serializerP)
    {
        // Read scalar metadata
        {
            std::string meshRead;
            index dimRead{0}, sizeRead{0};
            int isPeriodicRead;
            serializerP->ReadString("mesh", meshRead);
            serializerP->ReadIndex("dim", dimRead);
            serializerP->ReadIndex("MPISize", sizeRead);
            serializerP->ReadInt("isPeriodic", isPeriodicRead);
            isPeriodic = bool(isPeriodicRead);
            DNDS_assert(meshRead == "UnstructuredMesh");
            DNDS_assert(dimRead == dim);
        }

        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: even-split read" << std::endl;

        auto innerCwd = serializerP->GetCurrentPath();

        auto readFatherAt = [&](auto &father, auto &son, const std::string &group)
        {
            serializerP->GoToPath(innerCwd);
            serializerP->GoToPath(group);
            EvenSplitReadFather(father, son, mpi, serializerP, "father");
        };

        auto readFatherAtTyped = [&](auto &father, auto &son, const std::string &group,
                                     MPI_Datatype commType, int commMult)
        {
            serializerP->GoToPath(innerCwd);
            serializerP->GoToPath(group);
            EvenSplitReadFather(father, son, mpi, serializerP, "father", commType, commMult);
        };

        readFatherAt(coords.father, coords.son, "coords");
        readFatherAt(cell2node.father, cell2node.son, "cell2node");
        readFatherAtTyped(cellElemInfo.father, cellElemInfo.son, "cellElemInfo",
                          ElemInfo::CommType(), ElemInfo::CommMult());
        readFatherAt(bnd2node.father, bnd2node.son, "bnd2node");
        readFatherAtTyped(bndElemInfo.father, bndElemInfo.son, "bndElemInfo",
                          ElemInfo::CommType(), ElemInfo::CommMult());

        if (isPeriodic)
        {
            readFatherAtTyped(cell2nodePbi.father, cell2nodePbi.son, "cell2nodePbi",
                              NodePeriodicBits::CommType(), NodePeriodicBits::CommMult());
            readFatherAtTyped(bnd2nodePbi.father, bnd2nodePbi.son, "bnd2nodePbi",
                              NodePeriodicBits::CommType(), NodePeriodicBits::CommMult());
            serializerP->GoToPath(innerCwd);
            periodicInfo.ReadSerializer(serializerP, "periodicInfo");
        }

        readFatherAt(cell2cellOrig.father, cell2cellOrig.son, "cell2cellOrig");
        readFatherAt(node2nodeOrig.father, node2nodeOrig.son, "node2nodeOrig");
        readFatherAt(bnd2bndOrig.father, bnd2bndOrig.son, "bnd2bndOrig");

        adjPrimaryState = Adj_PointToGlobal;
        cell2node.idx.markGlobal();
        bnd2node.idx.markGlobal();

        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: even-split read done, "
                  << "nCellLocal=" << cell2node.father->Size()
                  << " nNodeLocal=" << coords.father->Size()
                  << " nBndLocal=" << bnd2node.father->Size() << std::endl;
    }

    // =================================================================
    // Step 2+3: Build facial cell2cell
    // =================================================================

    ssp<tAdj::element_type> UnstructuredMesh::
        ReadDistributed_BuildFacialCell2Cell()
    {
        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: building distributed cell2cell" << std::endl;

        // Step 2: build node-neighbor cell2cell and bnd2cell via DSL.
        RecoverNode2CellAndNode2Bnd();
        RecoverCell2CellAndBnd2Cell();

        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: building facial cell2cell" << std::endl;

        // Ghost-pull cell2node and cellElemInfo for all neighbor cells.
        {
            cell2cell.TransAttach();
            cell2cell.trans.createFatherGlobalMapping();

            MeshConnectivity dagTmp;
            fillRegistry(dagTmp);

            GhostSpec cellSpec{{{EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell}}};
            auto cellResult = dagTmp.evaluateGhostTree(
                CompiledGhostTree::compile(cellSpec), mpi);

            auto it = cellResult.ghostIndices.find(EntityKind::Cell);
            std::vector<index> ghostCells;
            if (it != cellResult.ghostIndices.end())
                ghostCells = std::move(it->second);

            cell2node.TransAttach();
            cell2node.trans.createFatherGlobalMapping();
            cellElemInfo.TransAttach();
            cellElemInfo.trans.createFatherGlobalMapping();

            cell2cell.trans.createGhostMapping(ghostCells);
            cell2node.trans.BorrowGGIndexing(cell2cell.trans);
            cellElemInfo.trans.BorrowGGIndexing(cell2cell.trans);

            cell2cell.trans.createMPITypes();
            cell2node.trans.createMPITypes();
            cellElemInfo.trans.createMPITypes();

            cell2cell.trans.pullOnce();
            cell2node.trans.pullOnce();
            cellElemInfo.trans.pullOnce();
        }

        // Step 3: filter cell2cell to face-sharing neighbors (O1 vertex intersection >= dim).
        auto cell2cellFacial = make_ssp<tAdj::element_type>(ObjName{"cell2cellFacialTmp"}, mpi);
        cell2cellFacial->Resize(cell2cell.father->Size(), 6);

        for (index iCell = 0; iCell < cell2cell.father->Size(); iCell++)
        {
            auto elemI = Elem::Element{cellElemInfo(iCell, 0).getElemType()};
            rowsize nVertI = elemI.GetNumVertices();
            std::vector<index> vertI(cell2node[iCell].begin(),
                                     cell2node[iCell].begin() + nVertI);
            std::sort(vertI.begin(), vertI.end());

            std::vector<index> facialNeighbors;
            facialNeighbors.reserve(6);
            for (rowsize ic2c = 0; ic2c < cell2cell.father->RowSize(iCell); ic2c++)
            {
                index iCellOther = (*cell2cell.father)(iCell, ic2c);

                auto [found, rank, localAppend] =
                    cell2cell.trans.pLGhostMapping->search_indexAppend(iCellOther);
                DNDS_assert_info(found, "cell2cell neighbor not found in ghost mapping");

                auto elemJ = Elem::Element{cellElemInfo(localAppend, 0).getElemType()};
                rowsize nVertJ = elemJ.GetNumVertices();
                std::vector<index> vertJ(cell2node[localAppend].begin(),
                                         cell2node[localAppend].begin() + nVertJ);
                std::sort(vertJ.begin(), vertJ.end());

                std::vector<index> intersect;
                intersect.reserve(9);
                std::set_intersection(vertI.begin(), vertI.end(),
                                      vertJ.begin(), vertJ.end(),
                                      std::back_inserter(intersect));
                if (static_cast<int>(intersect.size()) >= dim)
                    facialNeighbors.push_back(iCellOther);
            }
            cell2cellFacial->ResizeRow(iCell, facialNeighbors.size());
            for (rowsize j = 0; j < static_cast<rowsize>(facialNeighbors.size()); j++)
                (*cell2cellFacial)(iCell, j) = facialNeighbors[j];
        }
        cell2cellFacial->Compress();

        return cell2cellFacial;
    }

    // =================================================================
    // Step 4: ParMetis repartition
    // =================================================================

    std::vector<MPI_int> UnstructuredMesh::
        ReadDistributed_PartitionParMetis(
            const ssp<tAdj::element_type> &cell2cellFacial,
            const PartitionOptions &partitionOptions)
    {
        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: ParMetis repartition" << std::endl;

        cell2cellFacial->createGlobalMapping();
        idx_t nPart = mpi.size;

        std::vector<idx_t> vtxdist(mpi.size + 1);
        for (MPI_int r = 0; r <= mpi.size; r++)
            vtxdist[r] = _METIS::indexToIdx(cell2cellFacial->pLGlobalMapping->ROffsets().at(r));

        std::vector<idx_t> xadj(cell2cellFacial->Size() + 1);
        for (index i = 0; i <= cell2cellFacial->Size(); i++)
            xadj[i] = _METIS::indexToIdx(cell2cellFacial->rowPtr(i) - cell2cellFacial->rowPtr(0));

        std::vector<idx_t> adjncy(xadj.back());
        for (index i = 0; i < xadj.back(); i++)
            adjncy[i] = _METIS::indexToIdx(cell2cellFacial->data()[i]);

        if (adjncy.empty())
            adjncy.resize(1, -1); // cope with zero-sized data

        std::vector<MPI_int> cellPartition(cell2cellFacial->Size());

        if (nPart > 1)
        {
            for (int i = 0; i < mpi.size; i++)
                DNDS_assert_info(vtxdist[i + 1] - vtxdist[i] > 0,
                                 "ParMetis requires > 0 cells on each proc");

            idx_t nCon{1};
            idx_t wgtflag{0}, numflag{0};
            std::vector<real_t> tpWeights(static_cast<size_t>(nPart) * nCon, 1.0 / nPart);
            real_t ubVec[1]{1.05};
            idx_t optsC[3]{1, 0, static_cast<idx_t>(partitionOptions.metisSeed)};
            idx_t objval;
            std::vector<idx_t> partOut(cell2cellFacial->Size());
            if (partOut.empty())
                partOut.resize(1, 0);

            int ret = ParMETIS_V3_PartKway(
                vtxdist.data(), xadj.data(), adjncy.data(),
                NULL, NULL, &wgtflag, &numflag,
                &nCon, &nPart, tpWeights.data(), ubVec, optsC,
                &objval, partOut.data(), &mpi.comm);
            DNDS_assert_info(ret == METIS_OK,
                             fmt::format("ParMETIS_V3_PartKway returned {}", ret));

            for (index i = 0; i < cell2cellFacial->Size(); i++)
                cellPartition[i] = static_cast<MPI_int>(partOut[i]);
        }
        else
        {
            std::fill(cellPartition.begin(), cellPartition.end(), 0);
        }

        // Print partition stats
        {
            std::vector<index> localPartCnt(nPart, 0);
            for (auto p : cellPartition)
                localPartCnt.at(p)++;
            std::vector<index> globalPartCnt(nPart, 0);
            MPI::Allreduce(localPartCnt.data(), globalPartCnt.data(), nPart, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            if (mpi.rank == 0)
            {
                auto [minIt, maxIt] = std::minmax_element(globalPartCnt.begin(), globalPartCnt.end());
                index nCellGlobal = 0;
                for (auto c : globalPartCnt)
                    nCellGlobal += c;
                log() << "UnstructuredMesh === ReadSerializeAndDistribute: partition done, "
                      << fmt::format("nCellGlobal [{}], ave [{}], min [{}], max [{}], ratio [{:.4f}]",
                                     nCellGlobal, real(nCellGlobal) / nPart, *minIt, *maxIt, real(*minIt) / *maxIt)
                      << std::endl;
            }
        }

        return cellPartition;
    }

    // =================================================================
    // Step 5a: Derive node and bnd partitions from cell partition
    // =================================================================

    UnstructuredMesh::EntityPartitions UnstructuredMesh::
        ReadDistributed_DeriveEntityPartitions(
            std::vector<MPI_int> cellPartition)
    {
        EntityPartitions result;
        result.cellPartition = std::move(cellPartition);

        // Node partition: each node goes to the min cell partition that claims it.
        result.nodePartition.assign(coords.father->Size(), static_cast<MPI_int>(INT32_MAX));
        for (index iCell = 0; iCell < cell2node.father->Size(); iCell++)
        {
            for (rowsize ic2n = 0; ic2n < cell2node.father->RowSize(iCell); ic2n++)
            {
                index iNode = (*cell2node.father)(iCell, ic2n);
                MPI_int rank;
                index val;
                bool found = coords.father->pLGlobalMapping->search(iNode, rank, val);
                DNDS_assert(found);
                if (rank == mpi.rank)
                    result.nodePartition[val] = std::min(result.nodePartition[val],
                                                         result.cellPartition.at(iCell));
            }
        }
        for (auto &p : result.nodePartition)
            if (p == static_cast<MPI_int>(INT32_MAX))
                p = 0;

        // Bnd partition: bnd goes to same rank as its owner cell (bnd2cell(iBnd, 0)).
        // Build a distributed lookup for the owner cell's target partition.
        auto cellPartArr = make_ssp<tAdj1::element_type>(ObjName{"cellPartArr"}, mpi);
        auto cellPartArrGhost = make_ssp<tAdj1::element_type>(ObjName{"cellPartArrGhost"}, mpi);
        cellPartArr->Resize(cell2node.father->Size());
        for (index i = 0; i < cellPartArr->Size(); i++)
            (*cellPartArr)(i, 0) = static_cast<index>(result.cellPartition.at(i));
        cellPartArr->createGlobalMapping();

        std::vector<index> bndOwnerGhostQuery;
        for (index iBnd = 0; iBnd < bnd2node.father->Size(); iBnd++)
        {
            index iOwnerCell = bnd2cell(iBnd, 0);
            MPI_int ownerRank;
            index ownerVal;
            bool found = cellPartArr->pLGlobalMapping->search(iOwnerCell, ownerRank, ownerVal);
            DNDS_assert(found);
            if (ownerRank != mpi.rank)
                bndOwnerGhostQuery.push_back(iOwnerCell);
        }
        ArrayTransformerType<tAdj1::element_type>::Type cellPartTrans;
        cellPartTrans.setFatherSon(cellPartArr, cellPartArrGhost);
        cellPartTrans.createGhostMapping(bndOwnerGhostQuery);
        cellPartTrans.createMPITypes();
        cellPartTrans.pullOnce();

        result.bndPartition.resize(bnd2node.father->Size());
        for (index iBnd = 0; iBnd < bnd2node.father->Size(); iBnd++)
        {
            index iOwnerCell = bnd2cell(iBnd, 0);
            DNDS_assert_info(iOwnerCell != UnInitIndex,
                             fmt::format("bnd {} has no owner cell after RecoverCell2CellAndBnd2Cell", iBnd));
            MPI_int ownerRank;
            index ownerVal;
            bool foundOwner = cellPartArr->pLGlobalMapping->search(iOwnerCell, ownerRank, ownerVal);
            DNDS_assert(foundOwner);
            index targetPart;
            if (ownerRank == mpi.rank)
                targetPart = (*cellPartArr)(ownerVal, 0);
            else
            {
                auto [found, gRank, gVal] = cellPartTrans.pLGhostMapping->search_indexAppend(iOwnerCell);
                DNDS_assert(found);
                if (gRank == -1)
                    targetPart = (*cellPartArr)(gVal, 0);
                else
                    targetPart = (*cellPartArrGhost)(gVal - cellPartArr->Size(), 0);
            }
            result.bndPartition[iBnd] = static_cast<MPI_int>(targetPart);
        }

        return result;
    }

    // =================================================================
    // Step 5b-5d: Redistribute arrays to new partition
    // =================================================================

    void UnstructuredMesh::
        ReadDistributed_Redistribute(
            const EntityPartitions &partitions)
    {
        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: redistributing" << std::endl;

        // Free temporary adjacencies no longer needed.
        cell2cell.father.reset();
        cell2cell.son.reset();
        node2cell.father.reset();
        node2cell.son.reset();
        node2bnd.father.reset();
        node2bnd.son.reset();
        bnd2cell.father.reset();
        bnd2cell.son.reset();

        // Compute push indices and serial-to-global reordering.
        std::vector<index> cell_push, cell_pushStart;
        std::vector<index> node_push, node_pushStart;
        std::vector<index> bnd_push, bnd_pushStart;
        Partition2LocalIdx(partitions.cellPartition, cell_push, cell_pushStart, mpi);
        Partition2LocalIdx(partitions.nodePartition, node_push, node_pushStart, mpi);
        Partition2LocalIdx(partitions.bndPartition, bnd_push, bnd_pushStart, mpi);

        std::vector<index> cell_Serial2Global, node_Serial2Global;
        Partition2Serial2Global(partitions.cellPartition, cell_Serial2Global, mpi, mpi.size);
        Partition2Serial2Global(partitions.nodePartition, node_Serial2Global, mpi, mpi.size);

        // Convert adjacency indices to new global numbering.
        ConvertAdjSerial2Global(cell2node.father, node_Serial2Global, mpi);
        ConvertAdjSerial2Global(bnd2node.father, node_Serial2Global, mpi);

        // Transfer all arrays to new partition.
        auto coordsOld = coords.father;
        auto cell2nodeOld = cell2node.father;
        auto cellElemInfoOld = cellElemInfo.father;
        auto bnd2nodeOld = bnd2node.father;
        auto bndElemInfoOld = bndElemInfo.father;
        auto cell2cellOrigOld = cell2cellOrig.father;
        auto node2nodeOrigOld = node2nodeOrig.father;
        auto bnd2bndOrigOld = bnd2bndOrig.father;

        coords.InitPair("coords", mpi);
        cell2node.InitPair("cell2node", mpi);
        cellElemInfo.InitPair("cellElemInfo", ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        bnd2node.InitPair("bnd2node", mpi);
        bndElemInfo.InitPair("bndElemInfo", ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        cell2cellOrig.InitPair("cell2cellOrig", mpi);
        node2nodeOrig.InitPair("node2nodeOrig", mpi);
        bnd2bndOrig.InitPair("bnd2bndOrig", mpi);

        TransferDataSerial2Global(coordsOld, coords.father, node_push, node_pushStart, mpi);
        TransferDataSerial2Global(node2nodeOrigOld, node2nodeOrig.father, node_push, node_pushStart, mpi);

        TransferDataSerial2Global(cell2nodeOld, cell2node.father, cell_push, cell_pushStart, mpi);
        TransferDataSerial2Global(cell2cellOrigOld, cell2cellOrig.father, cell_push, cell_pushStart, mpi);
        TransferDataSerial2Global(cellElemInfoOld, cellElemInfo.father, cell_push, cell_pushStart, mpi);

        TransferDataSerial2Global(bnd2nodeOld, bnd2node.father, bnd_push, bnd_pushStart, mpi);
        TransferDataSerial2Global(bndElemInfoOld, bndElemInfo.father, bnd_push, bnd_pushStart, mpi);
        TransferDataSerial2Global(bnd2bndOrigOld, bnd2bndOrig.father, bnd_push, bnd_pushStart, mpi);

        if (isPeriodic)
        {
            auto cell2nodePbiOld = cell2nodePbi.father;
            auto bnd2nodePbiOld = bnd2nodePbi.father;
            cell2nodePbi.InitPair("cell2nodePbi", NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
            bnd2nodePbi.InitPair("bnd2nodePbi", NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
            TransferDataSerial2Global(cell2nodePbiOld, cell2nodePbi.father, cell_push, cell_pushStart, mpi);
            TransferDataSerial2Global(bnd2nodePbiOld, bnd2nodePbi.father, bnd_push, bnd_pushStart, mpi);
        }

        adjPrimaryState = Adj_PointToGlobal;
        cell2node.idx.markGlobal();
        bnd2node.idx.markGlobal();
    }

} // namespace DNDS::Geom
