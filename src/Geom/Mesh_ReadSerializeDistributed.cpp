#include "Mesh.hpp"
#include "Mesh_PartitionHelpers.hpp"
#include "Metis.hpp"

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

        /************************************************************/
        // Step 0: read scalar metadata
        /************************************************************/
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

        /************************************************************/
        // Step 1: even-split read of all primary arrays
        /************************************************************/
        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: even-split read" << std::endl;

        // Navigate into "coords/father" etc. -- ArrayPair::ReadSerialize
        // expects to navigate into name/father. We replicate the read
        // logic but with EvenSplit offset.
        // NOTE: GoToPath("..") doesn't work in H5 serializer, so we
        // save/restore absolute paths using GetCurrentPath().
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

        serializerP->GoToPath(cwd);

        // At this point we have evenly-split data. Node indices in cell2node/bnd2node
        // point to the ORIGINAL global numbering (as written). The coords are evenly split
        // by the ORIGINAL node numbering. So cell2node already points to valid global node
        // indices, and coords.father[i] corresponds to global node index
        // (coordsGlobalStart + i). This is exactly the state adjPrimaryState == Adj_PointToGlobal.
        adjPrimaryState = Adj_PointToGlobal;

        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: even-split read done, "
                  << "nCellLocal=" << cell2node.father->Size()
                  << " nNodeLocal=" << coords.father->Size()
                  << " nBndLocal=" << bnd2node.father->Size() << std::endl;

        /************************************************************/
        // Step 2: build distributed cell2cell (node-neighbor)
        // Reuse existing methods that operate on distributed data
        // with adjPrimaryState == Adj_PointToGlobal.
        /************************************************************/
        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: building distributed cell2cell" << std::endl;

        RecoverNode2CellAndNode2Bnd();
        RecoverCell2CellAndBnd2Cell();

        // cell2cell is now the node-neighbor adjacency (global indices).
        // bnd2cell is also built.

        /************************************************************/
        // Step 3: filter cell2cell to facial neighbors only
        // (shared O1 vertices >= dim)
        // Need ghost cell2node and cellElemInfo to check intersection.
        /************************************************************/
        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: building facial cell2cell" << std::endl;

        // Temporarily pull ghost cell2node and cellElemInfo for the
        // cells referenced in cell2cell.
        {
            cell2cell.TransAttach();
            cell2cell.trans.createFatherGlobalMapping();

            std::vector<index> ghostCells;
            for (index iCell = 0; iCell < cell2cell.father->Size(); iCell++)
                for (rowsize ic2c = 0; ic2c < cell2cell.father->RowSize(iCell); ic2c++)
                {
                    index iCellOther = (*cell2cell.father)(iCell, ic2c);
                    MPI_int rank;
                    index val;
                    if (!cell2cell.trans.pLGlobalMapping->search(iCellOther, rank, val))
                        DNDS_assert_info(false, "search failed");
                    if (rank != mpi.rank)
                        ghostCells.push_back(iCellOther);
                }

            cell2node.TransAttach();
            cell2node.trans.createFatherGlobalMapping();
            cellElemInfo.TransAttach();
            cellElemInfo.trans.createFatherGlobalMapping();

            // Build temporary ghost mapping on cell2cell, borrow for cell2node/cellElemInfo
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

        // Now build the facial subset.
        // cell2cell.father + son are populated. For each local cell's
        // neighbor, get its vertex set via cell2node (father or son) and
        // check intersection size.
        tAdj cell2cellFacialTmp;
        DNDS_MAKE_SSP(cell2cellFacialTmp, mpi);
        cell2cellFacialTmp->Resize(cell2cell.father->Size(), 6);

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

                // Find iCellOther in the father+son appended index
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
            cell2cellFacialTmp->ResizeRow(iCell, facialNeighbors.size());
            for (rowsize j = 0; j < static_cast<rowsize>(facialNeighbors.size()); j++)
                (*cell2cellFacialTmp)(iCell, j) = facialNeighbors[j];
        }
        cell2cellFacialTmp->Compress();

        /************************************************************/
        // Step 4: ParMetis repartition
        /************************************************************/
        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: ParMetis repartition" << std::endl;

        cell2cellFacialTmp->createGlobalMapping();
        idx_t nPart = mpi.size;

        std::vector<idx_t> vtxdist(mpi.size + 1);
        for (MPI_int r = 0; r <= mpi.size; r++)
            vtxdist[r] = _METIS::indexToIdx(cell2cellFacialTmp->pLGlobalMapping->ROffsets().at(r));

        std::vector<idx_t> xadj(cell2cellFacialTmp->Size() + 1);
        for (index i = 0; i <= cell2cellFacialTmp->Size(); i++)
            xadj[i] = _METIS::indexToIdx(cell2cellFacialTmp->rowPtr(i) - cell2cellFacialTmp->rowPtr(0));

        std::vector<idx_t> adjncy(xadj.back());
        for (index i = 0; i < xadj.back(); i++)
            adjncy[i] = _METIS::indexToIdx(cell2cellFacialTmp->data()[i]);

        if (adjncy.empty())
            adjncy.resize(1, -1); // cope with zero-sized data

        std::vector<MPI_int> cellPartition(cell2cellFacialTmp->Size());

        if (nPart > 1)
        {
            // Check that every rank has > 0 cells for ParMetis
            for (int i = 0; i < mpi.size; i++)
                DNDS_assert_info(vtxdist[i + 1] - vtxdist[i] > 0,
                                 "ParMetis requires > 0 cells on each proc");

            idx_t nCon{1};
            idx_t wgtflag{0}, numflag{0};
            std::vector<real_t> tpWeights(static_cast<size_t>(nPart) * nCon, 1.0 / nPart);
            real_t ubVec[1]{1.05};
            idx_t optsC[3]{1, 0, static_cast<idx_t>(partitionOptions.metisSeed)};
            idx_t objval;
            std::vector<idx_t> partOut(cell2cellFacialTmp->Size());
            if (partOut.empty())
                partOut.resize(1, 0);

            int ret = ParMETIS_V3_PartKway(
                vtxdist.data(), xadj.data(), adjncy.data(),
                NULL, NULL, &wgtflag, &numflag,
                &nCon, &nPart, tpWeights.data(), ubVec, optsC,
                &objval, partOut.data(), &mpi.comm);
            DNDS_assert_info(ret == METIS_OK,
                             fmt::format("ParMETIS_V3_PartKway returned {}", ret));

            for (index i = 0; i < cell2cellFacialTmp->Size(); i++)
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

        /************************************************************/
        // Step 5: redistribute cells, nodes, bnds to new partition
        // using ArrayTransformer push-based transfer.
        // Reuse the helper templates from Mesh_PartitionHelpers.hpp.
        /************************************************************/
        if (mpi.rank == 0)
            log() << "UnstructuredMesh === ReadSerializeAndDistribute: redistributing" << std::endl;

        // 5a: derive node and bnd partitions
        // Node partition: each node goes to the first (min) cell partition that claims it.
        std::vector<MPI_int> nodePartition(coords.father->Size(), static_cast<MPI_int>(INT32_MAX));
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
                    nodePartition[val] = std::min(nodePartition[val], cellPartition.at(iCell));
            }
        }
        // Nodes not claimed by any local cell: assign to this rank
        // (rare with even-split; correct partition will be set by caller's rebuild).
        for (auto &p : nodePartition)
            if (p == static_cast<MPI_int>(INT32_MAX))
                p = 0;

        // Bnd partition: bnd goes to the same rank as its owner cell.
        // Bnd partition:
        // bnd2cell was built by RecoverCell2CellAndBnd2Cell in Step 2.
        // bnd2cell(iBnd, 0) is the global index of the owner cell, which
        // may be on another rank in the current even-split. Build a distributed
        // lookup: for each bnd's owner cell, find its target partition.
        //
        // Strategy: create a tAdj1 array holding cellPartition as global data,
        // pull ghosts for the owner cells needed by our bnds, then look up.
        tAdj1 cellPartArr, cellPartArrGhost;
        DNDS_MAKE_SSP(cellPartArr, mpi);
        DNDS_MAKE_SSP(cellPartArrGhost, mpi);
        cellPartArr->Resize(cell2node.father->Size());
        for (index i = 0; i < cellPartArr->Size(); i++)
            (*cellPartArr)(i, 0) = static_cast<index>(cellPartition.at(i));
        cellPartArr->createGlobalMapping();

        // Gather ghost cell indices needed by our bnds' owner cells
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

        std::vector<MPI_int> bndPartition(bnd2node.father->Size());
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
            bndPartition[iBnd] = static_cast<MPI_int>(targetPart);
        }

        // Now free temporary adjacencies no longer needed.
        cell2cell.father.reset();
        cell2cell.son.reset();
        cell2cellFacialTmp.reset();
        node2cell.father.reset();
        node2cell.son.reset();
        node2bnd.father.reset();
        node2bnd.son.reset();
        bnd2cell.father.reset();
        bnd2cell.son.reset();

        // 5b: compute push indices and serial-to-global reordering
        std::vector<index> cell_push, cell_pushStart;
        std::vector<index> node_push, node_pushStart;
        std::vector<index> bnd_push, bnd_pushStart;
        Partition2LocalIdx(cellPartition, cell_push, cell_pushStart, mpi);
        Partition2LocalIdx(nodePartition, node_push, node_pushStart, mpi);
        Partition2LocalIdx(bndPartition, bnd_push, bnd_pushStart, mpi);

        std::vector<index> cell_Serial2Global, node_Serial2Global;
        Partition2Serial2Global(cellPartition, cell_Serial2Global, mpi, mpi.size);
        Partition2Serial2Global(nodePartition, node_Serial2Global, mpi, mpi.size);

        // 5c: convert adjacency indices to new global numbering
        ConvertAdjSerial2Global(cell2node.father, node_Serial2Global, mpi);
        ConvertAdjSerial2Global(bnd2node.father, node_Serial2Global, mpi);

        // 5d: transfer all arrays to new partition
        // Save old fathers, create new ones, push/pull.
        auto coordsOld = coords.father;
        auto cell2nodeOld = cell2node.father;
        auto cellElemInfoOld = cellElemInfo.father;
        auto bnd2nodeOld = bnd2node.father;
        // bnd2cell is not serialized; rebuilt by caller's RecoverCell2CellAndBnd2Cell
        auto bndElemInfoOld = bndElemInfo.father;
        auto cell2cellOrigOld = cell2cellOrig.father;
        auto node2nodeOrigOld = node2nodeOrig.father;
        auto bnd2bndOrigOld = bnd2bndOrig.father;

        DNDS_MAKE_SSP(coords.father, mpi);
        DNDS_MAKE_SSP(coords.son, mpi);
        DNDS_MAKE_SSP(cell2node.father, mpi);
        DNDS_MAKE_SSP(cell2node.son, mpi);
        DNDS_MAKE_SSP(cellElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        DNDS_MAKE_SSP(cellElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        DNDS_MAKE_SSP(bnd2node.father, mpi);
        DNDS_MAKE_SSP(bnd2node.son, mpi);
        // bnd2cell is not serialized; rebuilt by caller's RecoverCell2CellAndBnd2Cell
        DNDS_MAKE_SSP(bndElemInfo.father, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        DNDS_MAKE_SSP(bndElemInfo.son, ElemInfo::CommType(), ElemInfo::CommMult(), mpi);
        DNDS_MAKE_SSP(cell2cellOrig.father, mpi);
        DNDS_MAKE_SSP(cell2cellOrig.son, mpi);
        DNDS_MAKE_SSP(node2nodeOrig.father, mpi);
        DNDS_MAKE_SSP(node2nodeOrig.son, mpi);
        DNDS_MAKE_SSP(bnd2bndOrig.father, mpi);
        DNDS_MAKE_SSP(bnd2bndOrig.son, mpi);

        TransferDataSerial2Global(coordsOld, coords.father, node_push, node_pushStart, mpi);
        TransferDataSerial2Global(node2nodeOrigOld, node2nodeOrig.father, node_push, node_pushStart, mpi);

        TransferDataSerial2Global(cell2nodeOld, cell2node.father, cell_push, cell_pushStart, mpi);
        TransferDataSerial2Global(cell2cellOrigOld, cell2cellOrig.father, cell_push, cell_pushStart, mpi);
        TransferDataSerial2Global(cellElemInfoOld, cellElemInfo.father, cell_push, cell_pushStart, mpi);

        TransferDataSerial2Global(bnd2nodeOld, bnd2node.father, bnd_push, bnd_pushStart, mpi);
        TransferDataSerial2Global(bndElemInfoOld, bndElemInfo.father, bnd_push, bnd_pushStart, mpi);
        TransferDataSerial2Global(bnd2bndOrigOld, bnd2bndOrig.father, bnd_push, bnd_pushStart, mpi);

        // periodic arrays
        if (isPeriodic)
        {
            auto cell2nodePbiOld = cell2nodePbi.father;
            auto bnd2nodePbiOld = bnd2nodePbi.father;
            DNDS_MAKE_SSP(cell2nodePbi.father, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
            DNDS_MAKE_SSP(cell2nodePbi.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
            DNDS_MAKE_SSP(bnd2nodePbi.father, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
            DNDS_MAKE_SSP(bnd2nodePbi.son, NodePeriodicBits::CommType(), NodePeriodicBits::CommMult(), mpi);
            TransferDataSerial2Global(cell2nodePbiOld, cell2nodePbi.father, cell_push, cell_pushStart, mpi);
            TransferDataSerial2Global(bnd2nodePbiOld, bnd2nodePbi.father, bnd_push, bnd_pushStart, mpi);
        }

        adjPrimaryState = Adj_PointToGlobal;

        /************************************************************/
        // Step 6: log final stats and return
        /************************************************************/
        // createGlobalMapping so NumXxxGlobal() works for logging
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

} // namespace DNDS::Geom
