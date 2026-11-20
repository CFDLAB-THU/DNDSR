#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Transition/SerialAdjReordering.hpp"
#include "Solver/Direct.hpp"
namespace DNDS::Geom
{
    std::size_t UnstructuredMesh::CoordinateHash()
    {
        auto read = AcquireRead();
        const auto view = read.HostView();
        auto rowsHash = [&](index first, index count)
        {
            std::size_t values = 0;
            for (index row = first; row < first + count; ++row)
                for (int component = 0; component < 3; ++component)
                    values ^= std::hash<real>{}(view.Coord(row)[component]);
            return array_hash<std::size_t, 3>{}({std::size_t(count), 0, values});
        };
        // Exact ArrayPair< ArrayEigenVector<3> > diagnostic contract: include
        // owned and appended ghost rows, then gather in communicator rank order.
        index local = array_hash<std::size_t, 2>{}({rowsHash(0, view.NumNode()),
                                                    rowsHash(view.NumNode(), view.NumNodeGhost())});
        std::vector<index> hashes(size_t(mpi.size));
        MPI::Allgather(&local, 1, DNDS_MPI_INDEX, hashes.data(), 1, DNDS_MPI_INDEX, mpi.comm);
        return vector_hash<index>{}(hashes);
    }
    tLocalMatStruct UnstructuredMesh::GetCell2CellFaceVLocal(bool local)
    {
        auto read = AcquireRead(MeshReadRequirements::CellFlux());
        return CurrentNativeState()->BuildLocalCellNeighbors(local);
    }
    /// Keep worker blocks contiguous in *factor* order even when their physical
    /// cell numbers occupy two intervals. Preserve the selected order inside each block.
    static void GroupFactorRows(const UnstructuredMesh &mesh, std::vector<index> &newToOld, std::vector<index> &oldToNew)
    {
        const index count = mesh.NumCell();
        std::vector<int> owner(count);
        std::vector<index> next(mesh.NLocalParts());
        index offset = 0;
        for (int part = 0; part < mesh.NLocalParts(); ++part)
        {
            const auto worker = mesh.LocalPartRanges(part);
            next[part] = offset;
            offset += worker.Size();
            for (auto range : {worker.first, worker.second})
                for (index row = range.begin; row < range.end; ++row)
                    owner[row] = part;
        }
        std::vector<index> ordered(count);
        oldToNew.resize(count);
        for (index pos = 0; pos < count; ++pos)
        {
            const auto row = newToOld.empty() ? pos : newToOld[pos];
            const auto target = next[owner.at(row)]++;
            ordered[target] = row;
            oldToNew[row] = target;
        }
        newToOld = std::move(ordered);
    }
    void UnstructuredMesh::ObtainLocalFactFillOrdering(Direct::SerialSymLUStructure &symLU, Direct::DirectPrecControl control)
    {
        if (!control.useDirectPrec)
            return;
        _nativeState = _nativeState->WithLocalCellNeighbors(GetCell2CellFaceVLocal(true));
        _nativeSmoothing.reset();

        const auto &graph = LocalCellNeighbors();
        auto &localFillOrderingNew2Old = symLU.localFillOrderingNew2Old;
        auto &localFillOrderingOld2New = symLU.localFillOrderingOld2New;
        if (!this->NumCell())
        {
            // Nonempty peers report bandwidth through the same two reductions
            // below. Empty owners must remain in that collective schedule.
            if (control.getOrderingCode() == 3 || control.getOrderingCode() == 4)
            {
                index oldBandwidth = 0, newBandwidth = 0;
                MPI::AllreduceOneIndex(oldBandwidth, MPI_MAX, mpi);
                MPI::AllreduceOneIndex(newBandwidth, MPI_MAX, mpi);
            }
            return;
        }
        if (control.getOrderingCode() == -1)
        {
            localFillOrderingNew2Old.reserve(this->NumCell());
            for (index i = 0; i < this->NumCell(); i++)
                if (i % 2 == 0)
                    localFillOrderingNew2Old.push_back(i);
            for (index i = 0; i < this->NumCell(); i++)
                if (i % 2 != 0)
                    localFillOrderingNew2Old.push_back(i);
            localFillOrderingOld2New.resize(this->NumCell());
            for (index i = 0; i < this->NumCell(); i++)
                localFillOrderingOld2New.at(localFillOrderingNew2Old.at(i)) = i;
        }
        else if (control.getOrderingCode() == 0)
        {
            // do nothing, natural order
        }
        else if (control.getOrderingCode() == 1) // Metis
        {

            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling metis" << std::endl;
            {
                auto [New2Old, Old2New] = ReorderSerialAdj_Metis(graph);
                localFillOrderingNew2Old = std::move(New2Old);
                localFillOrderingOld2New = std::move(Old2New);
            }
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): metis done" << std::endl;
        }
        else if (control.getOrderingCode() == 2) // MMD
        {
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling boost::minimum_degree_ordering" << std::endl;
            {
                auto [New2Old, Old2New] = ReorderSerialAdj_BoostMMD(graph);
                localFillOrderingNew2Old = std::move(New2Old);
                localFillOrderingOld2New = std::move(Old2New);
            }
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): boost done" << std::endl;
        }
        else if (control.getOrderingCode() == 3) // RCM
        {
            index bandWidthOld{0}, bandWidthNew{0};
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling boost::cuthill_mckee_ordering" << std::endl;
            {
                auto [New2Old, Old2New] = ReorderSerialAdj_BoostRCM(graph, bandWidthOld, bandWidthNew);
                localFillOrderingNew2Old = std::move(New2Old);
                localFillOrderingOld2New = std::move(Old2New);
            }
            MPI::AllreduceOneIndex(bandWidthOld, MPI_MAX, this->mpi);
            MPI::AllreduceOneIndex(bandWidthNew, MPI_MAX, this->mpi);
            if (mpi.rank == mRank)
                log() << fmt::format("UnstructuredMesh::ObtainLocalFactFillOrdering(): boost done, old BW [{}] new BW [{}] ", bandWidthOld, bandWidthNew) << std::endl;
            // for (auto v : localFillOrderingOld2New)
            //     std::cout << v << ", ";
            // std::cout << std::endl;
        }
        else if (control.getOrderingCode() == 4) // CorrectRCM
        {
            index bandWidthOld{0}, bandWidthNew{0};
            if (mpi.rank == mRank)
                log() << "UnstructuredMesh::ObtainLocalFactFillOrdering(): start calling CorrectRCM::CuthillMcKeeOrdering" << std::endl;
            {
                auto [New2Old, Old2New] = ReorderSerialAdj_CorrectRCM(
                    graph.begin(), graph.end(), bandWidthOld, bandWidthNew);
                localFillOrderingNew2Old = std::move(New2Old);
                localFillOrderingOld2New = std::move(Old2New);
            }
            MPI::AllreduceOneIndex(bandWidthOld, MPI_MAX, this->mpi);
            MPI::AllreduceOneIndex(bandWidthNew, MPI_MAX, this->mpi);
            if (mpi.rank == mRank)
                log() << fmt::format("UnstructuredMesh::ObtainLocalFactFillOrdering(): CorrectRCM done, old BW [{}] new BW [{}] ", bandWidthOld, bandWidthNew) << std::endl;
            // for (auto v : localFillOrderingOld2New)
            //     std::cout << v << ", ";
            // std::cout << std::endl;
        }
        else
        {
            DNDS_assert_info(false, "No such ordering code");
        }
        if (NLocalParts() > 1)
        {
            bool disjoint = false;
            for (int part = 0; part < NLocalParts(); ++part)
                disjoint |= LocalPartRanges(part).second.Size() != 0;
            if (control.getOrderingCode() != 0 || disjoint)
                GroupFactorRows(*this, localFillOrderingNew2Old, localFillOrderingOld2New);
        }
    }
}
