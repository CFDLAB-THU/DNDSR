#include "DNDS/MPI/Failure.hpp"
#include "Geom/Transition/NativeMeshTransition.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "Geom/Transition/CellPermutation.hpp"
#include "Geom/Transition/LocalCellOrdering.hpp"
#include "Geom/Unstructured/ReaderPartitionGraph.hpp"
#include "Geom/Transition/NativeMeshPartitionGraph.hpp"
#include "Geom/Transition/MeshAdjacencyBindings.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <set>
#include <unordered_set>
#include <tuple>

namespace DNDS::Geom
{
    namespace
    {
        using RArray = Runtime::Array<index>;
        using MeshTransitionDetail::Bindings;
        using MeshTransitionDetail::FacadeKey;
        RArray BuildIndexRows(const std::vector<std::vector<index>> &rows)
        {
            return Runtime::BuildArrayRows<index>(rows.size(), [&](size_t row)
                                                  { return rows[row].size(); }, [&](size_t row, auto output)
                                                  { std::copy(rows[row].begin(), rows[row].end(), output.begin()); });
        }
        template <class T, class MapValue>
        Runtime::Array<T> ReorderOwnedRows(const Runtime::Array<T> &source, const std::vector<index> &oldToNew,
                                           bool reorderRows, bool remapValues, bool sortValues, MapValue mapValue)
        {
            DNDS_check_throw_info(!reorderRows || oldToNew.size() == source.RowCount(),
                                  "owned reorder permutation extent differs from array rows");
            // Non-cell descriptors retain their rows and values. Published
            // storage is immutable, so aliases can share the source handle
            // instead of cloning every relation unrelated to the local order.
            if (!reorderRows && !remapValues && !sortValues)
                return source;
            auto layout = source.Layout();
            if (reorderRows && layout.IsVariable())
            {
                // The caller supplies a validated permutation. When every row
                // keeps the destination's existing length, the complete shape
                // is unchanged (including uniform CSR/padded rows). Share its
                // immutable metadata; values are still moved independently.
                bool sameShape = true;
                for (index iRow = 0; sameShape && iRow < layout.RowCount(); ++iRow)
                    sameShape = layout.RowSize(iRow) == layout.RowSize(oldToNew[size_t(iRow)]);
                if (!sameShape)
                {
                    std::vector<size_t> sizes(layout.RowCount());
                    for (index iRow = 0; iRow < layout.RowCount(); ++iRow)
                        sizes[size_t(oldToNew[size_t(iRow)])] = size_t(layout.RowSize(iRow));
                    layout = layout.IsPadded() ? Runtime::RowLayout::Padded(sizes, layout.Stride())
                                               : Runtime::RowLayout::Ragged(sizes);
                }
            }
            Runtime::Array<T> result(layout);
            auto input = source.Read();
            auto output = result.Write();
            const auto inputView = input.View();
            const auto outputView = output.View();
            if (!remapValues && !sortValues)
            {
                int invalid = 0;
#pragma omp parallel for if (inputView.RowCount() >= 4096) schedule(static) reduction(| : invalid)
                for (index oldRow = 0; oldRow < inputView.RowCount(); ++oldRow)
                {
                    const size_t newRow = reorderRows ? size_t(oldToNew[oldRow]) : oldRow;
                    const auto count = inputView.RowSize(oldRow);
                    if (newRow >= outputView.RowCount() || count != outputView.RowSize(newRow))
                    {
                        invalid = 1;
                        continue;
                    }
                    std::copy_n(inputView.Row(oldRow), count, outputView.Row(newRow));
                }
                DNDS_check_throw_info(!invalid, "owned reorder changed a row layout");
                return result;
            }
            int invalid = 0;
#pragma omp parallel for if (inputView.RowCount() >= 4096) schedule(static) reduction(| : invalid)
            for (index oldRow = 0; oldRow < inputView.RowCount(); ++oldRow)
            {
                const size_t newRow = reorderRows ? size_t(oldToNew[oldRow]) : oldRow;
                // The read lease pins row metadata. Keep its bound outside the
                // element loop; output writes may otherwise inhibit compiler
                // hoisting of the generic CSR/padded row-size loads.
                const auto count = inputView.RowSize(oldRow);
                if (newRow >= outputView.RowCount() || count != outputView.RowSize(newRow))
                {
                    invalid = 1;
                    continue;
                }
                auto *target = outputView.Row(newRow);
                const auto *sourceRow = inputView.Row(oldRow);
                try
                {
                    for (index slot = 0; slot < count; ++slot)
                        target[slot] = remapValues ? mapValue(sourceRow[slot]) : sourceRow[slot];
                    if constexpr (std::is_same_v<T, index>)
                        if (sortValues)
                            std::sort(target, target + count);
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("native mesh transition preparation failed");
                    invalid = 1;
                }
            }
            DNDS_check_throw_info(!invalid, "owned reorder changed a row layout");
            return result;
        }
        PermutationTransfer LocalPermutationTransfer(const std::vector<index> &old2new, index globalCount,
                                                     const MPIInfo &mpi)
        {
            std::vector<index> counts(size_t(mpi.size));
            const index owned = index(old2new.size());
            MPI_Allgather(&owned, 1, DNDS_MPI_INDEX, counts.data(), 1, DNDS_MPI_INDEX, mpi.comm);
            PermutationTransfer result;
            result.isLocalOnly = true;
            result.localOld2New = old2new;
            result.targetRanks.assign(size_t(owned), MPI_int(mpi.rank));
            result.newGlobalOffsets.resize(size_t(mpi.size + 1));
            result.newGlobalOffsets[0] = 0;
            for (int rank = 0; rank < mpi.size; ++rank)
                result.newGlobalOffsets[size_t(rank + 1)] = result.newGlobalOffsets[size_t(rank)] + counts[size_t(rank)];
            DNDS_check_throw_info(result.newGlobalOffsets.back() == globalCount,
                                  "local permutation transfer global extent is inconsistent");
            const index first = result.newGlobalOffsets[size_t(mpi.rank)];
            result.newGlobalIndices.resize(size_t(owned));
            for (size_t row = 0; row < size_t(owned); ++row)
                result.newGlobalIndices[row] = first + old2new[row];
            result.pushStart.assign(size_t(mpi.size + 1), 0);
            result.pushStart[size_t(mpi.rank + 1)] = owned;
            for (int rank = mpi.rank + 2; rank <= mpi.size; ++rank)
                result.pushStart[size_t(rank)] = owned;
            result.pushIndex.resize(size_t(owned));
            std::iota(result.pushIndex.begin(), result.pushIndex.end(), index(0));
            return result;
        }
        /// Pre-face setup ordering needs face neighbors, but enumerating the
        /// whole sub-entity topology is unnecessary. Match the legacy node-set
        /// filter directly over the already-published vertex-sharing Cell2Cell
        /// rows: shared vertices >= dimension means a topological face.
        tLocalMatStruct PrimaryFaceNeighborsFromCellGraph(const NativeOwnedMeshData &source)
        {
            const auto cells = source.spaces.at(EntityKind::Cell);
            DNDS_check_throw_info(cells.NumGhost() == 0 && source.relations[size_t(MeshRelation::CellNodes)] &&
                                      source.relations[size_t(MeshRelation::CellCells)] && source.cellInfo,
                                  "primary local ordering requires owned cell/node/neighbor rows");
            const auto &cellNodes = *source.relations[size_t(MeshRelation::CellNodes)];
            const auto &cellCells = *source.relations[size_t(MeshRelation::CellCells)];
            auto nodes = cellNodes.Read();
            auto neighbors = cellCells.Read();
            auto metadata = source.cellInfo->Read();
            std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> nodeBits;
            if (source.periodic)
            {
                DNDS_check_throw_info(source.cellNodesPbi, "periodic primary local ordering requires cell-node frame bits");
                nodeBits.emplace(source.cellNodesPbi->Read());
            }
            tLocalMatStruct graph(size_t(cells.NumOwned()));
            // Hex/tet-boundary graphs have at most six facial neighbors in 3D
            // and four in 2D. Reserving once avoids repeated small-row heap
            // growth while CompactAdjacencyRow still supplies exact storage.
            for (auto &row : graph)
                row.reserve(6);
            constexpr size_t maxVertices = size_t(Elem::CellNumNodeMax);
            const size_t cellCount = size_t(cells.NumOwned());
            DNDS_check_throw_info(cellCount == 0 || maxVertices <= SIZE_MAX / cellCount,
                                  "primary local ordering vertex scratch exceeds address space");
            std::vector<index> vertices(cellCount * maxVertices);
            std::vector<uint8_t> frames;
            if (nodeBits)
                frames.resize(cellCount * maxVertices);
            std::vector<uint8_t> vertexCounts(cellCount);
            for (size_t cell = 0; cell < cellCount; ++cell)
            {
                const auto element = Elem::Element{metadata.View()(cell, 0).getElemType()};
                const int nVertices = element.GetNumVertices();
                DNDS_check_throw_info(nVertices > 0 && size_t(nVertices) <= maxVertices &&
                                          nodes.View().RowSize(cell) == size_t(element.GetNumNodes()),
                                      "primary local ordering cell has an invalid element cone");
                vertexCounts[cell] = uint8_t(nVertices);
                auto *sorted = vertices.data() + cell * maxVertices;
                std::copy_n(nodes.View().Row(cell), nVertices, sorted);
                if (nodeBits)
                {
                    auto *frame = frames.data() + cell * maxVertices;
                    for (int vertex = 0; vertex < nVertices; ++vertex)
                        frame[vertex] = uint8_t(nodeBits->View()(cell, size_t(vertex)));
                    std::array<size_t, maxVertices> order{};
                    for (int vertex = 0; vertex < nVertices; ++vertex)
                        order[size_t(vertex)] = size_t(vertex);
                    std::sort(order.begin(), order.begin() + nVertices, [&](size_t left, size_t right)
                              { return std::pair(sorted[left], frame[left]) < std::pair(sorted[right], frame[right]); });
                    std::array<index, maxVertices> sortedCopy{};
                    std::array<uint8_t, maxVertices> frameCopy{};
                    for (int slot = 0; slot < nVertices; ++slot)
                    {
                        sortedCopy[size_t(slot)] = sorted[order[size_t(slot)]];
                        frameCopy[size_t(slot)] = frame[order[size_t(slot)]];
                    }
                    std::copy_n(sortedCopy.begin(), nVertices, sorted);
                    std::copy_n(frameCopy.begin(), nVertices, frame);
                }
                else
                    std::sort(sorted, sorted + nVertices);
            }
            for (index cell = 0; cell < cells.NumOwned(); ++cell)
            {
                const auto *current = vertices.data() + size_t(cell) * maxVertices;
                const size_t nVertices = vertexCounts[size_t(cell)];
                for (auto neighborGlobal : neighbors.View()[size_t(cell)])
                {
                    if (neighborGlobal == UnInitIndex)
                        continue;
                    const auto neighbor = cells.GlobalToLocal(neighborGlobal);
                    if (!neighbor || *neighbor >= cells.NumOwned() || *neighbor == cell)
                        continue;
                    const auto *other = vertices.data() + size_t(*neighbor) * maxVertices;
                    const size_t otherVertices = vertexCounts[size_t(*neighbor)];
                    bool facial = false;
                    if (!nodeBits)
                    {
                        size_t a = 0, b = 0, shared = 0;
                        while (a < nVertices && b < otherVertices)
                        {
                            if (current[a] < other[b])
                                ++a;
                            else if (other[b] < current[a])
                                ++b;
                            else
                            {
                                ++shared;
                                ++a;
                                ++b;
                            }
                        }
                        facial = shared >= size_t(source.dimension);
                    }
                    else
                    {
                        size_t rawA = 0, rawB = 0, rawShared = 0;
                        while (rawA < nVertices && rawB < otherVertices)
                        {
                            if (current[rawA] < other[rawB])
                                ++rawA;
                            else if (other[rawB] < current[rawA])
                                ++rawB;
                            else
                            {
                                ++rawShared;
                                ++rawA;
                                ++rawB;
                            }
                        }
                        if (rawShared < size_t(source.dimension))
                            continue;
                        const auto *currentFrames = frames.data() + size_t(cell) * maxVertices;
                        const auto *otherFrames = frames.data() + size_t(*neighbor) * maxVertices;
                        uint8_t tried = 0;
                        for (size_t currentVertex = 0; currentVertex < nVertices && !facial; ++currentVertex)
                        {
                            for (size_t otherVertex = 0; otherVertex < otherVertices; ++otherVertex)
                            {
                                if (current[currentVertex] != other[otherVertex])
                                    continue;
                                const uint8_t mask = currentFrames[currentVertex] ^ otherFrames[otherVertex];
                                if (tried & (uint8_t(1) << mask))
                                    continue;
                                tried |= uint8_t(1) << mask;
                                size_t a = 0, b = 0, shared = 0;
                                while (a < nVertices && b < otherVertices)
                                {
                                    const auto currentKey = std::pair(current[a], currentFrames[a]);
                                    const auto otherKey = std::pair(other[b], uint8_t(otherFrames[b] ^ mask));
                                    if (currentKey < otherKey)
                                        ++a;
                                    else if (otherKey < currentKey)
                                        ++b;
                                    else
                                    {
                                        ++shared;
                                        ++a;
                                        ++b;
                                    }
                                }
                                if (shared >= size_t(source.dimension))
                                {
                                    facial = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (facial)
                        graph[size_t(cell)].push_back(*neighbor);
                }
                detail::CompactAdjacencyRow(graph[size_t(cell)]);
            }
            return graph;
        }
        /// Move row payloads. allowLocalPermutation must be rank-uniform and is
        /// true only for an explicitly requested full local permutation. The
        /// local branch preserves independent output storage while avoiding
        /// self-MPI and payload-plan construction.
        template <class T>
        Runtime::Array<T> Move(const Runtime::Array<T> &source, const Runtime::RowTransferPlan &route,
                               const Runtime::ElementSchema &schema, const MPIInfo &mpi,
                               bool allowLocalPermutation, std::vector<Runtime::TransferOperation> *pending = nullptr)
        {
            const auto &send = route.SendRows(), &receive = route.ReceiveRows();
            bool localOnly = allowLocalPermutation && source.GetDevice() == Runtime::Device::Host() && !route.HasCollisions() &&
                             send.size() == size_t(mpi.size) && receive.size() == size_t(mpi.size);
            for (int peer = 0; localOnly && peer < mpi.size; ++peer)
                if (peer != mpi.rank && (!send[size_t(peer)].empty() || !receive[size_t(peer)].empty()))
                    localOnly = false;
            if (localOnly)
            {
                const auto &sendRows = send[size_t(mpi.rank)], &receiveRows = receive[size_t(mpi.rank)];
                localOnly = sendRows.size() == source.RowCount() &&
                            receiveRows.size() == source.RowCount() && route.DestinationRows() == source.RowCount();
                if (localOnly)
                {
                    bool identity = true;
                    for (size_t row = 0; row < source.RowCount(); ++row)
                        identity = identity && sendRows[row] == receiveRows[row];
                    if (identity)
                        return source.Clone();

                    std::vector<size_t> sizes(source.RowCount());
                    for (size_t row = 0; row < source.RowCount(); ++row)
                        sizes[receiveRows[row]] = source.RowSize(sendRows[row]);
                    const auto sourceLayout = source.Layout();
                    auto localLayout = !sourceLayout.IsVariable()
                                           ? Runtime::RowLayout::Dense(source.RowCount(), sourceLayout.Width(), sourceLayout.Stride())
                                       : sourceLayout.IsPadded()
                                           ? Runtime::RowLayout::Padded(sizes, sourceLayout.Stride())
                                           : Runtime::RowLayout::Ragged(sizes);
                    Runtime::Array<T> output(localLayout);
                    auto input = source.Read(), target = output.Write();
                    auto inputView = input.View(), outputView = target.View();
                    for (size_t row = 0; row < source.RowCount(); ++row)
                    {
                        const auto from = sendRows[row], to = receiveRows[row];
                        std::copy(inputView.Row(from), inputView.Row(from) + sizes[to], outputView.Row(to));
                    }
                    return output;
                }
            }

            auto payload = Runtime::PayloadPlan(route, source.Layout(), schema);
            Runtime::Array<T> output(payload.DestinationLayout());
            Runtime::TransferOptions options;
            options.mode = Runtime::TransferMode::Transient;
            options.context = route.Context();
            if (pending)
            {
                // The transition visits the same payload sequence on every
                // rank. Distinct tags permit retained direct transfers to
                // overlap the next payload's CPU remap without communicator
                // duplication or a pack buffer. The operation owns all leases.
                DNDS_check_throw_info(pending->size() < size_t(INT_MAX), "too many concurrent transition payloads");
                options.tag = int(pending->size()) + 1;
                pending->push_back(payload.Bind(source, output, Runtime::TransferTransport::Direct, options).Begin());
            }
            else
                payload.Bind(source, output, Runtime::TransferTransport::Direct, options).Begin().Wait();
            return output;
        }
        void FinishMoves(std::vector<Runtime::TransferOperation> &pending)
        {
            for (auto &operation : pending)
                operation.Wait();
            pending.clear(); // Retire binding tags as well as completed leases.
        }
        struct EntityRemap
        {
            index first = 0;
            const std::vector<index> *newGlobals = nullptr;
            std::vector<std::pair<index, index>> remote;

            index operator()(index global) const
            {
                if (global >= first && global - first < index(newGlobals->size()))
                    return (*newGlobals)[size_t(global - first)];
                const auto found = std::lower_bound(remote.begin(), remote.end(), global,
                                                    [](const auto &item, index target)
                                                    { return item.first < target; });
                DNDS_check_throw_info(found != remote.end() && found->first == global,
                                      "native transition remap is incomplete");
                return found->second;
            }
        };
        /// Most aliases remain ordered after numbering changes. Preserve their
        /// shared payload when possible; otherwise sort an independent receiver
        /// copy so topological slot order is never modified through an alias.
        RArray SortOwnedTargets(const RArray &values)
        {
            auto read = values.Read();
            const auto view = read.View();
            bool sorted = true;
            for (index iRow = 0; sorted && iRow < view.RowCount(); ++iRow)
            {
                const auto row = view[iRow];
                sorted = std::is_sorted(row.begin(), row.end());
            }
            if (sorted)
                return values;
            const std::vector<index> noPermutation;
            return ReorderOwnedRows(values, noPermutation, false, false, true, [](index value)
                                    { return value; });
        }
        NativeOwnedMeshTransitionResult ReorderOwnedPrimaryRows(
            const NativeOwnedMeshData &source, const MPIInfo &mpi, const std::vector<index> &cellOld2New,
            index first)
        {
            const auto oldCells = source.spaces.at(EntityKind::Cell);
            DNDS_check_throw_info(oldCells.NumGhost() == 0 && cellOld2New.size() == size_t(oldCells.NumOwned()),
                                  "primary owned reorder requires a complete owned-cell permutation");
            std::vector<index> referenced;
            std::unordered_set<uintptr_t> scannedTargets;
            auto collectTargets = [&](const AdjacencyDescriptor &key, const RArray &rows)
            {
                if (key.to != EntityKind::Cell)
                    return;
                if (!scannedTargets.insert(reinterpret_cast<uintptr_t>(rows.SharedStorage().get())).second)
                    return;
                auto read = rows.Read();
                const auto view = read.View();
                for (index iRow = 0; iRow < view.RowCount(); ++iRow)
                    for (auto global : view[iRow])
                        // Only remote numbering needs a closure transfer.
                        // Keep invalid remote IDs for checked owner lookup.
                        if (global != UnInitIndex && (global < first || global - first >= oldCells.NumOwned()))
                            referenced.push_back(global);
            };
            for (size_t slot = 0; slot < source.relations.size(); ++slot)
                if (source.relations[slot])
                    collectTargets(Bindings()[slot], *source.relations[slot]);
            for (const auto &[key, rows] : source.namedRelations)
                if (!(key == NativeSetupFacialCellNeighborsKey()))
                    collectTargets(key, rows);
            std::sort(referenced.begin(), referenced.end());
            referenced.erase(std::unique(referenced.begin(), referenced.end()), referenced.end());

            std::vector<index> remote;
            std::vector<Runtime::RowRequest> requests;
            for (auto global : referenced)
                if (global < first || global - first >= oldCells.NumOwned())
                {
                    requests.push_back({oldCells.Owner(global), oldCells.OwnerLocalRow(global), index(remote.size())});
                    remote.push_back(global);
                }
            int anyRemote = remote.empty() ? 0 : 1;
            MPI_Allreduce(MPI_IN_PLACE, &anyRemote, 1, MPI_INT, MPI_MAX, mpi.comm);
            RArray newGlobals(Runtime::RowLayout::Dense(size_t(oldCells.NumOwned()), 1));
            std::vector<index> newGlobalValues(size_t(oldCells.NumOwned()));
            {
                auto write = newGlobals.Write();
                const auto view = write.View<Runtime::FixedRows<1, 1>>();
                for (index iRow = 0; iRow < view.RowCount(); ++iRow)
                {
                    DNDS_check_throw_info(cellOld2New[iRow] >= 0 && cellOld2New[iRow] < oldCells.NumOwned(),
                                          "owned cell permutation target is out of range");
                    view(iRow, 0) = newGlobalValues[iRow] = first + cellOld2New[iRow];
                }
            }
            RArray remoteNewGlobals(Runtime::RowLayout::Dense(0, 1));
            if (anyRemote)
            {
                Runtime::RowTransferPlan route(source.communication ? source.communication : std::make_shared<Runtime::CommunicationContext>(mpi.comm), newGlobals.RowCount(), remote.size(), requests);
                remoteNewGlobals = Move(newGlobals, route, Runtime::ElementSchema::Arithmetic<index>(), mpi, false);
                auto read = remoteNewGlobals.Read();
                DNDS_check_throw_info(read.View().RowCount() == remote.size(), "primary owned reorder lost remote cell targets");
            }
            EntityRemap remap;
            remap.first = first;
            remap.newGlobals = &newGlobalValues;
            if (anyRemote)
            {
                auto read = remoteNewGlobals.Read();
                const auto view = read.View();
                remap.remote.reserve(remote.size());
                for (size_t row = 0; row < remote.size(); ++row)
                    remap.remote.emplace_back(remote[row], view(row, 0));
            }
            auto mappedCell = [&](index value)
            { return value == UnInitIndex ? value : remap(value); };

            NativeOwnedMeshTransitionResult result;
            result.sourceSpaces = source.spaces;
            auto owned = source;
            // Fixed and named descriptors commonly alias the same immutable
            // payload. Apply a cell-row permutation once per payload and
            // transformation, then let aliases retain the same result handle.
            using ReorderKey = std::tuple<uintptr_t, bool, bool, bool>;
            std::map<ReorderKey, RArray> reordered;
            auto reorder = [&](const RArray &rows, const AdjacencyDescriptor &key) -> RArray
            {
                const bool sourceCell = key.from == EntityKind::Cell;
                const bool targetCell = key.to == EntityKind::Cell;
                const bool sorted = key.order == RelationOrder::SortedUnique;
                const ReorderKey identity{reinterpret_cast<uintptr_t>(rows.SharedStorage().get()),
                                          sourceCell, targetCell, sorted};
                auto found = reordered.find(identity);
                if (found != reordered.end())
                    return found->second;
                auto moved = ReorderOwnedRows(rows, cellOld2New, sourceCell, targetCell, sorted, mappedCell);
                reordered.emplace(identity, moved);
                return moved;
            };
            auto nodes = source.spaces.at(EntityKind::Node);
            owned.spaces.insert_or_assign(EntityKind::Cell,
                                          nodes.CreateSibling(EntityKind::Cell, oldCells.NumOwned()));
            for (size_t slot = 0; slot < source.relations.size(); ++slot)
                if (source.relations[slot])
                {
                    const auto key = FacadeKey(slot);
                    auto moved = reorder(*source.relations[slot], key);
                    owned.relations[slot] = std::move(moved);
                }
            owned.namedRelations.clear();
            for (const auto &[key, rows] : source.namedRelations)
            {
                if (key == NativeSetupFacialCellNeighborsKey())
                    continue;
                auto moved = reorder(rows, key);
                owned.namedRelations.emplace(key, std::move(moved));
            }
            if (source.cellInfo)
                owned.cellInfo = ReorderOwnedRows(*source.cellInfo, cellOld2New, true, false, false, [](const ElemInfo &value)
                                                  { return value; });
#define DNDS_REORDER_CELL_COMPANION(member) \
    if (source.member)                      \
    owned.member = ReorderOwnedRows(*source.member, cellOld2New, true, false, false, [](const NodePeriodicBits &value) { return value; })
            DNDS_REORDER_CELL_COMPANION(cellNodesPbi);
            DNDS_REORDER_CELL_COMPANION(cellFacesPbi);
            DNDS_REORDER_CELL_COMPANION(cellEdgesPbi);
#undef DNDS_REORDER_CELL_COMPANION
            if (source.provenance)
                for (auto &[kind, ids] : owned.provenance->sourceIds)
                    if (kind == EntityKind::Cell)
                        ids = ReorderOwnedRows(ids, cellOld2New, true, false, false, [](index value)
                                               { return value; });
            owned.ids[EntityKind::Cell] = ReorderOwnedRows(source.ids.at(EntityKind::Cell), cellOld2New, true, false, false,
                                                           [](index value)
                                                           { return value; });

            for (const auto &[kind, old] : source.spaces)
            {
                std::vector<index> permutation(cellOld2New);
                if (kind != EntityKind::Cell)
                {
                    permutation.resize(size_t(old.NumOwned()));
                    std::iota(permutation.begin(), permutation.end(), index(0));
                }
                result.transfers.emplace(kind, LocalPermutationTransfer(permutation, old.NumGlobal(), mpi));
            }
            result.owned = std::move(owned);
            return result;
        }
        void Agree(const MPIInfo &mpi, const std::string &signature, const std::shared_ptr<Runtime::CommunicationContext> &communication)
        {
            int length = int(signature.size());
            MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
            std::string expected = mpi.rank == 0 ? signature : std::string(size_t(length), '\0');
            MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            { DNDS_check_throw_info(signature == expected, "native transition schedules differ"); });
        }
    }

    NativeMeshTransitionResult NativeMeshTransition::Build(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                                           const ReorderInput &input, int ghostLayers)
    {
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !source.PreparationCommunication() || source.PreparationCommunication()->ChecksCollectivePreflight());
        // Reconstruction first removes only the requested native families.
        // Retained relations, IDs and ancestry then use the normal exact move.
        // New target IDs begin at the old family's high-water mark.
        const std::set<EntityKind> requested(input.destroyKinds.begin(), input.destroyKinds.end());
        std::string reconstruction;
        FieldTransferDetail::LocalStage(mpi, source.PreparationCommunication(), [&]
                                        {
            source.Validate();
            DNDS_check_throw_info(ghostLayers >= 1,"native transition requires positive closure");
            for (auto kind : requested)
            {
                DNDS_check_throw_info((kind == EntityKind::Face || kind == EntityKind::Edge) && source.HasEntities(kind),
                                      "only existing native Face or Edge families can be reconstructed");
                reconstruction += std::to_string(int(kind))+";";
            }
            for (const auto &map : input.explicitMaps)
                DNDS_check_throw_info(!input.destroyKinds.count(map.kind),"cannot place a reconstructed family");
            for (const auto &follow : input.follows)
                DNDS_check_throw_info(!input.destroyKinds.count(follow.follower) && !input.destroyKinds.count(follow.leader),
                                      "cannot follow through a reconstructed family"); });
        Agree(mpi, reconstruction, source.PreparationCommunication());
        if (input.destroyKinds.empty())
            return BuildImpl(source, mpi, input, ghostLayers, nullptr);
        auto stripped = NativeMeshAssembler::CaptureOwned(source, mpi);
        const auto &bindings = Bindings();
        std::map<EntityKind, index> starts;
        FieldTransferDetail::LocalStage(mpi, source.PreparationCommunication(), [&]
                                        {
            for (auto kind : requested)
            {
                starts.emplace(kind,source.NextEntityId(kind));
                stripped.spaces.erase(kind); stripped.ids.erase(kind); stripped.identity.nextIds.erase(kind);
                for (size_t slot = 0; slot < stripped.relations.size(); ++slot)
                    if (bindings[slot].from == kind || bindings[slot].to == kind) stripped.relations[slot].reset();
                for (auto item = stripped.namedRelations.begin(); item != stripped.namedRelations.end();)
                {
                    const auto path = item->first.Path();
                    if (std::find(path.begin(),path.end(),kind) != path.end()) item = stripped.namedRelations.erase(item);
                    else ++item;
                }
                if (stripped.provenance)
                {
                    stripped.provenance->sourceIds.erase(kind); stripped.provenance->sourceKinds.erase(kind);
                    if (stripped.provenance->sourceIds.empty()) stripped.provenance.reset();
                }
                if (kind == EntityKind::Face)
                {
                    stripped.faceInfo.reset(); stripped.cellFacesPbi.reset(); stripped.faceNodesPbi.reset();
                    stripped.relations[size_t(MeshRelation::CellFaceCells)].reset();
                    stripped.requirements.faces = stripped.requirements.cellFaceNeighbors = stripped.requirements.allFaceCellsLocal = false;
                }
                else
                {
                    stripped.edgeInfo.reset(); stripped.cellEdgesPbi.reset(); stripped.edgeNodesPbi.reset();
                    stripped.requirements.edges = false;
                }
            } });
        auto retained = input;
        retained.destroyKinds.clear();
        auto moved = BuildOwnedImpl(stripped, mpi, retained, nullptr);
        // Field source associations are the original publication, not the
        // temporary owned-family projection with equivalent numbering.
        FieldTransferDetail::LocalStage(mpi, source.PreparationCommunication(), [&]
                                        {
            moved.sourceSpaces.clear();
            for (const auto &[kind,family] : source._spaces) moved.sourceSpaces.emplace(kind,family.identity); });
        moved.owned = NativeMeshTopology::Build(std::move(moved.owned), mpi, input.destroyKinds.count(EntityKind::Face),
                                                input.destroyKinds.count(EntityKind::Edge), starts);
        NativeMeshTransitionResult result;
        result.transfers = std::move(moved.transfers);
        result.sourceSpaces = std::move(moved.sourceSpaces);
        result.state = AssembleNativeMesh(mpi, std::move(moved.owned), ghostLayers);
        return result;
    }

    NativeMeshTransitionResult NativeMeshTransition::ReorderLocalCells(UnstructuredMesh &source,
                                                                       const LocalCellOrderingSettings &settings, int ghostLayers)
    {
        Runtime::CommunicationDetail::AgreePolicy(source.getMPI().comm, !source.PreparationCommunication() || source.PreparationCommunication()->ChecksCollectivePreflight());
        std::optional<Runtime::VersionTracker::ReadLease> pin;
        FieldTransferDetail::LocalStage(source.getMPI(), source.PreparationCommunication(), [&]
                                        { pin.emplace(source.versions.Read()); });
        const auto ordering = BuildLocalCellOrdering(source, settings);
        ReorderInput input;
        input.localPermutations.emplace(EntityKind::Cell, ordering.oldToNew);
        auto result = Build(*source.CurrentNativeState(), source.getMPI(), input, ghostLayers);
        result.state = result.state->WithLocalPartitions(ordering.subregionStarts);
        return result;
    }

    NativeMeshTransitionResult NativeMeshTransition::ReorderLocalCells(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                                                       int nParts, int nPartsInner, int ghostLayers)
    {
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !source.PreparationCommunication() || source.PreparationCommunication()->ChecksCollectivePreflight());
        nParts = std::max(nParts, 1);
        nPartsInner = std::max(nPartsInner, 1);
        detail::CellPermutationResult permutation;
        FieldTransferDetail::LocalStage(mpi, source.PreparationCommunication(), [&]
                                        {
            source.Validate();
            DNDS_check_throw_info(int64_t(nParts) * nPartsInner < std::numeric_limits<int>::max() && ghostLayers >= 1,
                                  "native local reorder partition count or closure is invalid");
            source.CheckRequirements(source.Requirements().faces ? MeshReadRequirements::CellFlux() : MeshReadRequirements::Primary());
            const auto &view = source.BorrowHostView();
            // Private-cell sorting below intentionally uses full node-neighbor
            // connectivity, including remote vertex-only contacts.
            for (index cell = 0; cell < view.NumCell(); ++cell)
                for (auto other : view.CellCells(cell))
                    DNDS_check_throw_info(other >= 0 && other < view.NumCellProc(),
                                          "native local reorder requires local owned CellCells closure"); });
        Agree(mpi, "local:" + std::to_string(nParts) + ":" + std::to_string(nPartsInner) + ":" + std::to_string(ghostLayers), source.PreparationCommunication());
        FieldTransferDetail::LocalStage(mpi, source.PreparationCommunication(), [&]
                                        {
            auto graph = source.Requirements().faces ? source.BuildLocalCellNeighbors()
                                                     : detail::BuildPrimaryLocalCellNeighbors(source);
            // Source setup historically reorders before face interpolation:
            // its vertex-derived facial graph is sorted and duplicate-free.
            // Face-slot order, including repeated periodic neighbors, must
            // not change the METIS/RCM permutation for that setup path.
            for (auto &row : graph)
            {
                std::sort(row.begin(), row.end());
                row.erase(std::unique(row.begin(), row.end()), row.end());
            }
            struct CellRows
            {
                const UnstructuredMeshView &view;
                auto operator[](index row) const { return view.CellCells(row); }
            } rows{source.BorrowHostView()};
            if (source.NumOwned(EntityKind::Cell) == 0)
                permutation.localPartitionStarts.assign(size_t(nParts) + 1, 0);
            else
                permutation = detail::ComputeCellPermutation(graph, rows, source.NumOwned(EntityKind::Cell), nParts, nPartsInner); });
        auto result = BuildImpl(source, mpi, {}, ghostLayers, &permutation.cellOld2New);
        FieldTransferDetail::LocalStage(mpi, source.PreparationCommunication(), [&]
                                        { result.state = result.state->WithLocalPartitions(std::move(permutation.localPartitionStarts)); });
        return result;
    }

    NativeMeshTransitionResult NativeMeshTransition::BuildImpl(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                                               const ReorderInput &input, int ghostLayers,
                                                               const std::vector<index> *cellOld2New)
    {
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !source.PreparationCommunication() || source.PreparationCommunication()->ChecksCollectivePreflight());
        auto moved = BuildOwnedImpl(NativeMeshAssembler::CaptureOwned(source, mpi), mpi, input, cellOld2New);
        NativeMeshTransitionResult result;
        result.transfers = std::move(moved.transfers);
        // Owned capture has deliberately detached ghost mappings. External
        // fields must still validate against the original published spaces.
        FieldTransferDetail::LocalStage(mpi, source.PreparationCommunication(), [&]
                                        {
            for (const auto &[kind, family] : source._spaces)
                result.sourceSpaces.emplace(kind, family.identity); });
        result.state = AssembleNativeMesh(mpi, std::move(moved.owned), ghostLayers);
        return result;
    }

    NativeOwnedMeshTransitionResult NativeMeshTransition::ReorderOwnedCells(
        const NativeOwnedMeshData &source, const MPIInfo &mpi, int nParts, int nPartsInner)
    {
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !source.communication || source.communication->ChecksCollectivePreflight());
        nParts = std::max(nParts, 1);
        nPartsInner = std::max(nPartsInner, 1);
        Agree(mpi, "owned-local:" + std::to_string(nParts) + ":" + std::to_string(nPartsInner), source.communication);
        source.LocalStage(mpi, [&]
                          { DNDS_check_throw_info(int64_t(nParts) * nPartsInner < std::numeric_limits<int>::max(), "owned local partition count overflows"); });
        auto cachedFacial = source.namedRelations.find(NativeSetupFacialCellNeighborsKey());
        std::optional<detail::NativePartitionGraph> facial;
        if (cachedFacial != source.namedRelations.end())
        {
            facial.emplace();
            auto rows = cachedFacial->second.Read();
            const auto view = rows.View();
            const auto nGlobal = source.spaces.at(EntityKind::Cell).NumGlobal();
            DNDS_check_throw_info(view.RowCount() == source.spaces.at(EntityKind::Cell).NumOwned(),
                                  "cached facial graph row count differs from owned cells");
            facial->offsets.resize(view.RowCount() + 1, 0);
            for (index row = 0; row < view.RowCount(); ++row)
            {
                const auto count = view.RowSize(row);
                DNDS_check_throw_info(facial->offsets[row] <= std::numeric_limits<idx_t>::max() - idx_t(count),
                                      "cached facial graph adjacency exceeds METIS extent");
                facial->offsets[row + 1] = facial->offsets[row] + METIS::indexToIdx(count);
            }
            facial->adjacency.resize(size_t(facial->offsets.back()));
            int invalid = 0;
#pragma omp parallel for schedule(static) reduction(| : invalid)
            for (index row = 0; row < view.RowCount(); ++row)
            {
                const auto sourceRow = view[row];
                auto *target = facial->adjacency.data() + facial->offsets[size_t(row)];
                // The private descriptor is SortedUnique. Relation validation
                // has already proved strictly increasing rows under its lease;
                // copying or re-sorting them would repeat that proof per row.
                for (size_t slot = 0; slot < sourceRow.size(); ++slot)
                {
                    const auto global = sourceRow[slot];
                    if (global < 0 || global >= nGlobal)
                    {
                        invalid = 1;
                        break;
                    }
                    target[slot] = METIS::indexToIdx(global);
                }
            }
            DNDS_check_throw_info(!invalid, "cached facial graph contains an invalid cell");
        }
        else
            facial.emplace(detail::BuildOwnedPartitionGraph(source, mpi));
        detail::CellPermutationResult permutation;
        source.LocalStage(mpi, [&]
                          {
            const auto cells = source.spaces.at(EntityKind::Cell);
            tLocalMatStruct graph(size_t(cells.NumOwned()));
            std::vector<uint8_t> hasRemote(size_t(cells.NumOwned()), 0);
            auto all = source.relations.at(size_t(MeshRelation::CellCells))->Read();
            const auto allView = all.View();
            if (cachedFacial == source.namedRelations.end() && !source.requirements.faces)
                graph = PrimaryFaceNeighborsFromCellGraph(source);
            else
            {
                int invalid = 0;
#pragma omp parallel for schedule(static)
                for (index cell = 0; cell < cells.NumOwned(); ++cell)
                {
                    auto &row = graph[size_t(cell)];
                    const auto begin = facial->offsets[size_t(cell)];
                    const auto end = facial->offsets[size_t(cell + 1)];
                    row.reserve(size_t(end - begin));
                    for (idx_t slot = facial->offsets[size_t(cell)]; slot < facial->offsets[size_t(cell + 1)]; ++slot)
                        if (const auto local = cells.GlobalToLocalPinned(index(facial->adjacency[size_t(slot)])))
                            row.push_back(*local);
                }
#pragma omp parallel for reduction(| : invalid) schedule(static)
                for (index cell = 0; cell < cells.NumOwned(); ++cell)
                {
                    const auto row = allView[cell];
                    for (auto global : row)
                    {
                        if (global < 0 || global >= cells.NumGlobal())
                        {
                            invalid = 1;
                            continue;
                        }
                        const auto local = cells.GlobalToLocalPinned(global);
                        if (!local || *local >= cells.NumOwned())
                            hasRemote[size_t(cell)] = 1;
                    }
                }
                DNDS_check_throw_info(!invalid, "owned cell neighbor is unavailable");
            }
            if (cells.NumOwned() == 0)
                permutation.localPartitionStarts.assign(size_t(nParts) + 1, 0);
            else
            {
                const std::vector<std::vector<index>> unusedRows;
                permutation = detail::ComputeCellPermutation(graph, unusedRows, cells.NumOwned(), nParts, nPartsInner,
                                                             &hasRemote);
            } });
        auto result = !source.requirements.faces && !source.spaces.count(EntityKind::Face)
                          ? ReorderOwnedPrimaryRows(source, mpi, permutation.cellOld2New,
                                                    source.spaces.at(EntityKind::Cell)._global->ROffsets()[size_t(mpi.rank)])
                          : BuildOwnedImpl(source, mpi, {}, &permutation.cellOld2New);
        result.owned.namedRelations.erase(NativeSetupFacialCellNeighborsKey());
        result.owned.localPartitionStarts = std::move(permutation.localPartitionStarts);
        return result;
    }

    NativeOwnedMeshTransitionResult NativeMeshTransition::BuildOwned(const NativeOwnedMeshData &source, const MPIInfo &mpi,
                                                                     const ReorderInput &input)
    {
        return BuildOwnedImpl(source, mpi, input, nullptr);
    }

    NativeOwnedMeshTransitionResult NativeMeshTransition::BuildOwnedImpl(const NativeOwnedMeshData &source, const MPIInfo &mpi,
                                                                         const ReorderInput &input,
                                                                         const std::vector<index> *cellOld2New)
    {
        Runtime::CommunicationDetail::AgreePolicy(mpi.comm, !source.communication || source.communication->ChecksCollectivePreflight());
        NativeOwnedMeshTransitionResult result;
        std::map<EntityKind, EntitySpace> oldSpaces, spaces;
        std::map<AdjacencyDescriptor, NativeNamedRelation> original;
        std::map<EntityKind, std::vector<MPI_int>> placements;
        std::string schedule;
        RArray noGhostRows;
        source.LocalStage(mpi, [&]
                          {
            noGhostRows = BuildIndexRows({});
            DNDS_check_throw_info(input.destroyKinds.empty(), "owned transition requires retained entity kinds");
            int rank = -1, size = 0;
            MPI_Comm_rank(mpi.comm, &rank);
            MPI_Comm_size(mpi.comm, &size);
            DNDS_check_throw_info(rank == mpi.rank && size == mpi.size, "native transition MPIInfo differs from communicator");
            for (const auto &[kind, family] : source.spaces)
            {
                family.Validate();
                EntitySpace::CheckCommunicator(family._mpi.comm, mpi.comm);
                DNDS_check_throw_info(family.Kind() == kind && family.NumGhost() == 0,
                                      "owned transition requires matching owned-only entity spaces");
                oldSpaces.emplace(kind, family);
            }
            for (size_t slot = 0; slot < source.relations.size(); ++slot)
                if (source.relations[slot])
                    original.emplace(FacadeKey(slot), NativeNamedRelation{{*source.relations[slot], noGhostRows}, Adj_PointToGlobal});
            for (const auto &[key, relation] : source.namedRelations)
                DNDS_check_throw_info(original.emplace(key, NativeNamedRelation{{relation, noGhostRows}, Adj_PointToGlobal}).second,
                                      "reserved native transition descriptor collision");
            for (const auto &map : input.explicitMaps)
            {
                DNDS_check_throw_info(oldSpaces.count(map.kind) && map.targetRanks.size() == size_t(oldSpaces.at(map.kind).NumOwned()) &&
                                          placements.emplace(map.kind, map.targetRanks).second, "invalid native placement family or extent");
                for (auto target : map.targetRanks) DNDS_check_throw_info(target >= 0 && target < mpi.size, "invalid native placement rank");
                schedule += "m" + std::to_string(int(map.kind)) + ";";
            }
            DNDS_check_throw_info(input.localPermutations.empty() ||
                                      (input.explicitMaps.empty() && input.follows.empty() && !cellOld2New),
                                  "local entity permutations cannot mix with placement/follow or cell-specialized ordering");
            for (const auto &[kind, permutation] : input.localPermutations)
            {
                DNDS_check_throw_info(oldSpaces.count(kind) && permutation.size() == size_t(oldSpaces.at(kind).NumOwned()),
                                      "local entity permutation extent or kind differs");
                std::vector<uint8_t> seen(permutation.size(), 0);
                for (auto target : permutation)
                {
                    DNDS_check_throw_info(target >= 0 && target < index(permutation.size()) && !seen[target],
                                          "local entity permutation is not bijective");
                    seen[target] = 1;
                }
                schedule += "l" + std::to_string(int(kind)) + ";";
            }
            for (const auto &follow : input.follows)
                schedule += "f" + std::to_string(int(follow.follower)) + ":" + std::to_string(int(follow.leader)) + ":" + follow.follower2leader.Signature();
            schedule += cellOld2New ? "local" : "partition";
            for (const auto &[key, unused] : original) schedule += key.Signature(); });
        Agree(mpi, schedule, source.communication);
        // SharedStorage identifies the complete immutable shape/value record,
        // unlike the allocation identity, which can back distinct row views.
        using PayloadKey = std::tuple<uintptr_t, uintptr_t, EntityKind, EntityKind, MeshAdjState>;
        auto payloadKey = [](const AdjacencyDescriptor &key, const NativeNamedRelation &values)
        {
            return PayloadKey{reinterpret_cast<uintptr_t>(values.rows.owned.SharedStorage().get()),
                              reinterpret_cast<uintptr_t>(values.rows.ghost.SharedStorage().get()),
                              key.from, key.to, values.state};
        };
        // Keep one validated adjacency scope per descriptor for this complete
        // transition. Follow placement, remote-map discovery and remapping all
        // consume immutable input rows; reacquiring the same scope would rescan
        // unchanged values. The adjacency layer proves shared content checks;
        // every descriptor still receives its own checked retained scope.
        std::map<AdjacencyDescriptor, std::unique_ptr<MeshAdjacencyRead>> inputReads;
        auto acquireInput = [&](const AdjacencyDescriptor &key) -> const MeshAdjacencyRead &
        {
            auto found = inputReads.find(key);
            if (found == inputReads.end())
            {
                const auto &values = original.at(key);
                const auto identity = payloadKey(key, values);
                std::vector<AdjacencyDescriptor> aliases;
                for (const auto &[candidate, payload] : original)
                    if (payloadKey(candidate, payload) == identity)
                        aliases.push_back(candidate);
                auto reads = MeshAdjacencyRead::AcquireAliases(aliases, oldSpaces.at(key.from), oldSpaces.at(key.to),
                                                               values.state, values.rows.owned, &values.rows.ghost);
                for (index iAlias = 0; iAlias < index(aliases.size()); ++iAlias)
                    inputReads.emplace(aliases[iAlias], std::make_unique<MeshAdjacencyRead>(std::move(reads[iAlias])));
                found = inputReads.find(key);
            }
            return *found->second;
        };
        source.LocalStage(mpi, [&]
                          { result.sourceSpaces = oldSpaces; });
        const int cellOwnerLocal = oldSpaces.at(EntityKind::Cell).NumOwned() ? mpi.rank : -1;
        int serialOwner = -1;
        MPI_Allreduce(&cellOwnerLocal, &serialOwner, 1, MPI_INT, MPI_MAX, mpi.comm);
        int serialOwnedLocal = serialOwner >= 0;
        for (const auto &[kind, old] : oldSpaces)
            if ((mpi.rank == serialOwner && old.NumOwned() != old.NumGlobal()) ||
                (mpi.rank != serialOwner && old.NumOwned() != 0))
                serialOwnedLocal = 0;
        int serialOwnedMin = 0;
        MPI_Allreduce(&serialOwnedLocal, &serialOwnedMin, 1, MPI_INT, MPI_MIN, mpi.comm);
        const bool serialOwned = !cellOld2New && serialOwnedMin;
        // Missing published inverse slots need not force a legacy recovery.
        // Their sparse native inverses are setup scratch only.
        for (auto slot : {MeshRelation::NodeCells, MeshRelation::NodeBoundaries})
            if (!original.count(FacadeKey(size_t(slot))))
            {
                const auto coneSlot = slot == MeshRelation::NodeCells ? MeshRelation::CellNodes : MeshRelation::BoundaryNodes;
                const auto &cone = original.at(FacadeKey(size_t(coneSlot)));
                const auto &key = Bindings()[size_t(coneSlot)];
                MeshAdjacencyRead read(key, oldSpaces.at(key.from), oldSpaces.at(key.to), cone.state,
                                       cone.rows.owned, &cone.rows.ghost);
                auto inverse = AdjacencyRegistry::Inverse(read, mpi);
                original.emplace(FacadeKey(size_t(slot)), NativeNamedRelation{
                                                              {inverse.Owned(), BuildIndexRows({})}, Adj_PointToGlobal});
            }
        const auto boundaryPlacement = source.periodic
                                           ? AdjacencyDescriptor::Composition(Adj::Bnd2Node, Adj::Cell2Node.Inverted(), RelationSelection::Exact, false,
                                                                              "geom.boundary.physical-frame-face", 1, RelationPeriodicPolicy::FrameMatched)
                                           : Adj::Bnd2Cell;
        if (source.periodic && !original.count(boundaryPlacement))
            source.LocalStage(mpi, [&]
                              {
                // Full snapshots/archives may predate the optional named
                // exact relation. The physical facade contract already puts
                // the boundary's own side in slot 0. Publish a distinct exact
                // side relation; never change/truncate its coupled support.
                const auto &physical = original.at(FacadeKey(size_t(MeshRelation::BoundaryCells)));
                MeshAdjacencyRead read(Adj::Bnd2Cell, oldSpaces.at(EntityKind::Bnd), oldSpaces.at(EntityKind::Cell),
                                        physical.state, physical.rows.owned, &physical.rows.ghost);
                std::vector<std::vector<index>> sides(size_t(read.NumOwned()));
                for (index boundary = 0; boundary < read.NumOwned(); ++boundary)
                {
                    DNDS_check_throw_info(read.RowSize(boundary) >= 1 && !read.ReadEntry(boundary,0).absent,
                                          "periodic boundary placement requires its physical side in slot zero");
                    sides[size_t(boundary)] = {read.ReadEntry(boundary,0).global};
                }
                original.emplace(boundaryPlacement,NativeNamedRelation{{BuildIndexRows(sides),BuildIndexRows({})},Adj_PointToGlobal}); });
        std::vector<FollowSpec> follows = input.follows;
        if (follows.empty() && placements.count(EntityKind::Cell))
            for (auto kind : {EntityKind::Node, EntityKind::Bnd})
                if (!placements.count(kind))
                    follows.push_back({kind, EntityKind::Cell, kind == EntityKind::Node ? Adj::Node2Cell : boundaryPlacement});
        for (const auto &follow : follows)
        {
            if (serialOwned)
            {
                const MeshAdjacencyRead *relation = nullptr;
                source.LocalStage(mpi, [&]
                                  {
                    DNDS_check_throw_info(placements.count(follow.leader) && !placements.count(follow.follower) &&
                                              follow.follower2leader.from == follow.follower && follow.follower2leader.to == follow.leader,
                                          "invalid native follow dependency");
                    auto found = original.find(follow.follower2leader);
                    if (found == original.end())
                        for (size_t slot = 0; slot < Bindings().size(); ++slot)
                            if (Bindings()[slot] == follow.follower2leader) { found = original.find(FacadeKey(slot)); break; }
                    DNDS_check_throw_info(found != original.end(), "native follow relation is unavailable");
                    relation = &acquireInput(found->first); });
                const auto &leaderPlacements = placements.at(follow.leader);
                const auto leaderFirst = (*oldSpaces.at(follow.leader)._global)(mpi.rank, 0);
                std::vector<MPI_int> assignment(size_t(relation->NumOwned()), mpi.rank);
                std::vector<index> converted;
                source.LocalStage(mpi, [&]
                                  {
                    for (index row = 0; row < relation->NumOwned(); ++row)
                    {
                        int target = mpi.size;
                        converted.resize(relation->RowSize(row));
                        relation->ConvertRowTo(row, Adj_PointToGlobal, converted.data());
                        for (auto value : converted)
                        {
                            const auto global = value;
                            if (global == UnInitIndex) continue;
                            DNDS_check_throw_info(global >= leaderFirst && global - leaderFirst < index(leaderPlacements.size()),
                                                  "native serial follow target is not locally available");
                            target = std::min(target, int(leaderPlacements[size_t(global - leaderFirst)]));
                        }
                        if (target < mpi.size) assignment[size_t(row)] = target;
                    } });
                placements.emplace(follow.follower, std::move(assignment));
                continue;
            }
            const MeshAdjacencyRead *relation = nullptr;
            std::vector<index> requested;
            std::vector<Runtime::RowRequest> requests;
            source.LocalStage(mpi, [&]
                              {
                DNDS_check_throw_info(placements.count(follow.leader) && !placements.count(follow.follower) &&
                                          follow.follower2leader.from == follow.follower && follow.follower2leader.to == follow.leader,
                                      "invalid native follow dependency");
                auto found = original.find(follow.follower2leader);
                if (found == original.end())
                    for (size_t slot = 0; slot < Bindings().size(); ++slot)
                        if (Bindings()[slot] == follow.follower2leader) { found = original.find(FacadeKey(slot)); break; }
                DNDS_check_throw_info(found != original.end(), "native follow relation is unavailable");
                relation = &acquireInput(found->first);
                std::vector<index> globals;
                for (index row = 0; row < relation->NumOwned(); ++row)
                    for (auto value : relation->GlobalRow(row)) if (value != UnInitIndex) globals.push_back(value);
                std::sort(globals.begin(), globals.end());
                globals.erase(std::unique(globals.begin(), globals.end()), globals.end());
                requested = globals;
                const auto &leader = oldSpaces.at(follow.leader);
                for (size_t row = 0; row < requested.size(); ++row)
                    requests.push_back({leader.Owner(requested[row]), leader.OwnerLocalRow(requested[row]), index(row)}); });
            RArray ranks(Runtime::RowLayout::Dense(placements.at(follow.leader).size(), 1));
            {
                auto write = ranks.Write();
                for (size_t i = 0; i < ranks.RowCount(); ++i)
                    write.View()(i, 0) = placements.at(follow.leader)[i];
            }
            Runtime::RowTransferPlan route(source.communication ? source.communication : std::make_shared<Runtime::CommunicationContext>(mpi.comm), ranks.RowCount(), requested.size(), requests);
            auto received = Move(ranks, route, Runtime::ElementSchema::Arithmetic<index>(), mpi, false);
            source.LocalStage(mpi, [&]
                              {
                auto read = received.Read();
                std::vector<MPI_int> targets(requested.size());
                for (size_t i = 0; i < requested.size(); ++i)
                    targets[i] = MPI_int(read.View()(i, 0));
                std::vector<MPI_int> assignment(size_t(relation->NumOwned()), mpi.rank);
                for (index row = 0; row < relation->NumOwned(); ++row)
                {
                    int target = mpi.size;
                    for (auto global : relation->GlobalRow(row))
                        if (global != UnInitIndex)
                        {
                            const auto found = std::lower_bound(requested.begin(), requested.end(), global);
                            DNDS_check_throw_info(found != requested.end() && *found == global,
                                                  "native follow target fetch lost a leader row");
                            target = std::min(target, int(targets[size_t(found - requested.begin())]));
                        }
                    if (target < mpi.size) assignment[size_t(row)] = target;
                }
                placements.emplace(follow.follower, std::move(assignment)); });
        }
        std::map<EntityKind, Runtime::RowTransferPlan> routes;
        std::map<EntityKind, Runtime::RowTransferPlan> adjacencyRoutes;
        std::map<EntityKind, std::vector<index>> sendOld2New;
        std::map<EntityKind, PermutationTransfer> allTransfers;
        const auto anchor = oldSpaces.begin()->second;
        for (const auto &[kind, old] : oldSpaces)
        {
            auto assigned = placements.find(kind);
            const auto targets = assigned == placements.end() ? std::vector<MPI_int>(size_t(old.NumOwned()), mpi.rank) : assigned->second;
            PermutationTransfer transfer;
            if (cellOld2New || !input.localPermutations.empty())
            {
                std::vector<index> identity(size_t(old.NumOwned()));
                std::iota(identity.begin(), identity.end(), index(0));
                const auto selected = input.localPermutations.find(kind);
                const auto &permutation = cellOld2New                                 ? (kind == EntityKind::Cell ? *cellOld2New : identity)
                                          : selected == input.localPermutations.end() ? identity
                                                                                      : selected->second;
                transfer = PermutationTransfer::fromLocalPermutation(permutation,
                                                                     old._global, mpi);
            }
            else
                transfer = PermutationTransfer::fromPartition(targets, old._global, mpi);
            auto route = NativeRowTransfer(transfer, mpi, source.communication);
            spaces.emplace(kind, anchor.CreateSibling(kind, index(route.DestinationRows())));
            if (!cellOld2New)
            {
                // Remapping already creates independent owned adjacency data.
                // Give that array the sender's partition-bucket row order in
                // the same pass, so each peer reads contiguous rows through
                // direct MPI datatypes. This is an explicit row permutation,
                // not a byte-packing workspace or a different transport.
                auto ordered = route.ReorderSourceByPeer();
                adjacencyRoutes.emplace(kind, std::move(ordered.first));
                sendOld2New.emplace(kind, std::move(ordered.second));
            }
            routes.emplace(kind, std::move(route));
            if (cellOld2New || assigned != placements.end() || !input.localPermutations.empty())
                result.transfers.emplace(kind, transfer);
            allTransfers.emplace(kind, std::move(transfer));
        }
        std::map<EntityKind, EntityRemap> remap;
        std::map<EntityKind, EntityRemap> serialRemaps;
        if (serialOwned)
            for (const auto &[kind, old] : oldSpaces)
            {
                serialRemaps[kind].first = (*old._global)(mpi.rank, 0);
                serialRemaps[kind].newGlobals = &allTransfers.at(kind).newGlobalIndices;
            }
        else
            for (const auto &[kind, old] : oldSpaces)
            {
                // Only Cell numbering changes during a local cell permutation.
                // Per-descriptor validation below still checks every input row;
                // building identity maps for other target families adds no proof.
                if (cellOld2New && kind != EntityKind::Cell)
                    continue;
                std::vector<index> globals;
                std::vector<Runtime::RowRequest> requests;
                std::vector<index> converted;
                source.LocalStage(mpi, [&]
                                  {
                std::vector<index> referenced;
                std::set<PayloadKey> scanned;
                for (const auto &[key, values] : original) if (key.to == kind)
                {
                    const auto &read = acquireInput(key);
                    // Validate every descriptor even when its exact payload was
                    // already scanned under another semantic name.
                    if (!scanned.insert(payloadKey(key, values)).second) continue;
                    for (index row = 0; row < read.NumOwned(); ++row)
                    {
                        converted.resize(read.RowSize(row));
                        read.ConvertRowTo(row, Adj_PointToGlobal, converted.data());
                        for (auto value : converted)
                        {
                            const auto global = value;
                            if (global != UnInitIndex) referenced.push_back(global);
                        }
                    }
                }
                std::sort(referenced.begin(), referenced.end());
                referenced.erase(std::unique(referenced.begin(), referenced.end()), referenced.end());
                const auto first = (*old._global)(mpi.rank, 0);
                const auto &newGlobals = allTransfers.at(kind).newGlobalIndices;
                auto &mapping = remap[kind];
                mapping.first = first;
                mapping.newGlobals = &newGlobals;
                for (auto global : referenced)
                {
                    if (global >= first && global - first < old.NumOwned())
                        continue;
                    else
                    {
                        requests.push_back({old.Owner(global), old.OwnerLocalRow(global), index(globals.size())});
                        globals.push_back(global);
                    }
                } });
                RArray values(Runtime::RowLayout::Dense(size_t(old.NumOwned()), 1));
                {
                    auto write = values.Write();
                    for (size_t i = 0; i < values.RowCount(); ++i)
                        write.View()(i, 0) = allTransfers.at(kind).newGlobalIndices[i];
                }
                Runtime::RowTransferPlan route(source.communication ? source.communication : std::make_shared<Runtime::CommunicationContext>(mpi.comm), values.RowCount(), globals.size(), requests);
                auto received = Move(values, route, Runtime::ElementSchema::Arithmetic<index>(), mpi, false);
                auto read = received.Read();
                remap.at(kind).remote.reserve(globals.size());
                for (size_t i = 0; i < globals.size(); ++i)
                    remap.at(kind).remote.emplace_back(globals[i], read.View()(i, 0));
            }
        std::map<AdjacencyDescriptor, RArray> movedRelations;
        std::vector<Runtime::TransferOperation> pendingMoves;
        source.LocalStage(mpi, [&]
                          {
            // Validate every descriptor before posting the first payload. In
            // particular, the serial-owned remap does not need the discovery
            // scan above, so an invalid later alias could otherwise throw with
            // earlier transfers still active. Tests can reject this admission
            // coherently; production retains its terminal failure policy.
            for (const auto &[key, unused] : original)
                (void)acquireInput(key);
            pendingMoves.reserve(original.size() + source.spaces.size() * 3 + 12); });
        auto *pending = cellOld2New ? nullptr : &pendingMoves;
        // Routing preserves row slots. Sorting belongs to the destination's
        // semantic descriptor, not the wire payload: sorted and topological
        // aliases can share one remap/transfer without changing either order.
        std::map<PayloadKey, int> movedPayloads;
        std::vector<RArray> movedByOrdinal;
        std::map<uintptr_t, RArray> sortedPayloads;
        std::vector<index> cellNew2Old;
        if (cellOld2New)
        {
            cellNew2Old.assign(cellOld2New->size(), -1);
            for (size_t oldRow = 0; oldRow < cellOld2New->size(); ++oldRow)
            {
                const auto newRow = (*cellOld2New)[oldRow];
                DNDS_check_throw_info(newRow >= 0 && size_t(newRow) < cellNew2Old.size(),
                                      "owned cell permutation contains an out-of-range row");
                DNDS_check_throw_info(cellNew2Old[size_t(newRow)] < 0,
                                      "owned cell permutation contains a duplicate row");
                cellNew2Old[size_t(newRow)] = index(oldRow);
            }
            DNDS_check_throw_info(std::all_of(cellNew2Old.begin(), cellNew2Old.end(), [](index row)
                                              { return row >= 0; }),
                                  "owned cell permutation is incomplete");
        }
        for (const auto &[key, values] : original)
        {
            const auto identity = payloadKey(key, values);
            const MeshAdjacencyRead *read = nullptr;
            source.LocalStage(mpi, [&]
                              {
                // A byte alias does not establish this descriptor's semantic
                // constraints (e.g. SortedUnique or removeSelf).
                read = &acquireInput(key);
                DNDS_check_throw_info(read->State() == Adj_PointToGlobal, "owned transition requires global adjacency rows"); });
            auto publish = [&](RArray moved)
            {
                movedPayloads.emplace(identity, int(movedByOrdinal.size()));
                movedByOrdinal.push_back(moved);
                // Retain handles while receives are active. Reading/sorting
                // occurs only after the explicit join below.
                movedRelations.emplace(key, std::move(moved));
            };
            int candidate = -1;
            if (auto found = movedPayloads.find(identity); found != movedPayloads.end())
                candidate = found->second;
            // Local alias inventories can differ. All ranks must reuse the
            // SAME earlier collective output, not merely find some local hit.
            // A local cell permutation has no collective payload output: every
            // rank completes its local relation loop before the next assembly
            // collective, so no cross-rank selection is needed here.
            if (!cellOld2New)
                candidate = FieldTransferDetail::AgreeAliasCandidate(mpi, candidate, original.size());
            if (candidate >= 0)
            {
                source.LocalStage(mpi, [&]
                                  { publish(movedByOrdinal.at(size_t(candidate))); });
                continue;
            }
            if (cellOld2New && key.from != EntityKind::Cell && key.to != EntityKind::Cell)
            {
                source.LocalStage(mpi, [&]
                                  { publish(values.rows.owned); });
                continue;
            }
            std::optional<RArray> globalRows;
            source.LocalStage(mpi, [&]
                              {
                // Owned transition inputs are global and the retained scope
                // above has already validated their encodings. Reuse the row
                // layout and fuse remapping with copying, instead of rebuilding
                // CSR offsets and making a separate conversion pass per row.
                const EntityRemap *mapping = cellOld2New && key.to != EntityKind::Cell
                                                 ? nullptr
                                                 : serialOwned ? &serialRemaps.at(key.to) : &remap.at(key.to);
                const auto &permutation = cellOld2New ? *cellOld2New : sendOld2New.at(key.from);
                globalRows.emplace(ReorderOwnedRows(values.rows.owned, permutation,
                                                     !cellOld2New || key.from == EntityKind::Cell, bool(mapping),
                                                     false,
                                                     [&](index value) { return value == UnInitIndex ? value : (*mapping)(value); })); });
            auto moved = cellOld2New
                             ? *globalRows
                             : Move(*globalRows, adjacencyRoutes.at(key.from), Runtime::ElementSchema::Arithmetic<index>(), mpi,
                                    false, pending);
            source.LocalStage(mpi, [&]
                              { publish(std::move(moved)); });
        }
        FinishMoves(pendingMoves);
        source.LocalStage(mpi, [&]
                          {
            for (auto &[key, moved] : movedRelations)
                if (key.order == RelationOrder::SortedUnique)
                {
                    const auto storage = reinterpret_cast<uintptr_t>(moved.SharedStorage().get());
                    auto found = sortedPayloads.find(storage);
                    if (found == sortedPayloads.end())
                        found = sortedPayloads.emplace(storage, SortOwnedTargets(moved)).first;
                    moved = found->second;
                } });
        NativeOwnedMeshData owned;
        owned.communication = source.communication;
        owned.identity = source.identity;
        owned.requirements = source.requirements;
        owned.cellCellsComplete = source.cellCellsComplete;
        owned.audit = source.audit;
        owned.ghostSpec = source.ghostSpec;
        owned.dimension = source.dimension;
        owned.periodic = source.periodic;
        owned.periodicInfo = source.periodicInfo;
        owned.spaces = spaces;
        auto moveCompanion = [&](const auto &values, EntityKind kind, const Runtime::ElementSchema &schema)
        {
            if (cellOld2New)
                return Move(values, routes.at(kind), schema, mpi, true, pending);
            // Apply the same explicit sender permutation used by adjacency
            // arrays. These are ordinary independent arrays in partition-row
            // order, not an MPI packing workspace. Direct datatypes then see
            // contiguous peer intervals for coordinates, POD metadata and IDs.
            using Array = std::decay_t<decltype(values)>;
            std::optional<Array> ordered;
            source.LocalStage(mpi, [&]
                              { ordered.emplace(ReorderOwnedRows(values, sendOld2New.at(kind), true, false, false,
                                                                 [](auto value)
                                                                 { return value; })); });
            return Move(*ordered, adjacencyRoutes.at(kind), schema, mpi, false, pending);
        };
        if (source.provenance)
        {
            owned.provenance = *source.provenance;
            for (auto &[kind, values] : owned.provenance->sourceIds)
                if (!cellOld2New || kind == EntityKind::Cell)
                    values = moveCompanion(values, kind, Runtime::ElementSchema::Arithmetic<index>());
        }
        source.LocalStage(mpi, [&]
                          {
            for (size_t slot = 0; slot < source.relations.size(); ++slot)
                if (source.relations[slot])
                {
                    owned.relations[slot] = movedRelations.at(FacadeKey(slot));
                }
            for (const auto &[key, unused] : source.namedRelations)
                owned.namedRelations.emplace(key, movedRelations.at(key)); });
        auto companion = [&](auto &destination, const auto &originalPair, EntityKind kind, const Runtime::ElementSchema &schema)
        {
            if (originalPair)
                destination = cellOld2New && kind != EntityKind::Cell ? *originalPair
                                                                      : moveCompanion(*originalPair, kind, schema);
        };
        companion(owned.coords, source.coords, EntityKind::Node, Runtime::ElementSchema::Arithmetic<real>());
        companion(owned.nodeWallDist, source.nodeWallDist, EntityKind::Node, Runtime::ElementSchema::Arithmetic<real>());
        const auto elementSchema = Runtime::ElementSchema::For<ElemInfo>();
        companion(owned.cellInfo, source.cellInfo, EntityKind::Cell, elementSchema);
        companion(owned.bndInfo, source.bndInfo, EntityKind::Bnd, elementSchema);
        companion(owned.faceInfo, source.faceInfo, EntityKind::Face, elementSchema);
        companion(owned.edgeInfo, source.edgeInfo, EntityKind::Edge, elementSchema);
        const auto bits = Runtime::ElementSchema::For<NodePeriodicBits>();
        companion(owned.cellNodesPbi, source.cellNodesPbi, EntityKind::Cell, bits);
        companion(owned.bndNodesPbi, source.bndNodesPbi, EntityKind::Bnd, bits);
        companion(owned.cellFacesPbi, source.cellFacesPbi, EntityKind::Cell, bits);
        companion(owned.faceNodesPbi, source.faceNodesPbi, EntityKind::Face, bits);
        companion(owned.cellEdgesPbi, source.cellEdgesPbi, EntityKind::Cell, bits);
        companion(owned.edgeNodesPbi, source.edgeNodesPbi, EntityKind::Edge, bits);
        for (const auto &[kind, old] : oldSpaces)
        {
            const auto &originalIds = source.ids.at(kind);
            if (cellOld2New && kind != EntityKind::Cell)
            {
                // The combined ID allocation begins with the owned prefix.
                // Reuse that immutable prefix; assembly builds its new ghosts.
                source.LocalStage(mpi, [&]
                                  {
                    const auto layout = Runtime::RowLayout::Dense(size_t(old.NumOwned()), 1, originalIds.Layout().Stride());
                    owned.ids.emplace(kind, RArray::FromStorage(layout, originalIds.Storage().Subspan(
                                                                          0, CheckedSize::Multiply(layout.Elements(), sizeof(index))))); });
                continue;
            }
            const auto &baseRoute = routes.at(kind);
            // IDs normally have precisely the same owned source rows as all
            // other companions. Share that validated route; only snapshots
            // with an additional retained ID suffix need an adjusted extent.
            int sameExtent = originalIds.RowCount() == baseRoute.SourceRows();
            MPI_Allreduce(MPI_IN_PLACE, &sameExtent, 1, MPI_INT, MPI_MIN, mpi.comm);
            auto idRoute = sameExtent ? baseRoute : Runtime::RowTransferPlan::Known(baseRoute.Context(), originalIds.RowCount(), baseRoute.DestinationRows(), baseRoute.SendRows(), baseRoute.ReceiveRows());
            owned.ids.emplace(kind, sameExtent ? moveCompanion(originalIds, kind, Runtime::ElementSchema::Arithmetic<index>())
                                               : Move(originalIds, idRoute, Runtime::ElementSchema::Arithmetic<index>(), mpi,
                                                      bool(cellOld2New), pending));
        }
        FinishMoves(pendingMoves);
        owned.retainedRead = source.retainedRead;
        owned.retainedProvenance = source.retainedProvenance;
        result.owned = std::move(owned);
        return result;
    }
}
