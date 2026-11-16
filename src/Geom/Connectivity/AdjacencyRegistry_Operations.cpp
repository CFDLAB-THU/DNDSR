#include "DNDS/MPI/Failure.hpp"
#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Connectivity/ConnectivityComposeRows.hpp"
#include <algorithm>
#include <limits>
#include <type_traits>

namespace DNDS::Geom
{
    static Runtime::Array<index> BuildNativeRows(const std::vector<std::vector<index>> &rows)
    {
        return Runtime::BuildArrayRows<index>(rows.size(), [&](size_t row)
                                              { return rows[row].size(); }, [&](size_t row, Runtime::RowSpan<index> target)
                                              {
            if (!rows[row].empty())
                std::copy(rows[row].begin(), rows[row].end(), target.data()); });
    }

    MeshAdjacency AdjacencyRegistry::ComposeFiltered(const MeshAdjacencyRead &AB, const MeshAdjacencyRead &BC,
                                                     const AdjacencyDescriptor &resultKind,
                                                     const std::function<bool(index, index, int)> &predicate,
                                                     const std::function<bool(index, index, const std::vector<index> &)> &matchExtra)
    {
        AB.Validate();
        BC.Validate();
        DNDS_check_throw_info(AB.Target().SameOrder(BC.Source()), "composition intermediate entity/map order mismatch");
        DNDS_check_throw_info(bool(predicate), "composition predicate is missing");
        resultKind.Validate();
        DNDS_check_throw_info(!resultKind.IsLegacy() && resultKind == AdjacencyDescriptor::Composition(AB.Kind(), BC.Kind(),
                                                                                                       resultKind.selection, resultKind.removeSelf, resultKind.semanticName, resultKind.semanticVersion, resultKind.periodic),
                              "composition output descriptor differs from its input definitions");
        DNDS_check_throw_info(!resultKind.removeSelf || AB.Source().SameOrder(BC.Target()),
                              "removeSelf requires identical source and final target spaces");
        DNDS_check_throw_info(resultKind.selection == RelationSelection::Any || resultKind.selection == RelationSelection::All ||
                                  resultKind.selection == RelationSelection::Custom || resultKind.selection == RelationSelection::Exact,
                              "composition requires an any/all/custom/exact selection");
        DNDS_check_throw_info(!matchExtra || resultKind.selection == RelationSelection::Custom || resultKind.selection == RelationSelection::Exact,
                              "additional matching requires a named custom/exact selection");
        DNDS_check_throw_info(resultKind.periodic != RelationPeriodicPolicy::FrameMatched || bool(matchExtra),
                              "frame-matched selection requires an explicit periodic matcher");
        DNDS_check_throw_info(resultKind.selection != RelationSelection::Exact || bool(matchExtra),
                              "exact selection requires an explicit incidence/frame matcher");
        auto unique = [](std::vector<index> entries)
        {
            entries.erase(std::remove(entries.begin(), entries.end(), UnInitIndex), entries.end());
            std::sort(entries.begin(), entries.end());
            entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
            return entries;
        };
        // Node support is revisited by every incident cell. Normalize each
        // retained row once instead of allocating/sorting it at every hop.
        // These caches scale with local owned/appended rows, never global size.
        std::vector<std::vector<index>> sourceRows(size_t(AB.NumOwned())), intermediateRows(size_t(BC.RowCount()));
        OMP::ForSetupRows(0, AB.NumOwned(), [&](index iRow)
                          { sourceRows[size_t(iRow)] = unique(AB.GlobalRow(iRow)); });
        OMP::ForSetupRows(0, BC.RowCount(), [&](index iRow)
                          { intermediateRows[size_t(iRow)] = unique(BC.GlobalRow(iRow)); });
        auto selected = [&](index a, index c, int count)
        {
            if (resultKind.removeSelf && a == c)
                return false;
            if (resultKind.selection == RelationSelection::Any)
                return count >= 1;
            if (resultKind.selection == RelationSelection::All)
            {
                const auto local = AB.Source().GlobalToLocalPinned(a);
                DNDS_check_throw_info(local, "composition source row is unavailable");
                return count == int(sourceRows[size_t(*local)].size());
            }
            return predicate(a, c, count);
        };
        std::vector<std::vector<index>> rows(size_t(AB.NumOwned()));
        ConnectivityRows::Compose(AB.NumOwned(), [&](index row) -> const std::vector<index> &
                                  { return sourceRows[size_t(row)]; }, [&](index row) -> const std::vector<index> &
                                  { return intermediateRows[size_t(row)]; }, [&](index global)
                                  {
                auto row = BC.Source().GlobalToLocalPinned(global);
                DNDS_check_throw_info(row && *row < BC.RowCount(), "composition intermediate row is unavailable; transfer it explicitly");
                return *row; }, [&](index row)
                                  { return AB.Source().LocalToGlobalPinned(row); }, selected, [&](index row, const std::vector<index> &entries)
                                  { rows[size_t(row)] = entries; }, matchExtra, !matchExtra && (resultKind.selection == RelationSelection::Any || resultKind.selection == RelationSelection::All));
        return MeshAdjacency(resultKind, AB.Source(), BC.Target(),
                             BuildNativeRows(rows));
    }

    MeshAdjacency AdjacencyRegistry::Compose(const MeshAdjacencyRead &AB, const MeshAdjacencyRead &BC, bool removeSelf)
    {
        DNDS_check_throw_info(!removeSelf || AB.Source().SameOrder(BC.Target()),
                              "removeSelf requires identical source and final target spaces");
        return ComposeFiltered(AB, BC, AdjacencyDescriptor::Composition(AB.Kind(), BC.Kind(), RelationSelection::Any, removeSelf),
                               SharedCountPredicate{1, removeSelf});
    }

    MeshAdjacency AdjacencyRegistry::ComposeAll(const MeshAdjacencyRead &AB, const MeshAdjacencyRead &BC, bool removeSelf)
    {
        return ComposeFiltered(AB, BC, AdjacencyDescriptor::Composition(AB.Kind(), BC.Kind(), RelationSelection::All, removeSelf),
                               SharedCountPredicate{1, removeSelf});
    }

    MeshAdjacency AdjacencyRegistry::Inverse(const MeshAdjacencyRead &cone, const MPIInfo &mpi)
    {
        int ownerPolicy[]{cone.Source().NumOwned() ? mpi.rank : -1,
                          int(cone.Source().ChecksCollectivePreflight()), -int(cone.Source().ChecksCollectivePreflight())};
        MPI_Allreduce(MPI_IN_PLACE, ownerPolicy, 3, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(ownerPolicy[1] == -ownerPolicy[2], "inverse preparation policies differ between ranks");
        const int sourceOwner = ownerPolicy[0];
        int localOnly = sourceOwner < 0 ||
                        (mpi.rank == sourceOwner && cone.Source().NumOwned() == cone.Source().NumGlobal() &&
                         cone.Target().NumOwned() == cone.Target().NumGlobal()) ||
                        (mpi.rank != sourceOwner && cone.Source().NumOwned() == 0 && cone.Target().NumOwned() == 0);
        MPI_Allreduce(MPI_IN_PLACE, &localOnly, 1, MPI_INT, MPI_MIN, mpi.comm);
        if (localOnly)
        {
            int validLocal = 1;
            std::vector<std::vector<index>> support(size_t(cone.Target().NumOwned()));
            try
            {
                cone.Validate();
                cone.CheckCommunicator(mpi.comm);
                for (index sourceRow = 0; sourceRow < cone.NumOwned(); ++sourceRow)
                {
                    const auto sourceGlobal = cone.Source().LocalToGlobalPinned(sourceRow);
                    for (size_t slot = 0; slot < cone.RowSize(sourceRow); ++slot)
                    {
                        const auto entry = cone.ReadEntry(sourceRow, slot);
                        if (entry.absent)
                            continue;
                        const auto targetRow = cone.Target().GlobalToLocalPinned(entry.global);
                        DNDS_check_throw_info(targetRow && size_t(*targetRow) < support.size(),
                                              "local inverse target row is unavailable");
                        support[size_t(*targetRow)].push_back(sourceGlobal);
                    }
                }
                OMP::ForSetupRows(0, index(support.size()), [&](index iRow)
                                  {
                    auto &entries = support[size_t(iRow)];
                    std::sort(entries.begin(), entries.end());
                    entries.erase(std::unique(entries.begin(), entries.end()), entries.end()); });
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("native connectivity preparation failed");
                validLocal = 0;
            }
            cone.Source().Preflight(validLocal, "local inverse relation validation failed on a rank");
            DNDS_check_throw_info(validLocal, "local inverse relation/communicator validation failed on a rank");
            return MeshAdjacency(cone.Kind().Inverted(), cone.Target(), cone.Source(), BuildNativeRows(support));
        }
        int valid = 1;
        std::vector<std::vector<index>> batches;
        try
        {
            cone.Validate();
            cone.CheckCommunicator(mpi.comm);
            // Packing can detect an unmanaged corrupted entry even when the
            // retained scope's construction scan is cached. Agree on those
            // errors before any rank enters the exchange collectives.
            batches.resize(size_t(mpi.size));
            for (index source = 0; source < cone.NumOwned(); ++source)
            {
                const auto sourceGlobal = cone.Source().LocalToGlobalPinned(source);
                for (size_t slot = 0; slot < cone.RowSize(source); ++slot)
                {
                    const auto entry = cone.ReadEntry(source, slot);
                    if (entry.absent)
                        continue;
                    // The retained adjacency scope pins both immutable maps.
                    // Find owner and row together; public descriptor queries
                    // would each repeat the epoch lock and binary search.
                    MPI_int owner = -1;
                    index ownerRow = -1;
                    DNDS_check_throw_info(cone.Target()._global->search(entry.global, owner, ownerRow),
                                          "inverse target is outside its entity space");
                    batches[size_t(owner)].push_back(ownerRow);
                    batches[size_t(owner)].push_back(sourceGlobal);
                }
            }
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("native connectivity preparation failed");
            valid = 0;
        }
        cone.Source().Preflight(valid, "inverse relation/communicator validation failed on a rank");
        DNDS_check_throw_info(valid, "inverse relation/communicator validation failed on a rank");
        // Each peer receives one ragged batch of (target-owned row, source
        // global ID) contributions. Only the owning rank sees that incidence;
        // the all-to-all discovery communicates presence, never global topology.
        std::vector<int> sendPresence(size_t(mpi.size)), receivePresence(size_t(mpi.size));
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            sendPresence[size_t(peer)] = !batches[size_t(peer)].empty();
        }
        auto source = BuildNativeRows(batches);
        MPI_Alltoall(sendPresence.data(), 1, MPI_INT, receivePresence.data(), 1, MPI_INT, mpi.comm);
        std::vector<std::vector<index>> send(size_t(mpi.size)), receive(size_t(mpi.size));
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            if (sendPresence[size_t(peer)])
                send[size_t(peer)].push_back(index(peer));
            if (receivePresence[size_t(peer)])
                receive[size_t(peer)].push_back(index(peer));
        }
        auto context = cone.Source().Communication() ? cone.Source().Communication() : std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        auto route = Runtime::RowTransferPlan::Known(context, size_t(mpi.size), size_t(mpi.size),
                                                     std::move(send), std::move(receive),
                                                     {cone.Source().Stamp(), cone.Target().Stamp()});
        Runtime::PayloadPlan payload(route, source.Layout(), Runtime::ElementSchema::Arithmetic<index>());
        Runtime::Array<index> incoming(payload.DestinationLayout());
        Runtime::TransferOptions options;
        options.mode = Runtime::TransferMode::Transient;
        options.context = context;
        payload.Bind(source, incoming, Runtime::TransferTransport::Direct, options).Begin().Wait();

        std::vector<std::vector<index>> support(size_t(cone.Target().NumOwned()));
        auto values = incoming.Read();
        auto view = values.View();
        valid = 1;
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            const auto length = view.RowSize(size_t(peer));
            valid = valid && length % 2 == 0;
            const auto &sourceOffsets = cone.Source()._global->ROffsets();
            for (size_t slot = 0; slot + 1 < length; slot += 2)
            {
                const auto targetRow = view(size_t(peer), slot), sourceGlobal = view(size_t(peer), slot + 1);
                const bool entryValid = targetRow >= 0 && size_t(targetRow) < support.size() &&
                                        sourceGlobal >= sourceOffsets[size_t(peer)] &&
                                        sourceGlobal < sourceOffsets[size_t(peer + 1)];
                valid = valid && entryValid;
                if (entryValid)
                    support[size_t(targetRow)].push_back(sourceGlobal);
            }
        }
        cone.Source().Preflight(valid, "inverse relation/communicator validation failed on a rank");
        DNDS_check_throw_info(valid, "inverse contribution payload contains an invalid row or source entity");
        OMP::ForSetupRows(0, index(support.size()), [&](index iRow)
                          {
            auto &entries = support[size_t(iRow)];
            std::sort(entries.begin(), entries.end());
            entries.erase(std::unique(entries.begin(), entries.end()), entries.end()); });
        return MeshAdjacency(cone.Kind().Inverted(), cone.Target(), cone.Source(),
                             BuildNativeRows(support));
    }

}
