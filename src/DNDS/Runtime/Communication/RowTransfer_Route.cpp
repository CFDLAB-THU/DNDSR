#include "DNDS/Runtime/Communication/RowTransfer_Route.hpp"
#include <algorithm>
#include <climits>
#include <numeric>

namespace DNDS::Runtime
{
    namespace
    {
        std::vector<int> Counts(const std::vector<std::vector<index>> &rows)
        {
            std::vector<int> counts;
            for (const auto &peer : rows)
                counts.push_back(int(peer.size()));
            return counts;
        }
        std::vector<int> Offsets(const std::vector<int> &counts)
        {
            std::vector<int> result(counts.size() + 1);
            for (size_t i = 0; i < counts.size(); ++i)
                result[i + 1] = result[i] + counts[i];
            return result;
        }
        /// Collision detection needs one answer after route construction, not
        /// dynamic membership. Check complete permutations in-place, or sort
        /// sparse request scratch; never allocate a global/entity-sized bitmap.
        class SeenRows
        {
            std::vector<index> _rows;
            index _extent;

        public:
            SeenRows(index extent, size_t expected) : _extent(extent)
            {
                DNDS_check_throw_info(extent >= 0, "route row extent is negative");
                _rows.reserve(expected);
            }
            void Append(index row) { _rows.push_back(row); }
            bool HasDuplicates()
            {
                if (index(_rows.size()) == _extent &&
                    std::all_of(_rows.begin(), _rows.end(), [&](index row)
                                { return row >= 0 && row < _extent; }))
                {
                    // A complete in-range row list is a permutation iff every
                    // value can be placed at its own index. Each swap fixes a
                    // new slot; finding that slot already fixed proves a
                    // duplicate. This is linear and uses only request scratch,
                    // including for migrations whose source extent is large.
                    for (index iRow = 0; iRow < _extent; ++iRow)
                        while (_rows[iRow] != iRow)
                        {
                            const auto iTarget = _rows[iRow];
                            if (_rows[iTarget] == iTarget)
                                return true;
                            std::swap(_rows[iRow], _rows[iTarget]);
                        }
                    return false;
                }
                std::sort(_rows.begin(), _rows.end());
                return std::adjacent_find(_rows.begin(), _rows.end()) != _rows.end();
            }
        };
    }

    namespace Detail
    {
        std::vector<uint64_t> ExchangeLengths(const RowTransferPlan &route, const RowLayout &source, bool preflightDone)
        {
            if (!preflightDone)
                route.Context()->Preflight(source.RowCount() == route.SourceRows() && route.Valid(), "row transfer source layout or epoch mismatch");
            const auto sendCounts = Counts(route.SendRows()), receiveCounts = Counts(route.ReceiveRows());
            const auto sendOffsets = Offsets(sendCounts), receiveOffsets = Offsets(receiveCounts);
            std::vector<uint64_t> send, receive(receiveOffsets.back());
            for (const auto &peer : route.SendRows())
                for (auto row : peer)
                    send.push_back(source.RowSize(row));
            MPI_Alltoallv(send.data(), sendCounts.data(), sendOffsets.data(), MPI_UINT64_T,
                          receive.data(), receiveCounts.data(), receiveOffsets.data(), MPI_UINT64_T, route.Communicator());
            return receive;
        }
        RowLayout InferredLayout(const RowTransferPlan &route, const RowLayout &source,
                                 const std::vector<uint64_t> &received, TransferLayout policy)
        {
            if (auto table = policy.TableLayout(source, route.DestinationRows()))
            {
                for (auto width : received)
                    DNDS_check_throw_info(width == uint64_t(table->Width()), "received row does not fit transfer destination table");
                return *table;
            }
            std::vector<size_t> lengths(size_t(route.DestinationRows()), SIZE_MAX);
            size_t edge = 0;
            bool valid = true;
            for (const auto &peer : route.ReceiveRows())
                for (auto row : peer)
                {
                    valid = valid && (lengths[row] == SIZE_MAX || lengths[row] == received[edge]);
                    lengths[row] = received[edge++];
                }
            DNDS_check_throw_info(valid, "fan-in contributions have incompatible row widths");
            for (auto &n : lengths)
                if (n == SIZE_MAX)
                    n = 0;
            return policy.Apply(source, lengths);
        }
    }

    struct RowTransferPlan::State
    {
        std::shared_ptr<CommunicationContext> context;
        MPI_Comm comm = MPI_COMM_NULL;
        index sources = 0, destinations = 0;
        std::vector<std::vector<index>> send, receive;
        bool collisions = false, reverseCollisions = false;
        std::vector<VersionStamp> dependencies;
    };

    RowTransferPlan::RowTransferPlan(MPI_Comm comm, index sourceRows, index destinationRows,
                                     const std::vector<RowRequest> &requests, std::vector<VersionStamp> dependencies)
        : RowTransferPlan(std::make_shared<CommunicationContext>(comm), sourceRows, destinationRows, requests, std::move(dependencies)) {}

    RowTransferPlan::RowTransferPlan(std::shared_ptr<CommunicationContext> context, index sourceRows, index destinationRows,
                                     const std::vector<RowRequest> &requests, std::vector<VersionStamp> dependencies)
    {
        DNDS_check_throw_info(context, "row transfer needs a communication context");
        const auto comm = context->Communicator();
        auto state = std::make_shared<State>();
        state->context = std::move(context);
        state->comm = comm;
        int size = 0;
        MPI_Comm_size(comm, &size);
        bool valid = sourceRows >= 0 && destinationRows >= 0 && requests.size() <= INT_MAX;
        for (auto request : requests)
            valid = valid && request.sourceRank >= 0 && request.sourceRank < size && request.sourceRow >= 0 &&
                    request.destinationRow >= 0 && request.destinationRow < destinationRows;
        state->context->Preflight(valid, "invalid row transfer request");
        state->sources = sourceRows;
        state->destinations = destinationRows;
        state->dependencies = std::move(dependencies);
        state->send.resize(size);
        state->receive.resize(size);
        std::vector<std::vector<std::pair<index, index>>> grouped(size);
        SeenRows targets(destinationRows, requests.size());
        for (auto request : requests)
        {
            grouped[request.sourceRank].emplace_back(request.sourceRow, request.destinationRow);
            targets.Append(request.destinationRow);
        }
        state->collisions = targets.HasDuplicates();
        // Pair order defines the positional direct-MPI correspondence. Sorting
        // each peer's pairs by source row keeps both endpoints aligned while
        // collapsing contiguous source rows into one hindexed type segment.
        for (auto &peer : grouped)
            std::sort(peer.begin(), peer.end());
        std::vector<std::vector<int64_t>> requested(size);
        for (size_t peer = 0; peer < grouped.size(); ++peer)
            for (const auto &[sourceRow, destinationRow] : grouped[peer])
            {
                requested[peer].push_back(int64_t(sourceRow));
                state->receive[peer].push_back(destinationRow);
            }
        auto requestCounts = Counts(state->receive);
        std::vector<int> incomingCounts(size);
        MPI_Alltoall(requestCounts.data(), 1, MPI_INT, incomingCounts.data(), 1, MPI_INT, state->comm);
        auto incomingTotal = std::accumulate(incomingCounts.begin(), incomingCounts.end(), int64_t(0));
        state->context->Preflight(incomingTotal <= INT_MAX, "row transfer request table exceeds MPI int range");
        SeenRows sourcesSeen(sourceRows, size_t(incomingTotal));
        auto requestOffsets = Offsets(requestCounts), incomingOffsets = Offsets(incomingCounts);
        std::vector<int64_t> outgoing, incoming(size_t(incomingTotal), 0);
        for (const auto &peer : requested)
            outgoing.insert(outgoing.end(), peer.begin(), peer.end());
        MPI_Alltoallv(outgoing.data(), requestCounts.data(), requestOffsets.data(), MPI_INT64_T,
                      incoming.data(), incomingCounts.data(), incomingOffsets.data(), MPI_INT64_T, state->comm);
        for (int peer = 0; peer < size; ++peer)
            for (int j = incomingOffsets[peer]; j < incomingOffsets[peer + 1]; ++j)
            {
                valid = valid && incoming[j] >= 0 && uint64_t(incoming[j]) < sourceRows;
                state->send[peer].push_back(index(incoming[j]));
                sourcesSeen.Append(index(incoming[j]));
            }
        state->reverseCollisions = sourcesSeen.HasDuplicates();
        state->context->Preflight(valid, "row transfer source index does not exist");
        _state = std::move(state);
    }
    RowTransferPlan RowTransferPlan::Known(std::shared_ptr<CommunicationContext> context, index sourceRows, index destinationRows,
                                           std::vector<std::vector<index>> send, std::vector<std::vector<index>> receive,
                                           std::vector<VersionStamp> dependencies)
    {
        DNDS_check_throw_info(context, "known route needs a communication context");
        const auto comm = context->Communicator();
        int size = 0;
        MPI_Comm_size(comm, &size);
        bool valid = sourceRows >= 0 && destinationRows >= 0 &&
                     send.size() == size_t(size) && receive.size() == size_t(size);
        size_t ns = 0, nr = 0;
        for (const auto &peer : send)
        {
            valid = valid && peer.size() <= INT_MAX;
            ns = CheckedSize::Add(ns, peer.size());
            for (auto row : peer)
                valid = valid && row >= 0 && row < sourceRows;
        }
        for (const auto &peer : receive)
        {
            valid = valid && peer.size() <= INT_MAX;
            nr = CheckedSize::Add(nr, peer.size());
            for (auto row : peer)
                valid = valid && row >= 0 && row < destinationRows;
        }
        context->Preflight(valid && ns <= INT_MAX && nr <= INT_MAX, "invalid known row route");
        auto sc = Counts(send), rc = Counts(receive);
        std::vector<int> actual(size);
        MPI_Alltoall(sc.data(), 1, MPI_INT, actual.data(), 1, MPI_INT, comm);
        context->Preflight(actual == rc, "known row route peer counts disagree");
        auto state = std::make_shared<State>();
        state->context = std::move(context);
        state->comm = comm;
        state->sources = sourceRows;
        state->destinations = destinationRows;
        state->send = std::move(send);
        state->receive = std::move(receive);
        state->dependencies = std::move(dependencies);
        SeenRows sourceSeen(sourceRows, ns), destinationSeen(destinationRows, nr);
        for (const auto &peer : state->send)
            for (auto row : peer)
                sourceSeen.Append(row);
        for (const auto &peer : state->receive)
            for (auto row : peer)
                destinationSeen.Append(row);
        state->reverseCollisions = sourceSeen.HasDuplicates();
        state->collisions = destinationSeen.HasDuplicates();
        RowTransferPlan result;
        result._state = std::move(state);
        return result;
    }
    RowTransferPlan RowTransferPlan::Reverse() const
    {
        auto result = *this;
        result._reverse = !result._reverse;
        return result;
    }
    std::pair<RowTransferPlan, std::vector<index>> RowTransferPlan::ReorderSourceByPeer() const
    {
        index sent = 0;
        for (const auto &peer : SendRows())
            sent += index(peer.size());
        // The immutable route already proved bounds and collision flags.
        // Unique in-range rows whose count equals the source extent are a
        // complete permutation; do not sort/check the same indices again.
        const bool sourceCollisions = _reverse ? _state->collisions : _state->reverseCollisions;
        int controls[]{int(Valid() && !sourceCollisions && sent == SourceRows()), int(_reverse), -int(_reverse)};
        MPI_Allreduce(MPI_IN_PLACE, controls, 3, MPI_INT, MPI_MIN, Communicator());
        MPI::CheckPreflight(controls[0] && controls[1] == -controls[2],
                            "sender ordering requires uniform route direction and each source row exactly once");
        std::shared_ptr<State> state;
        std::vector<index> oldToNew;
        bool valid = true;
        try
        {
            state = std::make_shared<State>();
            state->context = Context();
            state->comm = Communicator();
            state->sources = SourceRows();
            state->destinations = DestinationRows();
            state->send = SendRows();
            state->receive = ReceiveRows();
            state->collisions = HasCollisions();
            state->dependencies = _state->dependencies;
            oldToNew.resize(size_t(SourceRows()));
            index next = 0;
            for (auto &peer : state->send)
                for (auto &row : peer)
                {
                    oldToNew[size_t(row)] = next;
                    row = next++;
                }
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("MPI row transfer preparation failed");
            valid = false;
        }
        Context()->Preflight(valid, "cannot allocate sender-ordered route");
        RowTransferPlan result;
        result._state = std::move(state);
        return {std::move(result), std::move(oldToNew)};
    }
    MPI_Comm RowTransferPlan::Communicator() const { return _state->comm; }
    std::shared_ptr<CommunicationContext> RowTransferPlan::Context() const { return _state->context; }
    index RowTransferPlan::SourceRows() const { return _reverse ? _state->destinations : _state->sources; }
    index RowTransferPlan::DestinationRows() const { return _reverse ? _state->sources : _state->destinations; }
    const std::vector<std::vector<index>> &RowTransferPlan::SendRows() const { return _reverse ? _state->receive : _state->send; }
    const std::vector<std::vector<index>> &RowTransferPlan::ReceiveRows() const { return _reverse ? _state->send : _state->receive; }
    bool RowTransferPlan::HasCollisions() const { return _reverse ? _state->reverseCollisions : _state->collisions; }
    bool RowTransferPlan::Valid() const
    {
        for (const auto &dependency : _state->dependencies)
            if (!dependency.Valid())
                return false;
        return true;
    }
    RowLayout RowTransferPlan::DestinationLayout(const RowLayout &source, TransferLayout layout) const
    {
        const auto lengths = Detail::ExchangeLengths(*this, source);
        std::optional<RowLayout> destination;
        try
        {
            destination = Detail::InferredLayout(*this, source, lengths, layout);
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("MPI row transfer preparation failed");
        }
        Context()->Preflight(bool(destination), "cannot infer destination row layout");
        return *destination;
    }
}
