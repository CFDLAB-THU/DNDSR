#include "DNDS/Runtime/Communication/RowTransfer_Route.hpp"
#include "DNDS/Runtime/Communication/DevicePacking.hpp"
#include "DNDS/Runtime/Communication/TransferAccess.hpp"
#include "DNDS/Runtime/Communication/CommunicationWorker.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "DNDS/MPI/Failure.hpp"
#include <algorithm>
#include <climits>
#include <optional>
#include <mutex>
#include <atomic>
#include <thread>

namespace DNDS::Runtime
{
    namespace
    {
        void Agree(const std::string &value, MPI_Comm comm, bool valid = true, index *uniformWidth = nullptr)
        {
            int rank = 0;
            MPI_Comm_rank(comm, &rank);
            // Root publishes an error sentinel instead of narrowing an invalid
            // length. Non-root oversized values fail the final exact comparison.
            int size = rank == 0 ? (value.size() <= INT_MAX ? int(value.size()) : -1) : 0;
            MPI_Bcast(&size, 1, MPI_INT, 0, comm);
            DNDS_check_throw_info(size >= 0, "transfer schema identifier too long");
            std::string reference(size, '\0');
            if (rank == 0)
                reference = value;
            MPI_Bcast(reference.data(), size, MPI_CHAR, 0, comm);
            // Payload construction can prove a common table width in this
            // existing agreement. Unknown/mixed shapes use -1 and retain the
            // ordinary per-edge width exchange; no new collective is added.
            int64_t controls[]{valid && value == reference, uniformWidth ? *uniformWidth : -1,
                               uniformWidth ? -*uniformWidth : 1};
            MPI_Allreduce(MPI_IN_PLACE, controls, uniformWidth ? 3 : 1, MPI_INT64_T, MPI_MIN, comm);
            MPI::CheckPreflight(controls[0], "transfer schema/policy or source layout/epoch differs between ranks");
            if (uniformWidth)
                *uniformWidth = controls[1] >= 0 && controls[1] == -controls[2] ? controls[1] : -1;
        }
        bool MPIAlive()
        {
            int final = 0;
            MPI_Finalized(&final);
            return !final;
        }
        struct Datatype
        {
            MPI_Datatype value = MPI_DATATYPE_NULL;
            ~Datatype()
            {
                if (value != MPI_DATATYPE_NULL && MPIAlive())
                    MPI_Type_free(&value);
            }
        };
        using Type = std::shared_ptr<Datatype>;

        Type MakeType(const std::vector<index> &rows, const RowLayout &layout, const ElementSchema &schema, size_t &segments, Type &emptyType)
        {
            std::vector<int> lengths;
            std::vector<MPI_Aint> displacements;
            const auto extent = schema.Extent();
            // Row-major logical wire order is independent of either endpoint's
            // physical format. Non-unit component stride contributes individual
            // elements; adjacent blocks are merged before committing MPI types.
            // No payload staging is introduced for ordinary copies.
            auto append = [&](index offset, index count)
            {
                while (count)
                {
                    const auto n = int(std::min(count, index(INT_MAX)));
                    DNDS_check_throw_info(offset >= 0 && size_t(offset) <=
                                                             size_t(std::numeric_limits<MPI_Aint>::max()) / extent,
                                          "MPI displacement overflow");
                    const auto displacement = MPI_Aint(size_t(offset) * extent);
                    if (!lengths.empty() && n <= INT_MAX - lengths.back() &&
                        displacement - displacements.back() == MPI_Aint(lengths.back()) * MPI_Aint(extent))
                        lengths.back() += n;
                    else
                    {
                        lengths.push_back(n);
                        displacements.push_back(displacement);
                    }
                    offset += n;
                    count -= n;
                }
            };
            for (auto iRow : rows)
                if (layout.IsContiguousRows())
                    append(layout.Offset(iRow), layout.RowSize(iRow));
                else
                    for (index iComp = 0; iComp < layout.RowSize(iRow); ++iComp)
                        append(layout.ElementOffset(iRow, iComp), 1);
            DNDS_check_throw_info(lengths.size() <= INT_MAX, "MPI datatype has too many segments");
            segments += lengths.size();
            // Empty peers share one committed zero-payload type per plan. A
            // sparse halo must not compile three identical types per idle rank.
            if (lengths.empty() && emptyType)
                return emptyType;
            auto type = std::make_shared<Datatype>();
            MPI_Type_create_hindexed(int(lengths.size()), lengths.data(), displacements.data(), schema.Type(), &type->value);
            MPI_Type_commit(&type->value);
            if (lengths.empty())
                emptyType = type;
            return type;
        }
    }

    struct PayloadPlan::State
    {
        RowTransferPlan route;
        RowLayout source, destination;
        ElementSchema schema;
        std::vector<Type> sendTypes, receiveTypes;
        size_t sendSegments = 0, receiveSegments = 0;
        State(RowTransferPlan r, RowLayout s, RowLayout d, ElementSchema e)
            : route(std::move(r)), source(std::move(s)), destination(std::move(d)), schema(std::move(e)) {}

        struct ReductionLayout
        {
            RowLayout contributions;
            std::vector<Type> receiveTypes;
            std::vector<Contribution> groups;
        };
        /// Reduction-only packing is a local derivation from the checked edge
        /// widths. Ordinary halo/migration copies must not allocate a second
        /// CSR, sort contribution groups or commit unused reduction datatypes.
        ReductionLayout CompileReduction() const
        {
            ReductionLayout result;
            std::vector<size_t> widths;
            for (const auto &peer : route.ReceiveRows())
                for (auto iRow : peer)
                    widths.push_back(size_t(destination.RowSize(iRow)));
            result.contributions = RowLayout::Ragged(widths);
            index edge = 0;
            Type emptyType;
            for (const auto &peer : route.ReceiveRows())
            {
                std::vector<index> slots;
                slots.reserve(peer.size());
                for (auto iRow : peer)
                {
                    slots.push_back(edge);
                    if (widths[edge] && destination.IsContiguousRows())
                        result.groups.push_back({size_t(result.contributions.Offset(edge)),
                                                 size_t(destination.Offset(iRow)), widths[edge]});
                    else
                        // Custom reducers keep their contiguous contribution
                        // contract. A strided destination is split into scalar
                        // contributions, sorted/grouped by actual destination.
                        for (index iComp = 0; iComp < index(widths[edge]); ++iComp)
                            result.groups.push_back({size_t(result.contributions.Offset(edge) + iComp),
                                                     size_t(destination.ElementOffset(iRow, iComp)), 1});
                    ++edge;
                }
                size_t segments = 0;
                result.receiveTypes.push_back(MakeType(slots, result.contributions, schema, segments, emptyType));
            }
            std::stable_sort(result.groups.begin(), result.groups.end(), [](auto a, auto b)
                             { return a.destinationElement < b.destinationElement; });
            return result;
        }
    };
    PayloadPlan::PayloadPlan(RowTransferPlan route, RowLayout source, RowLayout destination, ElementSchema schema)
        : PayloadPlan(std::move(route), std::move(source), std::optional<RowLayout>(std::move(destination)), std::move(schema), {}) {}
    PayloadPlan::PayloadPlan(RowTransferPlan route, RowLayout source, ElementSchema schema, TransferLayout layout)
        : PayloadPlan(std::move(route), std::move(source), std::nullopt, std::move(schema), layout) {}
    PayloadPlan::PayloadPlan(RowTransferPlan route, RowLayout source, std::optional<RowLayout> destination, ElementSchema schema, TransferLayout layout)
    {
        auto state = std::make_shared<State>(std::move(route), std::move(source), destination.value_or(RowLayout{}), std::move(schema));
        const auto comm = state->route.Communicator();
        // Both checks precede the same width exchange. Carry source validity in
        // exact schema agreement instead of a second consecutive Allreduce.
        index uniformWidth = state->source.Kind() == RowLayoutKind::Table ? state->source.Width() : -1;
        Agree(state->schema.Identity() + ":" + state->schema.Definition(), comm,
              state->source.RowCount() == state->route.SourceRows() && state->route.Valid(), &uniformWidth);
        std::vector<uint64_t> lengths;
        if (uniformWidth >= 0)
        {
            size_t count = 0;
            for (const auto &peer : state->route.ReceiveRows())
                count = CheckedSize::Add(count, peer.size());
            lengths.assign(count, uint64_t(uniformWidth));
        }
        else
            lengths = Detail::ExchangeLengths(state->route, state->source, true);
        bool valid = true;
        try
        {
            if (!destination)
                state->destination = Detail::InferredLayout(state->route, state->source, lengths, layout);
            valid = state->destination.RowCount() == state->route.DestinationRows();
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("MPI row transfer preparation failed");
            valid = false;
        }
        // Everything below is local (MPI datatype constructors are not
        // collective). Accumulate failures and agree once before publication,
        // instead of three collective barriers around the same local work.
        try
        {
            if (valid)
            {
                size_t edge = 0;
                for (const auto &peer : state->route.ReceiveRows())
                    for (auto row : peer)
                        valid = (state->destination.RowSize(row) == lengths[edge++]) && valid;
            }
            if (valid)
            {
                Type emptyType;
                for (size_t peer = 0; peer < state->route.SendRows().size(); ++peer)
                {
                    state->sendTypes.push_back(MakeType(state->route.SendRows()[peer], state->source, state->schema, state->sendSegments, emptyType));
                    // Colliding final destinations are never used as receive types.
                    if (!state->route.HasCollisions())
                        state->receiveTypes.push_back(MakeType(state->route.ReceiveRows()[peer], state->destination, state->schema, state->receiveSegments, emptyType));
                }
            }
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("MPI row transfer preparation failed");
            valid = false;
        }
        state->route.Context()->Preflight(valid, "row payload destination/edge width mismatch or MPI layout compilation failure");
        _state = std::move(state);
    }
    RowLayout PayloadPlan::DestinationLayout() const { return _state->destination; }
    size_t PayloadPlan::SendSegments() const { return _state->sendSegments; }
    size_t PayloadPlan::ReceiveSegments() const { return _state->receiveSegments; }
    bool PayloadPlan::Compatible(const std::string &type) const { return _state->schema.Compatible(type); }
    void PayloadPlan::CheckLayouts(const RowLayout &source, const RowLayout &destination, size_t bytes, bool elementCompatible) const
    {
        _state->route.Context()->Preflight(LayoutsMatch(source, destination, bytes, elementCompatible), "Array layout differs from compiled payload");
    }
    bool PayloadPlan::LayoutsMatch(const RowLayout &source, const RowLayout &destination, size_t bytes, bool elementCompatible) const
    {
        return elementCompatible && source == _state->source && destination == _state->destination && bytes == _state->schema.Extent();
    }

    struct TransferBinding::State
    {
        std::shared_ptr<const PayloadPlan::State> plan;
        std::optional<PayloadPlan::State::ReductionLayout> reductionLayout;
        Buffer source, destination, workspace, deviceGroups;
        Buffer hostSource, hostDestination;
        // Private fixed-capacity staging and numeric metadata. One active
        // operation owns these; no host payload mirror or MPI_Pack is used.
        Buffer deviceSend, deviceReceive, packRows, unpackRows, packFields;
        std::vector<index> sendElements, receiveElements;
        index packRowCount = 0, unpackRowCount = 0, fieldCount = 0;
        void *PackedPointer(bool receive, int peer) const
        {
            auto *base = static_cast<uint8_t *>((receive ? deviceReceive : deviceSend).LegacyPointer());
            const auto offset = (receive ? receiveElements : sendElements)[peer];
            return base ? base + size_t(offset) * plan->schema.Extent() : nullptr;
        }
        int PackedCount(bool receive, int peer) const
        {
            const auto &offsets = receive ? receiveElements : sendElements;
            return int(offsets[peer + 1] - offsets[peer]);
        }
        TransferTransport transport = TransferTransport::Direct;
        TransferOptions options;
        std::shared_ptr<const int> tag;
        std::vector<std::vector<uint8_t>> packedSend, packedReceive;
        std::vector<int> packedSendSizes;
        std::shared_ptr<const Reducer> reducer;
        MPI_Comm comm = MPI_COMM_NULL;
        std::vector<MPI_Request> sends, receives;
        bool active = false;
        bool emptyDirectPayload = false;
        bool collectivePreflight = true;
        std::mutex mutex;
        std::shared_ptr<AccessTrackingContext> tracking;
        uint64_t lane = 0;
        void Preflight(bool valid, const char *message) const
        {
            CommunicationDetail::Preflight(valid, comm, collectivePreflight, message);
        }
        ~State()
        {
            if (tracking)
                tracking->RetireLane(lane);
            if (MPIAlive())
            {
                for (auto &r : sends)
                    if (r != MPI_REQUEST_NULL)
                        MPI_Request_free(&r);
                for (auto &r : receives)
                    if (r != MPI_REQUEST_NULL)
                        MPI_Request_free(&r);
                if (comm != MPI_COMM_NULL && !options.context)
                    MPI_Comm_free(&comm);
            }
        }
    };
    TransferBinding PayloadPlan::Bind(Buffer source, Buffer destination, TransferTransport transport, TransferOptions options) const
    {
        return TransferBinding(_state, std::move(source), std::move(destination), {}, transport, std::move(options));
    }
    TransferBinding PayloadPlan::Bind(Buffer source, Buffer destination, Reducer reducer, TransferTransport transport, TransferOptions options) const
    {
        return TransferBinding(_state, std::move(source), std::move(destination), std::make_shared<const Reducer>(std::move(reducer)), transport, std::move(options));
    }
    TransferBinding::TransferBinding(std::shared_ptr<const PayloadPlan::State> plan, Buffer source, Buffer destination,
                                     std::shared_ptr<const Reducer> reducer, TransferTransport transport, TransferOptions options,
                                     bool layoutsValid)
    {
        auto state = std::make_shared<State>();
        state->plan = std::move(plan);
        state->source = std::move(source);
        state->destination = std::move(destination);
        state->reducer = std::move(reducer);
        state->transport = transport;
        const auto emptyRows = [](const auto &rows)
        { return std::all_of(rows.begin(), rows.end(), [](const auto &row)
                             { return row.empty(); }); };
        state->emptyDirectPayload = transport == TransferTransport::Direct && !state->reducer &&
                                    emptyRows(state->plan->route.SendRows()) && emptyRows(state->plan->route.ReceiveRows());
        state->options = std::move(options);
        state->collectivePreflight = (state->options.context ? state->options.context : state->plan->route.Context())->ChecksCollectivePreflight();
        if (state->options.preflight != TransferPreflight::Inherit)
            state->collectivePreflight = state->options.preflight == TransferPreflight::Collective;
        const auto routeComm = state->plan->route.Communicator();
        // Fixed control fields need no string broadcast. Min(x,-x) proves
        // exact agreement in one collective; widen before negating INT_MIN.
        const int64_t controls[]{int(state->options.mode), bool(state->options.context), state->options.tag,
                                 bool(state->reducer), int(transport),
                                 state->reducer ? int(state->reducer->initialization) : 0, bool(state->options.provider), state->options.allowStreamOrderedDeviceMemory,
                                 state->collectivePreflight, int(state->options.preflight), bool(state->options.communicationWorker), state->options.isolatedNamespace};
        constexpr index controlNumber = sizeof(controls) / sizeof(controls[0]);
        int64_t agreed[2 * controlNumber + 2];
        for (index i = 0; i < controlNumber; ++i)
        {
            agreed[2 * i] = controls[i];
            agreed[2 * i + 1] = -controls[i];
        }
        agreed[2 * controlNumber] = layoutsValid;
        agreed[2 * controlNumber + 1] = state->source.GetDevice().space != MemorySpace::SYCL &&
                                        state->destination.GetDevice().space != MemorySpace::SYCL;
        MPI_Allreduce(MPI_IN_PLACE, agreed, int(2 * controlNumber + 2), MPI_INT64_T, MPI_MIN, routeComm);
        bool controlsAgree = true;
        for (index i = 0; i < controlNumber; ++i)
            controlsAgree = controlsAgree && agreed[2 * i] == -agreed[2 * i + 1];
        MPI::CheckPreflight(controlsAgree, "transfer binding policy differs between ranks");
        MPI::CheckPreflight(agreed[2 * controlNumber], "Array layout differs from compiled payload");
        MPI::CheckPreflight(agreed[2 * controlNumber + 1], "SYCL device halo transfer is unsupported");
        if (state->options.communicationWorker)
            CommunicationDetail::Preflight(!state->options.provider && state->source.GetDevice() == Device::Host() &&
                                               state->destination.GetDevice() == Device::Host() &&
                                               (transport == TransferTransport::Direct || transport == TransferTransport::HostPacked),
                                           routeComm, true, "communication worker supports CPU MPI Direct/HostPacked only");
        DNDS_check_throw_info(state->options.preflight == TransferPreflight::Inherit ||
                                  state->options.preflight == TransferPreflight::Collective ||
                                  state->options.preflight == TransferPreflight::AbortOnFailure,
                              "invalid MPI transfer preflight policy");
        if (state->options.provider)
        {
            Agree(state->options.provider->Name(), routeComm);
            const auto &p = *state->plan;
            _provider = state->options.provider->Bind({p.route, p.source, p.destination, p.schema}, state->source,
                                                      state->destination, state->reducer, transport, state->options, layoutsValid);
            DNDS_check_throw_info(_provider, "transfer provider returned no binding");
            return;
        }
        if (state->reducer)
            Agree(state->reducer->identity, routeComm);
        CommunicationDetail::Preflight(transport != TransferTransport::Direct || state->options.allowStreamOrderedDeviceMemory ||
                                           (!state->source.StreamOrderedMemory() && !state->destination.StreamOrderedMemory()),
                                       routeComm, state->collectivePreflight,
                                       "Direct MPI pool/VMM memory requires verified allowStreamOrderedDeviceMemory; CUDA awareness alone is insufficient. Select explicit packing or ordinary preallocated device storage.");
        bool valid = true;
        const bool inherited = !state->options.context && !state->options.isolatedNamespace;
        if (inherited)
        {
            CommunicationDetail::Preflight(state->options.tag == 0 && !state->options.channel.empty(), routeComm,
                                           state->collectivePreflight, "inherited transfer needs a named channel and no explicit tag");
            Agree(state->options.channel, routeComm);
            state->options.context = state->plan->route.Context();
            state->tag = state->options.context->ReserveAgreed();
            state->options.tag = *state->tag;
        }
        CommunicationDetail::Preflight(!state->options.isolatedNamespace || !state->options.context, routeComm,
                                       state->collectivePreflight, "isolated transfer cannot also share a context");
        if (state->options.context)
        {
            int comparison = MPI_UNEQUAL;
            MPI_Comm_compare(routeComm, state->options.context->Communicator(), &comparison);
            valid = comparison == MPI_IDENT || comparison == MPI_CONGRUENT;
            state->comm = state->options.context->Communicator();
            try
            {
                if (valid && !inherited)
                    state->tag = state->options.context->Reserve(state->options.tag);
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("MPI row transfer preparation failed");
                valid = false;
            }
        }
        else
        {
            CommunicationDetail::Preflight(state->options.tag == 0, routeComm, state->collectivePreflight, "isolated transfer binding uses tag zero");
            MPI_Comm_dup(routeComm, &state->comm);
        }
        const auto &p = *state->plan;
        valid = valid && state->source.Identity() && state->destination.Identity() && state->source.Identity() != state->destination.Identity();
        valid = valid && state->source.Bytes() / p.schema.Extent() == p.source.Elements() && state->source.Bytes() % p.schema.Extent() == 0;
        valid = valid && state->destination.Bytes() / p.schema.Extent() == p.destination.Elements() && state->destination.Bytes() % p.schema.Extent() == 0;
        valid = valid && (!p.route.HasCollisions() || state->reducer);
        const bool gpu = state->source.GetDevice().space == MemorySpace::CUDA || state->destination.GetDevice().space == MemorySpace::CUDA;
        valid = valid && (transport == TransferTransport::Direct || transport == TransferTransport::HostPacked || transport == TransferTransport::DevicePacked);
        valid = valid && (transport == TransferTransport::HostPacked || !gpu || MPI::isCudaAware());
        if (transport == TransferTransport::DevicePacked)
        {
#ifndef DNDS_USE_CUDA
            valid = false;
#endif
            valid = valid && state->source.GetDevice().space == MemorySpace::CUDA && state->source.GetDevice() == state->destination.GetDevice() &&
                    !p.schema.Fields().empty() && p.schema.Extent() <= size_t(INT64_MAX) &&
                    state->source.Bytes() <= size_t(INT64_MAX) && state->destination.Bytes() <= size_t(INT64_MAX);
        }
        if (state->reducer)
            valid = valid && state->reducer->schema == p.schema.Identity() && state->reducer->elementBytes == p.schema.Extent() &&
                    (transport == TransferTransport::HostPacked || state->destination.GetDevice().space == MemorySpace::Host ? bool(state->reducer->host) : bool(state->reducer->device));
        // A rejected context can have a different group/order. Validate on the
        // route communicator; disabled safety agreement fails fast there too.
        CommunicationDetail::Preflight(valid, routeComm, state->collectivePreflight, transport == TransferTransport::DevicePacked ? "DevicePacked requires CUDA-aware MPI, one CUDA placement, a described schema, valid extents and collision resolution" : "invalid transfer binding: context/tag, collision, schema, placement, size or alias");
        try
        {
            if (state->reducer)
            {
                state->reductionLayout.emplace(p.CompileReduction());
                const auto &reduction = *state->reductionLayout;
                state->workspace = Buffer::Allocate(CheckedSize::Multiply(reduction.contributions.Elements(), p.schema.Extent()),
                                                    transport == TransferTransport::HostPacked ? Device::Host() : state->destination.GetDevice(), AllocationOptions{p.schema.Alignment()});
                if (transport != TransferTransport::HostPacked && state->destination.GetDevice().space == MemorySpace::CUDA)
                {
                    auto host = Buffer::Allocate(CheckedSize::Multiply(reduction.groups.size(), sizeof(Contribution)));
                    {
                        auto w = host.Write();
                        std::copy(reduction.groups.begin(), reduction.groups.end(), w.View<Contribution>().data);
                    }
                    state->deviceGroups = Buffer::Allocate(host.Bytes(), state->destination.GetDevice());
                    Copy(host, state->deviceGroups, {state->destination.GetDevice(), nullptr}).Wait();
                }
            }
            if (transport == TransferTransport::DevicePacked)
            {
                const auto device = state->source.GetDevice();
                auto upload = [device](const auto &values)
                {
                    using T = typename std::decay_t<decltype(values)>::value_type;
                    auto host = Buffer::Allocate(CheckedSize::Multiply(values.size(), sizeof(T)));
                    {
                        auto write = host.Write();
                        std::copy(values.begin(), values.end(), write.template View<T>().data);
                    }
                    auto result = Buffer::Allocate(host.Bytes(), device);
                    Copy(host, result, {device, nullptr}).Wait();
                    return result;
                };
                std::vector<Detail::DevicePackField> fields;
                for (const auto &field : p.schema.Fields())
                    fields.push_back({index(field.offset), index(CheckedSize::Multiply(field.count, ElementSchema::ScalarBytes(field.scalar)))});
                state->fieldCount = index(fields.size());
                state->packFields = upload(fields);
                auto compileRows = [&](const auto &peers, const RowLayout &layout, Buffer &metadata, index &rowCount, std::vector<index> &offsets, bool allocate = true)
                {
                    std::vector<Detail::DevicePackRow> rows;
                    offsets.push_back(0);
                    for (const auto &peer : peers)
                    {
                        index next = offsets.back();
                        for (auto iRow : peer)
                        {
                            const auto width = layout.RowSize(iRow);
                            DNDS_check_throw_info(width <= INT_MAX && next <= INT64_MAX - width, "device-packed logical extent overflow");
                            rows.push_back({width ? layout.ElementOffset(iRow, 0) : 0, layout.ComponentStride(), next, width});
                            next += width;
                        }
                        DNDS_check_throw_info(next - offsets.back() <= INT_MAX, "device-packed peer exceeds MPI element count");
                        offsets.push_back(next);
                    }
                    rowCount = index(rows.size());
                    metadata = upload(rows);
                    const auto bytes = CheckedSize::Multiply(size_t(offsets.back()), p.schema.Extent());
                    DNDS_check_throw_info(bytes <= size_t(INT64_MAX), "device-packed byte extent exceeds signed indexing");
                    return allocate ? Buffer::Allocate(bytes, device, AllocationOptions{p.schema.Alignment()}) : Buffer{};
                };
                state->deviceSend = compileRows(p.route.SendRows(), p.source, state->packRows, state->packRowCount, state->sendElements);
                // Reduction contributions already use peer-major contiguous rows;
                // retain their order and combine only after receive completion.
                state->deviceReceive = compileRows(p.route.ReceiveRows(), p.destination, state->unpackRows, state->unpackRowCount, state->receiveElements, !state->reducer);
                if (state->reducer)
                    state->deviceReceive = state->workspace;
            }
            if (transport == TransferTransport::HostPacked)
            {
                state->hostSource = Buffer::Allocate(state->source.Bytes());
                state->hostDestination = Buffer::Allocate(state->destination.Bytes());
                const size_t peers = p.route.SendRows().size();
                state->packedSend.resize(peers);
                state->packedReceive.resize(peers);
                state->packedSendSizes.resize(peers);
                for (size_t peer = 0; peer < peers; ++peer)
                {
                    auto receiveType = state->reducer ? state->reductionLayout->receiveTypes[peer] : p.receiveTypes[peer];
                    MPI_Count sendBytes = 0, receiveBytes = 0;
                    MPI_Type_size_x(p.sendTypes[peer]->value, &sendBytes);
                    MPI_Type_size_x(receiveType->value, &receiveBytes);
                    DNDS_check_throw_info(sendBytes < INT_MAX && receiveBytes < INT_MAX, "host-packed peer payload exceeds MPI int count");
                    int sendSize = 0, receiveSize = 0;
                    MPI_Pack_size(1, p.sendTypes[peer]->value, state->comm, &sendSize);
                    MPI_Pack_size(1, receiveType->value, state->comm, &receiveSize);
                    state->packedSend[peer].resize(std::max(sendSize, 1));
                    state->packedReceive[peer].resize(std::max(receiveSize, 1));
                    int position = 0;
                    if (sendBytes)
                        MPI_Pack(state->hostSource.LegacyPointer(), 1, p.sendTypes[peer]->value,
                                 state->packedSend[peer].data(), int(state->packedSend[peer].size()), &position, state->comm);
                    state->packedSendSizes[peer] = position;
                }
            }
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("MPI row transfer preparation failed");
            valid = false;
        }
        // Copy/direct bindings do not allocate a workspace. Reducer/transport
        // policy was agreed above, so all ranks take the same branch here.
        if (state->reducer || transport != TransferTransport::Direct)
            state->Preflight(valid, "cannot allocate/compile transfer workspace");
        state->receives.reserve(p.route.ReceiveRows().size());
        state->sends.reserve(p.route.SendRows().size());
        auto *receiveBase = state->reducer ? state->workspace.LegacyPointer() : state->destination.LegacyPointer();
        if (state->options.mode == TransferMode::Persistent)
            for (int peer = 0; peer < int(p.route.SendRows().size()); ++peer)
            {
                if (!p.route.ReceiveRows()[peer].empty())
                {
                    state->receives.push_back(MPI_REQUEST_NULL);
                    auto type = state->reducer ? state->reductionLayout->receiveTypes[peer] : p.receiveTypes[peer];
                    if (transport == TransferTransport::Direct)
                        MPI_Recv_init(receiveBase, 1, type->value, peer, state->options.tag, state->comm, &state->receives.back());
                    else if (transport == TransferTransport::DevicePacked)
                        MPI_Recv_init(state->PackedPointer(true, peer), state->PackedCount(true, peer), p.schema.Type(), peer, state->options.tag, state->comm, &state->receives.back());
                    else
                        MPI_Recv_init(state->packedReceive[peer].data(), int(state->packedReceive[peer].size()), MPI_PACKED,
                                      peer, state->options.tag, state->comm, &state->receives.back());
                }
                if (!p.route.SendRows()[peer].empty())
                {
                    state->sends.push_back(MPI_REQUEST_NULL);
                    if (transport == TransferTransport::Direct)
                        MPI_Send_init(state->source.LegacyPointer(), 1, p.sendTypes[peer]->value, peer, state->options.tag, state->comm, &state->sends.back());
                    else if (transport == TransferTransport::DevicePacked)
                        MPI_Send_init(state->PackedPointer(false, peer), state->PackedCount(false, peer), p.schema.Type(), peer, state->options.tag, state->comm, &state->sends.back());
                    else
                        MPI_Send_init(state->packedSend[peer].data(), state->packedSendSizes[peer], MPI_PACKED,
                                      peer, state->options.tag, state->comm, &state->sends.back());
                }
            }
        _state = std::move(state);
    }

    using Detail::TransferAccess;
    struct TransferOperation::State
    {
        std::shared_ptr<TransferBinding::State> binding;
        TransferAccess source, destination, workspace;
        ExecutionContext context;
        bool started = false, destinationReady = false;
        bool sendsStarted = false, unpackStarted = false, packingSubmitted = false;
        Completion packedReady, unpackReady;
        std::exception_ptr error;
        AccessTrackingContext::Submission checked;
        std::shared_ptr<const AccessOrder> order;
        bool Prepare(bool scheduled, bool checkedAccess, bool callerValid);
        void CancelPrepared();
        void Post();
        void PackHost();
        ~State() { checked.Complete(); }
    };
    struct TransferOperation::WorkerState
    {
        // Controller retains the operation (including MPI resources). The task
        // only borrows it, and publishes completion under this mutex before its
        // final access. Wait/destruction never free MPI owners on the worker.
        std::unique_ptr<TransferOperation> operation;
        mutable std::mutex mutex;
        std::condition_variable ready;
        bool destinationReady = false, done = false;
        void Wait(bool destinationOnly)
        {
            std::unique_lock<std::mutex> lock(mutex);
            ready.wait(lock, [&]
                       { return destinationOnly ? destinationReady : done; });
        }
        bool Test() const
        {
            std::lock_guard<std::mutex> lock(mutex);
            return done;
        }
    };
    struct TransferOperation::ManagedState final : ExecutionProgress, StreamEvent::State
    {
        TransferOperation operation;
        std::shared_ptr<const AccessOrder> order;
        const std::thread::id controller = std::this_thread::get_id();
        std::atomic<bool> done{false};
        bool armed = false;
        void CheckController() const
        {
            DNDS_check_throw_info(std::this_thread::get_id() == controller, "managed MPI transfer needs its controller thread");
        }
        void Retire()
        {
            operation = {}; // MPI/provider owners are dropped only here.
            done.store(true, std::memory_order_release);
            NotifyCompletion();
        }
        bool Poll() override
        {
            CheckController();
            if (done.load(std::memory_order_acquire))
                return true;
            if (!armed)
                return false;
            if (!operation.Test())
                return false;
            Retire();
            return true;
        }
        bool Ready() const override
        {
            if (std::this_thread::get_id() == controller)
                return const_cast<ManagedState *>(this)->Poll();
            return done.load(std::memory_order_acquire);
        }
        void Wait() const override
        {
            if (done.load(std::memory_order_acquire))
                return;
            CheckController();
            auto &self = *const_cast<ManagedState *>(this);
            DNDS_check_throw_info(armed, "managed MPI operation was not submitted");
            self.operation.Wait();
            self.Retire();
        }
    };

    bool TransferOperation::State::Prepare(bool scheduled, bool checkedAccess, bool callerValid)
    {
        auto &_state = binding;
        auto *op = this;
        const auto context = this->context;
        bool valid = callerValid && _state->plan->route.Valid();
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            valid = valid && !_state->active;
        }
        try
        {
            DNDS_check_throw_info(valid, "transfer binding is active or routing is stale");
            auto ctx = [context](const Buffer &b)
            { return b.GetDevice() == Device::Host() ? ExecutionContext::Host() : context; };
            op->source.Acquire(_state->source, ctx(_state->source), false, scheduled, checkedAccess);
            op->destination.Acquire(_state->destination, ctx(_state->destination), true, scheduled, checkedAccess);
            if (_state->reducer)
                op->workspace.Acquire(_state->workspace, ctx(_state->workspace), true, scheduled, checkedAccess);
            // Producers must have released their write leases. This event also
            // orders prior work on the caller's stream before portable MPI.
            if (_state->transport == TransferTransport::DevicePacked)
            {
                op->packingSubmitted = true;
                Detail::LaunchDevicePacking(context, op->source.View<uint8_t>().data, _state->deviceSend.LegacyPointer(),
                                            static_cast<const Detail::DevicePackRow *>(_state->packRows.LegacyPointer()), _state->packRowCount,
                                            static_cast<const Detail::DevicePackField *>(_state->packFields.LegacyPointer()), _state->fieldCount, index(_state->plan->schema.Extent()), false);
                op->packedReady = Completion::Record(context);
            }
            else
                Completion::Record(context).Wait();
            if (!_state->options.communicationWorker)
                PackHost();
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("MPI row transfer preparation failed");
            valid = false;
        }

        return valid;
    }
    void TransferOperation::State::CancelPrepared()
    {
        auto *op = this;
        // Some peers may have queued packing before collective rejection.
        // Drain that read-only prefix before releasing the source lease.
        if (op->packingSubmitted)
            Completion::Record(context).Wait();
        op->source.CancelUnmodified();
        op->destination.CancelUnmodified();
        op->workspace.CancelUnmodified();
    }
    void TransferOperation::State::PackHost()
    {
        auto &_state = binding;
        auto *op = this;
        const auto context = this->context;
        if (_state->transport == TransferTransport::HostPacked)
        {
            auto stage = [context](const void *src, Device sd, void *dst, size_t bytes)
            { Detail::CopyBytes(src, sd, dst, Device::Host(), bytes, sd == Device::Host() ? ExecutionContext::Host() : context).Wait(); };
            stage(op->source.View<uint8_t>().data, _state->source.GetDevice(), _state->hostSource.LegacyPointer(), _state->source.Bytes());
            stage(op->destination.View<uint8_t>().data, _state->destination.GetDevice(), _state->hostDestination.LegacyPointer(), _state->destination.Bytes());
            for (size_t peer = 0; peer < _state->packedSend.size(); ++peer)
            {
                if (!_state->packedSendSizes[peer])
                    continue;
                int position = 0;
                MPI_Pack(_state->hostSource.LegacyPointer(), 1, _state->plan->sendTypes[peer]->value,
                         _state->packedSend[peer].data(), int(_state->packedSend[peer].size()), &position, _state->comm);
                DNDS_check_throw_info(position == _state->packedSendSizes[peer], "MPI packed payload size changed");
            }
        }
    }
    void TransferOperation::State::Post()
    {
        auto &_state = binding;
        auto *op = this;
        if (_state->options.communicationWorker)
            PackHost();
        if (_state->transport == TransferTransport::DevicePacked)
            op->source.Complete(op->packedReady);
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            _state->active = true;
        }
        if (_state->options.mode == TransferMode::Persistent)
        {
            if (!_state->receives.empty())
                MPI_Startall(int(_state->receives.size()), _state->receives.data());
            if (_state->transport != TransferTransport::DevicePacked && !_state->sends.empty())
                MPI_Startall(int(_state->sends.size()), _state->sends.data());
        }
        else
        {
            auto &b = *_state;
            const auto &p = *b.plan;
            b.receives.clear();
            b.sends.clear();
            for (int peer = 0; peer < int(p.route.SendRows().size()); ++peer)
            {
                if (!p.route.ReceiveRows()[peer].empty())
                {
                    b.receives.push_back(MPI_REQUEST_NULL);
                    auto type = b.reducer ? b.reductionLayout->receiveTypes[peer] : p.receiveTypes[peer];
                    if (b.transport == TransferTransport::Direct)
                        MPI_Irecv(b.reducer ? op->workspace.View<uint8_t>().data : op->destination.View<uint8_t>().data, 1,
                                  type->value, peer, b.options.tag, b.comm, &b.receives.back());
                    else if (b.transport == TransferTransport::DevicePacked)
                        MPI_Irecv(b.PackedPointer(true, peer), b.PackedCount(true, peer), p.schema.Type(), peer, b.options.tag, b.comm, &b.receives.back());
                    else
                        MPI_Irecv(b.packedReceive[peer].data(), int(b.packedReceive[peer].size()), MPI_PACKED,
                                  peer, b.options.tag, b.comm, &b.receives.back());
                }
                if (b.transport != TransferTransport::DevicePacked && !p.route.SendRows()[peer].empty())
                {
                    b.sends.push_back(MPI_REQUEST_NULL);
                    if (b.transport == TransferTransport::Direct)
                        MPI_Isend(op->source.View<uint8_t>().data, 1, p.sendTypes[peer]->value,
                                  peer, b.options.tag, b.comm, &b.sends.back());
                    else
                        MPI_Isend(b.packedSend[peer].data(), b.packedSendSizes[peer], MPI_PACKED,
                                  peer, b.options.tag, b.comm, &b.sends.back());
                }
            }
        }
        op->sendsStarted = _state->transport != TransferTransport::DevicePacked;
        op->started = true;
    }
    TransferOperation TransferBinding::Begin(ExecutionContext context) const
    {
        return Start(context, false);
    }
    TransferOperation TransferBinding::BeginScheduled(ExecutionContext context, bool callerValid) const
    {
        return Start(context, true, {}, 0, callerValid);
    }
    TransferOperation TransferBinding::BeginScheduled(const ExecutionStream &stream, bool callerValid) const
    {
        return StartManaged(stream, false, callerValid, true);
    }
    TransferOperation TransferBinding::BeginChecked(const ExecutionStream &stream, bool callerValid) const
    {
        return StartManaged(stream, true, callerValid, true);
    }
    TransferOperation TransferBinding::BeginReadyScheduled(const ExecutionStream &stream, bool callerValid) const
    {
        return StartManaged(stream, false, callerValid, false);
    }
    TransferOperation TransferBinding::BeginReadyChecked(const ExecutionStream &stream, bool callerValid) const
    {
        return StartManaged(stream, true, callerValid, false);
    }
    TransferOperation TransferBinding::StartManaged(const ExecutionStream &stream, bool checked,
                                                    bool callerValid, bool observeProducer) const
    {
        DNDS_check_throw_info(_state, "managed MPI begin requires an MPI binding; native providers use Enqueue");
        auto managed = std::make_shared<TransferOperation::ManagedState>();
        bool valid = callerValid;
        ExecutionContext context;
        try
        {
            stream.RegisterProgress(managed); // Before preparing/packing/posting.
            context = stream.Context();
            if (observeProducer && !checked && context.device == Device::Host())
                stream.Record().WaitHost();
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("managed MPI registration failed");
            valid = false; // Fold into the existing launch preflight, no new allreduce.
        }
        try
        {
            managed->operation = checked ? StartChecked(stream, valid, observeProducer)
                                         : Start(context, true, {}, 0, valid);
            managed->order = managed->operation.CheckedOrder();
            managed->armed = true;
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("managed MPI submission failed");
            managed->done = true; // Rejected before posting; registry can retire it.
            throw;
        }
        TransferOperation result;
        result._managed = std::move(managed);
        return result;
    }
    TransferOperation TransferBinding::StartChecked(const ExecutionStream &stream,
                                                    bool callerValid, bool observeProducer) const
    {
        DNDS_check_throw_info(_state, "BeginChecked is an MPI bridge; native providers use Enqueue");
        auto tracking = stream._tracking;
        uint64_t lane = 0;
        bool valid = callerValid;
        try
        {
            DNDS_check_throw_info(tracking, "BeginChecked needs a checked execution stream");
            StreamEvent ready;
            if (observeProducer)
            {
                ready = stream.Record();
                if (stream.Context().device == Device::Host())
                    ready.WaitHost();
            }
            {
                std::lock_guard<std::mutex> lock(_state->mutex);
                DNDS_check_throw_info(!_state->active, "checked transfer binding is active");
                if (_state->tracking != tracking)
                {
                    if (_state->tracking)
                        _state->tracking->RetireLane(_state->lane);
                    _state->tracking = tracking;
                    _state->lane = tracking->NewLane();
                }
                lane = _state->lane;
            }
            if (observeProducer)
                tracking->WaitFor(lane, ready._order, [] {});
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("MPI row transfer preparation failed");
            valid = false;
        }
        _state->Preflight(valid, "checked MPI stream preflight failed on a rank");
        return Start(stream.Context(), true, std::move(tracking), lane);
    }
    void TransferBinding::Enqueue(EagerEngine &engine, const ExecutionStream &stream) const
    {
        DNDS_check_throw_info(_provider, "MPI transfer requires explicit host progress; Enqueue needs a stream-native provider");
        _provider->Enqueue(engine, stream);
    }
    TransferOperation TransferBinding::Start(ExecutionContext context, bool scheduled, std::shared_ptr<AccessTrackingContext> tracking, uint64_t lane, bool callerValid) const
    {
        if (_provider)
        {
            DNDS_check_throw_info(callerValid, "native transfer provider requires caller-coordinated preflight");
            return TransferOperation(_provider->Begin(context, scheduled));
        }
        auto op = std::make_unique<TransferOperation::State>();
        op->binding = _state;
        op->context = context;
        auto begin = [&]
        {
            const bool valid = op->Prepare(scheduled, bool(tracking), callerValid);
            try
            {
                _state->Preflight(valid, "transfer access/stream preflight failed on a rank");
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("MPI row transfer preparation failed");
                op->CancelPrepared();
                throw;
            }
            if (_state->options.communicationWorker)
            {
                std::lock_guard<std::mutex> lock(_state->mutex);
                _state->active = true; // Reserve before a worker can observe it.
            }
            else
                op->Post();
        };
        if (tracking)
        {
            std::vector<ScheduledBufferAccess> accesses{{_state->source, AccessMode::Read}, {_state->destination, AccessMode::Write}};
            // Private workspaces are retained by the binding and only one
            // operation may be active. Declare them as well for release proof.
            for (const auto *buffer : {&_state->workspace, &_state->deviceSend, &_state->deviceReceive, &_state->hostSource, &_state->hostDestination})
                if (buffer->Identity())
                    accesses.push_back({*buffer, AccessMode::Write});
            tracking->Submit(lane, accesses, [&](auto submission)
                             { op->checked = submission; begin(); }, [&](bool valid)
                             { _state->Preflight(valid, "unordered checked MPI access on a rank"); });
        }
        else
            begin();
        TransferOperation result(std::move(op));
        if (tracking)
            result._state->order = tracking->Record(lane, [] {});
        if (_state->options.communicationWorker)
        {
            TransferOperation queued;
            queued._worker = std::make_unique<TransferOperation::WorkerState>();
            queued._worker->operation = std::make_unique<TransferOperation>(std::move(result));
            auto *pending = queued._worker.get();
            try
            {
                _state->options.communicationWorker->Submit([pending]
                                                            {
                    auto &operation = *pending->operation;
                    operation._state->Post();
                    operation.WaitDestinationReady();
                    {
                        std::lock_guard<std::mutex> lock(pending->mutex);
                        pending->destinationReady = true;
                        pending->ready.notify_all();
                    }
                    operation.Wait();
                    {
                        std::lock_guard<std::mutex> lock(pending->mutex);
                        pending->done = true;
                        pending->ready.notify_all();
                    } });
            }
            catch (...)
            {
                MPI::Abort(_state->comm, "communication worker submission failed");
            }
            return queued;
        }
        return result;
    }
    TransferGroup::TransferGroup(std::shared_ptr<CommunicationContext> context, std::vector<TransferBinding> bindings)
        : _context(std::move(context)), _bindings(std::move(bindings))
    {
        DNDS_check_throw_info(_context, "transfer group needs a live coordinated communication context");
        bool valid = true;
        std::string signature;
        std::vector<const void *> owners;
        for (const auto &binding : _bindings)
        {
            const auto &state = binding._state;
            if (!state || binding._provider)
            {
                valid = false;
                continue;
            }
            valid = valid && !state->options.communicationWorker && state->options.context == _context &&
                    bool(_bindings.front()._state) &&
                    state->collectivePreflight == _bindings.front()._state->collectivePreflight &&
                    state->options.mode == TransferMode::Persistent &&
                    state->source.GetDevice() == Device::Host() && state->destination.GetDevice() == Device::Host() &&
                    (state->transport == TransferTransport::Direct || state->transport == TransferTransport::HostPacked) &&
                    std::find(owners.begin(), owners.end(), state.get()) == owners.end();
            owners.push_back(state.get());
            signature += std::to_string(state->options.tag) + ":";
        }
        // Ordered tags identify the exact member set in this shared namespace.
        // This required setup agreement is never a per-Begin policy handshake.
        Agree(signature, _context->Communicator(), valid);
    }
    std::vector<TransferOperation> TransferGroup::Begin() const
    {
        DNDS_check_throw_info(_context, "transfer group was moved");
        std::vector<TransferOperation> operations;
        if (_bindings.empty())
            return operations;
        bool valid = true;
        try
        {
            operations.reserve(_bindings.size());
            for (const auto &binding : _bindings)
            {
                auto state = std::make_unique<TransferOperation::State>();
                state->binding = binding._state;
                state->context = ExecutionContext::Host();
                operations.push_back(TransferOperation(std::move(state)));
                // Do not short circuit: all successfully acquired fields must
                // be present in the rollback set before agreement/posting.
                const bool prepared = operations.back()._state->Prepare(false, false, true);
                valid = prepared && valid;
            }
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("MPI row transfer preparation failed");
            valid = false;
        }
        try
        {
            _bindings.front()._state->Preflight(valid, "transfer group reservation failed on a rank");
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("MPI row transfer preparation failed");
            for (auto &operation : operations)
                operation._state->CancelPrepared();
            throw;
        }
        // Persistent requests and all payload/lease owners are already ready.
        // Keep field messages/tags separate; no coalescing or packing is added.
        for (auto &operation : operations)
            operation._state->Post();
        return operations;
    }
    TransferOperation::TransferOperation() = default;
    bool TransferBinding::HasEmptyDirectPayload() const
    {
        return _state && !_provider && _state->emptyDirectPayload;
    }
    TransferOperation TransferBinding::SubmitEmpty(EagerEngine &engine, const ExecutionStream &stream,
                                                   bool callerValid) const
    {
        // A locally empty participant still enters exactly the preflights of
        // its nonempty peers. No payload is read/written, so reserve no buffers
        // and require no CPU-visible producer prefix. Stream ordering alone
        // carries the halo publication dependency, for every backend/rank count.
        bool idle;
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            idle = !_state->active;
        }
        if (engine.ChecksAccess())
        {
            _state->Preflight(callerValid && idle && bool(stream._tracking),
                              "checked MPI stream preflight failed on a rank");
            // Nonempty checked peers next enter AccessTracking::Submit's
            // unconditional agreement callback, before payload preparation.
            // This route declares no payload accesses, but must participate in
            // that same middle agreement to preserve collective ordering.
            _state->Preflight(true, "unordered checked MPI access on a rank");
        }
        const auto context = stream.Context();
        const auto placed = [&](const Buffer &buffer)
        { return buffer.GetDevice() == Device::Host() || buffer.GetDevice() == context.device; };
        _state->Preflight(callerValid && idle && _state->plan->route.Valid() &&
                              placed(_state->source) && placed(_state->destination),
                          "transfer access/stream preflight failed on a rank");
        auto retained = _state;
        try
        {
            // No mutable requests/workspace are touched, so repeated ordered
            // empty publications may queue on this binding. A real active Begin
            // still fails the checks above. Retain MPI/context/plan owners until
            // native completion even when the public operation is discarded.
            engine.Submit(stream, [retained](ExecutionContext) {});
            TransferOperation result;
            result._nativeCompletion = std::make_shared<StreamEvent>(stream.Record());
            return result;
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("empty MPI transfer submission failed");
            throw;
        }
    }
    TransferOperation TransferBinding::SubmitReady(EagerEngine &engine, const ExecutionStream &communication,
                                                   bool callerValid) const
    {
        DNDS_check_throw_info(engine.Owns(communication), "transfer stream belongs to another runtime");
        if (HasEmptyDirectPayload())
            return SubmitEmpty(engine, communication, callerValid);
        if (!_provider)
            return engine.ChecksAccess() ? BeginReadyChecked(communication, callerValid)
                                         : BeginReadyScheduled(communication, callerValid);
        DNDS_check_throw_info(callerValid, "native transfer caller validation failed");
        Enqueue(engine, communication);
        TransferOperation operation;
        operation._nativeCompletion = std::make_shared<StreamEvent>(communication.Record());
        return operation;
    }
    TransferOperation TransferBinding::SubmitAfter(EagerEngine &engine, const ExecutionStream &communication,
                                                   StreamEvent producer, bool callerValid) const
    {
        DNDS_check_throw_info(engine.Owns(communication), "transfer stream belongs to another runtime");
        if (!_provider && !HasEmptyDirectPayload())
        {
            // Portable MPI posting needs the producer visible to its controller.
            // Do not enqueue a CPU waiter that would occupy a worker or call MPI
            // from a thread forbidden by the initialized MPI thread level.
            producer.WaitHost();
        }
        communication.WaitFor(std::move(producer));
        return SubmitReady(engine, communication, callerValid);
    }
    void TransferOperation::EnqueueWait(const ExecutionStream &consumer) const
    {
        auto completion = CompletionEvent();
        if (!_nativeCompletion && consumer.Context().device.space == MemorySpace::CUDA)
        {
            // Portable MPI has no native CUDA stream completion. Progress its
            // requests and any device unpack on the permitted controller, then
            // import the completed ordering record. This bridge is explicitly
            // non-capturable; ordinary Stream::WaitFor remains strict.
            completion.WaitHost();
        }
        consumer.WaitFor(std::move(completion));
    }
    StreamEvent TransferOperation::ManagedCompletion() const
    {
        if (_nativeCompletion)
            return *_nativeCompletion;
        if (!_managed)
            return {};
        StreamEvent result(_managed);
        result._order = _managed->order;
        return result;
    }
    StreamEvent TransferOperation::CompletionEvent() const
    {
        if (_nativeCompletion)
            return *_nativeCompletion;
        DNDS_check_throw_info(_managed,
                              "completion event requires a managed transfer");
        return ManagedCompletion();
    }
    std::shared_ptr<const AccessOrder> TransferOperation::CheckedOrder() const
    {
        if (_nativeCompletion)
            return _nativeCompletion->_order;
        if (_managed)
            return _managed->order;
        if (_worker)
            return _worker->operation->CheckedOrder();
        return _state ? _state->order : nullptr;
    }
    TransferOperation::TransferOperation(std::unique_ptr<State> state) : _state(std::move(state)) {}
    TransferOperation::TransferOperation(std::unique_ptr<Detail::TransferProviderOperation> provider) : _provider(std::move(provider)) {}
    TransferOperation::TransferOperation(TransferOperation &&) noexcept = default;
    TransferOperation &TransferOperation::operator=(TransferOperation &&other) noexcept
    {
        if (this != &other)
        {
            try
            {
                if (!_managed && !_nativeCompletion)
                    Wait();
            }
            catch (...)
            {
                MPI::Abort(_state ? _state->binding->comm : MPI_COMM_WORLD, "transfer replacement completion failed");
            }
            _managed = std::move(other._managed);
            _nativeCompletion = std::move(other._nativeCompletion);
            _state = std::move(other._state);
            _provider = std::move(other._provider);
            _worker = std::move(other._worker);
        }
        return *this;
    }
    TransferOperation::~TransferOperation()
    {
        if (_managed || _nativeCompletion)
            return; // Runtime inventory, not handle scope, owns retirement.
        // Waiting while unwinding a rank-local exception can strand this rank
        // before the application reaches its fatal handler: a peer may never
        // post the matching operation. Normal RAII retirement still drains;
        // exceptional retirement of active MPI work is unconditionally fatal.
        if (((_state && _state->started) || (_worker && !_worker->Test())) && std::uncaught_exceptions())
            DNDS::assert_false_info("no exception while an MPI transfer is active", __FILE__, __LINE__,
                                    "cannot drain an active MPI transfer during exception unwinding; aborting instead of risking unmatched MPI waits");
        try
        {
            Wait();
        }
        catch (...)
        {
            MPI::Abort(_state ? _state->binding->comm : MPI_COMM_WORLD, "transfer destruction completion failed");
        }
    }
    bool TransferOperation::PostPackedSends(bool wait)
    {
        auto &s = *_state;
        auto &b = *s.binding;
        if (s.sendsStarted)
            return true;
        try
        {
            if (wait)
                s.packedReady.Wait();
            else if (!s.packedReady.Ready())
                return false;
            if (b.options.mode == TransferMode::Persistent)
            {
                if (!b.sends.empty())
                    MPI_Startall(int(b.sends.size()), b.sends.data());
            }
            else
            {
                b.sends.clear();
                for (int peer = 0; peer < int(b.plan->route.SendRows().size()); ++peer)
                    if (!b.plan->route.SendRows()[peer].empty())
                    {
                        b.sends.push_back(MPI_REQUEST_NULL);
                        MPI_Isend(b.PackedPointer(false, peer), b.PackedCount(false, peer), b.plan->schema.Type(), peer, b.options.tag, b.comm, &b.sends.back());
                    }
            }
        }
        catch (const std::exception &error)
        {
            // Receives are posted but sends may be missing. Neither a retry,
            // a destructor drain nor an error Allreduce can be matched safely.
            MPI::Abort(b.comm, std::string("packed MPI send preparation failed after receives were posted: ") + error.what());
        }
        catch (...)
        {
            MPI::Abort(b.comm, "packed MPI send preparation failed after receives were posted");
        }
        s.sendsStarted = true;
        return true;
    }
    void TransferOperation::WaitDestinationReady()
    {
        if (_nativeCompletion)
        {
            _nativeCompletion->WaitHost();
            return;
        }
        if (_managed)
        {
            if (_managed->done.load(std::memory_order_acquire))
                return;
            _managed->CheckController();
            _managed->operation.WaitDestinationReady();
            return;
        }
        if (_worker)
        {
            _worker->Wait(true);
            return;
        }
        if (_provider)
            _provider->WaitDestinationReady();
        else
            CompleteDestination(true);
    }
    bool TransferOperation::CompleteDestination(bool wait)
    {
        if (_state && _state->error)
            std::rethrow_exception(_state->error);
        if (!_state || !_state->started || _state->destinationReady)
            return true;
        if (!PostPackedSends(wait))
            return false;
        auto &s = *_state;
        auto &b = *s.binding;
        const auto finish = [&]
        {
            if (!wait && !s.unpackReady.Ready())
                return false;
            s.unpackReady.Wait();
            s.workspace.Wait();
            s.destination.Wait();
            s.destinationReady = true;
            return true;
        };
        try
        {
            if (s.unpackStarted)
                return finish();
            if (wait)
                MPI_Waitall(int(b.receives.size()), b.receives.data(), MPI_STATUSES_IGNORE);
            else
            {
                int received = 0;
                MPI_Testall(int(b.receives.size()), b.receives.data(), &received, MPI_STATUSES_IGNORE);
                if (!received)
                    return false;
            }
            s.unpackReady = Completion::FromExternalCompletion(s.destination.context);
            if (b.transport == TransferTransport::DevicePacked)
            {
                if (b.reducer)
                    b.reducer->device(b.deviceReceive.LegacyPointer(), s.destination.View<std::byte>().data,
                                      static_cast<const Contribution *>(b.deviceGroups.LegacyPointer()), b.reductionLayout->groups.size(), s.context);
                else
                    Detail::LaunchDevicePacking(s.context, b.deviceReceive.LegacyPointer(), s.destination.View<uint8_t>().data,
                                                static_cast<const Detail::DevicePackRow *>(b.unpackRows.LegacyPointer()), b.unpackRowCount,
                                                static_cast<const Detail::DevicePackField *>(b.packFields.LegacyPointer()), b.fieldCount, index(b.plan->schema.Extent()), true);
                s.unpackReady = Completion::Record(s.context);
            }
            else
            {
                const bool packed = b.transport == TransferTransport::HostPacked;
                if (packed)
                    for (size_t peer = 0; peer < b.packedReceive.size(); ++peer)
                    {
                        if (b.plan->route.ReceiveRows()[peer].empty())
                            continue;
                        int position = 0;
                        auto type = b.reducer ? b.reductionLayout->receiveTypes[peer] : b.plan->receiveTypes[peer];
                        MPI_Count bytes = 0;
                        MPI_Type_size_x(type->value, &bytes);
                        if (!bytes)
                            continue;
                        MPI_Unpack(b.packedReceive[peer].data(), int(b.packedReceive[peer].size()), &position,
                                   b.reducer ? s.workspace.View<uint8_t>().data : b.hostDestination.LegacyPointer(), 1, type->value, b.comm);
                    }
                if (b.reducer)
                {
                    if (packed || b.destination.GetDevice() == Device::Host())
                        b.reducer->host(s.workspace.View<std::byte>().data, packed ? b.hostDestination.LegacyPointer() : s.destination.View<std::byte>().data,
                                        b.reductionLayout->groups, ExecutionContext::Host());
                    else
                        b.reducer->device(s.workspace.View<std::byte>().data, s.destination.View<std::byte>().data,
                                          static_cast<const Contribution *>(b.deviceGroups.LegacyPointer()), b.reductionLayout->groups.size(), s.context);
                    if (!packed && b.destination.GetDevice().space == MemorySpace::CUDA)
                        s.unpackReady = Completion::Record(s.context);
                }
                if (packed)
                    s.unpackReady = Detail::CopyBytes(b.hostDestination.LegacyPointer(), Device::Host(), s.destination.View<uint8_t>().data,
                                                      b.destination.GetDevice(), b.destination.Bytes(), b.destination.GetDevice() == Device::Host() ? ExecutionContext::Host() : s.context);
            }
            // MPI completion itself makes direct destinations ready. For device
            // unpack/copy, retain its exact event instead of fencing the later
            // stream tail. Test must leave unrelated controller work runnable.
            s.destination.Complete(s.unpackReady);
            if (b.reducer)
                s.workspace.Complete(s.unpackReady);
            s.unpackStarted = true;
            return finish();
        }
        catch (...)
        {
            MPI::Abort(b.comm, "MPI destination completion/reduction failed");
        }
    }
    void TransferOperation::Wait()
    {
        if (_nativeCompletion)
        {
            _nativeCompletion->WaitHost();
            return;
        }
        if (_managed)
        {
            _managed->Wait();
            return;
        }
        if (_worker)
        {
            _worker->Wait(false);
            return;
        }
        if (_provider)
        {
            _provider->Wait();
            return;
        }
        if (_state && _state->error)
            std::rethrow_exception(_state->error);
        if (!_state || !_state->started)
            return;
        WaitDestinationReady();
        auto &s = *_state;
        auto &b = *s.binding;
        MPI_Waitall(int(b.sends.size()), b.sends.data(), MPI_STATUSES_IGNORE);
        try
        {
            if (b.transport != TransferTransport::DevicePacked)
                s.source.Complete(Completion::FromExternalCompletion(s.source.context));
            // DevicePacked already attached packedReady when posting; sends
            // cannot complete before that producer. Do not submit it twice.
            s.source.Wait();
        }
        catch (...)
        {
            MPI::Abort(b.comm, "MPI source completion failed");
        }
        s.started = false;
        s.checked.Complete();
        {
            std::lock_guard<std::mutex> lock(b.mutex);
            b.active = false;
        }
        if (s.error)
            std::rethrow_exception(s.error);
    }
    bool TransferOperation::Test()
    {
        if (_nativeCompletion)
            return _nativeCompletion->Ready();
        if (_managed)
            return _managed->Ready();
        if (_worker)
            return _worker->Test();
        if (_provider)
            return _provider->Test();
        if (_state && _state->error)
            std::rethrow_exception(_state->error);
        if (!_state || !_state->started)
            return true;
        auto &b = *_state->binding;
        if (!CompleteDestination(false))
            return false;
        int send = 0;
        MPI_Testall(int(b.sends.size()), b.sends.data(), &send, MPI_STATUSES_IGNORE);
        if (send)
        {
            Wait();
            return true;
        }
        return false;
    }
#ifndef DNDS_USE_CUDA
    void Detail::LaunchDevicePacking(ExecutionContext, const void *, void *, const DevicePackRow *, index, const DevicePackField *, index, index, bool)
    {
        DNDS_check_throw_info(false, "device packing requires CUDA support");
    }
#endif
}
