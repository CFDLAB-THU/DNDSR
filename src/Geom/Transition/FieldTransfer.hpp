#pragma once

#include "Geom/Transition/ReorderPlan.hpp"
#include "DNDS/Runtime/Array/Field.hpp"
#include "DNDS/Runtime/Communication/Exchange.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/Runtime/Communication/TransferLayout.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>

namespace DNDS::Geom
{
    namespace FieldTransferDetail
    {
        /// Agree on a canonical earlier payload without assuming every rank's
        /// local alias inventory is the same.
        inline int AgreeAliasCandidate(const MPIInfo &mpi, int candidate, size_t relationCount)
        {
            return MPI::AgreeCandidate(mpi, candidate, relationCount);
        }

        /// Every rank runs local work before agreement and enters the next
        /// collective only after all ranks succeeded. Payload/route constructors
        /// retain their own collective schema and extent checks.
        template <class F>
        void LocalStage(const MPIInfo &mpi, F &&action)
        {
            MPI::LocalStage(mpi, std::forward<F>(action), "field preparation failed before collective payload; source retained");
        }
        template <class F>
        void LocalStage(const MPIInfo &mpi, const std::shared_ptr<Runtime::CommunicationContext> &communication, F &&action)
        {
            Runtime::CommunicationDetail::LocalStage(mpi.comm, !communication || communication->ChecksCollectivePreflight(),
                                                     std::forward<F>(action), "field preparation failed before collective payload; source retained");
        }

        template <class Value>
        struct Prepared final : PreparedCompanion
        {
            std::shared_ptr<Value> destination;
            Value next;
            Prepared(std::shared_ptr<Value> source, Value value) : destination(std::move(source)), next(std::move(value)) {}
            void Publish() noexcept override
            {
                static_assert(std::is_nothrow_move_assignable_v<Value>, "field publication must not throw");
                *destination = std::move(next);
            }
        };
    }

    /// Extract the exact correspondence once, then share it across native Arrays
    /// (including separate SoA components). No geometry or interpolation in DNDS.
    /// All ranks select the same policy at this setup boundary. A null context
    /// preserves enabled preflight; off mode still validates locally and aborts
    /// on failure. The returned route retains the context for payload launches.
    inline Runtime::RowTransferPlan NativeRowTransfer(const PermutationTransfer &transfer, const MPIInfo &mpi,
                                                      std::shared_ptr<Runtime::CommunicationContext> communication = {})
    {
        // Agree on the mode before taking branches with different collectives.
        const bool checks = !communication || communication->ChecksCollectivePreflight();
        int controls[]{int(transfer.isLocalOnly), -int(transfer.isLocalOnly), int(checks), -int(checks)};
        MPI_Allreduce(MPI_IN_PLACE, controls, 4, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(controls[0] == -controls[1] && controls[2] == -controls[3], "native row transfer mode/policy differs between ranks");
        std::vector<std::vector<index>> send, receive;
        std::vector<int> sendCounts, receiveCounts;
        index destinationRows = 0;
        FieldTransferDetail::LocalStage(mpi, communication, [&]
                                        {
            const index rows = index(transfer.size());
            send.resize(mpi.size);
            receive.resize(mpi.size);
            if (transfer.isLocalOnly)
            {
                DNDS_check_throw_info(transfer.localOld2New.size() == rows, "invalid native local permutation extent");
                std::vector<bool> seen(rows, false);
                send[mpi.rank].reserve(rows);
                receive[mpi.rank].reserve(rows);
                for (index i = 0; i < rows; ++i)
                {
                    const auto target = transfer.localOld2New[i];
                    DNDS_check_throw_info(target >= 0 && size_t(target) < rows && !seen[size_t(target)], "invalid native local permutation");
                    seen[size_t(target)] = true;
                    send[mpi.rank].push_back(i);
                    receive[mpi.rank].push_back(target);
                }
                destinationRows = rows;
            }
            else
            {
                DNDS_check_throw_info(transfer.pushStart.size() == size_t(mpi.size + 1) &&
                                          transfer.newGlobalOffsets.size() == size_t(mpi.size + 1) &&
                                          transfer.pushStart.front() == 0 && transfer.pushIndex.size() == rows &&
                                          transfer.pushStart.back() == transfer.size(), "invalid native distributed route extent");
                sendCounts.resize(mpi.size);
                receiveCounts.resize(mpi.size);
                std::vector<bool> seen(size_t(rows), false);
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    const auto first = transfer.pushStart[peer], last = transfer.pushStart[peer + 1];
                    DNDS_check_throw_info(first >= 0 && last >= first && size_t(last) <= rows && last - first <= INT_MAX,
                                          "invalid native push offsets");
                    DNDS_check_throw_info(transfer.newGlobalOffsets[peer] >= 0 &&
                                              transfer.newGlobalOffsets[peer + 1] >= transfer.newGlobalOffsets[peer],
                                          "invalid native destination offsets");
                    sendCounts[peer] = int(last - first);
                    send[peer].reserve(size_t(sendCounts[peer]));
                    for (DNDS::index j = first; j < last; ++j)
                    {
                        const auto row = transfer.pushIndex[size_t(j)];
                        DNDS_check_throw_info(row >= 0 && size_t(row) < rows && !seen[size_t(row)] &&
                                                  transfer.targetRanks[size_t(row)] == peer, "invalid native push row");
                        seen[size_t(row)] = true;
                        send[peer].push_back(row);
                    }
                }
                destinationRows = transfer.newGlobalOffsets[mpi.rank + 1] - transfer.newGlobalOffsets[mpi.rank];
            } });
        if (!transfer.isLocalOnly)
        {
            MPI_Alltoall(sendCounts.data(), 1, MPI_INT, receiveCounts.data(), 1, MPI_INT, mpi.comm);
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                // ArrayTransformer push receives contiguous sender-rank buckets,
                // retaining each sender's pushIndex order within its bucket.
                index next = 0;
                for (int peer = 0; peer < mpi.size; ++peer)
                {
                    receive[peer].reserve(size_t(receiveCounts[peer]));
                    for (int j = 0; j < receiveCounts[peer]; ++j)
                        receive[peer].push_back(next++);
                }
                DNDS_check_throw_info(next == destinationRows, "native destination offsets disagree with received rows"); });
        }
        return Runtime::RowTransferPlan::Known(communication ? std::move(communication) : std::make_shared<Runtime::CommunicationContext>(mpi.comm),
                                               index(transfer.size()), destinationRows, std::move(send), std::move(receive));
    }

    /// Register owned rows for detached preparation and publication. The default
    /// policy preserves the source storage family and stride/capacity. Registration
    /// retains an owning publication slot; ordinary Array aliases remain storage aliases.
    /// The destination storage policy is explicit and never inferred from a
    /// static accessor. Padding is not copied. Failed preparation retains source.
    /// Arrays have no mesh communication policy: pass the mesh's context to
    /// inherit its preflight mode. Null retains enabled compatibility behavior.
    template <class T>
    CompanionEntry RegisterArray(std::shared_ptr<Runtime::Array<T>> slot, EntityKind kind, std::string name,
                                 Runtime::ElementSchema schema = Runtime::ElementSchema::Arithmetic<T>(),
                                 Runtime::ExecutionContext context = Runtime::ExecutionContext::Host(),
                                 CompanionDisposition disposition = CompanionDisposition::TransferIfReordered,
                                 Runtime::TransferLayout destinationLayout = {},
                                 std::shared_ptr<Runtime::CommunicationContext> communication = {})
    {
        DNDS_check_throw_info(slot, "field publication requires an owning slot");
        CompanionEntry entry{kind, {}, std::move(name)};
        entry.disposition = disposition;
        entry.preflight = [slot, schema, context](const PermutationTransfer &transfer)
        {
            const auto &array = *slot;
            context.Validate();
            DNDS_check_throw_info(array.RowCount() == size_t(transfer.size()) && array.Storage().IsIdle() && array.IsValid() &&
                                      schema.Extent() == sizeof(T) && schema.Compatible(Runtime::ElementSchema::TypeIdentity<T>()) &&
                                      context.device == array.GetDevice(),
                                  "native Array migration requires idle valid compatible owned rows/context");
        };
        entry.attachment = [slot, schema]()
        {
            const auto &array = *slot;
            DNDS_check_throw_info(array.IsValid() && array.Storage().IsIdle(), "native Array attachment requires valid idle storage");
            return CompanionAttachment{schema.Identity() + ":" + Runtime::ElementSchema::TypeIdentity<T>() + ":" +
                                           std::to_string(static_cast<int>(array.GetDevice().space)),
                                       {},
                                       false,
                                       0,
                                       0,
                                       slot.get(),
                                       array.Identity(),
                                       array.Version(),
                                       array.ShapeVersion(),
                                       size_t(array.RowCount()),
                                       size_t(array.RowCount()),
                                       0};
        };
        entry.prepare = [slot, schema, context, destinationLayout, communication](const PermutationTransfer &transfer, const MPIInfo &mpi)
        {
            auto &array = *slot;
            auto route = NativeRowTransfer(transfer, mpi, communication);
            Runtime::PayloadPlan payload(route, array.Layout(), schema, destinationLayout);
            std::unique_ptr<FieldTransferDetail::Prepared<Runtime::Array<T>>> next;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            { next = std::make_unique<FieldTransferDetail::Prepared<Runtime::Array<T>>>(slot,
                                                                                                                        Runtime::Array<T>(payload.DestinationLayout(), array.GetDevice())); });
            payload.Bind(array, next->next).Begin(context).Wait();
            return next;
        };
        auto validate = entry.preflight;
        auto prepare = entry.prepare;
        entry.fn = [validate, prepare, disposition, communication](const PermutationTransfer &transfer, const MPIInfo &mpi)
        {
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                DNDS_check_throw_info(disposition == CompanionDisposition::TransferIfReordered, "field disposition prohibits reorder");
                validate(transfer); });
            auto next = prepare(transfer, mpi);
            next->Publish();
        };
        return entry;
    }

    /// Adapt an owned native field to the existing mesh permutation machinery.
    /// Stable IDs, states and time histories are ordinary registered fields.
    /// The initial implementation is host-only; ghosts are rebuilt separately.
    /// Registration retains the supplied owning field slot through publication.
    /// Pass the agreed mesh communication context explicitly; a null context
    /// keeps collective preflight enabled, independently of array execution.
    template <class T>
    CompanionEntry RegisterField(std::shared_ptr<Runtime::Field<T>> slot, EntityKind kind, std::string name,
                                 CompanionDisposition disposition = CompanionDisposition::TransferIfReordered,
                                 std::shared_ptr<Runtime::CommunicationContext> communication = {})
    {
        DNDS_check_throw_info(slot, "field publication requires an owning slot");
        CompanionEntry entry{kind, {}, std::move(name)};
        entry.disposition = disposition;
        entry.preflight = [slot, kind](const PermutationTransfer &transfer)
        {
            const auto &field = *slot;
            std::string expected = entityKindName(kind);
            std::transform(expected.begin(), expected.end(), expected.begin(), [](unsigned char c)
                           { return char(std::tolower(c)); });
            DNDS_check_throw_info(field.Values().GetDevice() == Runtime::Device::Host() &&
                                      field.Layout().rows == size_t(transfer.size()) &&
                                      field.GetDistribution().ghosts == 0 &&
                                      field.GetDistribution().owned == field.Layout().rows &&
                                      field.GetDistribution().epoch != UINT64_MAX && field.Values().IsIdle() &&
                                      field.Values().IsValid() && field.Association() == expected,
                                  "native field migration requires idle valid host owned rows with matching entity association");
        };
        entry.attachment = [slot, kind]()
        {
            const auto &field = *slot;
            std::string expected = entityKindName(kind);
            std::transform(expected.begin(), expected.end(), expected.begin(), [](unsigned char c)
                           { return char(std::tolower(c)); });
            DNDS_check_throw_info(field.Association() == expected, "field association differs from attached entity kind");
            DNDS_check_throw_info(field.Values().IsValid() && field.Values().IsIdle(), "native Field attachment requires valid idle storage");
            return CompanionAttachment{Runtime::ElementSchema::TypeIdentity<T>() + ":" + std::to_string(field.Layout().components) + ":" +
                                           std::to_string(field.Layout().stride),
                                       field.Association(), true,
                                       field.GetDistribution().domain, field.GetDistribution().epoch, slot.get(),
                                       field.Values().Identity(), field.Values().Version(), 0, field.Layout().rows,
                                       field.GetDistribution().owned, field.GetDistribution().ghosts};
        };
        entry.prepare = [slot, communication](const PermutationTransfer &transfer, const MPIInfo &mpi)
        {
            auto &field = *slot;
            // Move source addresses with the mesh's exact correspondence,
            // including local permutation and distributed receive order.
            auto route = NativeRowTransfer(transfer, mpi, communication);
            auto layout = field.Layout();
            layout.rows = route.DestinationRows();
            auto distribution = field.GetDistribution();
            distribution.owned = layout.rows;
            ++distribution.epoch;
            std::unique_ptr<FieldTransferDetail::Prepared<Runtime::Field<T>>> next;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            { next = std::make_unique<FieldTransferDetail::Prepared<Runtime::Field<T>>>(slot,
                                                                                                                        Runtime::Field<T>(layout, distribution, field.Association())); });
            // Dense Field includes stride padding in its exact row
            // history. Byte schema retains compatibility for POD fields.
            std::optional<Runtime::Array<uint8_t>> source, destination;
            std::optional<Runtime::ElementSchema> bytes;
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                        const auto width = CheckedSize::Multiply(layout.stride, sizeof(T));
                        source.emplace(Runtime::Array<uint8_t>::FromStorage(Runtime::RowLayout::Dense(field.Layout().rows, width), field.Values()));
                        destination.emplace(Runtime::Array<uint8_t>::FromStorage(Runtime::RowLayout::Dense(layout.rows, width), next->next.Values()));
                        bytes.emplace(Runtime::ElementSchema::Arithmetic<uint8_t>()); });
            Runtime::PayloadPlan(route, source->Layout(), destination->Layout(), *bytes)
                .Bind(*source, *destination)
                .Begin()
                .Wait();
            return next;
        };
        auto validate = entry.preflight;
        auto prepare = entry.prepare;
        entry.fn = [validate, prepare, disposition, communication](const PermutationTransfer &transfer, const MPIInfo &mpi)
        {
            FieldTransferDetail::LocalStage(mpi, communication, [&]
                                            {
                DNDS_check_throw_info(disposition == CompanionDisposition::TransferIfReordered, "field disposition prohibits reorder");
                validate(transfer); });
            auto next = prepare(transfer, mpi);
            next->Publish();
        };
        return entry;
    }
}
