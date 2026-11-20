#pragma once

#include "Cauchy/IO/OutputMeasurement.hpp"
#include "DNDS/Runtime/Array/Array.hpp"

namespace DNDS::Cauchy
{
    /// Pinned payloads owned by one serialized file writer. Each file uses a
    /// fresh Batch, after the previous file has released its Array views.
    /// A queued snapshot owns its device image and never borrows these slots.
    class OutputStaging
    {
        std::vector<Runtime::Buffer> _slots;
        size_t _reservedBytes = 0;

    public:
        class Batch
        {
            OutputStaging &_owner;
            size_t _next = 0;

        public:
            explicit Batch(OutputStaging &owner) : _owner(owner) {}
            Batch(const Batch &) = delete;
            Batch &operator=(const Batch &) = delete;
            template <class T>
            Runtime::Array<T> HostValues(const Runtime::Array<T> &source, Runtime::ExecutionContext copy,
                                         const std::string &name)
            {
                if (source.GetDevice() == Runtime::Device::Host())
                    return source;
                DNDS_check_throw_info(copy.device == source.GetDevice(), "output staging stream and field placement differ");
                const auto layout = source.Layout();
                const size_t bytes = CheckedSize::Multiply(layout.Elements(), sizeof(T));
                if (_next == _owner._slots.size())
                    _owner._slots.emplace_back();
                auto &slot = _owner._slots[_next++];
                if (!slot.Identity() || slot.Bytes() < bytes)
                {
                    _owner._reservedBytes -= slot.Bytes();
                    slot = {};
                    const OutputMeasurement allocation("pinned_reserve", _owner._reservedBytes + bytes);
                    slot = Runtime::Buffer::AllocatePinnedHost(bytes);
                    _owner._reservedBytes += slot.Bytes();
                }
                auto host = Runtime::Array<T>::FromStorage(layout, slot.Subspan(0, bytes));
                host.SetName("Cauchy/output/pinned/" + name, DNDS_ALLOCATION_SITE());
                const OutputMeasurement transfer("d2h", bytes);
                host.CopyFrom(source, copy).Wait();
                return host;
            }
        };
    };
}
