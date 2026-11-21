#pragma once

#include "Cauchy/IO/OutputMeasurement.hpp"
#include "DNDS/Runtime/Array/Array.hpp"

namespace DNDS::Cauchy
{
    /// Pinned host slots for one output batch. Only the controller stages
    /// values. Rewind after the preceding write and manifest have completed.
    class OutputStaging
    {
        std::vector<Runtime::Buffer> _slots;
        size_t _reservedBytes = 0, _next = 0;

    public:
        void Rewind() { _next = 0; }
        template <class T>
        Runtime::Array<T> HostValues(const Runtime::Array<T> &source, Runtime::ExecutionContext copy,
                                     const std::string &name, bool detachHost = false)
        {
            if (source.GetDevice() == Runtime::Device::Host())
                return detachHost ? source.Clone() : source;
            DNDS_check_throw_info(copy.device == source.GetDevice(), "output staging stream and field placement differ");
            const auto layout = source.Layout();
            const size_t bytes = CheckedSize::Multiply(layout.Elements(), sizeof(T));
            if (_next == _slots.size())
                _slots.emplace_back();
            auto &slot = _slots[_next++];
            if (!slot.Identity() || slot.Bytes() < bytes)
            {
                _reservedBytes -= slot.Bytes();
                slot = {};
                const OutputMeasurement allocation("pinned_reserve", _reservedBytes + bytes);
                slot = Runtime::Buffer::AllocatePinnedHost(bytes);
                _reservedBytes += slot.Bytes();
            }
            auto host = Runtime::Array<T>::FromStorage(layout, slot.Subspan(0, bytes));
            host.SetName("Cauchy/output/pinned/" + name, DNDS_ALLOCATION_SITE());
            const OutputMeasurement transfer("d2h", bytes);
            host.CopyFrom(source, copy).Wait();
            return host;
        }
    };
}
