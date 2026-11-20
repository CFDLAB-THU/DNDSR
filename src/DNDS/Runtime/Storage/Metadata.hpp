#pragma once

#include "DNDS/Runtime/Storage/Buffer.hpp"
#include "DNDS/Core/Span.hpp"
#include <algorithm>
#include <vector>
#include <utility>

namespace DNDS::Runtime
{
    /// Published backend metadata. Copies share the allocation; only const spans
    /// escape. Construction copies once into an ordinary tracked Buffer. Shape
    /// editors publish a new instance instead of editing an escaped descriptor.
    template <class T>
    class MetadataBuffer
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>,
                      "backend metadata must be plain records; retain services and resource owners outside it");
        Buffer _buffer;
        const T *_data = nullptr;
        size_t _size = 0;
        bool _host = true;

    public:
        MetadataBuffer() = default;
        MetadataBuffer(const MetadataBuffer &) = default;
        MetadataBuffer &operator=(const MetadataBuffer &) = default;
        MetadataBuffer(MetadataBuffer &&other) noexcept
            : _buffer(std::move(other._buffer)), _data(std::exchange(other._data, nullptr)),
              _size(std::exchange(other._size, 0)), _host(other._host) {}
        MetadataBuffer &operator=(MetadataBuffer &&other) noexcept
        {
            if (this != &other)
            {
                _buffer = std::move(other._buffer);
                _data = std::exchange(other._data, nullptr);
                _size = std::exchange(other._size, 0);
                _host = other._host;
            }
            return *this;
        }
        explicit MetadataBuffer(const std::vector<T> &values)
            : _buffer(Buffer::Allocate(CheckedSize::Multiply(values.size(), sizeof(T)))), _size(values.size())
        {
            DNDS_check_throw_info(_size <= size_t(INT64_MAX),
                                  "metadata element count exceeds signed index range");
            auto write = _buffer.Write();
            auto view = write.template View<T>();
            if (_size)
                std::copy(values.begin(), values.end(), view.data);
            _data = view.data;
        }
        /// Fill the final allocation in ordinal order, then publish read-only
        /// metadata. The generator returns values; no mutable pointer escapes.
        /// An exception abandons the unpublished allocation.
        template <class F>
        static MetadataBuffer Generate(size_t count, F &&valueAt)
        {
            DNDS_check_throw_info(count <= size_t(INT64_MAX), "metadata element count exceeds signed index range");
            MetadataBuffer result;
            result._buffer = Buffer::Allocate(CheckedSize::Multiply(count, sizeof(T)));
            result._size = count;
            auto write = result._buffer.Write();
            auto view = write.template View<T>();
            for (index i = 0; i < index(count); ++i)
                view.data[i] = valueAt(i);
            result._data = view.data;
            return result;
        }
        const T *data() const
        {
            DNDS_check_throw_info(_host,
                                  "host metadata access requires host placement");
            return _data;
        }
        size_t size() const { return _size; }
        bool empty() const { return !_size; }
        const T *begin() const { return data(); }
        const T *end() const { return _size ? data() + _size : data(); }
        const T &operator[](size_t i) const
        {
            // Residency is a host access contract even in bounds-unchecked builds.
            // Numerical loops can acquire HostView() once before indexing its span.
            const auto *values = data();
            if constexpr (SpanBoundsChecks)
                DNDS_HD_assert(i < _size);
            return values[i];
        }
        const T &back() const { return (*this)[_size - 1]; }
        Device GetDevice() const { return _buffer.GetDevice(); }
        /// Diagnostic alias only; immutable metadata values/ownership do not change.
        void SetDiagnosticName(const std::string &name, AllocationSite site = {}) const
        {
            if (_buffer.Identity())
                _buffer.SetDiagnosticName(name, site);
        }
        /// Explicit immutable publication on another backend. Copies are setup
        /// operations and complete before the returned descriptor is exposed.
        /// This copies object bytes; nested views must already reference the
        /// destination backend. Pointer relocation is never inferred here.
        MetadataBuffer CopyTo(Device device, ExecutionContext context) const
        {
            MetadataBuffer result;
            result._size = _size;
            result._host = device == Device::Host();
            result._buffer = Buffer::Allocate(_buffer.Bytes(), device);
            if (_buffer.Identity())
                Copy(_buffer, result._buffer, context).Wait();
            auto read = result._buffer.Read(device == Device::Host()
                                                ? ExecutionContext::Host()
                                                : context);
            result._data = read.template View<T>().data;
            return result;
        }
        /// CPU metadata view; retain this immutable owner through every use.
        SpanView<const T> HostView() const
        {
            DNDS_check_throw_info(_size <= size_t(INT64_MAX),
                                  "metadata extent exceeds signed range");
            return {data(), index(_size)};
        }
        uint64_t Identity() const { return _buffer.Identity(); }
        /// Read-only accounting of retained allocation capacity, without an access lease.
        template <class F>
        void VisitAllocations(F &&visit) const
        {
            if (_buffer.Identity())
                visit(_buffer.Identity(), _buffer.AllocationHandle().Bytes(), _buffer.GetDevice());
        }
        bool IsUnique() const { return !_buffer.Identity() || _buffer.IsUnique(); }
        ReadAccess Read(ExecutionContext context = ExecutionContext::Host()) const
        {
            return _buffer.Read(context);
        }
        /// Borrowed whole-buffer descriptor of this immutable publication.
        /// Construction/CopyTo finish initialization before publication, and
        /// there is no mutable alias API. Retain a MetadataBuffer copy (or its
        /// owning publication) through queued completion; no per-use read lease
        /// or completion event is needed. This does not apply to mutable Arrays.
        SpanView<const T> View(Device device) const
        {
            DNDS_check_throw_info(device == GetDevice(), "immutable metadata view placement mismatch");
            return {_data, index(_size)};
        }
        bool operator==(const MetadataBuffer &other) const
        {
            DNDS_check_throw_info(GetDevice() == Device::Host() &&
                                      other.GetDevice() == Device::Host(),
                                  "metadata value comparison requires host placement");
            return _size == other._size && (_data == other._data || std::equal(begin(), end(), other.begin()));
        }
    };
}
