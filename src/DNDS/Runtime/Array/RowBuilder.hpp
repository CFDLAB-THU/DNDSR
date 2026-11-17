#pragma once

#include "DNDS/Runtime/Array/Array.hpp"
#include <atomic>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>

namespace DNDS::Runtime
{
    template <class Fill>
    void FillKnownRows(size_t rows, bool parallel, Fill &&fill)
    {
        if (!parallel || rows < 4096)
        {
            for (size_t row = 0; row < rows; ++row)
                fill(row);
            return;
        }
        std::exception_ptr failure;
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(rows); ++row)
            try
            {
                fill(size_t(row));
            }
            catch (...)
            {
#pragma omp critical(DNDS_FillKnownRowsFailure)
                if (!failure)
                    failure = std::current_exception();
            }
        if (failure)
            std::rethrow_exception(failure);
    }

    /// CPU setup for known row sizes: allocate the final table/CSR once and
    /// initialize it under one writer. Unlike incremental RowBuilder this needs
    /// no allocation/controller per row. Callbacks must not retain borrowed rows;
    /// a failed fill never publishes an output. Every size is queried once.
    template <class T, class Size, class Fill>
    Array<T> BuildArrayRows(size_t rows, Size &&size, Fill &&fill, std::optional<size_t> width = {},
                            bool parallel = false)
    {
        if (width)
        {
            Array<T> result(RowLayout::Dense(rows, *width));
            auto write = result.Write();
            const auto view = write.View();
            FillKnownRows(rows, parallel, [&](size_t row)
                          {
                DNDS_check_throw_info(size(row) == *width, "known table row width mismatch");
                fill(row, view[row]); });
            return result;
        }
        std::vector<size_t> lengths(rows);
        FillKnownRows(rows, parallel, [&](size_t row)
                      { lengths[row] = size(row); });
        Array<T> result(RowLayout::Ragged(lengths));
        {
            auto write = result.Write();
            const auto view = write.View();
            FillKnownRows(rows, parallel, [&](size_t row)
                          { fill(row, view[row]); });
        }
        return result;
    }

    /// Fill an already known physical shape, preserving table stride, padded
    /// capacity and shared immutable row metadata. Only logical slots are filled.
    template <class T, class Fill>
    Array<T> BuildArrayRows(const RowLayout &shape, Fill &&fill, bool parallel = false)
    {
        Array<T> result(shape);
        auto write = result.Write();
        const auto view = write.View();
        FillKnownRows(size_t(shape.RowCount()), parallel, [&](size_t row)
                      { fill(row, view[index(row)]); });
        return result;
    }

    /// Exclusive CPU construction form. Distinct rows may be resized/filled by
    /// different workers; Commit follows their join. Borrowed row pointers expire
    /// on that row's resize or on Commit. No published Array aliases this draft.
    /// @pre Workers edit distinct rows; the caller joins them before Commit or
    /// destruction. No row pointer may escape its resize/publication boundary.
    /// A builder does not serialize concurrent writes to the same row.
    class RowBuilder
    {
        friend class ExclusiveRows;
        struct State
        {
            size_t rows, elementBytes, capacity, width;
            RowLayoutKind kind;
            Buffer lengths, dense;
            WriteAccess lengthAccess, denseAccess;
            size_t *sizes;
            struct Slab
            {
                Buffer buffer;
                std::atomic<size_t> used = 0;
                explicit Slab(Buffer storage) : buffer(std::move(storage)) {}
            };
            std::vector<std::shared_ptr<Slab>> slabs;
            std::atomic<Slab *> currentSlab = nullptr;
            std::vector<std::byte *> rowPointers;
            std::vector<size_t> rowCapacity;
            std::mutex slabMutex;

            State(size_t n, size_t bytes, RowLayoutKind layout, size_t cap, Buffer values = {}, WriteAccess access = {})
                : rows(n), elementBytes(bytes), capacity(cap), width(cap), kind(layout)
            {
                DNDS_check_throw_info(n <= size_t(INT64_MAX) && cap <= size_t(INT64_MAX),
                                      "row builder shape exceeds signed index range");
                DNDS_check_throw_info(bytes && (layout == RowLayoutKind::CSR || layout == RowLayoutKind::Padded || layout == RowLayoutKind::Table),
                                      "invalid row builder element size/layout");
                lengths = Buffer::Allocate(CheckedSize::Multiply(n, sizeof(size_t)));
                lengthAccess = lengths.Write();
                sizes = lengthAccess.View<size_t>().data;
                if (kind == RowLayoutKind::CSR)
                {
                    rowPointers.resize(n);
                    rowCapacity.resize(n);
                }
                else
                {
                    dense = values.Identity() ? std::move(values) : Buffer::Allocate(CheckedSize::Multiply(CheckedSize::Multiply(n, cap), bytes));
                    denseAccess = access.Active() ? std::move(access) : dense.Write();
                    if (kind == RowLayoutKind::Table && n)
                        std::fill_n(sizes, n, cap);
                }
            }
            std::byte *Pointer(size_t row) const
            {
                if (kind == RowLayoutKind::CSR)
                    return rowPointers[row];
                auto *base = denseAccess.View<std::byte>().data;
                return base ? base + row * capacity * elementBytes : nullptr;
            }
            static std::byte *ReserveInSlab(Slab &slab, size_t request)
            {
                auto *base = static_cast<std::byte *>(slab.buffer.LegacyPointer());
                size_t observed = slab.used.load(std::memory_order_relaxed);
                for (;;)
                {
                    const auto address = reinterpret_cast<uintptr_t>(base + observed);
                    const auto padding = (64 - address % 64) % 64;
                    const auto needed = CheckedSize::Add(padding, request);
                    if (needed > slab.buffer.Bytes() || observed > slab.buffer.Bytes() - needed)
                        return nullptr;
                    if (slab.used.compare_exchange_weak(observed, observed + needed,
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_relaxed))
                        return base + observed + padding;
                }
            }
            Slab *NewSlab(size_t wanted)
            {
                auto slab = std::make_shared<Slab>(Buffer::Allocate(wanted));
                auto *live = slab.get();
                slabs.push_back(std::move(slab));
                currentSlab.store(live, std::memory_order_release);
                return live;
            }
            std::byte *AllocateRow(size_t row, size_t bytes)
            {
                // CSR construction often creates thousands of tiny rows. One
                // allocation per node/cell dominates serial ingress and generic
                // topology setup, so carve stable row spans from large slabs.
                // Oversized rows receive a dedicated slab. A resize still
                // invalidates the old borrowed pointer, as documented above.
                constexpr size_t slabBytes = 64 * 1024;
                constexpr size_t alignment = 64;
                const size_t request = std::max(bytes, size_t(1));
                const size_t wanted = std::max(slabBytes, CheckedSize::Add(request, alignment));
                Slab *slab = currentSlab.load(std::memory_order_acquire);
                std::byte *pointer = slab ? ReserveInSlab(*slab, request) : nullptr;
                if (!pointer)
                {
                    // Distinct rows can be allocated concurrently. The common
                    // path reserves through the active slab's atomic frontier;
                    // the mutex is reached only when a new slab is published.
                    std::lock_guard<std::mutex> lock(slabMutex);
                    slab = currentSlab.load(std::memory_order_acquire);
                    if (slab)
                        pointer = ReserveInSlab(*slab, request);
                    if (!pointer)
                    {
                        slab = NewSlab(wanted);
                        pointer = ReserveInSlab(*slab, request);
                        DNDS_check_throw_info(pointer, "row builder could not reserve a slab row");
                    }
                }
                rowPointers[row] = pointer;
                rowCapacity[row] = bytes;
                return pointer;
            }
        };
        std::unique_ptr<State> _state;
        State &Live() const
        {
            DNDS_check_throw_info(_state, "row builder already committed or moved");
            return *_state;
        }
        explicit RowBuilder(std::unique_ptr<State> state) : _state(std::move(state)) {}

    public:
        RowBuilder(size_t rows, size_t elementBytes, RowLayoutKind kind = RowLayoutKind::CSR, size_t capacity = 0)
            : _state(std::make_unique<State>(rows, elementBytes, kind, capacity)) {}
        RowBuilder(const RowBuilder &) = delete;
        RowBuilder &operator=(const RowBuilder &) = delete;
        RowBuilder(RowBuilder &&) noexcept = default;
        RowBuilder &operator=(RowBuilder &&) noexcept = default;
        bool Active() const { return bool(_state); }
        /// Explicit detached edit: snapshot logical rows under one source lease.
        /// The original storage and all retained views remain unchanged. Pending
        /// source writes are rejected by the existing access contract.
        static RowBuilder FromStorage(const ArrayStorage &source)
        {
            const auto &shape = source.Layout();
            DNDS_check_throw_info(shape.IsContiguousRows(), "incremental row builder requires contiguous rows; convert explicitly");
            RowBuilder draft(shape.RowCount(), source.ElementBytes(), shape.Kind(),
                             shape.IsVariable() ? shape.Stride() : shape.Width());
            const auto device = source.Values().GetDevice();
            auto read = source.Read({device, nullptr});
            const auto *values = read.View<std::byte>().data;
            for (size_t i = 0; i < shape.RowCount(); ++i)
            {
                draft.ResizeRow(i, shape.RowSize(i), false);
                auto row = draft.RowBytes(i);
                if (row.size)
                    Detail::CopyBytes(values + shape.Offset(i) * source.ElementBytes(), device, row.data,
                                      Device::Host(), row.size, {device, nullptr})
                        .Wait();
            }
            return draft;
        }
        index RowCount() const { return index(Live().rows); }
        index Rows() const { return RowCount(); }
        size_t ElementBytes() const { return Live().elementBytes; }
        size_t RowSize(size_t row) const
        {
            const auto &s = Live();
            DNDS_check_throw_info(row < s.rows, "builder row out of range");
            return s.sizes[row];
        }
        void ResizeRow(size_t row, size_t length, bool preserve = true)
        {
            auto &s = Live();
            const auto previous = RowSize(row);
            DNDS_check_throw_info(length <= size_t(INT64_MAX), "builder row length exceeds signed index range");
            const auto bytes = CheckedSize::Multiply(length, s.elementBytes);
            DNDS_check_throw_info(s.kind == RowLayoutKind::CSR || length <= s.capacity, "row size exceeds padded capacity");
            DNDS_check_throw_info(s.kind != RowLayoutKind::Table || length == s.width, "table row width is fixed");
            if (s.kind == RowLayoutKind::CSR && bytes > s.rowCapacity[row])
            {
                auto *previousPointer = s.Pointer(row);
                auto *replacement = s.AllocateRow(row, bytes);
                if (preserve && previous)
                    std::memcpy(replacement, previousPointer, std::min(previous, length) * s.elementBytes);
            }
            else if (length && (!preserve || length > previous))
            {
                const auto start = preserve ? previous : 0;
                std::memset(s.Pointer(row) + start * s.elementBytes, 0, (length - start) * s.elementBytes);
            }
            s.sizes[row] = length;
        }
        Span<std::byte> RowBytes(size_t row) const
        {
            const auto n = RowSize(row);
            return {Live().Pointer(row), CheckedSize::Multiply(n, Live().elementBytes), MemorySpace::Host};
        }
        template <class T>
        Span<T> Row(size_t row) const
        {
            static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
            DNDS_check_throw_info(sizeof(T) == Live().elementBytes, "builder element size mismatch");
            auto bytes = RowBytes(row);
            DNDS_check_throw_info(reinterpret_cast<uintptr_t>(bytes.data) % alignof(T) == 0, "builder element alignment mismatch");
            return {reinterpret_cast<T *>(bytes.data), bytes.size / sizeof(T), MemorySpace::Host};
        }
        ArrayStorage::Shared Commit()
        {
            auto &s = Live();
            std::vector<size_t> lengths;
            if (s.rows)
                lengths.assign(s.sizes, s.sizes + s.rows);
            auto layout = s.kind == RowLayoutKind::CSR ? RowLayout::Ragged(lengths) : s.kind == RowLayoutKind::Padded ? RowLayout::Padded(lengths, s.capacity)
                                                                                                                      : RowLayout::Dense(s.rows, s.width, s.capacity);
            Buffer values = s.dense;
            if (s.kind == RowLayoutKind::CSR)
            {
                values = Buffer::Allocate(CheckedSize::Multiply(layout.Elements(), s.elementBytes));
                auto write = values.Write();
                auto *target = write.View<std::byte>().data;
                for (size_t i = 0; i < s.rows; ++i)
                    if (lengths[i])
                        std::memcpy(target + layout.Offset(i) * s.elementBytes, s.Pointer(i), lengths[i] * s.elementBytes);
            }
            auto result = ArrayStorage::FromBuffer(std::move(layout), std::move(values), s.elementBytes);
            s.denseAccess.Wait();
            s.lengthAccess.Wait();
            _state.reset();
            return result;
        }
    };

    /// A unique published Array enters exclusive setup state. Alias creation,
    /// normal access and payload binding fail until Commit. Metadata descriptors
    /// remain immutable; new sizes/offsets are published once. Padded/table values
    /// are edited in place; CSR values pack once. Abandonment marks shape invalid.
    class ExclusiveRows
    {
        ArrayStorage::Shared _source;
        std::unique_ptr<RowBuilder> _draft;
        WriteAccess _oldValues;

        void Abort() noexcept
        {
            if (!_source)
                return;
            try
            {
                auto source = std::move(_source);
                std::lock_guard<std::mutex> lock(source->_shapeMutex);
                source->_shapeValid = false;
                source->_editing = false;
                _draft.reset();
                _oldValues.Fail();
            }
            catch (...)
            {
                std::terminate();
            }
        }

    public:
        explicit ExclusiveRows(const ArrayStorage::Shared &source)
        {
            DNDS_check_throw_info(source, "exclusive edit requires storage");
            std::lock_guard<std::mutex> lock(source->_shapeMutex);
            source->CheckPublished();
            DNDS_check_throw_info(source->_layout->rows.IsContiguousRows(), "exclusive shape edit requires contiguous rows");
            DNDS_check_throw_info(source->_values.GetDevice() == Device::Host(), "exclusive row editing requires host storage");
            DNDS_check_throw_info(source.use_count() == 1 && source->_layout.use_count() == 1 &&
                                      source->_layout->rows.Offsets().IsUnique() && source->_layout->rows.Sizes().IsUnique() &&
                                      source->_values.IsUnique(),
                                  "exclusive edit requires unique unpinned storage and metadata; detach explicitly");
            auto write = source->_values.Write();
            try
            {
                const auto &shape = source->_layout->rows;
                const auto bytes = source->_elementBytes;
                if (shape.IsRagged())
                {
                    _draft = std::make_unique<RowBuilder>(shape.RowCount(), bytes);
                    auto *values = write.View<std::byte>().data;
                    for (size_t i = 0; i < shape.RowCount(); ++i)
                    {
                        _draft->ResizeRow(i, shape.RowSize(i));
                        auto row = _draft->RowBytes(i);
                        if (row.size)
                            std::memcpy(row.data, values + shape.Offset(i) * bytes, row.size);
                    }
                    _oldValues = std::move(write);
                }
                else
                {
                    auto state = std::make_unique<RowBuilder::State>(shape.RowCount(), bytes, shape.Kind(),
                                                                     shape.IsPadded() ? shape.Stride() : shape.Width(),
                                                                     source->_values, std::move(write));
                    // Preserve physical table stride, even when width < capacity.
                    state->capacity = shape.Stride();
                    state->width = shape.Width();
                    for (size_t i = 0; i < shape.RowCount(); ++i)
                        state->sizes[i] = shape.RowSize(i);
                    _draft.reset(new RowBuilder(std::move(state)));
                }
                _source = source;
                source->_editing = true;
            }
            catch (...)
            {
                write.CancelUnmodified();
                throw;
            }
        }
        ExclusiveRows(const ExclusiveRows &) = delete;
        ExclusiveRows &operator=(const ExclusiveRows &) = delete;
        ExclusiveRows(ExclusiveRows &&) noexcept = default;
        ExclusiveRows &operator=(ExclusiveRows &&other) noexcept
        {
            if (this != &other)
            {
                Abort();
                _source = std::move(other._source);
                _draft = std::move(other._draft);
                _oldValues = std::move(other._oldValues);
            }
            return *this;
        }
        ~ExclusiveRows() { Abort(); }
        RowBuilder &Draft() const
        {
            DNDS_check_throw_info(_source && _draft, "exclusive editor is inactive");
            return *_draft;
        }
        void Commit()
        {
            auto result = Draft().Commit();
            auto source = std::move(_source);
            std::lock_guard<std::mutex> lock(source->_shapeMutex);
            _oldValues.CancelUnmodified();
            source->_values = result->_values;
            source->_layout = result->_layout;
            ++source->_shapeVersion;
            source->_editing = false;
            _draft.reset();
        }
    };

    template <class T>
    ExclusiveRows ArrayHandle<T>::EditRows() { return ExclusiveRows(_storage); }
}
