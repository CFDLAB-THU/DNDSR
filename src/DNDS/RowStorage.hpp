#pragma once

#include "Errors.hpp"
#include <algorithm>
#include <memory>
#include <vector>

namespace DNDS
{
    /// Uncompressed CSR row with std::vector resize semantics and owning leases.
    /// Element access uses cached storage; structural mutation detaches a leased
    /// row. Copies remain deep copies, moves transfer the allocation.
    template <class T>
    class RowStorage
    {
        std::shared_ptr<std::vector<T>> storage;
        T *ptr = nullptr;
        size_t count = 0;

        void sync() noexcept
        {
            ptr = storage ? storage->data() : nullptr;
            count = storage ? storage->size() : 0;
        }
        void detach()
        {
            if (!storage)
                storage = std::make_shared<std::vector<T>>();
            else if (storage.use_count() != 1)
                storage = std::make_shared<std::vector<T>>(*storage);
            sync();
        }

    public:
        using value_type = T;
        RowStorage() = default;
        explicit RowStorage(size_t n, const T &value = T{}) : storage(std::make_shared<std::vector<T>>(n, value)) { sync(); }
        RowStorage(const RowStorage &other)
            : storage(other.storage ? std::make_shared<std::vector<T>>(*other.storage) : nullptr) { sync(); }
        RowStorage(RowStorage &&other) noexcept : storage(std::move(other.storage))
        {
            sync();
            other.sync();
        }
        RowStorage &operator=(const RowStorage &other)
        {
            if (this != &other)
            {
                RowStorage copy(other);
                swap(copy);
            }
            return *this;
        }
        RowStorage &operator=(RowStorage &&other) noexcept
        {
            if (this != &other)
            {
                storage = std::move(other.storage);
                sync();
                other.sync();
            }
            return *this;
        }
        void swap(RowStorage &other) noexcept
        {
            storage.swap(other.storage);
            sync();
            other.sync();
        }
        size_t size() const noexcept { return count; }
        T *data() noexcept { return ptr; }
        const T *data() const noexcept { return ptr; }
        T &at(size_t i)
        {
            DNDS_check_throw(i < count);
            return ptr[i];
        }
        const T &at(size_t i) const
        {
            DNDS_check_throw(i < count);
            return ptr[i];
        }
        void resize(size_t n)
        {
            if (n == count)
                return;
            detach();
            storage->resize(n);
            sync();
        }
        void reserve(size_t n)
        {
            if (storage && n <= storage->capacity())
                return;
            detach();
            storage->reserve(n);
            sync();
        }
        template <class Iterator>
        void assign(Iterator first, Iterator last)
        {
            auto replacement = std::make_shared<std::vector<T>>(first, last);
            storage = std::move(replacement);
            sync();
        }
        std::shared_ptr<T> lease() { return {storage, ptr}; }
        std::shared_ptr<const T> lease() const { return {storage, ptr}; }
    };
}
