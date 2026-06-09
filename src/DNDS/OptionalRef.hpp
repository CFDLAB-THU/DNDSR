#pragma once
/// @file OptionalRef.hpp
/// @brief A lightweight non-owning nullable reference wrapper — safer than a
/// raw pointer for optional out/in-out parameters.

#include "Errors.hpp"
#include <cstddef>
#include <utility>

namespace DNDS
{

    template <typename T>
    class OptionalRef
    {
        T *mPtr = nullptr;

    public:
        OptionalRef() = default;

        OptionalRef(std::nullptr_t) {}

        explicit OptionalRef(T &ref) : mPtr(&ref) {}

        explicit operator bool() const { return mPtr != nullptr; }

        bool has_value() const { return mPtr != nullptr; }

        T &operator*() const
        {
            DNDS_assert(mPtr);
            return *mPtr;
        }

        T *operator->() const
        {
            DNDS_assert(mPtr);
            return mPtr;
        }

        T &value() const
        {
            DNDS_assert(mPtr);
            return *mPtr;
        }

        T *get() const { return mPtr; }

        void reset() { mPtr = nullptr; }

        template <typename U>
        void emplace(U &ref)
        {
            mPtr = &ref;
        }

        friend bool operator==(const OptionalRef &a, const OptionalRef &b)
        {
            return a.mPtr == b.mPtr;
        }
        friend bool operator!=(const OptionalRef &a, const OptionalRef &b)
        {
            return a.mPtr != b.mPtr;
        }
    };

} // namespace DNDS
