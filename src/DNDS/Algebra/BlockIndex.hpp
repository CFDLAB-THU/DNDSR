#pragma once

#include "DNDS/Core/Defines.hpp"

namespace DNDS::Algebra
{
    /// Logical dense-block entry, independent of the surrounding Array layout.
    /// N=0 uses runtime block size; a static N removes that inner-loop degree
    /// of freedom. The enclosing row view/adapter validates entry bounds.
    template <index N = 0>
    DNDS_DEVICE_CALLABLE constexpr index DenseBlockEntry(index i, index j, index runtimeSize, bool rowMajor)
    {
        const index size = N ? N : runtimeSize;
        return rowMajor ? i * size + j : i + size * j;
    }
}
