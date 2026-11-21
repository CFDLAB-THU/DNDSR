#pragma once
#include "DNDS/Runtime/Storage/Buffer.hpp"

namespace DNDS::Runtime::Detail
{
    /// Out-of-place operations reserve the operands separately. Subranges of
    /// one allocation share an access controller, and foreign imports with
    /// different controllers may still cover the same physical bytes.
    inline bool IndependentBuffers(const Buffer &source, const Buffer &destination)
    {
        return source.Identity() && destination.Identity() && source.Identity() != destination.Identity() &&
               !source.Overlaps(destination);
    }

    /// An in-place provider reserves a single read/write operand. Its mutation
    /// must be published to every handle representing that operand.
    inline bool SameBufferOperand(const Buffer &source, const Buffer &destination)
    {
        return source.Identity() && source.Identity() == destination.Identity() && source.SameRange(destination);
    }
}
