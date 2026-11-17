#pragma once

#include "DNDS/Algebra/AmgXPreconditioner.hpp"
#include <string>

namespace DNDS::Algebra
{
    /// Compact portable diagnostic object, excluding raw pointers/allocations.
    /// Generating the report does not synchronize; obtaining its provider
    /// snapshot is the explicit diagnostic completion boundary.
    std::string AmgXMemoryReport(const AmgXMemorySnapshot &);
}
