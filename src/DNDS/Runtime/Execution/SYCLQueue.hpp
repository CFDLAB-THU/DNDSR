#pragma once

#include "DNDS/Runtime/Execution/SYCLExecution.hpp"
#include <sycl/sycl.hpp>

namespace DNDS::Runtime
{
    /// Backend kernel boundary. Obtain a queue handle sharing the Runtime
    /// lane. Submit before completing the access lease for these operands.
    sycl::queue SYCLQueue(const ExecutionContext &context);
}
