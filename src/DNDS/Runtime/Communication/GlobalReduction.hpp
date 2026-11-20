#pragma once

#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Runtime
{
    /// A collective bound to specific Array allocations during coordinated setup.
    /// Submit on the communication stream in identical collective order across ranks.
    /// Use stream events for producer and consumer dependencies. Dropping the binding
    /// after submission is safe; the engine retains the participating resources.
    class GlobalReductionBinding
    {
    public:
        virtual ~GlobalReductionBinding() = default;
        virtual StreamEvent Submit(EagerEngine &engine, const ExecutionStream &stream) const = 0;
    };
}
