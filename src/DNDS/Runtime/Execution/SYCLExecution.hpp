#pragma once

#include "DNDS/Runtime/Storage/Buffer.hpp"

namespace DNDS::Runtime
{
    /// Select a local device by backend and per-backend ordinal. Runtime owns
    /// one USM context per device. Queues created for that device share it.
    enum class SYCLTarget
    {
        CPU,
        CUDA
    };
    Device SelectSYCLDevice(SYCLTarget target, int ordinal = 0);
    ExecutionContext CreateSYCLQueue(Device device);

    namespace SYCLDetail
    {
        class Event;
        std::shared_ptr<Event> Record(const ExecutionContext &context);
        void Wait(const Event &event);
        bool Ready(const Event &event);
        void Synchronize(const ExecutionContext &context);
        std::unique_ptr<AllocationResource> Allocate(size_t bytes, size_t alignment, Device device);
        void Copy(void *destination, const void *source, size_t bytes, const ExecutionContext &context);
    }
}
