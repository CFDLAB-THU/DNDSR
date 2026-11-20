#include "DNDS/Runtime/Storage/AllocationDiagnostics.hpp"
#include "DNDS/Core/CheckedSize.hpp"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#if defined(__unix__) || defined(__APPLE__)
#    include <unistd.h>
#endif

namespace DNDS::Runtime
{
    namespace
    {
        struct TrackedAllocation
        {
            AllocationRecord record;
            std::function<AllocationActivity()> activity;
        };
        struct Registry
        {
            std::atomic<bool> enabled{false};
            std::atomic<bool> captureStacks{false};
            std::mutex mutex;
            std::unordered_map<uint64_t, TrackedAllocation> records;
            size_t live = 0, peak = 0;
        };
        Registry &GetRegistry()
        {
            // Process-lifetime diagnostic metadata avoids static teardown order
            // dependencies with late Array owners. No payload is retained here.
            static Registry *registry = new Registry;
            return *registry;
        }
    }

    void AllocationDiagnostics::Enable(bool enabled)
    {
        auto &registry = GetRegistry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (!enabled)
        {
            registry.records.clear();
            registry.live = registry.peak = 0;
            registry.captureStacks.store(false, std::memory_order_relaxed);
        }
        else if (const auto *value = std::getenv("DNDS_RUNTIME_ALLOCATION_STACKS"))
            registry.captureStacks.store(std::strcmp(value, "1") == 0, std::memory_order_relaxed);
        registry.enabled.store(enabled, std::memory_order_release);
    }
    void AllocationDiagnostics::CaptureStacks(bool enabled)
    {
        GetRegistry().captureStacks.store(enabled, std::memory_order_release);
    }
    bool AllocationDiagnostics::Enabled() noexcept
    {
        return GetRegistry().enabled.load(std::memory_order_acquire);
    }
    void Detail::TrackAllocation(uint64_t identity, size_t bytes, Device device,
                                 bool external, bool streamOrdered, bool pinnedHost,
                                 std::function<AllocationActivity()> activity)
    {
        auto &registry = GetRegistry();
        if (!registry.enabled.load(std::memory_order_acquire))
            return;
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (!registry.enabled.load(std::memory_order_relaxed))
            return;
        AllocationRecord record;
        record.identity = identity;
        record.bytes = bytes;
        record.device = device;
        record.externalProvider = external;
        record.streamOrderedMemory = streamOrdered;
        record.pinnedHostMemory = pinnedHost;
        if (registry.captureStacks.load(std::memory_order_acquire))
            record.creationStack = DNDS::getTraceString();
        const auto live = CheckedSize::Add(registry.live, bytes);
        const auto inserted = registry.records.emplace(identity, TrackedAllocation{std::move(record), std::move(activity)}).second;
        if (inserted)
        {
            registry.live = live;
            registry.peak = std::max(registry.peak, live);
        }
    }
    void Detail::ForgetAllocation(uint64_t identity) noexcept
    {
        auto &registry = GetRegistry();
        if (!registry.enabled.load(std::memory_order_acquire))
            return;
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = registry.records.find(identity);
        if (found != registry.records.end())
        {
            registry.live -= found->second.record.bytes;
            registry.records.erase(found);
        }
    }
    void Detail::NameAllocation(uint64_t identity, const std::string &name, AllocationSite site)
    {
        auto &registry = GetRegistry();
        if (!registry.enabled.load(std::memory_order_acquire))
            return;
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = registry.records.find(identity);
        if (found == registry.records.end())
            return;
        auto &record = found->second.record;
        if (!name.empty() && std::find(record.names.begin(), record.names.end(), name) == record.names.end())
            record.names.push_back(name);
        if (record.file.empty() && site.file && *site.file)
        {
            record.file = site.file;
            record.function = site.function ? site.function : "";
            record.line = site.line;
        }
    }
    AllocationSnapshot AllocationDiagnostics::Snapshot()
    {
        auto &registry = GetRegistry();
        std::vector<TrackedAllocation> copied;
        AllocationSnapshot result;
        {
            std::lock_guard<std::mutex> lock(registry.mutex);
            result.liveBytes = registry.live;
            result.peakBytes = registry.peak;
            copied.reserve(registry.records.size());
            for (const auto &entry : registry.records)
                copied.push_back(entry.second);
        }
        // Do not hold registry and allocation mutexes together. A snapshot is a
        // local observation; concurrent allocation/activity changes may follow it.
        result.allocations.reserve(copied.size());
        for (auto &entry : copied)
        {
            if (entry.activity)
                entry.record.activity = entry.activity();
            if (entry.record.pinnedHostMemory)
                result.pinnedHostBytes = CheckedSize::Add(result.pinnedHostBytes, entry.record.bytes);
            result.allocations.push_back(std::move(entry.record));
        }
        std::sort(result.allocations.begin(), result.allocations.end(), [](const auto &a, const auto &b)
                  { return a.bytes != b.bytes ? a.bytes > b.bytes : a.identity < b.identity; });
        return result;
    }
    void AllocationSnapshot::Dump(std::ostream &output) const
    {
        output << "Runtime allocation ownership: live_bytes=" << liveBytes << " peak_bytes=" << peakBytes
               << " pinned_host_bytes=" << pinnedHostBytes;
#if defined(__unix__) || defined(__APPLE__)
        output << " pid=" << ::getpid();
#endif
        output << '\n';
        for (const auto &entry : allocations)
        {
            output << entry.identity << ' ' << (entry.device.space == MemorySpace::SYCL ? "sycl:" : entry.device.space == MemorySpace::CUDA ? "cuda:"
                                                                                                                                            : "host:")
                   << entry.device.ordinal << " bytes=" << entry.bytes
                   << " provider=" << (entry.externalProvider ? "external" : "native")
                   << " stream_ordered=" << entry.streamOrderedMemory
                   << " pinned_host=" << entry.pinnedHostMemory
                   << " pending_reads=" << entry.activity.pendingReaders
                   << " pending_writes=" << entry.activity.pendingWriters;
            for (const auto &name : entry.names)
                output << " name=" << name;
            if (!entry.file.empty())
                output << " site=" << entry.file << ':' << entry.line << ' ' << entry.function;
            output << '\n';
            if (!entry.creationStack.empty())
                output << "allocation_stack identity=" << entry.identity << '\n'
                       << entry.creationStack << "end_allocation_stack\n";
        }
    }
}
