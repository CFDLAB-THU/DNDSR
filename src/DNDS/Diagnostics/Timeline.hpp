#pragma once
#include <cstdint>

namespace DNDS
{
    /** @brief Opt-in Caliper region for standalone reports and CUDA correlation.
     * @details DNDS_TIMELINE=1 enables annotations at startup. Scopes measure host
     * execution/submission, not GPU completion, and introduce no synchronization.
     * Builds without Caliper are no-ops. A scope stays on its creating thread; payloads
     * identify steps, stages, iterations or row selections at both region boundaries. No execution/lifetime contract relies
     * on this diagnostic. Names must remain valid through scope destruction.
     */
    class TimelineScope
    {
        bool _active = false;
        const char *_name = nullptr;
        std::uint64_t _attribute = ~std::uint64_t(0);

    public:
        explicit TimelineScope(const char *name, std::int64_t payload = -1) noexcept;
        ~TimelineScope();
        TimelineScope(const TimelineScope &) = delete;
        TimelineScope &operator=(const TimelineScope &) = delete;
        TimelineScope(TimelineScope &&) = delete;
        TimelineScope &operator=(TimelineScope &&) = delete;
    };
}
