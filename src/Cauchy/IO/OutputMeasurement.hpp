#pragma once

#include <chrono>
#include <cstddef>

namespace DNDS::Cauchy
{
    /// Optional rank-local wall time. Capture and file work may overlap on
    /// different threads, so elapsed times are not additive solver costs.
    class OutputMeasurement
    {
        using Clock = std::chrono::steady_clock;
        const char *_phase;
        size_t _bytes;
        Clock::time_point _start;

    public:
        explicit OutputMeasurement(const char *phase, size_t bytes = 0);
        ~OutputMeasurement();
        OutputMeasurement(const OutputMeasurement &) = delete;
        OutputMeasurement &operator=(const OutputMeasurement &) = delete;
    };
}
