#include "Cauchy/IO/OutputMeasurement.hpp"
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <unistd.h>

namespace DNDS::Cauchy
{
    OutputMeasurement::OutputMeasurement(const char *phase, size_t bytes) : _phase(phase), _bytes(bytes)
    {
        static const bool enabled = []
        {
            const auto *value = std::getenv("DNDS_CAUCHY_OUTPUT_PROFILE");
            return value && std::string_view(value) == "1";
        }();
        if (enabled)
            _start = Clock::now();
        else
            _phase = nullptr;
    }
    OutputMeasurement::~OutputMeasurement()
    {
        if (_phase)
        {
            const auto end = Clock::now();
            const double seconds = std::chrono::duration<double>(end - _start).count();
            const auto nanos = [](auto t)
            { return static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count()); };
            std::fprintf(stderr, "CAUCHY_OUTPUT_PROFILE pid=%ld phase=%s seconds=%.9f bytes=%zu start_ns=%lld end_ns=%lld\n", long(getpid()), _phase, seconds, _bytes, nanos(_start), nanos(end));
        }
    }
}
