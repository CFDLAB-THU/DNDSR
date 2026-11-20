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
            const double seconds = std::chrono::duration<double>(Clock::now() - _start).count();
            std::fprintf(stderr, "CAUCHY_OUTPUT_PROFILE pid=%ld phase=%s seconds=%.9f bytes=%zu\n", long(getpid()), _phase, seconds, _bytes);
        }
    }
}
