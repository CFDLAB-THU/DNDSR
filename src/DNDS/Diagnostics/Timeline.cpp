#include "DNDS/Diagnostics/Timeline.hpp"
#include <cstdlib>
#ifdef DNDS_USE_CALIPER
#    include <caliper/cali.h>
#    include <caliper/Caliper.h>
#endif

namespace DNDS
{
    TimelineScope::TimelineScope(const char *name, std::int64_t payload) noexcept
    {
#ifdef DNDS_USE_CALIPER
        static const bool enabled = []
        {
            const auto *value = std::getenv("DNDS_TIMELINE");
            return value && value[0] == '1' && value[1] == '\0';
        }();
        _active = enabled;
        if (_active)
        {
            _name = name;
            cali_begin_region(name);
            // Per-region typed context avoids confusing a step number with an
            // iteration number in a nested report. Attribute lookup is cached
            // by Caliper; identifiers remain valid through process lifetime.
            if (payload >= 0)
            {
                _attribute = cali_create_attribute(name, CALI_TYPE_INT,
                                                   CALI_ATTR_ASVALUE | CALI_ATTR_SCOPE_THREAD);
                auto caliper = cali::Caliper::instance();
                caliper.begin(caliper.get_attribute(_attribute),
                              cali::Variant(CALI_TYPE_INT, &payload, sizeof(payload)));
            }
        }
#else
        (void)name;
        (void)payload;
#endif
    }
    TimelineScope::~TimelineScope()
    {
#ifdef DNDS_USE_CALIPER
        if (_active)
        {
            if (_attribute != CALI_INV_ID)
                cali_end(_attribute);
            cali_end_region(_name);
        }
#endif
    }
}
