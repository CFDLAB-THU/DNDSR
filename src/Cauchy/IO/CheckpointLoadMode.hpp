#pragma once

namespace DNDS::Cauchy
{
    enum class CheckpointLoadMode
    {
        /// Restore the saved clock and compatible history, including an active solve when present.
        Resume,
        /// Keep the clock and means. Use the active trial of an internal steady checkpoint.
        MeansOnly,
    };
}
