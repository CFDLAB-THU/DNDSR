#pragma once

#include "DNDS/Config/ConfigParam.hpp"

namespace DNDS::CFV
{
    /** @brief Launch dimensions for retained neighbor-buffered CUDA sweeps.
     * @details SIMT uses a block per cell; FP64 MMA uses a warp per cell.
     * Both stage each neighbor matrix once for all modes and components,
     * using two cp.async slots. Mapping/buffering are implementation contracts,
     * not independent configuration axes. Numerical weights are unchanged.
     */
    struct ReconstructionSweepLaunch
    {
        int threads = 0, gridBlocks = 0;
        DNDS_DECLARE_CONFIG(ReconstructionSweepLaunch)
        {
            DNDS_FIELD(threads, "Zero selects SIMT 32 / MMA 128; otherwise a multiple of 32 in [32,256]");
            DNDS_FIELD(gridBlocks, "Grid-stride block count; zero chooses a wide grid");
        }
        void Validate() const
        {
            DNDS_check_throw_info((threads == 0 || (threads >= 32 && threads <= 256 && threads % 32 == 0)) && gridBlocks >= 0,
                                  "invalid CUDA VR launch dimensions");
        }
    };
}
