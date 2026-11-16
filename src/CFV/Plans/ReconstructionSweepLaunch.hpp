#pragma once

#include "DNDS/Config/ConfigParam.hpp"

namespace DNDS::CFV
{
    /** @brief Explicit CUDA VR scan mapping and staging policy.
     * @details A neighbor stage holds one complete modal matrix and all
     * component coefficients. A cellRow stage holds every incidence of the
     * next grid-stride cell. Neither policy reloads a matrix per output tile.
     * These choices affect execution, never the reconstruction functional.
     */
    struct ReconstructionSweepLaunch
    {
        std::string mapping = "block", buffering = "neighbor";
        int threads = 128, gridBlocks = 0;
        bool asynchronous = true;
        DNDS_DECLARE_CONFIG(ReconstructionSweepLaunch)
        {
            DNDS_FIELD(mapping, "block or warp per cell; tensor arithmetic requires warp");
            DNDS_FIELD(buffering, "Double-buffer one neighbor or a complete cellRow");
            DNDS_FIELD(threads, "Threads per block, a multiple of 32 in [32,256]");
            DNDS_FIELD(gridBlocks, "Grid-stride block count; zero chooses a wide grid");
            DNDS_FIELD(asynchronous, "Use cp.async to overlap staging with the current product");
        }
        void Validate() const
        {
            DNDS_check_throw_info(mapping == "block" || mapping == "warp", "unknown CUDA VR cell mapping");
            DNDS_check_throw_info(buffering == "neighbor" || buffering == "cellRow", "unknown CUDA VR buffering scope");
            DNDS_check_throw_info(threads >= 32 && threads <= 256 && threads % 32 == 0 && gridBlocks >= 0,
                                  "invalid CUDA VR launch dimensions");
        }
    };
}
