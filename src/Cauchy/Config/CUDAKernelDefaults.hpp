#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/Runtime/Execution/Device.hpp"

namespace DNDS::Cauchy
{
    /// @brief Query the selected device during setup, never inside a queued kernel.
    bool IsA100Device(Runtime::Device device);

    /** @brief Resolve automatic policies before constructing resident operators.
     * @details Measured A100 P3 flow5 defaults use the neighbor-pipelined
     * FP64 SIMT VR sweep and warp-per-cell PP with 512 threads.
     * RHS selection is resolved separately when its workset is bound.
     * The hardware flag is separate so policy tests need no CUDA device.
     */
    void ResolveAutomaticCUDAKernels(DomainSettings &settings, bool a100, bool flow5);

    /// Resolve the automatic inverse provider once when binding a linear solver.
    std::string ResolveAutomaticCUDAInverse(const std::string &selection, Runtime::Device device);
}
