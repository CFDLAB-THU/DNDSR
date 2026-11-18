#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/Runtime/Execution/Device.hpp"

namespace DNDS::Cauchy
{
    /// @brief Query the selected device during setup, never inside a queued kernel.
    bool IsA100Device(Runtime::Device device);

    /** @brief Resolve automatic policies before constructing resident operators.
     * @details Measured A100 P3 flow5 defaults use segmented CTA RHS, the
     * neighbor-pipelined FP64 SIMT VR sweep and warp-per-cell PP with 512
     * threads. Other FP64 shapes use the same CTA RHS catalog and general VR.
     * Measured A100 P3 ACM uses thread-per-face in 2D, CTA packages in 3D,
     * and the pipelined VR sweep for its four-component profile.
     * Reduced-precision RHS comparisons retain thread mapping. Explicit selectors
     * and launch dimensions are preserved; no storage or precision is changed.
     * The hardware flag is separate so policy tests need no CUDA device.
     */
    void ResolveAutomaticCUDAKernels(DomainSettings &settings, bool a100, bool withoutRANS);

    /// Resolve the automatic inverse provider once when binding a linear solver.
    std::string ResolveAutomaticCUDAInverse(const std::string &selection, Runtime::Device device);
}
