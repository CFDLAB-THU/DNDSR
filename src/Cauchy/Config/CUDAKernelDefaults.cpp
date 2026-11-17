#include "Cauchy/Config/CUDAKernelDefaults.hpp"
#ifdef DNDS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif

namespace DNDS::Cauchy
{
    bool IsA100Device(Runtime::Device device)
    {
        if (device.space != Runtime::MemorySpace::CUDA)
            return false;
#ifdef DNDS_USE_CUDA
        cudaDeviceProp properties{};
        const auto status = cudaGetDeviceProperties(&properties, device.ordinal);
        DNDS_check_throw_info(status == cudaSuccess, "cannot resolve CUDA kernel defaults for selected device");
        return properties.major == 8 && properties.minor == 0 &&
               std::string(properties.name).find("A100") != std::string::npos;
#else
        DNDS_check_throw_info(false, "CUDA kernel defaults require a CUDA build");
        return false;
#endif
    }
    void ResolveAutomaticCUDAKernels(DomainSettings &settings, bool a100, bool flow5)
    {
        auto &execution = settings.execution;
        auto &kernels = execution.kernels;
        const auto &numerics = settings.numerics;
        const bool measuredShape = a100 && flow5 && settings.mesh.dimension == 3 &&
                                   numerics.reconstruction == "variational" && numerics.cfv.maxOrder == 3 &&
                                   numerics.cfv.intOrder <= 5 && execution.layout == "aos" &&
                                   execution.fieldGrouping == "coupled";
        if (kernels.rhsMapping == "auto")
        {
            const bool fp64 = kernels.polynomialArithmetic == "fp64" && kernels.dissipationArithmetic == "fp64" &&
                              kernels.fluxArithmetic == "fp64";
            // The consolidated FP64 catalog has one mapping for every valid
            // C/M/D shape. This is capability selection; only measuredShape
            // carries the A100 performance evidence used by other defaults.
            kernels.rhsMapping = fp64 ? "ctaSegments" : "thread";
        }
        if (kernels.rhsReduction == "auto")
            kernels.rhsReduction = kernels.rhsMapping == "ctaSegments" ? "segmented" : "serial";
        if (execution.reconstructionKernel == "auto")
            // Only the measured A100 case is promoted automatically. The
            // matrix-shared selector is now a fixed comparison fixture, not a
            // general fallback for another shape/physics/storage profile.
            execution.reconstructionKernel = !numerics.reconstructionSolver.UsesAMG() && numerics.limiter != "LVR" && measuredShape ? "fp64Pipelined" : "fp64";
        if (!kernels.polynomialPPThreads)
            kernels.polynomialPPThreads = measuredShape ? 512 : 128;
    }
    std::string ResolveAutomaticCUDAInverse(const std::string &selection, Runtime::Device device)
    {
        if (selection != "auto")
            return selection;
        // Guarded LU uses the existing condition guard and filtered SVD
        // fallback. It is an implementation choice, not a new inverse policy.
        return IsA100Device(device) ? "guardedLU" : "batchedSVD";
    }
}
