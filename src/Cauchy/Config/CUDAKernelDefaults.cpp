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
    void ResolveAutomaticCUDAKernels(DomainSettings &settings, bool a100, bool withoutRANS)
    {
        auto &execution = settings.execution;
        auto &kernels = execution.kernels;
        const auto &numerics = settings.numerics;
        const bool acm = settings.equation.system == "incompressibleACM";
        const bool a100Polynomial = a100 && numerics.reconstruction == "variational" &&
                                    execution.layout == "aos" && execution.fieldGrouping == "coupled";
        const bool a100P3 = a100Polynomial && numerics.cfv.maxOrder == 3 && numerics.cfv.intOrder <= 5;
        const bool measuredFlow5 = a100P3 && !acm && withoutRANS && settings.mesh.dimension == 3;
        if (acm && kernels.rhsMapping == "auto")
        {
            const bool fp64 = kernels.polynomialArithmetic == "fp64" && kernels.dissipationArithmetic == "fp64" &&
                              kernels.fluxArithmetic == "fp64";
            // Low-work 3D profiles tie or favor thread-per-face. Require a
            // clear warmed-update gain before paying for CTA face packages.
            const int degree = numerics.cfv.maxOrder, rule = numerics.RHSIntegrationOrder();
            const bool acmCTA = a100Polynomial && settings.mesh.dimension == 3 && !numerics.firstOrderConvection &&
                                ((degree == 2 && rule >= 4) || (degree == 3 && rule >= 2));
            kernels.rhsMapping = fp64 && acmCTA ? "ctaSegments" : "thread";
        }
        if (acm && kernels.rhsReduction == "auto")
            kernels.rhsReduction = "serial";
        if (execution.reconstructionKernel == "auto")
        {
            // The matrix-shared selector remains a fixed comparison fixture.
            const bool measuredFlow4 = a100P3 && acm && withoutRANS;
            const bool usePipeline = (measuredFlow5 || measuredFlow4) &&
                                     !numerics.reconstructionSolver.UsesAMG() && numerics.limiter != "LVR";
            execution.reconstructionKernel = usePipeline ? "fp64Pipelined" : "fp64";
        }
        if (!kernels.polynomialPPThreads)
            kernels.polynomialPPThreads = measuredFlow5 ? 512 : 128;
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
